import os
import sys
import pickle
import numpy as np
import time
import argparse


class Streamer(object):
    def __init__(self, opts):
        self._options = opts
        self._unk = '<unk>'
        self._freq_words, self._word2id, self._id2word, raw_data = self._load_data()
        self._freq_words, self._word2id, self._id2word, self._uniform_p, \
            self._sub_sampling_p, self._neg_sampling_p = self._update_words_info()
        opts.vocab_size = len(self._word2id)
        self._i_words, self._o_words = self._formalize(raw_data)
        print('Shape of i_words and o_words are : {} and {} respectively'.
              format(self._i_words.shape, self._o_words.shape))
        self._train_size = len(self._i_words)
        self._train_iter = -1

    def get_id2word(self):
        return self._id2word

    def get_word2id(self):
        return self._word2id

    def _load_data(self):
        opts = self._options
        with open(opts.data_path, 'rb') as f:
            data = pickle.load(f)
        f.close()
        return data

    def _formalize(self, data):
        opts, word2id, unk = \
            self._options, self._word2id, self._unk
        unk_num = word2id.get(unk)
        max_num = 0

        def _formalize_parallel(sentence):
            pad_sentences = []
            pad_sentence = [unk_num] * opts.window_size + sentence + [unk_num] * opts.window_size
            for i in range(2*opts.window_size+1):
                if i == opts.window_size:
                    continue
                else:
                    start_idx, end_idx = i, i + len(sentence)
                    pad_sentences.append(pad_sentence[start_idx: end_idx])
            sentence_si_words = np.expand_dims(np.array(sentence), axis=-1)
            sentence_so_words = np.array(pad_sentences).transpose([1, 0])
            return sentence_si_words, sentence_so_words

        word_num = len(word2id)
        i_words, o_words = [], []
        t1 = time.time()
        for t, sentence in enumerate(data):
            si_words, so_words = _formalize_parallel(sentence)
            si_words = np.array(np.array(si_words) < word_num).astype(np.int32) * si_words
            so_words = np.array(np.array(so_words) < word_num).astype(np.int32) * so_words
            if max_num < np.max(si_words):
                max_num = np.max(si_words)
            for si, so in zip(si_words, so_words):
                i_words.append(si)
                o_words.append(so)
        t2 = time.time()
        t = t2 - t1
        print('Total time: {}'.format(t))
        word_num = len(word2id)
        i_words = np.array(i_words)
        o_words = np.array(o_words)
        max_i_word, max_o_word = np.max(i_words), np.max(o_words)
        print('Max num: {},  word_num is {}, Maximum number in i_words and o_words are {} and {} respectively'.
              format(max_num, word_num, max_i_word, max_o_word))
        return i_words, o_words

    def _update_words_info(self):
        opts, freq_words, word2id, id2word = \
            self._options, self._freq_words, self._word2id, self._id2word
        # Get the number of reserved word
        if opts.num_units is not None:
            num_units = opts.num_units
        elif opts.min_count is not None:
            num_units = -1
            for i, (word, freq) in enumerate(freq_words):
                if freq < opts.min_count:
                    num_units = i
                    break
        else:
            num_units = len(freq_words)
        # Get the nre frequent words
        clip_freq_words = freq_words[:num_units]
        # Summarize number of word in data
        word_sum, unk_sum = 0, 0
        for i, (word, freq) in enumerate(freq_words):
            word_sum += freq
            if i >= num_units:
                unk_sum += freq
        for i in range(num_units, len(freq_words)):
            word = id2word.get(i)
            del word2id[word]
            del id2word[i]
        clip_freq_words = [(self._unk, unk_sum)] + clip_freq_words
        word2id[self._unk] = 0
        id2word[0] = self._unk
        assert len(word2id) == len(id2word) == len(clip_freq_words)
        print('Total number of word: {}, length of word2id: {}, that of id2word: {}, '
              'length of clip_freq_words: {}'.
              format(word_sum, len(word2id), len(id2word), len(clip_freq_words)))

        uniform_p = []
        for i, (word, freq) in enumerate(clip_freq_words):
            uniform_p.append(float(freq) / float(word_sum))
        uniform_p = np.array(uniform_p)
        print('The shape pf uniform_p: {}'.format(uniform_p.shape))
        sub_sampling_p = 1 - np.sqrt(opts.sub_sampling_p_factor/uniform_p)
        sub_sampling_p = np.clip(sub_sampling_p, 0.0, 1.0)
        neg_sampling_p = np.power(uniform_p, opts.neg_sampling_p_factor)
        neg_sampling_p = neg_sampling_p / np.sum(neg_sampling_p)
        print('The shape of sub_sampling_p: {}, that of neg_sampling_p: {}'.
              format(sub_sampling_p.shape, neg_sampling_p.shape))
        return clip_freq_words, word2id, id2word, uniform_p, sub_sampling_p, neg_sampling_p

    def _update_data(self, data):
        opts, word2id = self._options, self._word2id
        for line in data:
            for i, v in enumerate(line):
                n = word2id.get(v, 0)
                line[i] = n
        return data

    def get_next_batch(self):
        opts, word2id, sub_sampling_p, neg_sampling_p = \
            self._options, self._word2id, self._sub_sampling_p, self._neg_sampling_p
        if self._train_iter == -1:
            index = list(range(len(self._i_words)))
            np.random.shuffle(index)
            self._i_words = self._i_words[index]
            self._o_words = self._o_words[index]
        self._train_iter += 1
        train_i_words, train_o_words, train_neg_words = [], [], []
        batch_num = 0
        while batch_num < opts.batch_size and self._train_iter < self._train_size:
            i_word = self._i_words[self._train_iter][0]
            if np.random.random() < sub_sampling_p[i_word]:
                train_i_words.append(self._i_words[self._train_iter])
                train_o_words.append(self._o_words[self._train_iter])
                batch_num += 1
            self._train_iter += 1
        if self._train_iter >= self._train_size:
            return self.get_next_batch()
        else:
            word_num = len(neg_sampling_p)
            train_neg_samples = np.random.choice(a=list(range(word_num)),
                p=neg_sampling_p, replace=True,
                size=[opts.batch_size, opts.neg_sampling_num*2*opts.window_size])
            train_i_words = np.array(train_i_words)
            train_o_words = np.array(train_o_words)
            return train_i_words, train_o_words, train_neg_samples

    def score(self):
        pass


def main():
    opts = read_commands()
    t1 = time.time()
    streamer = Streamer(opts)
    t2 = time.time()
    t = t2 - t1
    print('Total time is : {}'.format(t))
    for i in range(1000):
        for data in streamer.get_next_batch():
            print(type(data))
            print(data.shape)


def read_commands():
    parser = argparse.ArgumentParser(usage='Train word2vec model!')
    root = os.path.abspath('../..')
    data_root = os.path.join(root, 'Data')
    parser.add_argument('--model', type=str, default=os.path.join(data_root, 'util'))
    parser.add_argument('--data_path', type=str, default=os.path.join(data_root, 'train/imdb_words_parallel.pkl'))
    parser.add_argument('--save_dir', type=str, default=os.path.join(data_root, 'model'))
    parser.add_argument('--log_dir', type=str, default=os.path.join(data_root, 'log'))
    parser.add_argument('--util_dir', type=str, default=os.path.join(data_root, 'util'))
    parser.add_argument('--neg_sampling_num', type=int, default=10)
    parser.add_argument('--neg_sampling_p_factor', type=float, default=0.75)
    parser.add_argument('--sub_sampling_p_factor', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--window_size', type=int, default=5)
    parser.add_argument('--num_units', type=int, default=None)
    parser.add_argument('--min_count', type=int, default=5)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
