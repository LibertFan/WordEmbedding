import os
import argparse
import collections
import numpy as np
import pickle
import multiprocessing
import time
import nltk
data_root = os.path.abspath('../../Data')


class BuildVocab(object):
    def __init__(self, opts):
        self._options = opts
        self._unk = '<unk>'
        data = self._load_data(opts.data_path)
        self._freq_words, self._word2id, self._id2word = self._build_vocab(data)
        # raw_i_words, raw_o_words = self._formalize(data)
        # print(type(raw_i_words), type(raw_o_words))
        # print(raw_i_words.shape, raw_o_words.shape)
        self._encode_data = self._word_encode(data)
        save_data = [self._freq_words, self._word2id, self._id2word, self._encode_data]
        self._save_pickle_data(opts.save_path, save_data)

    @staticmethod
    def _load_data(data):
        corpus = []
        with open(data, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                words = nltk.tokenize.word_tokenize(line.strip().lower())
                corpus.append(words)
        print('Load data finish!')
        return corpus

    def _build_vocab(self, data):
        # Collect most common word in data
        raw_word_freq = collections.Counter()
        for v in data:
            raw_word_freq.update(v)
        sort_freq_words = sorted(raw_word_freq.items(), key=lambda x: x[1], reverse=True)
        word2id = dict()
        id2word = dict()
        word_sum = 0
        for i, (word, freq) in enumerate(sort_freq_words):
            word2id[word] = i+1
            id2word[i+1] = word
            word_sum += freq
        print('Number of words: {}'.format(word_sum))
        print('Demonstration:')
        print('Length of word2id, id2word are {} and {} respectively'.
              format(len(word2id), len(id2word)))
        for i in range(10):
            raw_num = i * 1000 + i
            word = id2word.get(raw_num)
            num = word2id.get(word)
            print(i, raw_num, word, num)
        return sort_freq_words, word2id, id2word

    def _word_encode(self, data):
        opts, word2id = self._options, self._word2id
        encode_data = []
        unk_num = word2id.get(self._unk, 0)
        print('UNK is: {}'.format(unk_num))
        for sentence in data:
            encode_sentence = []
            for w in sentence:
                n = word2id.get(w, unk_num)
                encode_sentence.append(n)
            encode_data.append(encode_sentence)
        return encode_data

    def _formalize(self, data):
        opts, word2id, unk = \
            self._options, self._word2id, self._unk

        def _formalize_parallel(self, sentence):
            opts, word2id, unk = \
                self._options, self._word2id, self._unk
            si_words, so_words = [], []
            pad_sentence = [unk] * opts.window_size + sentence + [unk] * opts.window_size
            for i, w in enumerate(sentence):
                si_words.append([sentence[i]])
                so_word = pad_sentence[i: i + opts.window_size] + \
                          pad_sentence[i + opts.window_size + 1: i + 2 * opts.window_size]
                so_words.append(so_word)
            return si_words, so_words

        i_words, o_words = [], []
        t1 = time.time()
        for sentence in data:
            si_words, so_words = _formalize_parallel(sentence)
            i_words.extend(si_words), o_words.extend(so_words)
        t2 = time.time()
        t = t2 - t1
        print('Total time: {}'.format(t))
        return np.array(i_words), np.array(o_words)

    @staticmethod
    def _save_pickle_data(path, data):
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        f.close()


def test():
    opts = read_commands()
    vocab = BuildVocab(opts)


def read_commands():
    parser = argparse.ArgumentParser(usage='Pre-processing data set')
    root = os.path.abspath('../..')
    raw_root = os.path.join(root, 'Data')
    parser.add_argument('--data_path', type=str, default=os.path.join(raw_root, 'raw/imdb.txt'))
    parser.add_argument('--save_path', type=str, default=os.path.join(raw_root, 'train/imdb_words.pkl'))
    parser.add_argument('--window_size', type=int, default=5)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    test()
