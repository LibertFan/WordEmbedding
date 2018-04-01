import os
import shutil
import time
import torch as th
import numpy as np
import torchvision as tv
import torch.nn as nn
from torch import FloatTensor
from torch.autograd import Variable
import torch.optim as optim


class SkipGramTrainer(object):
    def __init__(self, opts, streamer, model):
        self._options = opts
        self._streamer = streamer
        self._model = model
        if th.cuda.is_available():
            self._model.cuda()
        print(model.parameters(), opts.learning_rate)
        self._optimizer = optim.SGD(model.parameters(), lr=opts.learning_rate)
        self.save_name = opts.data_set_name + '_' + opts.graph_name + '_' + opts.tag \
            if opts.tag is not None else opts.graph_name
        self.util_file = os.path.join(opts.util_dir, self.save_name + '_' +
                                      time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())))
        self.log_dir = os.path.join(opts.log_dir, self.save_name)
        print('Log directory: {}'.format(self.log_dir))
        if opts.is_training:
            if os.path.exists(self.log_dir):
                print('Log directory exists! Delete it ?')
                command = bool(eval(input()))
                if command:
                    shutil.rmtree(self.log_dir)
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        # Setting log
        self.save_dir = os.path.join(opts.save_dir, self.save_name)
        print('Save directory: {}'.format(self.save_dir))
        if opts.is_training:
            if os.path.exists(self.save_dir):
                print('Save directory exists! Delete it ?')
                command = bool(eval(input()))
                if command:
                    shutil.rmtree(self.save_dir)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

    def adjust_learning_rate(self, e):
        opts, optimizer = self._options, self._optimizer
        learning_rate = opts.learning_rate * np.power(opts.learning_rate_factor, int((e+1)/500))
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

    def train(self):
        opts, streamer, model, optimizer = \
            self._options, self._streamer, self._model, self._optimizer
        for e in range(opts.epochs):
            self.adjust_learning_rate(e)
            i_words, o_words, n_words = streamer.get_next_batch()
            if np.mod(e, opts.display_every) == 0:
                self.visualize(i_words, o_words, n_words)
            i_words = Variable(th.LongTensor(i_words))
            o_words = Variable(th.LongTensor(o_words))
            n_words = Variable(th.LongTensor(n_words))
            if th.cuda.is_available():
                i_words = i_words.cuda()
                o_words = o_words.cuda()
                n_words = n_words.cuda()
            optimizer.zero_grad()
            loss = model(i_words, o_words, n_words)
            loss.backward()
            optimizer.step()
            if np.mod(e+1, opts.display_every) == 0:
                print('epoch: {}, loss: {}'.format(e, loss.data))
            if np.mod(e+1, opts.save_every) == 0:
                model.save_embedding(os.path.join(self.save_dir, 'word2vec-epoch-{}.txt'.format(e)),
                                     streamer.get_id2word())
                print('Model of epoch {} has been saved!'.format(e+1))

    def visualize(self, i_words, o_words, n_words):
        id2word = self._streamer.get_id2word()
        batch_size = i_words.shape[0]
        index = np.random.randint(0, batch_size-1)
        i_word, o_word, n_word = i_words[index], o_words[index], n_words[index]
        ei_word, eo_word, en_word = [], [], []
        for word in i_word:
            ei_word.append(id2word[word])
        for word in o_word:
            eo_word.append(id2word[word])
        for word in n_word:
            en_word.append(id2word[word])
        print('Index: {}|| i_word: {}|| o_words: {}|| n_words: {}'.
              format(index, ' '.join(ei_word), ' '.join(eo_word), ' '.join(en_word)))

    def test(self):
        pass
