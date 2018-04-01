import torch as th
import torchvision as tv
import torch.nn as nn
import torch.nn.functional as F


class SkipGram(nn.Module):
    def __init__(self, opts):
        super(SkipGram, self).__init__()
        self._options = opts
        self.i_emb_mat = nn.Embedding(opts.vocab_size, opts.emb_dim, sparse=True)
        self.o_emb_mat = nn.Embedding(opts.vocab_size, opts.emb_dim, sparse=True)
        self._init_emb()

    def _init_emb(self):
        opts = self._options
        if not isinstance(opts.emb_dim, int):
            raise Exception('argument `emb_dim` is not defined here!')
        upper = 0.5 / opts.emb_dim
        lower = -upper
        self.i_emb_mat.weight.data.uniform_(lower, upper)
        self.o_emb_mat.weight.data.uniform_(-0, 0)

    def forward(self, i_words, o_words, n_words):
        batch_size, sent = i_words.size(0), i_words.size(1)
        embed_i = self.i_emb_mat(i_words)
        embed_o = self.o_emb_mat(o_words)
        embed_o = th.transpose(embed_o, 1, 2)
        score = th.matmul(embed_i, embed_o).squeeze(dim=1)
        log_target = F.logsigmoid(score).sum(dim=1)
        neg_embed_o = self.o_emb_mat(n_words)
        neg_embed_o = th.transpose(neg_embed_o, 1, 2)
        neg_score = th.matmul(embed_i, neg_embed_o).squeeze(dim=1)
        sum_log_sampled = F.logsigmoid(-1 * neg_score).sum(dim=1)
        loss = log_target + sum_log_sampled
        loss = -1 * loss.sum(dim=0) / (batch_size * sent)
        return loss

    def input_embeddings(self):
        return self.i_embeddings.weight.data.cpu().numpy()

    def save_embedding(self, file_name, id2word):
        embeds = self.i_embeddings.weight.data
        with open(file_name, 'w') as f:
            for idx in range(len(embeds)):
                word = id2word(idx)
                embed = ' '.join(embeds[idx])
                f.write(word + ' ' + embed + '\n')
        f.close()
