#! -*- coding:utf-8 -*-
import numpy as np
from utiles import OurTokenizer


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


class data_generator:
    def __init__(self, data, token_dict, maxlen=100, batch_size=32):
        self.data = data
        self.tokenizer = OurTokenizer(token_dict)
        self.maxlen = maxlen
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self, mod="multi"):
        if mod == "multi":
            while True:
                idxs = list(range(len(self.data)))
                np.random.shuffle(idxs)
                we, pe, label, label2 = [], [], [], []
                for i in idxs:
                    d = self.data[i]
                    text = d[0][:self.maxlen]
                    word_emb, position_emb = self.tokenizer.encode(first=text)
                    we.append(word_emb)
                    pe.append(position_emb)
                    label.append(d[1])
                    label2.append(d[2])
                    if len(we) == self.batch_size or i == idxs[-1]:
                        we = seq_padding(we)
                        pe = seq_padding(pe)
                        yield [we, pe], [label, label2]
                        we, pe, label, label2 = [], [], [], []

        else:
            while True:
                idxs = list(range(len(self.data)))
                np.random.shuffle(idxs)
                we, pe, label = [], [], []
                for i in idxs:
                    d = self.data[i]
                    text = d[0][:self.maxlen]
                    word_emb, position_emb = self.tokenizer.encode(first=text)
                    we.append(word_emb)
                    pe.append(position_emb)
                    # attention! process different label
                    label.append(d[2])
                    if len(we) == self.batch_size or i == idxs[-1]:
                        we = seq_padding(we)
                        pe = seq_padding(pe)
                        yield [we, pe], [label]
                        we, pe, label = [], [], []
