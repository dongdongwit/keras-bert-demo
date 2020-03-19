#! -*- coding:utf-8 -*-
import keras
import keras_bert
import pandas as pd
import numpy as np


class LossHistory(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.losses = []

    def on_train_begin(self, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append([logs.get('loss'), logs.get('acc'),
                            logs.get('val_loss'), logs.get('val_acc')])


class OurTokenizer(keras_bert.Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        return R


def parser_excel(excel_path, class_num, label1_id, label2_id):
    train_excel = pd.read_excel(excel_path, header=None, skiprows=[0])
    label1 = []
    for row in train_excel[1]:
        label1.append(label1_id[row])
    label2 = []
    for row in train_excel[2]:
        label2.append(label2_id[row])
    data = []
    for index in range(len(train_excel[4])):
        data.append((train_excel[4][index], np.eye(class_num[0])[label1[index]],
                     np.eye(class_num[1])[label2[index]]))
    return data


def get_class_weight(label_list, class_num):
    class_weight = len(label_list) / np.histogram(label_list, class_num)[0]
    return class_weight
