#! -*- coding:utf-8 -*-

import jieba
import pandas as pd
import fasttext
from random import shuffle

def get_fasttext_data(excel_path, label_id):
    excel_data = pd.read_excel(excel_path, header=None, skiprows=[0])
    label_list = []
    for row in excel_data[3]:
        label_list .append(label_id[row])
    sequence = []
    for index in range(len(excel_data[4])):
        text = excel_data[4][index].split("\\n")[0]
        seg_test = jieba.cut(text)
        outline = " ".join(seg_test)
        outline = outline + "\t__label__" + str(label_list[index]) + "\n"
        sequence.append(outline)
    return sequence

format_id = {"Liquid": 0, "Powder": 1, "Bar": 2, "Capsule": 3}
train_data = get_fasttext_data("data/data_clean_label1_train.xlsx", format_id)
valid_data = get_fasttext_data("data/data_clean_label1_valid.xlsx", format_id)

def save_data(data_path, list_format_data):
    with open(data_path, "w", encoding="utf-8") as f:
        shuffle(list_format_data)
        for data in list_format_data:
            f.write(data)

train_path = "data/data_clean_fasttext_train.txt"
valid_path = "data/data_clean_fasttext_valid.txt"
# save_data(train_path, train_data)
# save_data(valid_path, valid_data)
classifier = fasttext.train_supervised(train_path, label_prefix="__label__")

result = classifier.test(valid_path)
print('precision：', result[1])



