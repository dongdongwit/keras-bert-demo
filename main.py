#! -*- coding:utf-8 -*-

import codecs
import pickle

from data_generator import data_generator
from model import my_model
from utiles import *

maxlen = 100
config_path = 'pre_model/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'pre_model/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'pre_model/chinese_L-12_H-768_A-12/vocab.txt'

with open("data/label1_id.pkl", 'rb') as f:
    label1_id = pickle.load(f)
with open("data/label2_id.pkl", 'rb') as f:
    label2_id = pickle.load(f)
class_num = [len(label1_id), len(label2_id)]

train_data = parser_excel("data/data_clean_label1_train.xlsx", class_num, label1_id, label2_id)
valid_data = parser_excel("data/data_clean_label1_valid.xlsx", class_num, label1_id, label2_id)
label1_list = [np.argmax(d[1]) for d in train_data]
label2_list = [np.argmax(d[2]) for d in train_data]
multi_class_weight = [get_class_weight(label1_list, class_num[0]), get_class_weight(label2_list, class_num[1])]
single_class_weight = get_class_weight(label2_list, class_num[1])

token_dict = {}
with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)
train_gen = data_generator(train_data, token_dict)
valid_gen = data_generator(valid_data, token_dict)

if __name__ == "__main__":
    history = LossHistory()
    filepath = 'trained_model/my_model.h5'
    ModelCheckpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, verbose=1)
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
    tbCallBack = keras.callbacks.TensorBoard(log_dir='logs/label1',  # log 目录
                             histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                             # batch_size=32,     # 用多大量的数据计算直方图
                             write_graph=True,  # 是否存储网络结构图
                             write_grads=True,  # 是否可视化梯度直方图
                             write_images=True,  # 是否可视化参数
                             embeddings_freq=0,
                             embeddings_layer_names=None,
                             embeddings_metadata=None)
    # single_model = my_model(config_path, checkpoint_path, class_num, trainable=True).get_single_model()
    # single_model.fit_generator(train_gen.__iter__(mod="single"), steps_per_epoch=len(train_gen),
    #                     callbacks=[history, ModelCheckpoint, tbCallBack], class_weight=single_class_weight,
    #                     verbose=1, epochs=100, validation_data=valid_gen.__iter__(mod="single"),
    #                     validation_steps=len(valid_gen))

    multi_model = my_model(config_path, checkpoint_path, class_num, trainable=True).get_multi_model()
    multi_model.fit_generator(train_gen.__iter__(mod="multi"), steps_per_epoch=len(train_gen),
                              callbacks=[history, ModelCheckpoint, tbCallBack], class_weight=multi_class_weight,
                              verbose=1, epochs=100, validation_data=valid_gen.__iter__(mod="multi"),
                              validation_steps=len(valid_gen))

