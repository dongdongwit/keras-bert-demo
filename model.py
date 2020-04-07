#! -*- coding:utf-8 -*-
from keras.layers import *
from keras.models import Model
from keras_bert import load_trained_model_from_checkpoint, calc_train_steps, AdamWarmup
from keras.optimizers import Adam
from keras import regularizers


class my_model:
    def __init__(self, config_path, checkpoint_path, class_num, trainable=False):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.class_num = class_num
        self.trainable = trainable

    def get_multi_model(self, weight=None):
        bert_model = load_trained_model_from_checkpoint(self.config_path, self.checkpoint_path, seq_len=None)
        le = 1e-3
        if self.trainable:
            le = 1e-4
            for l in bert_model.layers:
                l.trainable = True
        x1_in = Input(shape=(None,))
        x2_in = Input(shape=(None,))

        x = bert_model([x1_in, x2_in])
        x = Lambda(lambda x: x[:, 0])(x)
        if not self.trainable:
            x = Dense(128, activation='relu', name="fc1", kernel_regularizer=regularizers.l2(0.01))(x)
            x = Dropout(0.3)(x)
        predict1 = Dense(self.class_num[0], activation='softmax', name="label1")(x)
        predict2 = Dense(self.class_num[1], activation='softmax', name="label2")(x)

        model = Model([x1_in, x2_in], [predict1, predict2], name="multi_model")
        model.summary()
        if weight != None:
            print('loading pre_train weight...')
            model.load_weights(weight, by_name=True)
            print('Done!')

        model.compile(
            loss=['categorical_crossentropy', 'categorical_crossentropy'],
            optimizer=Adam(le),  # 用足够小的学习率
            metrics=['accuracy', 'accuracy']
        )

        return model

    def get_single_model(self, weight=None):
        bert_model = load_trained_model_from_checkpoint(self.config_path, self.checkpoint_path, seq_len=None)
        le = 1e-3
        if self.trainable:
            le = 1e-4
            for l in bert_model.layers:
                l.trainable = True
        x1_in = Input(shape=(None,))
        x2_in = Input(shape=(None,))

        x = bert_model([x1_in, x2_in])
        x = Lambda(lambda x: x[:, 0])(x)
        if not self.trainable:
            x = Dense(128, activation='relu', name="fc1", kernel_regularizer=regularizers.l2(0.01))(x)
            x = Dropout(0.3)(x)
        predict = Dense(self.class_num[1], activation='softmax', name="label")(x)

        model = Model([x1_in, x2_in], predict, name="single_model")
        model.summary()
        if weight != None:
            print('loading pre_train weight...')
            model.load_weights(weight, by_name=True)
            print('Done!')
        model.compile(loss=['categorical_crossentropy'],
                      optimizer=Adam(le),  # 用足够小的学习率
                      metrics=['accuracy'])

        return model
