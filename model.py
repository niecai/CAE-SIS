import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import Callback
import numpy as np
import random


class Evaluate(Callback):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def on_epoch_end(self, epoch, logs=None):
        if self.evaluate():
            print("Epoch %d: early stopping" % epoch)
            self.model.stop_training = True

    def evaluate(self):
        predict = np.asarray(self.model.predict(self.x), dtype=np.uint8)
        return (predict == self.y).all()


class AutoEncoder:
    def __init__(self, input_shape, encoded_shape, feature_map=64, ks=23, step=15):
        mid_feature = (input_shape[0] // step) * (input_shape[1] // step) * feature_map
        input_img = Input(shape=input_shape)

        conv = Conv2D(feature_map, (ks, ks), strides=(step, step),
                      activation='relu', input_shape=input_shape,
                      name='conv2d')(input_img)
        fla = Flatten()(conv)

        encoded = Dense(encoded_shape, activation='relu', name='encoded')(fla)

        decoded = Dense(mid_feature, activation='relu', name='decode')(encoded)
        resha = Reshape((input_shape[0] // step, input_shape[1] // step, feature_map))(decoded)
        res = Conv2DTranspose(1, (ks, ks), strides=(step, step), activation='relu', name='transconv2d')(resha)
        self.autoencoder = Model(input_img, res)

        self.encoder = Model(input_img, encoded)

        encoded_input = Input(shape=(encoded_shape, ))
        decoder_layer0 = self.autoencoder.layers[-3]
        decoder_layer1 = self.autoencoder.layers[-2]
        decoder_layer2 = self.autoencoder.layers[-1]
        self.decoder = Model(encoded_input, decoder_layer2(decoder_layer1(decoder_layer0(encoded_input))))
        self.first_AE = Model(input_img, decoder_layer2(conv))
        self.feature = encoded_shape

        self.predict_model = Model(input_img, resha)
        feature_map_input = Input(shape=(input_shape[0] // step, input_shape[1] // step, feature_map))
        self.share_model = Model(feature_map_input, decoder_layer2(feature_map_input))

    def train(self, data, my_epochs=100, pre_train_model='pre_train.h5'):
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        evaluator = Evaluate(data, data)
        self.autoencoder.compile(optimizer=adam, loss='mse')
        self.autoencoder.load_weights(pre_train_model)

        self.autoencoder.fit(data, data, epochs=my_epochs, shuffle=True, verbose=2, callbacks=[evaluator])

        encode_code = self.encoder.predict(data)
        have_fea = set()
        for i in range(self.feature):
            if encode_code[0][i] != 0.:
                have_fea.add(i)

        if len(have_fea) == self.feature:
            return self.predict_model.predict(data)

        # we = self.autoencoder.layers[-3].get_weights()
        # for i in range(self.feature):
        #     if i not in have_fea:
        #         updata = np.zeros(we[0][i, :].shape)
        #         for j in have_fea:
        #             p_j = random.random() * encode_code[0][j]
        #             encode_code[0][i] += p_j
        #             updata += we[0][j, :] * p_j
        #             encode_code[0][j] -= p_j
        #         we[0][i, :] = updata / encode_code[0][i]
        # self.autoencoder.layers[-3].set_weights(we)
        #
        # self.decoder.compile(optimizer=adam, loss='mse')
        # self.decoder.summary()
        # evaluator = Evaluate(encode_code, data)
        # self.decoder.fit(encode_code, data, epochs=my_epochs,
        #                  shuffle=True, verbose=2, callbacks=[evaluator])
        # self.decoder.save_weights('AE.h5')
        return self.predict_model.predict(data)

    def pre_train(self, data, my_epochs=100, model_name=None):
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

        self.first_AE.compile(optimizer=adam, loss='mse')
        self.first_AE.summary()
        self.first_AE.fit(data, data, epochs=my_epochs // 4, shuffle=True, verbose=0)

        self.autoencoder.layers[-1].trainable = False
        self.autoencoder.layers[-6].trainable = False
        self.autoencoder.compile(optimizer=adam, loss='mse')
        self.autoencoder.summary()
        self.autoencoder.fit(data, data, epochs=my_epochs // 4, shuffle=True, verbose=0)

        self.autoencoder.layers[-1].trainable = True
        self.autoencoder.layers[-6].trainable = True
        self.autoencoder.compile(optimizer=adam, loss='mse')
        self.autoencoder.summary()
        self.autoencoder.fit(data, data, epochs=my_epochs, shuffle=True, verbose=2)

        comp = [0.] * self.feature
        for x in self.encoder.predict(data):
            for i in range(self.feature):
                comp[i] += x[i]
        index = sorted(list(range(self.feature)), key=lambda x: -comp[x])
        best_i = set(index[:1])
        for i in range(1, self.feature):
            if comp[index[i]] < comp[index[i - 1]] / 10:
                break
            best_i.add(index[i])

        if len(best_i) < 1:
            print('Random initialization failed.')
            return
        elif len(best_i) == self.feature:
            self.autoencoder.save_weights(model_name)
            return
        else:
            print('Start processing features.')

        for x in self.encoder.predict(data):
            print(x)

        we = self.autoencoder.layers[-4].get_weights()
        for i in range(self.feature):
            if i not in best_i:
                updata = np.zeros(we[0][:, 0].shape)
                for j in best_i:
                    updata += random.random() * we[0][:, j]
                we[0][:, i] = updata
        self.autoencoder.layers[-4].set_weights(we)

        self.autoencoder.layers[-4].trainable = False
        self.autoencoder.layers[-6].trainable = False
        self.autoencoder.compile(optimizer=adam, loss='mse')
        self.autoencoder.summary()
        self.autoencoder.fit(data, data, epochs=my_epochs, shuffle=True, verbose=0)

        self.autoencoder.layers[-4].trainable = True
        self.autoencoder.layers[-6].trainable = True
        self.autoencoder.compile(optimizer=adam, loss='mse')
        self.autoencoder.summary()
        self.autoencoder.fit(data, data, epochs=my_epochs, shuffle=True, verbose=0)

        self.autoencoder.save_weights(model_name)

