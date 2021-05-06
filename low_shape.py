import cv2
from model import *
from secretProcessing import *
import h5py
import pickle


def img2share(img, k, n, share_road, direc_generation=False, encryption_method=None,
              feature_map=64, encoded_shape=6, img_name=None, pre_train_model=None,
              ks=23, step=15):
    data = cv2.imread(img, 0)
    AE = AutoEncoder((data.shape[0], data.shape[1], 1), encoded_shape,
                     feature_map=feature_map, ks=ks, step=step)
    AE.autoencoder.summary()

    feature = AE.train(data[np.newaxis, :, :, np.newaxis],
                       my_epochs=50000, pre_train_model=pre_train_model)
    # parameter = h5py.File('AE.h5', 'r')
    # reconv_bias = parameter['transconv2d']['transconv2d']['bias:0'][()]
    # reconv_kernel = parameter['transconv2d']['transconv2d']['kernel:0'][()]
    # dense_bias = parameter['decode']['decode']['bias:0'][()]
    # dense_kernel = parameter['decode']['decode']['kernel:0'][()]
    # parameter.close()

    reconv_kernel, reconv_bias = AE.autoencoder.layers[-1].get_weights()
    # dense_kernel, dense_bias = AE.autoencoder.layers[-3].get_weights()
    data = np.concatenate([feature.flatten(), reconv_bias, reconv_kernel.flatten()], axis=0)
    share_list = decomposition(data, k, n,
                               direc_generation=direc_generation,
                               encryption_method=encryption_method,
                               feature=len(data))
    for i, s in enumerate(share_list):
        with open(share_road + "loss2_{}_share{}".format(img_name, i), 'wb') as f:
            pickle.dump(s, f)

    print(feature)


def share2img(share, img_shape, encoded_shape, encryption_method=None, output="xxx.jpg",
              feature_map=64, ks=23, step=15):
    share_list = []
    for file in share:
        with open(file, 'rb') as f:
            data = pickle.load(f)
            share_list.append(data)
    reconstruction_data = reconstruction(share_list, encryption_method=encryption_method)

    mid_feature = (img_shape[0] // step) * (img_shape[1] // step) * feature_map
    feature = reconstruction_data[:mid_feature].reshape((1, img_shape[0] // step, img_shape[1] // step, feature_map))

    # dense_bias = reconstruction_data[encoded_shape:encoded_shape + mid_feature]
    #
    # r = encoded_shape + mid_feature * (encoded_shape + 1)
    # dense_kernel = reconstruction_data[encoded_shape + mid_feature:r].reshape((encoded_shape, mid_feature))

    reconv_bias = reconstruction_data[mid_feature:mid_feature + 1]
    reconv_kernel = reconstruction_data[mid_feature + 1:].reshape((ks, ks, 1, feature_map))

    # parameter = h5py.File('AE.h5', 'r+')
    # parameter['transconv2d']['transconv2d']['bias:0'][()][...] = reconv_bias
    # parameter['transconv2d']['transconv2d']['kernel:0'][()][...] = reconv_kernel
    # parameter['decode']['decode']['bias:0'][()][...] = dense_bias
    # parameter['decode']['decode']['kernel:0'][()][...] = dense_kernel
    # parameter.close()

    AE = AutoEncoder((img_shape[0], img_shape[1], 1), encoded_shape, feature_map=feature_map, ks=ks, step=step)
    # AE.decoder.load_weights('AE.h5')
    AE.autoencoder.layers[-1].set_weights([reconv_kernel, reconv_bias])
    # AE.autoencoder.layers[-3].set_weights([dense_kernel, dense_bias])
    img = AE.share_model.predict(feature)
    cv2.imwrite(output, np.asarray(img, dtype=np.uint8)[0])


fea_map = 25
fea = 3
img_name = 'lena'
pre_train_model = 'pre_train_loss2.h5'
ks = 16
step = 16

train_x = []
root = 'pre_traindata/'
for p in os.listdir(root):
    data = cv2.imread(root + p, 0)
    train_x.append(data)
train_x = np.asarray(train_x)[:, :, :, np.newaxis]

AE = AutoEncoder((128, 128, 1), fea, feature_map=fea_map, ks=ks, step=step)
AE.autoencoder.summary()
AE.pre_train(train_x, my_epochs=5000, model_name=pre_train_model)

for x in AE.encoder.predict(train_x):
    print(x)

img2share('testdata/{}.png'.format(img_name), 3, 4, 'share_result/', direc_generation=True,
          encryption_method='AE', feature_map=fea_map, encoded_shape=fea,
          img_name=img_name, pre_train_model=pre_train_model, ks=ks, step=step)
data = ['share_result/loss2_{}_share0'.format(img_name),
        'share_result/loss2_{}_share1'.format(img_name),
        'share_result/loss2_{}_share2'.format(img_name),
        'share_result/loss2_{}_share3'.format(img_name)]
ch = []
num = 0


def choose(now, n):
    global num, img_name
    if n == 4:
        if now != 0 and now != 4:
            share2img(ch, (128, 128), fea, encryption_method='AE',
                      output='reconsitution/{}/loss2_{}_{}.png'.format(now, img_name, num),
                      feature_map=fea_map, ks=ks, step=step)
            print('play {}_loss2 in {}'.format(num, str(ch)))
            num += 1
    else:
        choose(now, n + 1)
        ch.append(data[n])
        choose(now + 1, n + 1)
        ch.pop()


choose(0, 0)
