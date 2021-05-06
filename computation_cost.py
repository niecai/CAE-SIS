import cv2
from model import *
from secretProcessing import *
import pickle
import time


def img2share(img, k, n, share_road, direc_generation=False, encryption_method=None,
              feature_map=64, encoded_shape=6, img_name=None, pre_train_model=None,
              ks=23, step=15):
    data = cv2.imread(img, 0)
    AE = AutoEncoder((data.shape[0], data.shape[1], 1), encoded_shape,
                     feature_map=feature_map, ks=ks, step=step)
    AE.autoencoder.summary()

    start = time.time()

    feature = AE.train(data[np.newaxis, :, :, np.newaxis],
                       my_epochs=5000, pre_train_model=pre_train_model)

    reconv_kernel, reconv_bias = AE.autoencoder.layers[-1].get_weights()
    data = np.concatenate([feature.flatten(), reconv_bias, reconv_kernel.flatten()], axis=0)
    share_list = decomposition(data, k, n,
                               direc_generation=direc_generation,
                               encryption_method=encryption_method,
                               feature=len(data))
    # for i, s in enumerate(share_list):
    #     with open(share_road + "{}_share{}".format(img_name, i), 'wb') as f:
    #         pickle.dump(s, f)
    return time.time() - start


def share2img(share, img_shape, encoded_shape, encryption_method=None, output="xxx.jpg",
              feature_map=64, ks=23, step=15):
    share_list = []
    for file in share:
        with open(file, 'rb') as f:
            data = pickle.load(f)
            share_list.append(data)
    start = time.time()
    reconstruction_data = reconstruction(share_list, encryption_method=encryption_method)
    res = time.time() - start

    mid_feature = (img_shape[0] // step) * (img_shape[1] // step) * feature_map
    feature = reconstruction_data[:mid_feature].reshape((1, img_shape[0] // step, img_shape[1] // step, feature_map))

    reconv_bias = reconstruction_data[mid_feature:mid_feature + 1]
    reconv_kernel = reconstruction_data[mid_feature + 1:].reshape((ks, ks, 1, feature_map))


    AE = AutoEncoder((img_shape[0], img_shape[1], 1), encoded_shape, feature_map=feature_map, ks=ks, step=step)

    AE.autoencoder.layers[-1].set_weights([reconv_kernel, reconv_bias])
    start = time.time()
    img = AE.share_model.predict(feature)
    # cv2.imwrite(output, np.asarray(img, dtype=np.uint8)[0])
    return res + time.time() - start


fea_map = 25
fea = 3
img_name = 'lena'
pre_train_model = 'pre_train_loss2.h5'
ks = 16
step = 16

cae_share = img2share('testdata/{}.png'.format(img_name), 3, 4, 'share_result/', direc_generation=True,
                      encryption_method='AE', feature_map=fea_map, encoded_shape=fea, img_name=img_name,
                      pre_train_model=pre_train_model, ks=ks, step=step)
data = ['share_result/loss2_{}_share0'.format(img_name),
        'share_result/loss2_{}_share1'.format(img_name),
        'share_result/loss2_{}_share2'.format(img_name),
        'share_result/loss2_{}_share3'.format(img_name)]

cae_rec = share2img(data[:3], (128, 128), fea, encryption_method='AE', output='test.png',
                    feature_map=fea_map, ks=ks, step=step)

data = cv2.imread('testdata/lena.png', 0)
start = time.time()
share_list = decomposition(data.flatten(), 3, 4, direc_generation=False, encryption_method='SMIESIS')
smie_share = time.time() - start

start = time.time()
img = reconstruction(share_list[:3], encryption_method='SMIESIS')
smie_rec = time.time() - start

print(cae_share, cae_rec, smie_share, smie_rec)


