import cv2
from model import *
from secretProcessing import *
import pickle


def share2img(share, img_shape, encoded_shape, encryption_method=None, output="xxx.jpg",
              feature_map=64, ks=23, step=15):
    share_list = []
    for file in share:
        with open(file, 'rb') as f:
            data = pickle.load(f)
            share_list.append(data)
    fake_shape = np.random.randint(0, 1 << 32, size=(len(share_list[0])), dtype='uint32')
    share_list.append(fake_shape)
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


fea_map = 64
fea = 3
ks = 16
step = 16
img_name = 'lena'
ch = ['share_result/{}_share0'.format(img_name),
      'share_result/{}_share1'.format(img_name),
      'share_result/{}_share2'.format(img_name)
      ]
share2img(ch, (128, 128), fea, encryption_method='AE',
                      output='fake_image.png',
                      feature_map=fea_map, ks=ks, step=step)
