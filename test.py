import cv2
import h5py
import pickle
import numpy as np
import random

from secretProcessing import *
# from model import *

# lena = cv2.imread('lena512.bmp', 0)
# lena = cv2.resize(lena, (128, 128))
# cv2.imwrite("testdata/lena.png", lena)
#
# Baboon = cv2.imread('dataset/4.2.03.tiff')
# Baboon = cv2.cvtColor(Baboon, cv2.COLOR_BGR2GRAY)
# Baboon = cv2.resize(Baboon, (128, 128))
# cv2.imwrite("testdata/Baboon.png", Baboon)
#
# house = cv2.imread('dataset/house.tiff')
# house = cv2.cvtColor(house, cv2.COLOR_BGR2GRAY)
# house = cv2.resize(house, (128, 128))
# cv2.imwrite("testdata/house.png", house)
#
# earth = cv2.imread('dataset/2.1.11.tiff')
# earth = cv2.cvtColor(earth, cv2.COLOR_BGR2GRAY)
# earth = cv2.resize(earth, (128, 128))
# cv2.imwrite("testdata/earth.png", earth)
#
#
# # dense_2
# # dense_2/dense_2
# # dense_2/dense_2/bias:0
# # dense_2/dense_2/kernel:0
# # input_2
# parameter = h5py.File('AE.h5', 'r')
#
#
# def show(data):
#     print('name ---', data.name)
#     if 'value' in dir(data):
#         # 有值的话直接打出
#         # print(data.shape)
#
#         print(data)
#     else:
#         # 是一个group的话则继续深入
#         for k in data:
#             show(data[k])
#
#
# print(parameter['conv2d_transpose_1']['conv2d_transpose_1']['bias:0'][()])
# print(parameter['conv2d_transpose_1']['conv2d_transpose_1']['kernel:0'][()])
# print(parameter['dense_2']['dense_2']['bias:0'][()])
# print(parameter['dense_2']['dense_2']['kernel:0'][()])
#
# with open('share_result/share0', 'rb') as f:
#     data = pickle.load(f)
# print(data.shape)

# sw = np.asarray([[0,0,0,1,1,1,0,0], [0,1,1,0,0,1,0,1], [1,1,0,1,0,0,1,1], [1,0,1,0,1,0,1,0]])
# t = np.asarray([1, 2, 4, 8, 16, 32, 64, 128])[:, np.newaxis]
# brep = (np.matmul(sw, t))
#
# sf = np.asarray([[0,0,0,1,1,1,0], [0,1,1,0,0,1,0], [1,1,0,1,0,0,1], [1,0,1,0,1,0,1]])
# sfin = sf * 255
# sfin[:, np.asarray([False, True, False, True, True, False, False], dtype=np.bool)] = brep
# data = np.asarray([129, 24, 194,  5,  82,  241,  149])
#
# for i in range(4):
#     print(np.bitwise_and(data, sfin[i]))


# def print_keras_wegiths(weight_file_path):
#     f = h5py.File(weight_file_path)  # 读取weights h5文件返回File类
#     try:
#         if len(f.attrs.items()):
#             print("{} contains: ".format(weight_file_path))
#             print("Root attributes:")
#         for key, value in f.attrs.items():
#             print("  {}: {}".format(key, value))  # 输出储存在File类中的attrs信息，一般是各层的名称
#
#         for layer, g in f.items():  # 读取各层的名称以及包含层信息的Group类
#             print("  {}".format(layer))
#             print("    Attributes:")
#             for key, value in g.attrs.items(): # 输出储存在Group类中的attrs信息，一般是各层的weights和bias及他们的名称
#                 print("      {}: {}".format(key, value))
#
#     finally:
#         f.close()
#
#
# print_keras_wegiths('AE.h5')
#
# parameter = h5py.File('AE.h5', 'r')
# reconv_bias = parameter['transconv2d']['transconv2d']['bias:0'][()]
# reconv_kernel = parameter['transconv2d']['transconv2d']['kernel:0'][()]
# dense_bias = parameter['decode']['decode']['bias:0'][()]
# dense_kernel = parameter['decode']['decode']['kernel:0'][()]

def get_random_list(l):
    res = list(range(l))
    for i in range(l - 1, 0, -1):
        x = random.randint(0, i - 1)
        res[x], res[i] = res[i], res[x]
    return res


data = np.asarray([5, 8, 7, 16])[:, np.newaxis]

print(data[get_random_list(4), :])
