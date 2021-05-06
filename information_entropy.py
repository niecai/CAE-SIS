import cv2
import numpy as np
import math
import pickle
from secretProcessing import *


def comp_ent(data, base):
    record = [0] * 256

    for x in data:
        if x:
            for j in range(math.ceil(base / 8)):
                record[x % 256] += 1
                x //= 256

    e = 0
    s = sum(record)
    for j in range(256):
        p = record[j] / s
        if p > 0:
            e -= p * math.log(p, 2.0)
    return e


data = cv2.imread('testdata/lena.png', 0)
share_list = decomposition(data.flatten(), 3, 4, direc_generation=False, encryption_method='SMIESIS')

es = 0.0
es += comp_ent(reconstruction([share_list[0], share_list[1]], encryption_method='SMIESIS'), 8)
es += comp_ent(reconstruction([share_list[0], share_list[2]], encryption_method='SMIESIS'), 8)
es += comp_ent(reconstruction([share_list[0], share_list[3]], encryption_method='SMIESIS'), 8)
es += comp_ent(reconstruction([share_list[1], share_list[2]], encryption_method='SMIESIS'), 8)
es += comp_ent(reconstruction([share_list[1], share_list[3]], encryption_method='SMIESIS'), 8)
es += comp_ent(reconstruction([share_list[2], share_list[3]], encryption_method='SMIESIS'), 8)
print(es / 6)