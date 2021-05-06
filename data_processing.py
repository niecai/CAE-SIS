import os
import cv2

root = 'dataset/'
for i, f in enumerate(os.listdir(root)):
    data = cv2.imread('dataset/{}'.format(f))
    data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    data = cv2.resize(data, (128, 128))
    cv2.imwrite("pre_traindata/{}.png".format(i), data)

lena = cv2.imread('lena512.bmp', 0)
lena = cv2.resize(lena, (128, 128))
cv2.imwrite("pre_traindata/210.png", lena)
