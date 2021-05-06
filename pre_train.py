import cv2
import os
from model import *


fea = 3
fea_map = 64
pre_train_model = 'pre_train.h5'
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
