# coding=utf-8

import time

import numpy as np
# ================  Load Data ===================
from keras.utils import np_utils

from generateData import getData2

img_rows, img_cols = 28, 28
num_classes = 10
ontime = time.time()


X_train, Y_train, X_test, Y_test, X_valid, Y_valid = getData2()


y_train_onehot_labels = np_utils.to_categorical(Y_train, 7)
y_test_onehot_lablels = np_utils.to_categorical(Y_test, 7)
y_validation_onehot_labels = np_utils.to_categorical(Y_valid, 7)


# side loss 的设置为0，是因为我们可以在call的时候会根据分类重新计算
# N x 1 为 side loss的维度

y_train_origin_centers = np.zeros((len(X_train), 1))
y_test_origin_centers = np.zeros((len(X_test), 1))
y_valid_origin_centers = np.zeros((len(X_valid), 1))

outup = time.time()
print('Consumption time', outup - ontime)

# ================ Construct Model ===================
from modelGenerator import generateModel2

# model_centerloss = generateModel1()
model_centerloss = generateModel2(lambda_c=0.2)
model_centerloss.summary()



# # ================  Model Train & Predict ===================

batch_size = 64
epochs = 25


# 0.65
model_centerloss.fit([X_train, y_train_onehot_labels], [y_train_onehot_labels, y_train_origin_centers], batch_size=batch_size,
                     epochs = epochs, verbose=1,
                     validation_data=([X_valid, y_validation_onehot_labels], [y_validation_onehot_labels, y_valid_origin_centers])
                     )
model_centerloss.evaluate([X_test, y_test_onehot_lablels], [y_test_onehot_lablels, y_valid_origin_centers], verbose=1)


