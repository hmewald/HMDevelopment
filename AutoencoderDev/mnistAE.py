import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import shutil, os
from os import listdir
from os.path import isfile, join
from ae_models import *

import math
import sys
import argparse

# Load MNIST dataset from Keras
from keras.datasets import mnist




parser = argparse.ArgumentParser()
parser.add_argument('-c', '--classname', help='code for class being fitted', type=str, default="mnist")
parser.add_argument('-m', '--modelpath', help='path for saving autoencoder model', type=str, default="/hmewald/Autoencoders/ModelMNIST/")
parser.add_argument('-e', '--epochnum', help='number of training info', type=int, default=10)
args = parser.parse_args()

class_name = args.classname
model_path = args.modelpath
epoch_int = args.epochnum

try:
    shutil.rmtree(model_path)
except OSError:
    pass

os.mkdir(model_path)
os.mkdir(model_path + "TrainingImages")
os.mkdir(model_path + "TestImages")


input_x = 243
input_y = 243
input_chan = 1

mark_size = 100

batch_int = 64
n_train = 8192
n_val = 64



def computeAEloss(in_im, recon_im):
    diff_im = recon_im - in_im
    K_int = np.prod(diff_im.shape)
    loss = np.sum(np.power(diff_im,2))/K_int
    return loss


def write_list(out_list, out_path):
    f_out = open(out_path,"w+")

    for point in out_list:
        f_out.write(str(point) + "\n")


def loadMark():
    mark_big = misc.imread("mark_raw.jpeg" ,mode='L')
    mark_res = misc.imresize(mark_big, (input_x, input_y))

    mark_im = mark_res.reshape([1, input_x, input_y, 1])

    return mark_im


model, model_enc, model_dec = newConvAE((input_y,input_x,input_chan))



# pre-shuffled train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape)
print(X_test.shape)

X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

n_train_set = X_train.shape[0]
n_test_set = X_test.shape[0]

X_train = X_train.reshape([n_train_set, 28, 28, 1])
X_test = X_test.reshape([n_test_set, 28, 28, 1])

X_train_big = np.zeros((n_train, input_x, input_y, 1))
X_test_big = np.zeros((n_val, input_x, input_y, 1))

for i in range(n_train):
    X_train_big[i,:,:,0] = misc.imresize(X_train[i,:,:].reshape([28,28]),(input_y, input_x))

for i in range(n_val):
    X_test_big[i,:,:,0] = misc.imresize(X_test[i,:,:].reshape([28,28]),(input_y, input_x))

X_train = X_train_big
X_test = X_test_big

mark_im = loadMark()

# X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
# X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))
#
# print(X_train.shape)
# print(X_test.shape)

model.fit(X_train[:n_train], X_train[:n_train], epochs=epoch_int, batch_size=batch_int, shuffle=True, validation_data=(X_test,X_test))
model.save(model_path + class_name + "_ae_model.h5")

# encoded_ims = model_enc.predict(X_test)
# decoded_ims = model_dec.predict(encoded_ims)

decoded_training_ims = model.predict(X_train[:n_val])
decoded_ims = model.predict(X_test)

n_show = n_val
loss_list = []
for i in range(n_show):
    if(input_chan == 3):
        train_im_i = X_train[i].reshape(input_y,input_x,input_chan)
        train_res_i = decoded_training_ims[i].reshape(input_y,input_x,input_chan)
        val_im_i = X_test[i].reshape(input_y,input_x,input_chan)
        dec_im_i = decoded_ims[i].reshape(input_y,input_x,input_chan)
    elif(input_chan == 1):
        train_im_i = X_train[i].reshape(input_y,input_x)
        train_res_i = decoded_training_ims[i].reshape(input_y,input_x)
        val_im_i = X_test[i].reshape(input_y,input_x)
        dec_im_i = decoded_ims[i].reshape(input_y,input_x)

    misc.imsave(model_path + "TrainingImages/ae_training" + format(i, '04d') + ".png", np.concatenate([train_im_i, train_res_i, train_res_i - train_im_i], axis=0))


    loss_list.append(computeAEloss(val_im_i, dec_im_i))
    comparison_ex_i = np.concatenate([val_im_i, dec_im_i, np.absolute(dec_im_i - val_im_i)], axis=0)
    misc.imsave(model_path + "TestImages/ae_example" + format(i, '04d') + ".png", comparison_ex_i)
    # misc.imsave(model_path + "ae_reconstruction" + str(i) + ".png", decoded_ims[i].reshape(input_y,input_x))

write_list(loss_list, model_path + "losses.txt")

decoded_mark = model.predict(mark_im)
misc.imsave(model_path + "ae_mark.png", np.concatenate([mark_im, decoded_mark, mark_im - decoded_mark], axis=0))
