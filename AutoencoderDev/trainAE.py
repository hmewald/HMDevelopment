import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from os import listdir
from os.path import isfile, join
from ae_models import *

import math
import sys
import argparse
import shutil, os




parser = argparse.ArgumentParser()
parser.add_argument('-c', '--classname', help='code for class being fitted', type=str, default="cf")
parser.add_argument('-m', '--modelpath', help='path for saving autoencoder model', type=str, default="/hmewald/Autoencoders/ModelCF/")
parser.add_argument('-it', '--imtrain', help='path with training images', type=str, default="/hmewald/DatasetTraining/VideoCF")
parser.add_argument('-iv', '--imval', help='path with test/validation images', type=str, default="/hmewald/DatasetTest/VideoCF")
parser.add_argument('-n', '--numset', help='number of images to load', type=str, default=1024)
results = parser.parse_args()

class_name = results.classname
model_path = results.modelpath
im_path = results.imtrain
im_test_path = results.imval
num_set = results.numset


try:
    shutil.rmtree(model_path)
except OSError:
    pass

os.mkdir(model_path)
os.mkdir(model_path + "TrainingImages")
os.mkdir(model_path + "TestImages")


encoding_dim = 64
input_x = 243
input_y = 243
input_chan = 1

mark_size = 100

epoch_int = 10
batch_int = 64
n_val = 64



def importDatasetX(im_path, n_max):
    # Read images and convert to grayscale
    im_files = [f for f in listdir(im_path) if isfile(join(im_path, f))]
    im_files.sort()

    n_set = min([len(im_files),n_max])

    if(input_chan == 3):
        X_arr = np.empty((n_set,input_y, input_x,input_chan), dtype='uint8')
        train_progress = 0
        for i in range(0,n_set):
            im_big = misc.imread(join(im_path,im_files[i]),mode='RGB')
            im_res = misc.imresize(im_big,(input_y, input_x))
            del im_big

            X_arr[i,:,:,:] = im_res

            if (math.ceil(100*i/n_set) > train_progress):
                train_progress = math.ceil(100*i/n_set)
                print("Image set reading at " + str(train_progress) + " percent completion")

    elif(input_chan == 1):
        X_arr = np.empty((n_set,input_y, input_x, input_chan), dtype='uint8')
        train_progress = 0
        for i in range(0,n_set):
            im_big = misc.imread(join(im_path,im_files[i]),mode='L')
            im_res = misc.imresize(im_big,(input_y, input_x))
            del im_big

            X_arr[i,:,:,0] = im_res

            if (math.ceil(10*i/n_set) > train_progress):
                train_progress = math.ceil(10*i/n_set)
                print("Image set reading at " + str(10*train_progress) + " percent completion")

    return X_arr, n_set


def computeAEloss(in_im, recon_im):
    diff_im = recon_im - in_im
    K_int = np.prod(diff_im.shape)
    loss = np.sum(np.power(diff_im,2))/K_int
    return loss


def write_list(out_list, out_path):
    f_out = open(out_path,"w+")

    for point in out_list:
        f_out.write(str(point) + "\n")


def markImset(im_set):
    mark_big = misc.imread("mark_raw.jpeg" ,mode='L')
    mark_res = misc.imresize(mark_big, (mark_size, mark_size))

    n_set = im_set.shape[0]
    for i in range(mark_size):
        for j in range(mark_size):
            if (mark_res[i,j] > 0.9):
                im_set[:,i,j] = np.ones((n_set,1))

    return im_set



model, model_enc, model_dec = splitConvAE((input_y,input_x,input_chan))


print("Reading training dataset:")
X_train, n_train = importDatasetX(im_path, num_set)
print("Reading test dataset:")
X_test, n_test = importDatasetX(im_test_path, n_val)

print(X_train.shape)
print(X_test.shape)

X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.
# X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
# X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))
#
# print(X_train.shape)
# print(X_test.shape)

model.fit(X_train, X_train, epochs=epoch_int, batch_size=batch_int, shuffle=True, validation_data=(X_test,X_test))
model.save(model_path + class_name + "_ae_model.h5")

# encoded_ims = model_enc.predict(X_test)
# decoded_ims = model_dec.predict(encoded_ims)
X_test = markImset(X_test)

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
