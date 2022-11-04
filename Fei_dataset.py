from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#from keras.optimizers import SGD
#from scipy.misc import imsave as ims
#from tensorflow.examples.tutorials.mnist import input_data

from tensorflow.keras.datasets import mnist

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import  tensorflow as tf
from utils import *

import tensorflow as tf
import tensorflow.keras
import scipy.io as sio
from tensorflow.keras import utils as np_utils

def GetMNIST_DataSet():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    image_size = x_train.shape[1]
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    return x_train,y_train,x_test,y_test

def GetSVHN_DataSet(isResize = False):
    file1 = 'data/svhn_train.mat'
    file2 = 'data/svhn_test.mat'
    train_data = sio.loadmat(file1)
    test_data = sio.loadmat(file2)

    x_train_hv = train_data['X']
    y_train_hv = train_data['y']
    x_test_hv = test_data["X"]
    y_test_hv = test_data["y"]
    
    x_train_hv = x_train_hv.transpose(3, 0, 1, 2)
    x_test_hv = x_test_hv.transpose(3, 0, 1, 2)

    if isResize:
        x_train_hv = tf.image.resize_images(x_train_hv, (28, 28))
        x_test_hv = tf.image.resize_images(x_test_hv, (28, 28))
        x_train_hv = tf.image.rgb_to_grayscale(x_train_hv)
        x_test_hv = tf.image.rgb_to_grayscale(x_test_hv)

        x_train_hv = tf.Session().run(x_train_hv)
        x_test_hv = tf.Session().run(x_test_hv)

    for h1 in range(np.shape(y_test_hv)[0]):
        y_test_hv[h1] = y_test_hv[h1]-1
    for h1 in range(np.shape(y_train_hv)[0]):
        y_train_hv[h1] = y_train_hv[h1]-1

    x_train_hv = x_train_hv.astype('float32') / 255
    x_test_hv = x_test_hv.astype('float32') / 255

    #y_test_hv = keras.utils.to_categorical(y_test_hv)
    return x_train_hv,y_train_hv,x_test_hv,y_test_hv

