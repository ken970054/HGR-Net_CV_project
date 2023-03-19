from sklearn.feature_extraction import image
import os
import numpy as np
from PIL import Image
import random

from keras.layers import Dense, Dropout, Activation, \
                         Flatten, Convolution2D, MaxPooling2D, \
                         BatchNormalization, Conv2D, Input, AveragePooling2D, \
                         Concatenate, add, ReLU, DepthwiseConv2D, Add, \
                         ZeroPadding2D, GlobalAveragePooling2D

from keras.models import Model
from BilinearUpSampling import *
from keras import optimizers

def MobileNetV2(shape):
    inputNode = Input(shape=shape)
    
    i = Conv2D(32, 3, 2, use_bias=False, padding='same')(inputNode)
    i = BatchNormalization()(i)
    i = ReLU(max_value=6)(i) 
    i = DepthwiseConv2D(3, use_bias=False, padding='same')(i)
    i = BatchNormalization()(i)
    i = ReLU(max_value=6)(i)
    i = Conv2D(16, 1, use_bias=False, padding='same')(i)
    i = BatchNormalization()(i)
    ## bottleneck (Inverted residuals)
    i = bottleneck(i, 16, 24, (2, 2), shortcut=False, zero_pad=True)
    i = bottleneck(i, 24, 24, (1, 1), shortcut=True)

    i = bottleneck(i, 24, 32, (2, 2), shortcut=False, zero_pad=True)
    i = bottleneck(i, 32, 32, (1, 1), shortcut=True)
    i = bottleneck(i, 32, 32, (1, 1), shortcut=True)

    i = bottleneck(i, 32, 64, (2, 2), shortcut=False, zero_pad=True)
    i = bottleneck(i, 64, 64, (1, 1), shortcut=True)
    i = bottleneck(i, 64, 64, (1, 1), shortcut=True)
    i = bottleneck(i, 64, 64, (1, 1), shortcut=True)

    i = bottleneck(i, 64, 96, (1, 1), shortcut=False)
    i = bottleneck(i, 96, 96, (1, 1), shortcut=True)
    i = bottleneck(i, 96, 96, (1, 1), shortcut=True)

    i = bottleneck(i, 96, 160, (2, 2), shortcut=False, zero_pad=True)
    i = bottleneck(i, 160, 160, (1, 1), shortcut=True)
    i = bottleneck(i, 160, 160, (1, 1), shortcut=True)

    i = bottleneck(i, 160, 320, (1, 1), shortcut=False)
    ## bottleneck 

    i = Conv2D(1280, 1, use_bias=False, padding='same')(i)
    i = BatchNormalization()(i)
    i = ReLU(max_value=6)(i)

    return inputNode, i

def bottleneck(i, filters, output, strides, shortcut=True, zero_pad=False):
    
    padding = 'valid' if zero_pad else 'same'
    shortcut_i = i

    i = Conv2D(filters * 6, 1, use_bias=False, padding='same')(i)
    i = BatchNormalization()(i)
    i = ReLU(max_value=6)(i)

    if zero_pad:
        i = ZeroPadding2D(padding=((0, 1), (0, 1)))(i)
    i = DepthwiseConv2D(3, strides=strides, use_bias=False, padding=padding)(i)
    i = BatchNormalization()(i)
    i = ReLU(max_value=6)(i)
    i = Conv2D(output, 1, use_bias=False, padding='same')(i)
    i = BatchNormalization()(i)
    # stride = 1 -> use shortcut
    if shortcut:
        i = Add()([i, shortcut_i])
    
    return i
