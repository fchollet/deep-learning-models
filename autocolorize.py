from __future__ import print_function

import numpy as np
import warnings

from keras.layers import merge, Input, Dropout, Reshape, Deconvolution2D
from keras.layers import Dense, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
from keras.utils.layer_utils import convert_all_kernels_in_model, print_summary
from keras.utils.data_utils import get_file
from imagenet_utils import decode_predictions, preprocess_input

def AutoColorize(include_top=True, weights='imagenet',
                input_tensor=None):
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        if include_top:
            input_shape = (1, 514, 514)
        else:
            input_shape = (1, None, None)
    else:
        if include_top:
            input_shape = (514, 514, 1)
        else:
            input_shape = (None, None, 1)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor)
        else:
            img_input = input_tensor

    # Block 1
    conv1_1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='conv1_1')(img_input)
    print(conv1_1._keras_shape)
    conv1_2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='conv1_2')(conv1_1)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(conv1_2)

    # Block 2
    conv2_1 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='conv2_1')(pool1)
    print(conv2_1._keras_shape)
    conv2_2 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='conv2_2')(conv2_1)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2_2)

    # Block 3
    conv3_1 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='conv3_1')(pool2)
    print(conv3_1._keras_shape)
    conv3_2 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='conv3_2')(conv3_1)
    conv3_3 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='conv3_3')(conv3_2)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_3)

    # Block 4
    conv4_1 = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv4_1')(pool3)
    print(conv4_1._keras_shape)
    conv4_2 = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv4_2')(conv4_1)
    conv4_3 = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv4_3')(conv4_2)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_3)

    # Block 5
    conv5_1 = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv5_1')(pool4)
    print(conv5_1._keras_shape)
    conv5_2 = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv5_2')(conv5_1)
    conv5_3 = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv5_3')(conv5_2)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(conv5_3)

    # Block 6
    fc6 = Convolution2D(4096, 7, 7, activation='relu', border_mode='same', name='fc6')(x)
    print(fc6._keras_shape)
    dropout_fc6 = Dropout(0.5, name='dropout_fc6')(fc6)

    # Block 7
    fc7 = Convolution2D(4096, 1, 1, activation='relu', border_mode='same', name='fc7')(dropout_fc6)
    print(fc7._keras_shape)
    dropout_fc7 = Dropout(0.5, name='dropout_fc7')(fc7)
    fc7_reshaped = Reshape((4096,16,16), name='fc7_reshaped')(dropout_fc7)
    print(fc7_reshaped._keras_shape)
    fc7_full_reshaped = Deconvolution2D(4096, 16, 16, (1, 4096, 16, 16), subsample=(8,8), border_mode='same', name='fc7_full_reshaped')(fc7_reshaped)
    print(fc7_full_reshaped._keras_shape)


    #if include_top:
        # Classification block


    # Create model
    model = Model(img_input, fc7_full_reshaped)

    # load weights

    return model


if __name__ == '__main__':
    model = AutoColorize(include_top=True, weights='imagenet')
    img_path = 'cat.jpg'
    img = image.load_img(img_path, target_size=(514, 514))
    x = image.img_to_array(img)

    preds = model.predict(x)
    img_color = image.array_to_img(preds)