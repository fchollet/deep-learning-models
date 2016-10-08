from __future__ import print_function

import numpy as np
import warnings

from keras.layers import merge, Input, Dropout, Reshape, Deconvolution2D, TimeDistributed
from keras.layers import Dense, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
from keras.utils.layer_utils import convert_all_kernels_in_model, print_summary
from keras.utils.data_utils import get_file
from imagenet_utils import decode_predictions, preprocess_input
import keras.utils.visualize_util as vutil


def AutoColorize(include_top=True, weights='imagenet',
                 input_tensor=None):
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        if include_top:
            input_shape = (1, 512, 512)
        else:
            input_shape = (1, None, None)
    else:
        if include_top:
            input_shape = (512, 512, 1)
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
    conv1_2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='conv1_2')(conv1_1)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(conv1_2)

    # Block 2
    conv2_1 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='conv2_1')(pool1)
    conv2_2 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='conv2_2')(conv2_1)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2_2)

    # Block 3
    conv3_1 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='conv3_1')(pool2)
    conv3_2 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='conv3_2')(conv3_1)
    conv3_3 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='conv3_3')(conv3_2)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_3)

    # Block 4
    conv4_1 = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv4_1')(pool3)
    conv4_2 = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv4_2')(conv4_1)
    conv4_3 = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv4_3')(conv4_2)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_3)

    # Block 5
    conv5_1 = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv5_1')(pool4)
    conv5_2 = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv5_2')(conv5_1)
    conv5_3 = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv5_3')(conv5_2)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(conv5_3)

    # Block 6
    fc6 = Convolution2D(4096, 7, 7, activation='relu', border_mode='same', name='fc6')(x)
    dropout_fc6 = Dropout(0.5, name='dropout_fc6')(fc6)

    # Block 7
    fc7 = Convolution2D(4096, 1, 1, activation='relu', border_mode='same', name='fc7')(dropout_fc6)
    dropout_fc7 = Dropout(0.5, name='dropout_fc7')(fc7)

    # Dense Hypercolumn
    data_full = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), name='data_full')(img_input)

    conv1_1_full = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), name='conv1_1_full')(conv1_1)
    conv1_2_full = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), name='conv1_2_full')(conv1_2)

    conv2_1_full = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), name='conv2_1_full')(conv2_1)
    conv2_2_full = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), name='conv2_2_full')(conv2_2)

    conv4_1_reshaped = Reshape((512, 1, 64, 64), name='conv4_1_reshaped')(conv4_1)
    conv4_1_full_reshaped = TimeDistributed(
        Deconvolution2D(1, 4, 4, subsample=(2, 2), output_shape=(None, 1, 128, 128), border_mode='same'),
        input_shape=(None, 512, 1, 64, 64), name='conv4_1_full_reshaped')(conv4_1_reshaped)
    conv4_1_full = Reshape((512, 128, 128), name='conv4_1_full')(conv4_1_full_reshaped)

    conv4_2_reshaped = Reshape((512, 1, 64, 64), name='conv4_2_reshaped')(conv4_2)
    conv4_2_full_reshaped = TimeDistributed(
        Deconvolution2D(1, 4, 4, subsample=(2, 2), output_shape=(None, 1, 128, 128), border_mode='same'),
        input_shape=(None, 512, 1, 64, 64), name='conv4_2_full_reshaped')(conv4_2_reshaped)
    conv4_2_full = Reshape((512, 128, 128), name='conv4_2_full')(conv4_2_full_reshaped)

    conv4_3_reshaped = Reshape((512, 1, 64, 64), name='conv4_3_reshaped')(conv4_3)
    conv4_3_full_reshaped = TimeDistributed(
        Deconvolution2D(1, 4, 4, subsample=(2, 2), output_shape=(None, 1, 128, 128), border_mode='same'),
        input_shape=(None, 512, 1, 64, 64), name='conv4_3_full_reshaped')(conv4_3_reshaped)
    conv4_3_full = Reshape((512, 128, 128), name='conv4_3_full')(conv4_3_full_reshaped)

    conv5_1_reshaped = Reshape((512, 1, 32, 32), name='conv5_1_reshaped')(conv5_1)
    conv5_1_full_reshaped = TimeDistributed(
        Deconvolution2D(1, 8, 8, subsample=(4, 4), output_shape=(None, 1, 128, 128), border_mode='same'),
        input_shape=(None, 512, 1, 32, 32), name='conv5_1_full_reshaped')(conv5_1_reshaped)
    conv5_1_full = Reshape((512, 128, 128), name='conv5_1_full')(conv5_1_full_reshaped)

    conv5_2_reshaped = Reshape((512, 1, 32, 32), name='conv5_2_reshaped')(conv5_2)
    conv5_2_full_reshaped = TimeDistributed(
        Deconvolution2D(1, 8, 8, subsample=(4, 4), output_shape=(None, 1, 128, 128), border_mode='same'),
        input_shape=(None, 512, 1, 32, 32), name='conv5_2_full_reshaped')(conv5_2_reshaped)
    conv5_2_full = Reshape((512, 128, 128), name='conv5_2_full')(conv5_2_full_reshaped)

    conv5_3_reshaped = Reshape((512, 1, 32, 32), name='conv5_3_reshaped')(conv5_3)
    conv5_3_full_reshaped = TimeDistributed(
        Deconvolution2D(1, 8, 8, subsample=(4, 4), output_shape=(None, 1, 128, 128), border_mode='same'),
        input_shape=(None, 512, 1, 32, 32), name='conv5_3_full_reshaped')(conv5_3_reshaped)
    conv5_3_full = Reshape((512, 128, 128), name='conv5_3_full')(conv5_3_full_reshaped)

    fc6_reshaped = Reshape((4096, 1, 16, 16), name='fc6_reshaped')(dropout_fc6)
    fc6_full_reshaped = TimeDistributed(
        Deconvolution2D(1, 16, 16, subsample=(8, 8), output_shape=(None, 1, 128, 128), border_mode='same'),
        input_shape=(None, 4096, 1, 16, 16), name='fc6_full_reshaped')(fc6_reshaped)
    fc6_full = Reshape((4096, 128, 128), name='fc6_full')(fc6_full_reshaped)

    fc7_reshaped = Reshape((4096, 1, 16, 16), name='fc7_reshaped')(dropout_fc7)
    fc7_full_reshaped = TimeDistributed(
        Deconvolution2D(1, 16, 16, subsample=(8, 8), output_shape=(None, 1, 128, 128), border_mode='same'),
        input_shape=(None, 4096, 1, 16, 16), name='fc7_full_reshaped')(fc7_reshaped)
    fc7_full = Reshape((4096, 128, 128), name='fc7_full')(fc7_full_reshaped)

    dense_hypercolumn = merge(
        [fc7_full, fc6_full, conv5_3_full, conv5_2_full, conv5_1_full, conv4_3_full, conv4_2_full, conv4_1_full,
         conv3_3, conv3_2, conv3_1, conv2_2_full, conv2_1_full,
         conv1_2_full, conv1_1_full, data_full], mode='concat',
        name='dense_hypercolumn', concat_axis=1)

    # Fully connected
    h_fc1 = Convolution2D(1024, 1, 1, activation='relu', border_mode='same', name='h_fc1')(dense_hypercolumn)

    if include_top:
        # Classification block
        prediction_h = Convolution2D(32, 1, 1, border_mode='same', name='prediction_h')(h_fc1)

        prediction_h_softmax = TimeDistributed(
            Activation('softmax'), input_shape=(None, 32, 1, 128, 128), name='prediction_h_softmax')(Reshape((32, 1, 128, 128))(prediction_h))

        prediction_h_softmax_reshaped = Reshape((32, 1, 128, 128), name='prediction_h_softmax_reshaped')(prediction_h_softmax)
        prediction_h_full_reshaped = TimeDistributed(
            Deconvolution2D(1, 8, 8, subsample=(4, 4), output_shape=(None, 32, 512, 512), border_mode='same'),
            input_shape=(None, 32, 1, 128, 128), name='prediction_h_full_reshaped')(prediction_h_softmax_reshaped)
        prediction_h_full =  Reshape((32, 512, 512), name='prediction_h_full')(prediction_h_full_reshaped)

        prediction_c = Convolution2D(32, 1, 1, border_mode='same', name='prediction_c')(h_fc1)
        prediction_c_softmax = Activation('softmax', name='prediction_c_softmax')(Reshape((32, 128, 128))(prediction_c))
        prediction_c_softmax_reshaped = Reshape((32, 1, 128, 128), name='prediction_c_softmax_reshaped')(prediction_c_softmax)
        prediction_c_full_reshaped = TimeDistributed(
            Deconvolution2D(1, 8, 8, subsample=(4, 4), output_shape=(None, 32, 512, 512), border_mode='same'),
            input_shape=(None, 32, 1, 128, 128), name='prediction_c_full_reshaped')(prediction_c_softmax_reshaped)
        prediction_c_full = Reshape((32, 512, 512), name='prediction_c_full')(prediction_c_full_reshaped)

        model = Model(input=img_input, output=[prediction_h_full, prediction_c_full])
    else:
        # Create model
        model = Model(input=img_input, output=h_fc1)

    model.save_weights('test.h5')

    vutil.plot(model, show_shapes=True)

    # load weights

    return model


if __name__ == '__main__':
    model = AutoColorize(include_top=True, weights='imagenet')
    img_path = 'cat.jpg'
    img = image.load_img(img_path, target_size=(514, 514))
    x = image.img_to_array(img)

    preds = model.predict(x)
    img_color = image.array_to_img(preds)
