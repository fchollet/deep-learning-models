from __future__ import print_function

import numpy as np
import warnings

from keras.layers import merge, Input
from keras.layers import Dense, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from imagenet_utils import preprocess_input

TH_WEIGHTS_PATH = 'Keras_model_weights.h5'

def identity_block(input_tensor, kernel_size, filters, stage, block):
    '''The identity_block is the block that has no conv layer at shortcut

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    nb_filter1, nb_filter2, nb_filter3 = filters
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'conv_stage' + str(stage) + '_block' + str(block) + '_branch'
    bn_name_base = 'bn_stage' + str(stage) + '_block' + str(block) + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, kernel_size, kernel_size,
                      border_mode='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = merge([x, input_tensor], mode='sum')
    x = Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    '''conv_block is the block that has a conv layer at shortcut

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    '''
    nb_filter1, nb_filter2, nb_filter3 = filters
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'conv_stage' + str(stage) + '_block' + str(block) + '_branch'
    bn_name_base = 'bn_stage' + str(stage) + '_block' + str(block) + '_branch'
    shortcut_name_post = '_stage' + str(stage) + '_block' + str(block) + '_proj_shortcut'

    x = Convolution2D(nb_filter1, 1, 1, subsample=strides,
                      name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, kernel_size, kernel_size, border_mode='same',
                      name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Convolution2D(nb_filter3, 1, 1, subsample=strides,
                             name='conv' + shortcut_name_post)(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name='bn' + shortcut_name_post)(shortcut)

    x = merge([x, shortcut], mode='sum')
    x = Activation('relu')(x)
    return x

def OpenNsfw(include_top=True, weights='yahoo', input_tensor=None):
    if weights not in {'yahoo', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `yahoo` '
                         '(pre-training on yahoo NSFW data).')
    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        if include_top:
            input_shape = (3, 224, 224)
        else:
            input_shape = (3, None, None)
    else:
        if include_top:
            input_shape = (224, 224, 3)
        else:
            input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor)
        else:
            img_input = input_tensor
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D((3, 3))(img_input)
    x = Convolution2D(64, 7, 7, subsample=(2, 2), name='conv_1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_1')(x)
    x = Activation('relu', name='relu_1')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    x = conv_block(x, 3, [32, 32, 128], stage=0, block=0, strides=(1, 1))
    x = identity_block(x, 3, [32, 32, 128], stage=0, block=1)
    x = identity_block(x, 3, [32, 32, 128], stage=0, block=2)

    x = conv_block(x, 3, [64, 64, 256], stage=1, block=0)
    x = identity_block(x, 3, [64, 64, 256], stage=1, block=1)
    x = identity_block(x, 3, [64, 64, 256], stage=1, block=2)
    x = identity_block(x, 3, [64, 64, 256], stage=1, block=3)

    x = conv_block(x, 3, [128, 128, 512], stage=2, block=0)
    x = identity_block(x, 3, [128, 128, 512], stage=2, block=1)
    x = identity_block(x, 3, [128, 128, 512], stage=2, block=2)
    x = identity_block(x, 3, [128, 128, 512], stage=2, block=3)
    x = identity_block(x, 3, [128, 128, 512], stage=2, block=4)
    x = identity_block(x, 3, [128, 128, 512], stage=2, block=5)

    x = conv_block(x, 3, [256, 256, 1024], stage=3, block=0)
    x = identity_block(x, 3, [256, 256, 1024], stage=3, block=1)
    x = identity_block(x, 3, [256, 256, 1024], stage=3, block=2)

    x = AveragePooling2D((7, 7), strides=(1,1), name='pool')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(2, activation='softmax', name='fc_nsfw')(x)

    model = Model(img_input, x)

    # load weights
    import h5py

    caffe = h5py.File('Keras_model_weights.h5', 'r')
    orig = h5py.File('resnet_50_1by2_nsfw.h5', 'r')

    layer = model.get_layer('conv_1')
    layer.set_weights([orig['/data/conv_1/0'][:], orig['/data/conv_1/1'][:]])

    layer = model.get_layer('bn_1')
    layer.set_weights([caffe['scale_1/scale_1_gamma'][:], caffe['scale_1/scale_1_beta'][:], caffe['bn_1/bn_1_running_mean'][:], caffe['bn_1/bn_1_running_std'][:]])

    # stage 0
    for block in range(3):
        if block == 0:
            list = ['_branch2a', '_branch2b', '_branch2c', '_proj_shortcut']
        else:
            list = ['_branch2a', '_branch2b', '_branch2c']
        for branch in list:
            base_name = '_stage0_block' + str(block) + branch
            bn_name = 'bn' + base_name
            conv_name = 'conv' + base_name
            orig_name = '/data/' + conv_name
            scale_name = 'scale' + base_name

            print(orig_name)
            layer = model.get_layer(conv_name)
            layer.set_weights(
                [orig[orig_name + '/0'][:], orig[orig_name + '/1'][:]])

            layer = model.get_layer(bn_name)
            layer.set_weights(
                [caffe[scale_name + '/' + scale_name + '_gamma'][:], caffe[scale_name + '/' + scale_name + '_beta'][:], caffe[bn_name + '/' + bn_name + '_running_mean'][:],
                 caffe[bn_name + '/' + bn_name + '_running_std'][:]])

    # stage 1
    for block in range(3):
        if block == 0:
            list = ['_branch2a', '_branch2b', '_branch2c', '_proj_shortcut']
        else:
            list = ['_branch2a', '_branch2b', '_branch2c']
        for branch in list:
            base_name = '_stage1_block' + str(block) + branch
            bn_name = 'bn' + base_name
            conv_name = 'conv' + base_name
            orig_name = '/data/' + conv_name
            scale_name = 'scale' + base_name

            print(conv_name)
            layer = model.get_layer(conv_name)
            layer.set_weights(
                [orig[orig_name + '/0'][:], orig[orig_name + '/1'][:]])

            layer = model.get_layer(bn_name)
            layer.set_weights(
                [caffe[scale_name + '/' + scale_name + '_gamma'][:], caffe[scale_name + '/' + scale_name + '_beta'][:], caffe[bn_name + '/' + bn_name + '_running_mean'][:],
                 caffe[bn_name + '/' + bn_name + '_running_std'][:]])

    # stage 2
    for block in range(6):
        if block == 0:
            list = ['_branch2a', '_branch2b', '_branch2c', '_proj_shortcut']
        else:
            list = ['_branch2a', '_branch2b', '_branch2c']
        for branch in list:
            base_name = '_stage2_block' + str(block) + branch
            bn_name = 'bn' + base_name
            conv_name = 'conv' + base_name
            orig_name = '/data/' + conv_name
            scale_name = 'scale' + base_name

            print(conv_name)
            layer = model.get_layer(conv_name)
            layer.set_weights(
                [orig[orig_name + '/0'][:], orig[orig_name + '/1'][:]])

            layer = model.get_layer(bn_name)
            layer.set_weights(
                [caffe[scale_name + '/' + scale_name + '_gamma'][:], caffe[scale_name + '/' + scale_name + '_beta'][:], caffe[bn_name + '/' + bn_name + '_running_mean'][:],
                 caffe[bn_name + '/' + bn_name + '_running_std'][:]])

    # stage 3
    for block in range(3):
        if block == 0:
            list = ['_branch2a', '_branch2b', '_branch2c', '_proj_shortcut']
        else:
            list = ['_branch2a', '_branch2b', '_branch2c']
        for branch in list:
            base_name = '_stage3_block' + str(block) + branch
            bn_name = 'bn' + base_name
            conv_name = 'conv' + base_name
            orig_name = '/data/' + conv_name
            scale_name = 'scale' + base_name

            print(conv_name)
            layer = model.get_layer(conv_name)
            layer.set_weights(
                [orig[orig_name + '/0'][:], orig[orig_name + '/1'][:]])

            layer = model.get_layer(bn_name)
            layer.set_weights(
                [caffe[scale_name + '/' + scale_name + '_gamma'][:], caffe[scale_name + '/' + scale_name + '_beta'][:], caffe[bn_name + '/' + bn_name + '_running_mean'][:],
                 caffe[bn_name + '/' + bn_name + '_running_std'][:]])

    if include_top:
        layer = model.get_layer('fc_nsfw')
        layer.set_weights([caffe['fc_nsfw/fc_nsfw_W'][:], caffe['fc_nsfw/fc_nsfw_b'][:]])

    caffe.close()
    orig.close()

    model.save_weights('open_nsfw_weights_th_dim_ordering_th_kernels.h5')
    # #model.save_weights('open_nsfw_weights_th_dim_ordering_th_kernels_notop.h5')

    if weights == 'yahoo':
        print('K.image_dim_ordering:', K.image_dim_ordering())
        if K.image_dim_ordering() == 'th':
            if include_top:
                weights_path = 'open_nsfw_weights_th_dim_ordering_th_kernels.h5'
            else:
                weights_path = get_file('open_nsfw_weights_th_dim_ordering_th_kernels_notop.h5',
                                        TH_WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models',
                                        md5_hash='f64f049c92468c9affcd44b0976cdafe')
            model.load_weights(weights_path)
            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image dimension ordering convention '
                              '(`image_dim_ordering="th"`). '
                              'For best performance, set '
                              '`image_dim_ordering="tf"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
                convert_all_kernels_in_model(model)
        else:
            if include_top:
                weights_path = get_file('open_nsfw_weights_tf_dim_ordering_tf_kernels.h5',
                                        TF_WEIGHTS_PATH,
                                        cache_subdir='models',
                                        md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
            else:
                weights_path = get_file('open_nsfw_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                        TF_WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models',
                                        md5_hash='a268eb855778b3df3c7506639542a6af')
            model.load_weights(weights_path)
            if K.backend() == 'theano':
                convert_all_kernels_in_model(model)
    return model

if __name__ == '__main__':
    model = OpenNsfw(include_top=True, weights='yahoo')

    img_path = 'baywatchphoto.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    #x = preprocess_input(x)
    print('Input image shape:', x.shape)

    #x = 255.0 * (x - x.min()) / (x.max() - x.min())
    #print(x)

    preds = model.predict(x)
    print('Predicted:', preds)