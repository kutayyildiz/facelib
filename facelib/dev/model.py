"""Keras implementation of various models."""

import tensorflow as tf
from facelib.dev.layer import Bottleneck
from facelib.dev.image import ImageAugmentor

def mobile_face_net(input_shape=(112,112,3), activation=tf.keras.layers.LeakyReLU):
    """Keras implmentation of MobileFaceNet.
    
    Parameters
    ----------
    input_shape : tuple, optional
        by default (112,112,3)
    activation : [type], optional
        activations used in the model, by default tf.keras.layers.LeakyReLU
    
    Returns
    -------
    [type]
        [description]
    """
    input = tf.keras.layers.Input(input_shape)

    x = tf.keras.layers.Conv2D(
        64, (3,3), (2,2), padding='same')(input)
    x = activation()(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # depthwise conv3x3
    x = tf.keras.layers.DepthwiseConv2D((3, 3), padding='same')(x)
    x = activation()(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # bottleneck
    # t=2, c=64, n=5, s=2
    x = Bottleneck(64, 2, 2, activation)(x)
    x = Bottleneck(64, 2, 1, activation)(x)
    x = Bottleneck(64, 2, 1, activation)(x)
    x = Bottleneck(64, 2, 1, activation)(x)
    x = Bottleneck(64, 2, 1, activation)(x)

    # bottleneck
    # t=4, c=128, n=1, s=2
    x = Bottleneck(128, 4, 2, activation)(x)

    # bottleneck
    # t=2, c=128, n=6, s=1
    x = Bottleneck(128, 2, 1, activation)(x)
    x = Bottleneck(128, 2, 1, activation)(x)
    x = Bottleneck(128, 2, 1, activation)(x)
    x = Bottleneck(128, 2, 1, activation)(x)
    x = Bottleneck(128, 2, 1, activation)(x)
    x = Bottleneck(128, 2, 1, activation)(x)

    # bottleneck
    # t=4, c=128, n=1, s=2
    x = Bottleneck(128, 4, 2, activation)(x)

    # bottleneck
    # t=2, c=128, n=2, s=1
    x = Bottleneck(128, 2, 1, activation)(x)
    x = Bottleneck(128, 2, 1, activation)(x)

    # conv1x1
    x = tf.keras.layers.Conv2D(512, (1, 1), (1, 1), padding='valid')(x)
    x = activation()(x)
    x = tf.keras.layers.BatchNormalization()(x)

    #linear GDConv7x7
    x = tf.keras.layers.DepthwiseConv2D((7, 7), padding='valid')(x)
    x = tf.keras.layers.BatchNormalization()(x)
 
    #linear conv1x1
    x = tf.keras.layers.Conv2D(128, (1, 1), (1, 1), padding='same')(x)
    x = tf.keras.layers.Flatten(name='out_mfn')(x)
    model = tf.keras.models.Model(input, x, name='mobile_face_nets')
    return model