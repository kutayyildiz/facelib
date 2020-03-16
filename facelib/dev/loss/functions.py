"""Keras implementation of various loss functions."""

import tensorflow as tf


def rmse(y_true, y_pred):
    """Keras implementation of root mean squared error."""
    loss = tf.losses.mse(y_true, y_pred)
    loss = tf.sqrt(loss)
    loss = tf.reduce_mean(loss)
    return loss
