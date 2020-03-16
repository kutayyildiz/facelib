"""Keras implementation of various layers."""

import tensorflow as tf

class Bottleneck(tf.keras.layers.Layer):
    """Keras implementation of Bottleneck layer.
    
    Parameters
    ----------
    c : int
        number of output channels
    t : int
        expansion rate
    s : int
        strides
    activation : tf.keras.layers.Layer, optional
        activation function compatible with keras
        e.g., tf.keras.layers.LeakyRelu
    
    """

    def __init__(self, c, t, s, activation=tf.keras.layers.ReLU, **kwargs):
        super(Bottleneck, self).__init__(**kwargs)

        # c:output_channels / t:expansion / s:strides
        self.s = s
        tchannel = c * t

        # Input:    h, w, d
        # Operator: 1x1, conv2d
        # Output:   h, w, (c*t)
        self.layer_1 = tf.keras.layers.Conv2D(
            filters = tchannel,
            kernel_size = 1,
            strides=1,
            padding='same',
            use_bias=False)
        self.layer_2 = activation()
        self.layer_3 = tf.keras.layers.BatchNormalization()

        # Input:    h, w, (c*t)
        # Operator: 3x3, dwise, stride=s
        # Output:   h/s, w/s, (c*t)
        self.layer_4 = tf.keras.layers.DepthwiseConv2D(
            kernel_size=(3,3),
            strides=s,
            padding='same',
            depth_multiplier=1,
            use_bias=False)
        self.layer_5 = activation()
        self.layer_6 = tf.keras.layers.BatchNormalization()
        
        # Input:    h/s, w/s, (c*t)
        # Operator: 1x1, conv2d, linear
        # Output:   h/s, w/s, (c)
        self.layer_7 = tf.keras.layers.Conv2D(
            filters=c,
            kernel_size=1,
            padding='same',
            strides=1,
            use_bias=False)
        if s==1:
            self.layer_add = tf.keras.layers.Add()

    def call(self, inputs):
        """Return transformed inputs."""
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.layer_6(x)
        x = self.layer_7(x)
        if self.s==1:
            x = self.layer_add([x, inputs])
        return x

class DenseL2(tf.keras.layers.Dense):
    """Keras implementation of dense layer with l2 normalization.
    
    Parameters
    ----------
    units : int
        number of output channels
    use_bias : bool
    kernel_regularizer : tf.keras.regularizers.Regularizer
    
    Notes
    -----
    Can be used with margin losses.
    
    """

    def __init__(self, units, use_bias=False, kernel_regularizer=None, **kwargs):
        super(DenseL2, self).__init__(
            units,
            use_bias=use_bias,
            kernel_regularizer=kernel_regularizer,
            **kwargs)

    def call(self, inputs):
        """Return transformed inputs."""
        w_n = tf.nn.l2_normalize(self.kernel, axis=0)
        logits = tf.nn.l2_normalize(inputs, axis=-1)
        out = tf.matmul(logits, w_n)
        return out

class DisturbLabel(tf.keras.layers.Layer):
    """Keras implementation of DisturbLabel."""

    def __init__(self, alpha):
        super(DisturbLabel, self).__init__()
        self.alpha = alpha

    def call(self, y_true):
        """Return transformed inputs.
        
        Notes
        -----
        Should be used on ground truths.
        
        """
        batch_size = tf.shape(y_true)[0]
        num_classes = tf.shape(y_true)[1]
        mask = tf.keras.backend.random_binomial([batch_size], self.alpha)
        mask_r = tf.logical_not(tf.cast(mask, tf.bool))
        mask_r = tf.cast(mask_r, tf.float32)
        mask = tf.stack([mask, mask_r])
        mask = tf.cast(mask, tf.float32)
        y_true_random = tf.random.uniform([batch_size], 0, num_classes, dtype=tf.int32)
        y_true_random = tf.one_hot(y_true_random, num_classes)
        merged = tf.stack([y_true_random, y_true])
        y_true_disturbed = tf.reduce_sum(merged*tf.expand_dims(mask, -1), 0)
        return y_true_disturbed
