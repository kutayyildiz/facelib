"""Keras implementation of various margin losses."""

import tensorflow as tf

class CombinedMargin:
    """Combination of arcface, cosface and sphereface loss functions.

    Notes
    -----
    ref(git) : https://github.com/deepinsight/insightface
    formula : cos(theta * m1 + m2) - m3
    """

    def __init__(self, m1=1., m2=0.3, m3=0.2, scale=64.):
        self.s = tf.constant(scale, tf.float32)
        self.m1 = tf.constant(m1, tf.float32)
        self.m2 = tf.constant(m2, tf.float32)
        self.m3 = tf.constant(m3, tf.float32)

    def loss(self, y_true, y_pred):
        """Keras compatible combined margin loss."""
        original_target_logit = tf.multiply(y_true, y_pred)
        original_target_logit = tf.reduce_sum(original_target_logit, axis=-1)
        original_target_logit = tf.expand_dims(original_target_logit, -1)
        theta = tf.acos(original_target_logit)
        marginal_target_logit = tf.cos(theta * self.m1 + self.m2) - self.m3
        marginal_target_logit = tf.clip_by_value(marginal_target_logit, -0.99, 0.99)
        fc7 = y_pred + y_true * (marginal_target_logit - original_target_logit)
        fc7 = fc7 * self.s
        fc7 = tf.keras.backend.softmax(fc7)
        loss = tf.keras.backend.categorical_crossentropy(y_true, fc7)
        return loss


class ArcFace(CombinedMargin):
    """Arcface loss.

    Notes
    -----
    ref(paper) : https://arxiv.org/abs/1801.07698
    formula : cos(theta + m)
    """

    def __init__(self, m=0.5, scale=64.):
        super().__init__(m1=1., m2=m, m3=0., scale=scale)


class CosineFace(CombinedMargin):
    """CosineFace loss.

    Notes
    -----
    ref(paper) : https://arxiv.org/abs/1801.09414
    formula : cos(theta) - m
    """
    
    def __init__(self, m=0.35, scale=64.):
        super().__init__(m1=1., m2=0., m3=m, scale=scale)


class SphereFace(CombinedMargin):
    """SphereFace loss.

    Notes
    -----
    ref(paper) : https://arxiv.org/abs/1704.08063
    formula : cos(theta * m)
    """
    
    def __init__(self, m=4., scale=1.):
        super().__init__(m1=m, m2=0., m3=0., scale=scale)