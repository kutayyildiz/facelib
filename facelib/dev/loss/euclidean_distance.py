"""Keras implementation of euclidean distance loss functions with various norms."""

import tensorflow as tf


class EuclideanDistance:
    """Euclidean Distance with norms for facial landmark detection.
    
    Parameters
    ----------
    norm : int
        0 : without normalization
        1 : norm using outer corner of eyes
        2 : norm using related points

    Returns
    -------
    method
        Keras compatible loss function.
    """

    def __init__(self, norm=0, **kwargs):

        self.kwargs = kwargs
        self.norm = norm
        self._norm_dict = {
            0: self._norm_0,
            1: self._norm_1,
            2: self._norm_2, }

    def loss(self, y_true, y_pred):
        """Keras compatible loss function for landmark.

        Notes
        -----
        Input shape should be [batch, num_points * 2]
        e.g., [y0, x0, y1, x1, y2, x2...]
        """
        norm = self._norm_fn(y_true, y_pred)
        num_batches = tf.shape(y_true)[0]
        y_true = tf.reshape(y_true, [num_batches, -1, 2])
        y_pred = tf.reshape(y_pred, [num_batches, -1, 2])
        loss = tf.norm(y_true - y_pred, axis=-1)
        loss = loss / norm
        loss = tf.reduce_mean(loss)
        return loss

    @property
    def _norm_fn(self,):
        return self._norm_dict[self.norm]

    def _norm_0(self, y_true):
        return 1.

    def _norm_1(self, y_true):
        """Norm using outer eye points.

        References
        ----------
        NRMSE: https://link.springer.com/article/10.1186/s13640-018-0324-4
        
        Notes
        -----
        outer eye points should be the first 4 coordinate values.
        e.g., [left_eye_y, left_eye_x, right_eye_y, right_eye_x,...]

        Returns
        -------
        norm : tf.constant
            euclidean distance between outer eye points.
        """
        y0, x0, y1, x1 = self.kwargs['outer_eye_points']
        eye_0 = tf.stack([y_true[:, y0], y_true[:, x0]], -1)
        eye_1 = tf.stack([y_true[:, y1], y_true[:, x1]], -1)
        norm = tf.norm(eye_0 - eye_1, axis=-1)
        norm = tf.expand_dims(norm, -1)
        return norm

    def _norm_2(self, y_true):
        # Norm using pairs of points(EXPERIMENTAL).
        num_batches = tf.shape(y_true)[0]
        y_true = tf.reshape(y_true, [num_batches, -1, 2])
        norm = tf.norm(y_true[:, ::2, :] - y_true[:, 1::2, :], axis=-1)
        norm = tf.concat([tf.tile(norm[:, ::2], [1, 2]),
                          tf.tile(norm[:, 1::2], [1, 2])], -1)
        return norm
