"""Image manipulating functions/classes compatible with tensorflow/keras."""

import math
import os

import cv2
import numpy as np
import tensorflow as tf
from skimage import transform as skitrans


class ImageAugmentor:
    """Image augmentation with tensorflow(graph compatible)."""

    def __init__(self,
                 translation_range=(-0., 0.),
                 scale_range=(-0., 0.),
                 rotation_range=(-180, 180)):
        self.tr = tf.constant(translation_range, tf.float32)
        self.sr = tf.constant(scale_range, tf.float32)
        self.rr = tf.constant(rotation_range, tf.float32)
    # performance comparison of augment methods:
    # tests done in 64 batches using tfrecord iterator on i7-6700k
    #   opencv:     183 ms
    #   skimage:    314 ms
    #   keras:      410 ms
    #   scipy:      1410ms

    def augment_opencv(self, img, coordinates):
        """Augments the input image using class parameters.
        
        Notes
        -----
        Modifies coordinates to match with image augmentation.

        """
        img_shape = tf.shape(img)
        img_center = tf.cast(img_shape[:2], tf.float32) / 2.
        translate = tf.random.uniform([2], self.tr[0], self.tr[1]) * tf.cast(img_shape[:2][::-1], tf.float32)
        scale = 1 + tf.random.uniform([2], self.sr[0], self.sr[1])
        pi = tf.constant(math.pi)
        angle = tf.random.uniform([1], self.rr[0], self.rr[1])[0]
        angle = angle * pi / 180
        translate = tf.stack([translate[0], translate[1]])
        center_shift = tf.stack([
            [1., 0, img_center[1]],
            [0, 1, img_center[0]],
            [0, 0, 1]])
        scale_matrix = tf.stack([
            [scale[0], 0, 0.],
            [0, scale[1], 0],
            [0, 0, 1]])
        rotat_matrix = tf.stack([
            [tf.cos(angle), tf.sin(angle), 0.],
            [-tf.sin(angle), tf.cos(angle), 0],
            [0, 0, 1]])
        trans_matrix = tf.stack([
            [1., 0, translate[0]-img_center[1]],
            [0, 1, translate[1]-img_center[0]],
            [0, 0, 1]])
        M = tf.matmul(center_shift, scale_matrix)
        M = tf.matmul(M, rotat_matrix)
        M = tf.matmul(M, trans_matrix)
        M = M[:2]
        coor_shape = tf.shape(coordinates)
        coordinates = tf.reshape(coordinates, [1, -1, 2])
        coordinates = tf.reverse(coordinates, [-1])

        def transform(img, coor, M, img_shape):
            coor = cv2.transform(coor, M)
            img = cv2.warpAffine(img, M, tuple(img_shape), borderMode=cv2.BORDER_REPLICATE)
            return img, coor

        img, coordinates = tf.numpy_function(
            transform,
            [img, coordinates, M, img_shape[:2][::-1]],
            [tf.uint8, tf.float32])
        img = tf.reshape(img, img_shape)
        coordinates = tf.reverse(coordinates, [-1])
        coordinates = tf.reshape(coordinates, coor_shape)
        return (img, coordinates)

def align_custom_numpy(img, landmarks):
    """Image aligning function implemented in numpy.

    Notes
    -----
    Landmarks should be:
    [left_eye_y, left_eye_x, right_eye_y, right_eye_x,
    nosetip_y, nosetip_x, mouth_left_edge_y, mouth_left_edge_x,
    mouth_right_edge_y, mouth_right_edge_x]
    """
    img_shape = np.shape(img)
    landmarks = np.reshape(landmarks, [-1, 2])
    delta_eye = landmarks[1] - landmarks[0]
    angle = np.arctan2(delta_eye[0], delta_eye[1])
    angle = angle * 180 / math.pi
    eye_center = np.mean(landmarks[:2], axis=0)
    mouth_center = np.mean(landmarks[3:], axis=0)
    center = np.mean(
        np.concatenate([landmarks[:2], landmarks[3:]], 0), 0)
    reflect_nosetip = center * 2 - landmarks[2]
    center = (center + reflect_nosetip * [0, 1]) / [1, 2]
    ed_eye = np.linalg.norm(delta_eye) # euclidean distance between eyes
    ed_eye_mouth = np.linalg.norm(eye_center - mouth_center)
    yx = center - np.stack([ed_eye_mouth, ed_eye])
    tmp = np.abs(center - landmarks[2])
    width = np.max([ed_eye, tmp[0]], 0) * 2
    height = np.max([ed_eye_mouth, tmp[1]], 0) * 2
    bbox = np.stack([yx[0], yx[1], height, width])
    bbox = bbox.astype(np.int32)

    M = cv2.getRotationMatrix2D(tuple(center[::-1]), angle, 1)
    img = cv2.warpAffine(img, M, (img_shape[1], img_shape[0]))
    y, x, h, w = (bbox[0], bbox[1], bbox[2], bbox[3])
    h = h + y
    w = w + x
    y = np.max([y, 0])
    y = np.min([y, img_shape[0] - 1])
    x = np.max([x, 0])
    x = np.min([x, img_shape[1] - 1])
    h = np.max([h - y, 1])
    h = np.min([h, img_shape[0] - y])
    w = np.max([w - x, 1])
    w = np.min([w, img_shape[1] - x])
    dst = img[y:y+h, x:x+w, :]
    return dst

def align_custom_tensorflow(img, landmarks):
    """Tensorflow graph compatible version of method: align_custom_numpy."""
    img_shape = tf.shape(img)
    landmarks = tf.reshape(landmarks, [-1, 2])
    pi = tf.constant(math.pi)
    delta_eye = landmarks[1] - landmarks[0]
    angle = tf.atan2(delta_eye[0], delta_eye[1])# in radians
    angle = angle * 180 / pi
    eye_center = tf.reduce_mean(landmarks[:2], axis=0)
    mouth_center = tf.reduce_mean(landmarks[3:], axis=0)
    center = tf.reduce_mean(
        tf.concat([landmarks[:2], landmarks[3:]], 0), 0)
    reflect_nosetip = center * 2 - landmarks[2]
    center = (center + reflect_nosetip * [0, 1]) / [1, 2]
    ed_eye = tf.linalg.norm(delta_eye) #euclidean distance
    ed_eye_mouth = tf.linalg.norm(eye_center - mouth_center)
    yx = center - tf.stack([ed_eye_mouth, ed_eye])
    tmp = tf.math.abs(center - landmarks[2])
    width = tf.reduce_max([ed_eye, tmp[0]], 0) * 2
    height = tf.reduce_max([ed_eye_mouth, tmp[1]], 0) * 2
    bbox = tf.stack([yx[0], yx[1], height, width])
    bbox = tf.cast(bbox, tf.int32)
    def rotate(img, center, angle, shape):
        M = cv2.getRotationMatrix2D(tuple(center), angle, 1)
        dst = cv2.warpAffine(img, M, (shape[0], shape[1]))
        return dst
    img = tf.numpy_function(
        rotate,
        [img, center[::-1], angle, img_shape[:2][::-1]],
        tf.uint8)
    img = tf.reshape(img, img_shape)
    y, x, h, w = (bbox[0], bbox[1], bbox[2], bbox[3])
    h = h + y
    w = w + x
    y = tf.reduce_max([y, 0])
    y = tf.reduce_min([y, img_shape[0] - 1])
    x = tf.reduce_max([x, 0])
    x = tf.reduce_min([x, img_shape[1] - 1])
    h = tf.reduce_max([h - y, 1])
    h = tf.reduce_min([h, img_shape[0] - y])
    w = tf.reduce_max([w - x, 1])
    w = tf.reduce_min([w, img_shape[1] - x])
    bbox = tf.stack([y, x, h, w], axis=0)
    dst = tf.image.crop_to_bounding_box(
        img, bbox[0], bbox[1], bbox[2], bbox[3])
    return dst

def align_golden_ratio_numpy(img, landmarks):
    """Image aligning function implemented in numpy.

    Notes
    -----
    Landmarks should be:
    [left_eye_y, left_eye_x, right_eye_y, right_eye_x,
    nosetip_y, nosetip_x, mouth_left_edge_y, mouth_left_edge_x,
    mouth_right_edge_y, mouth_right_edge_x]
    """
    img_shape = np.shape(img)
    landmarks = np.reshape(landmarks, [-1, 2])
    delta_eye = landmarks[1] - landmarks[0]
    rad = np.arctan2(delta_eye[0], delta_eye[1])
    angle = rad * 180 / math.pi
    ed_eye = np.linalg.norm(delta_eye) # euclidean distance between eyes
    eye_center = np.mean(landmarks[:2], axis=0)
    center = np.array(img_shape[:2]) / 2
    center_shift = eye_center - center
    width = ed_eye * 1.865
    height = ed_eye * 3.018
    rotation_matrix = cv2.getRotationMatrix2D(tuple(eye_center[::-1]), angle, 1)
    rotation_matrix = np.concatenate([rotation_matrix, [[0,0,1]]], 0)
    translation_matrix = np.array([
        [1, 0, center_shift[1]],
        [0, 1, center_shift[0] + ed_eye * 0.3],
        [0, 0, 1]])
    translation_matrix = np.linalg.inv(translation_matrix)
    M = np.matmul(rotation_matrix, translation_matrix)
    M = M[:2]
    img = cv2.warpAffine(img, M, (img_shape[1], img_shape[0]))
    bbox = np.array([center[0]-height/2, center[1]-width/2, height, width], dtype=np.int32)
    y, x, h, w = bbox
    h = h + y
    w = w + x
    y = np.max([y, 0])
    y = np.min([y, img_shape[0] - 1])
    x = np.max([x, 0])
    x = np.min([x, img_shape[1] - 1])
    h = np.max([h - y, 1])
    h = np.min([h, img_shape[0] - y])
    w = np.max([w - x, 1])
    w = np.min([w, img_shape[1] - x])
    dst = img[y:y+h, x:x+w, :]
    return dst

def align_golden_ratio_tensorflow(img, landmarks):
    """Tensorflow graph compatible version of method: align_golden_ratio_numpy."""
    img_shape = tf.shape(img)
    landmarks = tf.reshape(landmarks, [-1, 2])
    pi = tf.constant(math.pi)
    delta_eye = landmarks[1] - landmarks[0]
    rad = tf.atan2(delta_eye[0], delta_eye[1])
    angle = rad * 180 / pi
    ed_eye = tf.linalg.norm(delta_eye) # euclidean distance between eyes
    eye_center = tf.reduce_mean(landmarks[:2], axis=0)
    center = tf.cast(img_shape[:2], tf.float32) / 2
    center_shift = eye_center - center
    width = ed_eye * 1.865
    height = ed_eye * 3.018
    translation_matrix = tf.convert_to_tensor([
        [1, 0, center_shift[1]],
        [0, 1, center_shift[0] + ed_eye * 0.3],
        [0, 0, 1]])
    translation_matrix = tf.linalg.inv(translation_matrix)
    def fn(img, center, eye_center, angle, img_shape, translation_matrix):
        rotation_matrix = cv2.getRotationMatrix2D(tuple(eye_center[::-1]), angle, 1)
        rotation_matrix = np.concatenate([rotation_matrix, [[0,0,1]]], 0)
        M = np.matmul(rotation_matrix, translation_matrix)
        M = M[:2]
        img = cv2.warpAffine(img, M, (img_shape[1], img_shape[0]))
        return img
    img = tf.numpy_function(
        fn,
        [img, center, eye_center, angle, img_shape, translation_matrix],
        tf.uint8)
    img = tf.reshape(img, img_shape)
    bbox = tf.convert_to_tensor(
        [center[0]-height/2, center[1]-width/2, height, width], dtype=tf.int32)
    y, x, h, w = bbox[0], bbox[1], bbox[2], bbox[3]
    h = h + y
    w = w + x
    y = tf.reduce_max([y, 0])
    y = tf.reduce_min([y, img_shape[0] - 1])
    x = tf.reduce_max([x, 0])
    x = tf.reduce_min([x, img_shape[1] - 1])
    h = tf.reduce_max([h - y, 1])
    h = tf.reduce_min([h, img_shape[0] - y])
    w = tf.reduce_max([w - x, 1])
    w = tf.reduce_min([w, img_shape[1] - x])
    dst = img[y:y+h, x:x+w, :]
    return dst

def align_similarity_numpy(img, lm):
    """Image aligning function implemented in numpy.

    Notes
    -----
    Landmarks should be:
    [left_eye_y, left_eye_x, right_eye_y, right_eye_x,
    nosetip_y, nosetip_x, mouth_left_edge_y, mouth_left_edge_x,
    mouth_right_edge_y, mouth_right_edge_x]
    """
    lm = np.reshape(lm, [-1, 2])
    img_shape = img.shape[:2]
    dst = np.array([
        [0.315, 0.46],
        [0.685, 0.46],
        [0.50,  0.64],
        [0.35,  0.825],
        [0.65,  0.825]], dtype=np.float32 )
    dst  = dst * img_shape[::-1]
    lm = np.flip(lm, -1)
    M = cv2.estimateAffinePartial2D(lm, dst)[0]
    warped = cv2.warpAffine(img,M,(img_shape[1],img_shape[0]), borderValue = 0.0)
    return warped

def align_similarity_tensorflow(img, lm):
    """Tensorflow graph compatible version of method: align_similarity_numpy."""
    img_shape = tf.shape(img)
    dst = tf.numpy_function(align_similarity_numpy, [img, lm], tf.uint8)
    dst = tf.reshape(dst, img_shape)
    return dst

def align_similarity_zoomed_numpy(img, lm):
    """Image aligning function implemented in numpy.

    Notes
    -----
    Landmarks should be:
    [left_eye_y, left_eye_x, right_eye_y, right_eye_x,
    nosetip_y, nosetip_x, mouth_left_edge_y, mouth_left_edge_x,
    mouth_right_edge_y, mouth_right_edge_x]
    """
    lm = np.reshape(lm, [-1, 2])
    img_shape = img.shape[:2]
    dst = np.array([
        [0.222, 0.37],
        [0.777, 0.37],
        [0.50,  0.64],
        [0.275, 0.9175],
        [0.275, 0.9175]], dtype=np.float32 )
    dst  = dst * img_shape[::-1]
    lm = np.flip(lm, -1)
    M = cv2.estimateAffinePartial2D(lm, dst)[0]
    warped = cv2.warpAffine(img,M,(img_shape[1],img_shape[0]), borderValue = 0.0)
    return warped

def align_similarity_zoomed_tensorflow(img, lm):
    """Tensorflow graph compatible version of method: align_similarity_zoomed_numpy."""
    img_shape = tf.shape(img)
    dst = tf.numpy_function(align_similarity_zoomed_numpy, [img, lm], tf.uint8)
    dst = tf.reshape(dst, img_shape)
    return dst

def align_similarity_v2_numpy(img, lm):
    """Image aligning function implemented in numpy.

    Notes
    -----
    Landmarks should be:
    [left_eye_y, left_eye_x, right_eye_y, right_eye_x,
    nosetip_y, nosetip_x, mouth_left_edge_y, mouth_left_edge_x,
    mouth_right_edge_y, mouth_right_edge_x]
    """
    lm = np.reshape(lm, [-1, 2])
    lm = lm[...,::-1]
    img_shape = img.shape[:2]
    dst = np.array([
        [0.34, 0.46],
        [0.66, 0.46],
        [0.5, 0.64],
        [0.37, 0.82],
        [0.63, 0.82] ], dtype=np.float32 )
    dst =  dst * img_shape[::-1]
    tform = skitrans.SimilarityTransform()
    tform.estimate(lm, dst)
    M = tform.params[0:2,:]
    warped = cv2.warpAffine(img,M,(img_shape[1],img_shape[0]), borderValue = 0.)
    return warped

def align_insightface_tensorflow(img, lm):
    """Tensorflow graph compatible version of method: align_insightface_numpy."""
    img_shape = tf.shape(img)
    dst = tf.numpy_function(align_insightface_numpy, [img, lm], tf.uint8)
    dst = tf.reshape(dst, img_shape)
    return dst

def align_insightface_numpy(img, lm):
    """Image aligning function implemented in numpy.

    Notes
    -----
    Landmarks should be:
    [left_eye_y, left_eye_x, right_eye_y, right_eye_x,
    nosetip_y, nosetip_x, mouth_left_edge_y, mouth_left_edge_x,
    mouth_right_edge_y, mouth_right_edge_x]
    """
    lm = np.reshape(lm, [-1, 2])
    lm = lm[...,::-1]
    img_shape = img.shape[:2]
    dst = np.array([
        [0.34191608, 0.46157411],
        [0.65653392, 0.45983393],
        [0.500225  , 0.64050538],
        [0.3709759 , 0.82469198],
        [0.631517  , 0.82325091]])
    dst =  dst * img_shape[::-1]
    tform = skitrans.SimilarityTransform()
    tform.estimate(lm, dst)
    M = tform.params[0:2,:]
    warped = cv2.warpAffine(img,M,(img_shape[1],img_shape[0]), borderValue = 0.)
    return warped

def align_similarity_v2_tensorflow(img, lm):
    """Tensorflow graph compatible version of method: align_similarity_v2_numpy."""
    img_shape = tf.shape(img)
    dst = tf.numpy_function(align_similarity_v2_numpy, [img, lm], tf.uint8)
    dst = tf.reshape(dst, img_shape)
    return dst
