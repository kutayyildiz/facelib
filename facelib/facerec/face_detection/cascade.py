"""Cascaded face detection using OpenCv."""

import cv2
import numpy as np
import pkg_resources


class CascadeClassifier:
    def __init__(self, name_xml, min_neighbors, scale, resize=None, input_img_channel='rgb'):
        path_xml = pkg_resources.resource_filename(
            'facelib.facerec.face_detection',
            'data/' + name_xml
        )
        self.face_cascade = cv2.CascadeClassifier(path_xml)
        self.min_neighbors = min_neighbors
        self.resize = resize
        assert input_img_channel in ['gray', 'bgr', 'rgb']
        self.input_img_channel = input_img_channel
        self.scale = scale

    def predict(self, img):
        if self.input_img_channel != 'gray':
            img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.resize is not None:
            img= cv2.resize(img, self.resize)
        bboxes = self.face_cascade.detectMultiScale(img, scaleFactor=self.scale, minNeighbors=self.min_neighbors)
        bboxes = np.array(bboxes)
        bboxes = bboxes.reshape(-1, 2, 2)[:,:,::-1]
        top_left = bboxes[:,0]
        bottom_right = np.sum(bboxes, -2)
        bboxes = np.concatenate([top_left, bottom_right], -1)
        bboxes = bboxes.reshape(-1, 2, 2)
        bboxes = bboxes / img.shape[:2]
        return bboxes


class LBPCascade(CascadeClassifier):
    def __init__(self, min_neighbors=3, scale=1.2, **kwargs):
        name_xml = 'lbpcascade_frontalface.xml'
        super(LBPCascade, self).__init__(
            name_xml,
            min_neighbors=min_neighbors,
            scale=scale,
            **kwargs,
        )

class HAARCascade(CascadeClassifier):
    def __init__(self, min_neighbors=5, scale=1.2, **kwargs):
        name_xml = 'haarcascade_frontalface.xml'
        super(HAARCascade, self).__init__(
            name_xml,
            min_neighbors=min_neighbors,
            scale=scale,
            **kwargs,
        )
