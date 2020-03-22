"""Quantized Single Shot Detector for face detection."""
from pathlib import Path

import cv2
import numpy as np
import pkg_resources
from tflite_runtime.interpreter import Interpreter, load_delegate

from facelib._utils import helper


class SSDFaceDetector:
    """SSD face detection(supports tpu).

    Notes
    -----
    ref(tflite) : http://download.tensorflow.org/models/object_detection/facessd_mobilenet_v2_quantized_320x320_open_image_v4.tar.gz

    Warnings
    --------
    If the image to be predicted is not square shaped, prediction accuracy is dramatically reduced.
    """

    def __init__(self, input_img_channel='rgb', resize=None, tpu=False):
        if tpu:
            name_model = 'ssd_int8_tpu'
            delegates = [load_delegate('libedgetpu.so.1')]
        else:
            name_model = 'ssd_int8_cpu'
            delegates = []
        path_tflite = helper.get_path('face_detection', name_model)
        self.face_detection_inference = Interpreter(
            model_path=str(path_tflite),
            experimental_delegates=delegates)
        self.face_detection_inference.allocate_tensors()
        self.index_input = self.face_detection_inference.get_input_details()[0]['index']
        self.index_boxes = self.face_detection_inference.get_output_details()[0]['index']
        self.index_probs = self.face_detection_inference.get_output_details()[2]['index']
        assert input_img_channel in ['bgr', 'rgb'], 'Incorrect input_img_channel'
        self.input_img_channel = input_img_channel

    def _preprocess(self, img):
        if self.input_img_channel == 'bgr':
            img = img[...,::-1] # bgr to rgb conversion
        img = cv2.resize(img, (320, 320))
        img = np.expand_dims(img, 0)
        return img

    @staticmethod
    def _extract_boxes(boxes, probs):
        threshold = 1e-1
        mask = np.less(probs, 1.)
        probs = probs[mask]
        mask = np.greater(probs, threshold)
        num_boxes = np.count_nonzero(mask)
        output = boxes[0][:num_boxes]
        return output

    def predict(self, img):
        """Detect faces inside an image.

        Parameters
        ----------
        img : ndarray
            3d array with shape (height,width,3) with dtype `uint8`.
        """
        original_shape = img.shape
        img = self.squarize_img(img)
        squared_shape = img.shape
        img = self._preprocess(img)
        self.face_detection_inference.set_tensor(self.index_input, img)
        self.face_detection_inference.invoke()
        pred_boxes = self.face_detection_inference.get_tensor(self.index_boxes)
        pred_probs = self.face_detection_inference.get_tensor(self.index_probs)
        bboxes = self._extract_boxes(pred_boxes, pred_probs)
        bboxes = bboxes.reshape(-1, 2, 2)
        fix = bboxes[:,1,1] - bboxes[:,0,1]
        fix = fix / 6.
        fix = np.expand_dims(fix, -1)
        fix = fix * [0., 1, 0, -1]
        fix = fix.reshape(-1, 2, 2)
        bboxes = bboxes + fix
        bboxes = self.normalize_bboxes(bboxes, original_shape, squared_shape)
        return bboxes

    @staticmethod
    def squarize_img(img):
        height, width = img.shape[:2]
        max_dim = max(height, width)
        new_shape = (max_dim, max_dim, 3)
        squarized = np.zeros(new_shape, dtype=img.dtype)
        squarized[:height,:width] += img
        return squarized

    @staticmethod
    def normalize_bboxes(bboxes, original_shape, squared_shape):
        change_ratio = np.array(squared_shape[:2], dtype=bboxes.dtype) / original_shape[:2]
        bboxes = bboxes * change_ratio
        return bboxes
