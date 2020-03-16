"""Quantized Single Shot Detector for face detection."""

import cv2
import numpy as np
import pkg_resources
from tflite_runtime.interpreter import Interpreter, load_delegate


class SSD:
    """SSD face detection(supports tpu).

    Notes
    -----
    ref(tflite) : http://download.tensorflow.org/models/object_detection/facessd_mobilenet_v2_quantized_320x320_open_image_v4.tar.gz
    Warnings
    --------
    If the image to be predicted is not square shaped prediction
        accuracy will be dramatically reduced.
    """

    def __init__(self, input_img_channel='rgb', resize=None, tpu=False):
        if tpu:
            name_tflite = 'mobilenet_ssd_v2_320x320_open_image_v4_int8_edgetpu.tflite'
            delegates = [load_delegate('libedgetpu.so.1')]
        else:
            name_tflite = 'mobilenet_ssd_v2_320x320_open_image_v4_int8.tflite'
            delegates = []
        path_tflite = pkg_resources.resource_filename(
            'facelib.facerec.face_detection',
            'data/' + name_tflite
        )

        self.face_detection_inference = Interpreter(
            model_path = path_tflite,
            experimental_delegates=delegates
        )
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

    def _extract_boxes(self, boxes, probs):
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
            3d array with shape (112,112,3) with dtype `uint8`.
        """
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
        return bboxes
