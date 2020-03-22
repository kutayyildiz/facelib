"""Script for facial landmark detection."""

from pathlib import Path

import cv2
import numpy as np
import pkg_resources
from tflite_runtime.interpreter import Interpreter, load_delegate

from facelib._utils import helper


class LandmarkDetector:
    def __init__(self, name, input_img_channel='rgb'):
        name_model, inference, tpu = name.split('_')
        tpu = (tpu == 'tpu')
        assert inference in ['fp32', 'int8'], 'Available inferences are: fp32, int8'
        name_model = name_model + '_' + inference
        if tpu:
            delegates = [load_delegate('libedgetpu.so.1')]
            name_model = name_model + '_tpu'
        else:
            name_model = name_model + '_cpu'
            delegates = []
        path_tflite = helper.get_path('landmark_detection', name_model)

        self.landmark_detection_inference = Interpreter(
            model_path = str(path_tflite),
            experimental_delegates = delegates,
        )
        self.landmark_detection_inference.allocate_tensors()
        self.input_index = self.landmark_detection_inference.get_input_details()[0]['index']
        self.output_index = self.landmark_detection_inference.get_output_details()[0]['index']

        assert input_img_channel in ['bgr', 'rgb'], 'Incorrect input_img_channel'
        self.input_img_channel = input_img_channel

    def predict(self, img):
        img = cv2.resize(img, (96, 96))
        if self.input_img_channel == 'bgr':
            img = img[...,::-1]
        img = np.expand_dims(img, 0)
        img = img.astype(np.float32)
        img = (img - 127.5) / 128.
        self.landmark_detection_inference.set_tensor(self.input_index, img)
        self.landmark_detection_inference.invoke()
        landmarks = self.landmark_detection_inference.get_tensor(self.output_index)[0]
        landmarks = landmarks.reshape(-1, 5, 2)
        return landmarks
