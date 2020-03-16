import cv2
import numpy as np
import pkg_resources
from tflite_runtime.interpreter import Interpreter, load_delegate


class FeatureExtractor:
    def __init__(self, name_model='mobilenet', inference='fp32', input_img_channel='rgb', tpu=False):
        assert inference in ['fp32', 'int8'], 'Available inferences are: fp32, int8'
        name_model = name_model + '_' + inference
        if tpu:
            delegates = [load_delegate('libedgetpu.so.1')]
            name_model = name_model + '_edgetpu'
        else:
            delegates = []
        name_tflite = name_model + '.tflite'
        path_tflite = pkg_resources.resource_filename(
            'facelib.facerec.feature_extraction',
            'data/' + name_tflite
        )

        self.feature_extraction_inference = Interpreter(
            model_path = path_tflite,
            experimental_delegates = delegates,
        )
        self.feature_extraction_inference.allocate_tensors()
        self.input_index = self.feature_extraction_inference.get_input_details()[0]['index']
        self.output_index = self.feature_extraction_inference.get_output_details()[0]['index']

        assert input_img_channel in ['bgr', 'rgb'], 'Incorrect input_img_channel'
        self.input_img_channel = input_img_channel

    def predict(self, img):
        img = cv2.resize(img, (112, 112))
        if self.input_img_channel == 'bgr':
            img = img[...,::-1]
        img = np.expand_dims(img, 0)
        img = img.astype(np.float32)
        img = (img - 127.5) / 128.
        self.feature_extraction_inference.set_tensor(self.input_index, img)
        self.feature_extraction_inference.invoke()
        features = self.feature_extraction_inference.get_tensor(self.output_index)[0]
        return features 
