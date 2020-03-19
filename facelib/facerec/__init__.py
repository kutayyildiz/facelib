"""Face recognition.

Includes face detection, facial landmark detection and facial feature extraction.
Extra Dependencies: tflite-runtime(pip)
"""
from .face_detection import HAARFaceDetector
from .face_detection import LBPFaceDetector
from .face_detection import SSDFaceDetector
from .feature_extraction import FeatureExtractor
from .landmark_detection import LandmarkDetector
from .pipeline import Pipeline

