"""Full face recognition pipeline."""

import cv2
import numpy as np
from skimage import transform as skitrans


class Pipeline:
    """Full face recognition pipeline that can be used
    to detect faces and extract features.

    Parameters
    ----------
    face_detector : Detector
        face detector with a function 'detect'
    landmark_detector : Detector
        landmar detector with a function 'detect'
    face_feature_extractor : Detector
        face feature extractor with a function 'detect'
    
    Examples
    --------
    >>> import cv2
    >>> from facelib import facerec
    >>> face_detector = facerec.SSDFaceDetector()
    >>> landmark_detector = facerec.LandmarkDetector()
    >>> feature_extractor = facerec.FeatureExtractor()
    >>> pipeline = facerec.Pipeline(
            face_detector,
            landmark_detector,
            feature_extractor
        )
    >>> img = cv2.imread('path_to_some_img')[...,::-1]
    >>> bboxes, landmarks, features = fr.predict(img)
    """
        
    def __init__(self, face_detector, face_landmark_detector, face_feature_extractor, scale=None):
        self.face_detector = face_detector
        self.landmark_detector = face_landmark_detector
        self.feature_extractor = face_feature_extractor
        self.scale = scale

    def predict(self, img):
        bboxes = self.face_detector.predict(img)
        if type(bboxes) == tuple:
            bboxes, landmarks = bboxes
            if len(bboxes) == 0:
                return [], [], []
        else:
            if len(bboxes) == 0:
                return [], [], []
            if self.scale is not None:
                bboxes = np.array([self.scale_bbox(bbox, self.scale) for bbox in bboxes])
            faces_cropped = [self.crop_to_bbox(img, bbox) for bbox in bboxes]
            landmarks = np.stack([self.landmark_detector.predict(face) for face in faces_cropped])
            landmarks = np.stack([self.normalize_landmark_coor(img, bbox, landmark) for bbox, landmark in zip(bboxes, landmarks)])
        # do not use np.stack for aligned faces since faces have different width/depth
        faces_aligned = [self.align(img, lm) for lm in landmarks]
        faces_feature = np.stack([self.feature_extractor.predict(fa) for fa in faces_aligned])
        return bboxes, landmarks, faces_feature

    def scale_bbox(self, bbox, scale = [1,1]):
        scale = np.array(scale)
        bbox = bbox.reshape([2, 2])
        height, width = bbox[1] - bbox[0]
        s = (scale - 1) / 2.
        bbox = bbox + [
            [-height * s[0], -width * s[1]], 
            [height * s[0], width * s[1]]]
        return bbox 

    def normalize_landmark_coor(self, img, bbox, landmark):
        # convert landmark coordinates relative to the whole image
        bbox = np.clip(bbox, 0, 1) # some values may be outside of range: [0,1]
        shape_img = img.shape[:2]
        bbox = np.reshape(bbox, [-1, 2]) * shape_img # fraction to pixel
        height, width = bbox[1] - bbox[0]
        landmark = np.reshape(landmark, [-1, 2])
        landmark = landmark * (height, width)
        landmark = landmark + bbox[0]
        landmark = landmark / img.shape[:2]
        return landmark

    def crop_to_bbox(self, img, bbox):
        shape_img = img.shape[:2]
        bbox = np.clip(bbox, 0., 1.)
        bbox = np.reshape(bbox, [-1, 2]) * shape_img
        bbox = bbox.flatten().astype(np.int32)
        img_cropped = img[bbox[0]:bbox[2], bbox[1]:bbox[3], :] # crop
        return img_cropped
    
    def align(self, img, lm):
        lm = np.reshape(lm, [-1, 2])
        lm = lm * img.shape[:2]
        dist = np.linalg.norm(lm[0] - lm[-1]) * 1.5
        dist = dist.astype(np.int32)
        center = np.mean(lm, 0).astype(np.int32)
        bbox = [center - dist, center + dist]
        bbox = np.clip(bbox, 0, img.shape[:2])
        img = img[
            bbox[0][0]: bbox[1][0],
            bbox[0][1]: bbox[1][1],
            :]
        lm = lm - bbox[0]
        lm = lm[...,::-1]
        img_shape = img.shape[:2]
        # coordinates are taken from:
        # https://github.com/deepinsight/insightface/blob/master/src/common/face_preprocess.py
        # normalized to fraction from pixel values
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
        aligned = cv2.warpAffine(img,M,(img_shape[1],img_shape[0]))
        return aligned
