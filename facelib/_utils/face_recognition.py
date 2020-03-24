from pathlib import Path
from textwrap import dedent
import sys
from joblib import dump, load
import pkg_resources
from collections import defaultdict

import cv2
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

from facelib import facerec
from facelib._utils import helper

def plot(img, bboxes, landmarks=None, persons_id=None):
    bboxes = bboxes.reshape(-1, 2, 2) * img.shape[:2]
    bboxes = bboxes[...,::-1].astype(np.int32)
    img = img.copy()
    for bbox in bboxes:
        cv2.rectangle(img, tuple(bbox[0]), tuple(bbox[1]), (0,255,0), 2)
    if landmarks is not None:
        landmarks = landmarks.reshape(-1, 5, 2) * img.shape[:2]
        landmarks = landmarks[...,::-1].astype(np.int32)
        for landmark in landmarks:
            for lm in landmark:
                cv2.circle(img, tuple(lm), 1, (0,0,255), 2)
    if persons_id is not None:
        font_face = cv2.FONT_HERSHEY_PLAIN
        font_scale = min(img.shape[:2]) / 500
        thickness = 1
        if bboxes is not None:
            for bbox, p_id in zip(bboxes, persons_id):
                size_text = cv2.getTextSize(p_id, font_face, font_scale, thickness)
                w, h = size_text[0]
                diff_w = int((w - (bbox[1][0] - bbox[0][0])) / 2)
                org = tuple(bbox[0] + [-diff_w, h])
                cv2.putText(img, p_id, org, font_face, font_scale,
                            (255,255,255), thickness, cv2.LINE_AA)
    return img

def get_default_pipeline_detector():
    """Return default pipeline(facelib.facerec.Pipeline)."""
    list_pipeline = helper.get_default_pipeline()
    print('Current pipeline: {}'.format(', '.join(list_pipeline)))
    name_fd, name_ld, name_fe = list_pipeline
    # Face Detector
    if name_fd == 'haarcascade':
        face_detector = facerec.HAARFaceDetector()
    elif name_fd == 'lbpcascade':
        face_detector = facerec.LBPFaceDetector()
    elif name_fd.rsplit('_', 1)[0] == 'ssd_int8':
        tpu = (name_fd.rsplit('_', 1)[-1] == 'tpu')
        face_detector = facerec.SSDFaceDetector(tpu=tpu)
    else:
        assert False, 'Failed to load face detection model.'
    # Landmark Detector
    landmark_detector = facerec.LandmarkDetector(name_ld)
    feature_extractor = facerec.FeatureExtractor(name_fe)
    pipeline = facerec.Pipeline(face_detector, landmark_detector, feature_extractor)
    return pipeline

def gen_folder(posix_path_folder):
    """Yield immediate subdirectories inside a folder."""
    for posix_path in posix_path_folder.iterdir():
        if posix_path.is_dir():
            yield posix_path

def gen_image(posix_path_folder):
    """Yield image files inside a folder."""
    extensions = ['bmp', 'dib', 'jpg', 'jpeg', 'jpe', 'png']
    extensions = ['*.' + ext for ext in extensions]
    for ext in extensions:
        for posix_path in posix_path_folder.glob(ext):
            path_img = str(posix_path)
            name_img = posix_path.name
            img = cv2.imread(path_img)[...,::-1] # bgr->rgb
            yield img, name_img

def augment_data(data, number, noise_range=1e-2):
    """Augments a list/array of lists/arrays."""
    std = np.std(data, axis=0)
    num_augments = number - len(data)
    if num_augments <= 0:
        return data
    diminish = noise_range / float(num_augments)
    augmented = data.copy()
    for _ in range(num_augments):
        rand = np.random.rand(std.shape[0]) * 2 - 1.
        noise = noise_range * rand
        fake_data = noise * std + data[np.random.randint(len(data))]
        fake_data = np.expand_dims(fake_data, axis=0)
        augmented = np.append(augmented, fake_data, axis=0)
        noise_range -= diminish
    return augmented

def train(detector, posix_path_folder, name_clf):
    """Train a classifier using image folders named as people names."""
    dict_features = defaultdict(lambda: np.empty((0, 128), np.float32))
    last_ftr = None
    for posix_path in gen_folder(posix_path_folder):
        name_person = posix_path.stem
        for img, _ in gen_image(posix_path):
            _, _, features = detector.predict(img)
            if len(features) == 0:
                continue
            if len(features) == 1:
                ftr = features[0]
                last_ftr = ftr
            if len(features) > 1:
                if last_ftr is not None:
                    distances = np.linalg.norm(features - last_ftr, axis=1)
                    closest_index = distances.argmin()
                    ftr = features[closest_index]
                else:
                    continue
            dict_features[name_person] = np.append(
                dict_features[name_person], np.expand_dims(ftr, 0), axis=0)
    X = []
    y = []
    for name in dict_features:
        num_augment = helper.get_num_augment()
        features = dict_features[name]
        num_augment = num_augment - len(features)
        for feature in augment_data(features, num_augment):
            X.append(feature)
            y.append(name)
    min_data_per_class = min(np.unique(y, return_counts=True)[1])
    n_neighbors = 3 if min_data_per_class < 5 else 5
    clf = KNeighborsClassifier(metric='cosine', n_neighbors=n_neighbors)
    clf.fit(X, y)
    helper.save_classifier(clf, name_clf)

def predict(detector, posix_path_folder, name_clf):
    config_settings = helper.get_settings_config()
    tolerance = float(config_settings['Predict']['tolerance'])
    clf = helper.get_classifier(name_clf)
    print('Predictions:')
    for img, name_img in gen_image(posix_path_folder):
        _, _, features = detector.predict(img)
        probas = clf.predict_proba(features)
        names_predicted = [
            clf.classes_[p.argmax()] if p.max() > (1 - tolerance) else None for p in probas]
        print('{}: {}'.format(name_img, names_predicted))