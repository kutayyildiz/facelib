from pathlib import Path
from joblib import dump, load
import pkg_resources

import cv2
from sklearn.svm import SVC
import numpy as np

from facelib import facerec
from facelib._utils import helper


class Predict:
    """Face recognition on image/images in folder/video."""

    def __init__(self,):
        self.detector = self.get_pipelined_detector()

    def gen_folder(self, path_folder):
        """Yield immediate subdirectories inside a folder."""
        p = Path(path_folder)
        for posix_path in p.iterdir():
            if posix_path.is_dir():
                yield posix_path
        

    def gen_image(self, posix_path_folder):
        """Yield image files inside a folder."""
        extensions = ['bmp', 'dib', 'jpg', 'jpeg', 'jpe', 'png']
        extensions = ['*.' + ext for ext in extensions]
        list_images = []
        for ext in extensions:
            for posix_path in posix_path_folder.glob(ext):
                path_img = str(posix_path)
                img = cv2.imread(path_img)[...,::-1] # bgr->rgb
                yield img

    def get_pipelined_detector(self,):
        """Return default pipeline(facelib.facerec.Pipeline)."""
        list_pipeline = helper.get_default_pipeline
        pipeline = facerec.pipeline(*list_pipeline)
    
    def train(self, path_folder, name_clf):
        """Train a classifier using image folders named as people names."""
        X = []
        y = []
        last_ftr = None
        for posix_path in self.gen_folder(path_folder):
            name_person = posix_path.stem
            for img in self.gen_image(posix_path):
                _, _, features = self.detector.predict(img)
                if not features:
                    continue
                if len(features) == 1:
                    ftr = features[0]
                    last_ftr = ftr
                if len(features) > 1:
                    if last_ftr is not None:
                        distances = np.linalg.norm(features - last_ftr, axis=1)
                        closest_index = distances.argmin()
                        ftr = features[closest_index]
                X.append(ftr)
                y.append(name_person)
        clf = SVC()
        clf.fit(X, y)
        path_posix_clf = helper.get_classifier_path(name_clf)
        if path_posix_clf.exists():
            print('Classifier named "{}" is overwritten.'.format(name_clf))
        dump(clf, str(path_posix_clf))

    def predict(self, posix_path_folder, name_clf):
        config_settings = helper.get_settings_config()
        tolerance = config_settings['Predict']['tolerance']
        posix_path_clf = helper.get_classifier_path(name_clf)
        assert posix_path_clf.exists(), 'Classifier name "{}" does not exist.'.format(name_clf)
        clf = load(str(posix_path_clf))
        for img in self.gen_image(posix_path_folder):
            _, _, features = self.detector.predict(img)
            probas = clf.predict_proba(features)
            names_predicted = [clf.classes_[p.argmax()] if p.max() > (1 - tolerance) else None for p in probas]
        return names_predicted









