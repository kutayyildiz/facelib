"""Entry point for facelib pip package."""
import argparse
import sys
from functools import wraps
from itertools import groupby
from pathlib import Path
from textwrap import dedent

import cv2

from facelib._utils import helper

try:
    from facelib._utils import face_recognition
except:
    print('Warning: Failed to load FaceRecognition.\n'
          'try installing tflite-runtime: $ facelib install --tflite-runtime')


class FaceLib:
    def __init__(self,):
        self.parser = argparse.ArgumentParser(
            description='Face Recognition Library',
            formatter_class=argparse.RawTextHelpFormatter)
        self.subparsers = self.parser.add_subparsers()
        self._parser_settings()
        self._parser_predict()
        self._parser_train()
        self._parser_model()
        self._parser_install()
        self._parse()

    def _parse(self,):
        args = sys.argv[1:]
        if not args:
            self.parser.parse_args(['-h'])
            sys.exit()
        parsed = self.parser.parse_args(args)
        parsed.func(parsed)

    def _parser_settings(self,):
        config = helper.get_settings_config()
        parser = self.subparsers.add_parser(name='setting',
                                            help='Adjust settings.',
                                            formatter_class=argparse.RawTextHelpFormatter)
        # Group: Predict
        config_predict = config['Predict']
        group_predict = parser.add_argument_group('Predict')
        group_predict.add_argument('-t', '--tolerance', type=float, help=dedent("""\
            Classification tolerance. range: (0:1]
            current:{}, recommended:0.5""".format(config_predict['tolerance'])))
        # Group: Train
        config_train = config['Train']
        group_train = parser.add_argument_group('Train')
        group_train.add_argument('-n', '--num-augment', type=int, help=dedent("""\
            Augment class images.
            current:{}, recommended:5""".format(config_train['num_augment'])))
        parser.set_defaults(func=self._settings)

    def _parser_train(self,):
        parser = self.subparsers.add_parser(name='train', help=(dedent("""\
                train a face classifier using images inside a folder
                training_folder/
                ├── alice/
                │   ├── img1.jpg
                ├── bob/
                │   ├── img1.jpg
                │   ├── img2.jpg""")))
        parser.add_argument('path', type=str, help='folder to be predicted')
        parser.add_argument('classifier', type=str, help='name of the classifier to be trained')
        parser.set_defaults(func=self._train)

    def _parser_predict(self,):
        parser = self.subparsers.add_parser(name='predict',
                                            help='Make predictions on images inside a folder.')
        parser.add_argument('path', type=str, help='folder to be predicted')
        parser.add_argument('-clf', '--classifier', type=str,
                            help=('predict person ids'))
        parser.add_argument('-v', '--verbose', action='store_true',
                            help='enable verbosity on console')
        parser.add_argument('-p', '--plot', action='store_true',
                            help='plot and save images to folder: (path)_facelib_plotted/')
        parser.add_argument('-c', '--crop', action='store_true',
                            help='crop faces and save to folder: (path)_facelib_cropped/')
        parser.set_defaults(func=self._predict)

    def _parser_model(self,):
        def _normalize(list_model):
            grouped_list = [list(g) for k, g in groupby(list_model, key=lambda x: x.split('_')[0])]
            normalized = '{'
            for group in grouped_list:
                normalized += ', '.join(group)
                normalized += ',\n'
            normalized = normalized[:-2] + '}'
            return normalized
        parser = self.subparsers.add_parser(name='model', help='model select/delete/load/save')
        subparsers_model = parser.add_subparsers()
        # Parser: select
        parser_select = subparsers_model.add_parser(name='select', help='select default models',
                                                    formatter_class=argparse.RawTextHelpFormatter)
        fd_models = helper.get_available_models('face_detection')
        parser_select.add_argument('-fd', '--face-detection', type=str, choices=fd_models,
                                   metavar='', help=_normalize(fd_models))
        lm_models = helper.get_available_models('landmark_detection')
        parser_select.add_argument('-ld', '--landmark-detection', type=str, choices=lm_models,
                                   metavar='', help=_normalize(lm_models))
        fe_models = helper.get_available_models('feature_extraction')
        parser_select.add_argument('-fe', '--feature-extraction', type=str, choices=fe_models,
                                   metavar='', help=_normalize(fe_models))
        parser_select.set_defaults(func=self._model_select)
        # Parser: load
        parser_load = subparsers_model.add_parser(name='load', help='select a pipeline template')
        parser_load.add_argument('name', type=str, help='name of the model')
        parser_load.set_defaults(func=self._model_load)
        # Parser: save
        parser_save = subparsers_model.add_parser(name='save', help='save current pipeline')
        parser_save.add_argument('name', type=str, help='name of the model')
        parser_save.set_defaults(func=self._model_save)
        # Parser: delete
        parser_delete = subparsers_model.add_parser(name='delete',
                                                    help='delete a pipeline template')
        parser_delete.add_argument('name', type=str, help='name of the model')
        parser_delete.set_defaults(func=self._model_delete)

    def _parser_install(self,):
        parser = self.subparsers.add_parser(name='install',
                                            help='install a package or dataset.')
        parser.add_argument('-tr', '--tflite-runtime', action='store_true',
                            help='install tflite-runtime(pip package)')
        parser.set_defaults(func=self._install)

    @staticmethod
    def _settings(args):
        config = helper.get_settings_config()
        # Group: Predict
        if args.tolerance is not None:
            config['Predict']['tolerance'] = str(args.tolerance)
        # Group: Train
        if args.num_augment is not None:
            config['Train']['num_augment'] = str(args.num_augment)
        helper.set_settings_config(config)

    @staticmethod
    def _train(args):
        posix_path_folder = Path(args.path)
        name_clf = args.classifier
        detector = face_recognition.get_default_pipeline_detector()
        face_recognition.train(detector, posix_path_folder, name_clf)

    @staticmethod
    def _predict(args):
        detector = face_recognition.get_default_pipeline_detector()
        posix_path_folder = Path(args.path) # test folder
        config_settings = helper.get_settings_config()
        # Load classifier(person id)
        if args.classifier is not None:
            tolerance = float(config_settings['Predict']['tolerance'])
            posix_path_clf = helper.get_classifier_path(args.classifier)
            if not posix_path_clf.exists():
                message = dedent("""\
                    Classifier name "{}" does not exist.
                    Available classifiers are: {}""").format(
                        args.classifier, ', '.join(helper.get_available_classifiers()))
                sys.exit(message)
            clf = face_recognition.load(str(posix_path_clf))
        # Walk through the images inside the test folder
        for img, name_img in face_recognition.gen_image(posix_path_folder):
            bboxes, landmarks, features = detector.predict(img)
            print(name_img)
            print('├───{} faces detected.'.format(len(bboxes)))
            # Predict person ids and verbose to terminal
            if args.classifier is not None:
                probas = clf.predict_proba(features)
                names_predicted = [
                    clf.classes_[p.argmax()] if p.max() > (1 - tolerance) else None for p in probas]
                print('├───{}'.format(names_predicted))
            # Plot and save test images
            if args.plot:
                path_plot = posix_path_folder.parent/(
                    posix_path_folder.stem + '_facelib_plotted')
                path_plot.mkdir(parents=True, exist_ok=True)
                path_plotted_img = path_plot/name_img
                if args.classifier is None:
                    names_predicted = None
                img_plotted = face_recognition.plot(img, bboxes, landmarks, names_predicted)
                cv2.imwrite(str(path_plotted_img), img_plotted[...,::-1])
            # Crop and save test faces
            if args.crop:
                path_crop = posix_path_folder.parent/(
                    posix_path_folder.stem + '_facelib_cropped')
                path_crop.mkdir(parents=True, exist_ok=True)
                faces_cropped = [detector.crop_to_bbox(img, bbox) for bbox in bboxes]
                if args.classifier is not None:
                    i = 0
                    for img, name in zip(faces_cropped, names_predicted):
                        if name is None:
                            name = '__Unknown__'
                        path_img = path_crop/name/(str(i) + '_' + name_img)
                        path_img.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(path_img), img[...,::-1])
                        i += 1
                else:
                    i = 0
                    for img in faces_cropped:
                        path_img = path_crop/(str(i) + '_' + name_img)
                        cv2.imwrite(str(path_img), img[...,::-1])
                        i += 1

    @staticmethod
    def _model_select(args):
        current_pipeline = helper.get_default_pipeline()
        if args.face_detection is not None:
            helper.set_default_model('face_detection', args.face_detection)
        if args.landmark_detection is not None:
            helper.set_default_model('landmark_detection', args.landmark_detection)
        if args.feature_extraction is not None:
            helper.set_default_model('feature_extraction', args.feature_extraction)
        new_pipeline = helper.get_default_pipeline()
        if current_pipeline == new_pipeline:
            print('WARNING: No updates made...')
        print('Face Recognition: {}\nLandmark Detection: {}\nFeature Extraction: {}'.format(
            *new_pipeline))

    @staticmethod
    def _model_load(args):
        helper.set_default_pipeline(args.name)

    @staticmethod
    def _model_save(args):
        pipeline = helper.get_default_pipeline()
        helper.save_pipeline(pipeline, args.name)
        print('Pipeline: [{}] saved as {}'.format(', '.join(pipeline), args.name))

    @staticmethod
    def _model_delete(args):
        helper.del_pipeline(args.name)

    @staticmethod
    def _install(args):
        if args.tflite_runtime:
            helper.install_tflite_runtime()


if __name__ == '__main__':
    FaceLib()
