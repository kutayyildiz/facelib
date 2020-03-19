"""Entry point for facelib pip package."""

import argparse
import logging
import sys
from itertools import groupby
sys.path.append('/media/disk_nvme/github/kutay/facelib')

from facelib._utils import helper

import logging
logging.getLogger().setLevel(logging.INFO)
try:
    from facelib import facerec
except:
    logging.warning('Failed to load facerec module, commands:[train, predict] will be unavailable.')
    logging.info('Check if tflite-runtime is installed. $ facelib install --tflite-runtime')


class FaceLib:
    def __init__(self,):
        self.parser = argparse.ArgumentParser(
            description='Face Recognition Library',
            formatter_class=argparse.RawTextHelpFormatter
        )
        self.subparsers = self.parser.add_subparsers()
        
        self.parser_predict()
        self.parser_train()
        self.parser_model()
        self.parser_install()

    def parse(self, args):
        parsed = self.parser.parse_args(args)
        return parsed.func(parsed)

    def parser_train(self,):
        parser = self.subparsers.add_parser(
            name='train',
            help="""train a face classifier using images inside a folder
training_folder/
├── alice/
│   ├── img1.jpg
├── bob/
│   ├── img1.jpg
│   ├── img2.jpg
"""
        )
        parser.set_defaults(func=self.train)

    def parser_predict(self,):
        parser = self.subparsers.add_parser(
            name='predict',
            help='predict on images inside a folder',
        )
        parser.add_argument(
            'path',
            type=str,
            help='folder to be predicted',
        )
        parser.add_argument(
            '-t', '--type',
            nargs='+',
            choices=['fd, ld, fe'],
            help="""default="fd ld fe"
fd: Face Detection
ld: Landmark Detection
fe: Feature Extraction
"""
        )
        parser.add_argument(
            '-v', '--verbose',
            type=int,
            choices=range(4),
            default=1,
            metavar='',
            help='select verbosity level(default=1, min=0, max=3).'
        )
        parser.add_argument(
            '-s', '--save',
            action='store_true',
            help='save predictions as csv',
        )
        parser.add_argument(
            '-p', '--plot',
            action='store_true',
            help='plot images and save them to folder',
        )
        parser.set_defaults(func=self.predict)

    def parser_model(self,):
        def _normalize(list_model):
            grouped_list = [list(g) for k, g in groupby(list_model, key=lambda x: x.split('_')[0])]
            normalized = '{'
            for group in grouped_list:
                normalized += ', '.join(group)
                normalized += ',\n'
            normalized = normalized[:-2] + '}'
            return normalized

        parser= self.subparsers.add_parser(
            name='model',
            help='select default models',
            formatter_class=argparse.RawTextHelpFormatter
        )
        parser.add_argument(
            '-d', '--delete',
            type=str,
            metavar='',
            help='delete the saved pipeline'
        )
        parser.add_argument(
            'l', '--load',
            type=str,
            metavar='',
            help='load a saved pipeline as default',
        )
        parser.add_argument(
            '-s', '--save',
            type=str,
            metavar='',
            help='save the pipeline for future use',
        )
        fd_models = helper.get_available_models('face_detection')
        parser.add_argument(
            '-fd', '--face-detection',
            type=str,
            choices=fd_models,
            metavar='',
            help=_normalize(fd_models),
        )
        lm_models = helper.get_available_models('landmark_detection') 
        parser.add_argument(
            '-lm', '--landmark-detection',
            type=str,
            choices=lm_models,
            metavar='',
            help=_normalize(lm_models),
        )
        fe_models = helper.get_available_models('feature_extraction')
        parser.add_argument(
            '-fe', '--feature-extraction',
            type=str,
            choices=fe_models,
            metavar='',
            help=_normalize(fe_models),
        )
        parser.set_defaults(func=self.model)
    
    def parser_install(self,):
        parser = self.subparsers.add_parser(
            name='install',
            help='install a package or dataset.',
        )
        parser.add_argument(
            '-tr', '--tflite-runtime',
            action='store_true',
            help='install tflite-runtime(pip package)',
        )
    def train(self, args):
        return args
    
    def predict(self, args):

        return args

    def model(self, args):
        if args.delete is not None:
            helper.del_pipeline(args.delete)
        assert not (args.load and args.save) is not None, 'Use -s and -l seperately.'
        fd, lm, fe = args.face_detection, args.landmark_detection, args.feature_extraction
        if args.load is not None:
            helper.set_default_pipeline(args.load)
        if args.save is not None:
            assert (fd and lm and fe) is not None, 'All type of models should be selected to save pipeline.'
            print('saving to {}'.format(args.save))
            helper.save_pipeline([fd, lm, fe], args.save)
        if fd is not None:
            helper.set_default_model('face_detection', fd)
        if lm is not None:
            helper.set_default_model('landmark_detection', lm)
        if fe is not None:
            helper.set_default_model('feature_extraction', fe)
        return args
        
    def install(self, args):
        if args.tflite_runtime:
            helper.install_tflite_runtime()
        
if __name__ =='__main__':
    fl = FaceLib()
    args = sys.argv
    fl.parse(args=args[1:])
