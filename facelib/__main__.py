"""Entry point for facelib pip package."""

import argparse
from .helper import install_tflite_runtime
parser = argparse.ArgumentParser(
    description='Face Recognition Library.')

parser.add_argument(
    '--install-tflite',
    action='store_true',
    help='Installs tflite-runtime pip package.'
)
args = parser.parse_args()

if args.install_tflite:
    install_tflite_runtime()