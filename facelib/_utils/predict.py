from pathlib import Path

import cv2

from facelib import facerec
from facelib._utils import helper

def gen_folder(path_folder):
    """Yield immediate subdirectories inside a folder."""
    p = Path(path_folder)
    for posix_path in p.iterdir():
        if posix_path.is_dir():
            yield str(posix_path)
    

def gen_image(path_folder):
    """Yield image files inside a folder."""
    p = Path(path_folder)
    extensions = ['bmp', 'dib', 'jpg', 'jpeg', 'jpe', 'png']
    extensions = ['*.' + ext for ext in extensions]
    list_images = []
    for ext in extensions:
        for posix_path in p.glob(ext):
            path_img = str(posix_path)
            img = cv2.imread(path_img)
            yield img

def get_pipelined_detector():
    """Return default pipeline(facelib.facerec.Pipeline)."""
    list_pipeline = helper.get_default_pipeline
    pipeline = facerec.pipeline(*list_pipeline)

