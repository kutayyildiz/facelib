"""Helper methods for facerec."""

from configparser import ConfigParser, ExtendedInterpolation
from urllib import request
from pathlib import Path
import pkg_resources


def install_data(type_model, name_model):
    types = [
        'face_detection'
        'landmark_detection',
        'feature_extraction',]
    assert type_model in types, 'Available types are: {}'.format(str(types))
    path_config = pkg_resources.resource_filename('facelib.facerec', 'links_to_models.ini')
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read(path_config)
    keys = list(config[type_model].keys())
    assert name_model in keys, 'Available {} {} models are:\n{}'.format(*type_model.split('_'), str(keys))
    link_data = config[type_model][name_model]
    extension = Path(link_data).suffix
    response = request.urlopen(link_data)
    path_file = pkg_resources.resource_filename(
        'facelib.facerec.' + type_model,
        'data/' + name_model + extension
    )
    with open(path_file, 'wb') as f:
        print('Downloading from url <{}>...'.format(link_data))
        f.write(response.read())
        print('File <{}> successfully created.'.format(path_file))
