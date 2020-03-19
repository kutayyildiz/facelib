import platform
import subprocess
import sys
from configparser import ConfigParser, ExtendedInterpolation
from pathlib import Path
from urllib import request

import pkg_resources


def get_tflite_runtime_link():
    """Get tflite_runtime whl package from google."""
    # Get system info
    assert_msg = 'tflite_runtime is not available for this platform'
    p_system = platform.system()
    if p_system ==  'Linux':
        plt = 'linux'
    elif p_system == 'Windows':
        plt = 'win'
    elif p_system == 'Darwin':
        assert platform.mac_ver()[0][:5] == '10.14', assert_msg
        plt = 'macosx_10_14'
    else:
        sys.exit(assert_msg)

    # Get python version
    major, minor = platform.python_version_tuple()[:2]
    python_version = major + minor

    # Get processor architecture
    arch = platform.machine()

    # Assertion
    assert python_version in ['35', '36', '37'] or arch in ['armv7l', 'aarch64', 'x86_64', 'amd64']

    url_base = 'https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp{0}-cp{0}m-{1}_{2}.whl'
    url_whl = url_base.format(python_version, plt, arch)
    return url_whl

def install_tflite_runtime():
    """Download tflite-runtime pip package."""
    url_whl = get_tflite_runtime_link()
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', url_whl])

def get_links_config():
    name_config = 'links_to_models.ini'
    path_config = pkg_resources.resource_filename('facelib._utils', name_config)
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read(path_config)
    assert config, 'Error opening config file: ' + name_config
    return config

def get_available_models(type_model):
    links = get_links_config()
    available_models = [key for key in links[type_model].keys() if not key.startswith('_')]
    return available_models

def assert_model_availability(type_model, name_model):
    types = [
        'face_detection',
        'landmark_detection',
        'feature_extraction',]
    assert type_model in types, 'Available types are: {}'.format(str(types))
    links = get_links_config()
    keys = [key for key in links[type_model].keys() if not key.startswith('_')]
    assert name_model in keys, 'Available {} {} models are:\n{}'.format(*type_model.split('_'), str(keys))

def get_path(type_model, name_model, force_download=True):
    assert_model_availability(type_model, name_model)
    links = get_links_config()
    link_model_data = links[type_model][name_model]
    filename = link_model_data.rsplit('/', 1)[-1]
    path_file = pkg_resources.resource_filename(
        'facelib.facerec' + type_model,
        'data/' + filename
    )
    exist = Path(path_file).exists()
    if force_download and not exist:
        install_data(type_model, name_model)
    return path_file

def get_link(type_model, name_model):
    links = get_links_config()
    url_link = links[type_model][name_model]
    return url_link

def install_data(type_model, name_model):
    path_file, exist = get_path(type_model, name_model, force_download=False)
    if exist:
        print('Model data already exists.')
        return
    link_data = get_link(type_model, name_model)
    response = request.urlopen(link_data)
    Path(path_file).parent.mkdir(parents=True, exist_ok=True)
    with open(path_file, 'wb') as f:
        print('Downloading from url <{}>...'.format(link_data))
        f.write(response.read())
        print('File <{}> successfully created.'.format(path_file))

def get_settings_config():
    name_config = 'settings.ini'
    path_config = pkg_resources.resource_filename('facelib._utils', name_config)
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read(path_config)
    return config

def set_settings_config(config):
    name_config = 'settings.ini'
    path_config = pkg_resources.resource_filename('facelib._utils', name_config)
    with open(path_config, 'w') as configfile:
        config.write(configfile)

def set_default_model(type_model, name_model):
    assert_model_availability(type_model, name_model)
    config = get_settings_config()
    config['DEFAULT'][type_model] = name_model
    set_settings_config(config)

def get_default_model(type_model):
    config = get_settings_config()
    config['DEFAULT'][type_model]
    return type_model

def get_default_pipeline():
    config = get_settings_config()
    default_config = config['DEFAULT']
    fd = default_config['face_detection']
    ld = default_config['landmark_detection']
    fe = default_config['feature_extraction']
    return [fd, ld, fe]

def set_default_pipeline(name_pipeline):
    fd, ld, fe = get_pipeline(name_pipeline)
    config = get_settings_config()
    default_config = config['DEFAULT']
    default_config['face_detection'] = fd 
    default_config['landmark_detection'] = ld
    default_config['feature_extraction'] = fe
    set_settings_config(config)

def get_pipeline(name_pipeline):
    config = get_settings_config()
    str_value = config['PipelineTemplates'][name_pipeline]
    list_value =  str_value.split(',')
    return list_value

def del_pipeline(name_pipeline):
    config = get_settings_config()
    assert name_pipeline in config['PipelineTemplates'].keys(), 'Pipeline does not exist.'
    del config['PipelineTemplates'][name_pipeline]
    set_settings_config(config)

def save_pipeline(list_pipeline, name_pipeline):
    assert len(list_pipeline) == 3, 'Number of models should be 3.'
    fd, ld, fe = list_pipeline
    # assert if available
    types = ['face_detection', 'landmark_detection', 'feature_extraction']
    for type_model, name_model in zip(types, list_pipeline):
        assert_model_availability(type_model, name_model)
    config = get_settings_config()
    config['PipelineTemplates'][name_pipeline] = ','.join(list_pipeline)
    set_settings_config(config)