"""Helper functions for facelib module."""
import platform
import subprocess
import sys
from configparser import ConfigParser, ExtendedInterpolation
from pathlib import Path
from urllib import request
from textwrap import dedent

import pkg_resources
from joblib import dump, load


# Tensorflow lite runtime
def get_tflite_runtime_link():
    """Get tflite_runtime whl package from google."""
    # Get system info
    assert_msg = 'tflite_runtime is not available for this platform'
    p_system = platform.system()
    if p_system == 'Linux':
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
    arch = platform.machine().lower()

    # Assertion
    assert python_version in ['35', '36', '37'] and arch in ['armv7l', 'aarch64', 'x86_64', 'amd64']

    url_base = ('https://dl.google.com/coral/python/'
                'tflite_runtime-2.1.0.post1-cp{0}-cp{0}m-{1}_{2}.whl')
    url_whl = url_base.format(python_version, plt, arch)
    return url_whl

def install_tflite_runtime():
    """Download tflite-runtime pip package."""
    url_whl = get_tflite_runtime_link()
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', url_whl])

# Template
def get_templates_config():
    """Get templates config."""
    name_config = 'templates.ini'
    path_config = pkg_resources.resource_filename('facelib._utils', name_config)
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read(path_config)
    return config

def set_templates_config(config):
    """Write config to templates config file."""
    name_config = 'templates.ini'
    path_config = pkg_resources.resource_filename('facelib._utils', name_config)
    with open(path_config, 'w') as configfile:
        config.write(configfile)

# Settings
def get_settings_config():
    """Set facelib module settings."""
    name_config = 'settings.ini'
    path_config = pkg_resources.resource_filename('facelib._utils', name_config)
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read(path_config)
    assert 1 >= float(config['Predict']['tolerance']) > 0, "Tolerance should be: (0, 1]"
    return config

def set_settings_config(config):
    """Get config with facelib module settings."""
    name_config = 'settings.ini'
    path_config = pkg_resources.resource_filename('facelib._utils', name_config)
    with open(path_config, 'w') as configfile:
        config.write(configfile)

# Model
def get_links_config():
    """Get config with url links to models datas."""
    name_config = 'links_to_models.ini'
    path_config = pkg_resources.resource_filename('facelib._utils', name_config)
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read(path_config)
    assert config, 'Error opening config file: ' + name_config
    return config

def set_default_model(type_model, name_model):
    """Set default model."""
    assert_model_availability(type_model, name_model)
    config = get_templates_config()
    config['CurrentTemplate'][type_model] = name_model
    set_templates_config(config)

def get_default_model(type_model):
    """Get name of the current model."""
    types = ['face_detection', 'landmark_detection', 'feature_extraction']
    assert type_model in types, 'Model type is wrong.'
    config = get_templates_config()
    model = config['CurrentTemplate'][type_model]
    return model

def get_available_models(type_model):
    """Get available model names as list."""
    links = get_links_config()
    available_models = [key for key in links[type_model].keys() if not key.startswith('_')]
    return available_models

def assert_model_availability(type_model, name_model):
    """Assert model availability."""
    types = [
        'face_detection',
        'landmark_detection',
        'feature_extraction',]
    assert type_model in types, 'Available types are: {}'.format(str(types))
    links = get_links_config()
    keys = [key for key in links[type_model].keys() if not key.startswith('_')]
    assert name_model in keys, 'Available {} {} models are:\n{}'.format(
        *type_model.split('_'), str(keys))

def get_path(type_model, name_model, force_download=True):
    """Get path to model data."""
    assert_model_availability(type_model, name_model)
    links = get_links_config()
    link_model_data = links[type_model][name_model]
    filename = link_model_data.rsplit('/', 1)[-1]
    path_file = pkg_resources.resource_filename(
        'facelib.facerec.' + type_model,
        'data/' + filename
    )
    exist = Path(path_file).exists()
    if force_download and not exist:
        install_data(type_model, name_model)
    return path_file

def get_link(type_model, name_model):
    """Get links to model data."""
    links = get_links_config()
    url_link = links[type_model][name_model]
    return url_link

def install_data(type_model, name_model):
    """Install model data."""
    path_file = get_path(type_model, name_model, force_download=False)
    link_data = get_link(type_model, name_model)
    response = request.urlopen(link_data)
    Path(path_file).parent.mkdir(parents=True, exist_ok=True)
    with open(path_file, 'wb') as _f:
        print('Downloading from url <{}>...'.format(link_data))
        _f.write(response.read())
        print('File <{}> successfully created.'.format(path_file))

# Pipeline
def print_templates_info():
    """Print current/available templates."""
    config = get_templates_config()
    current_pipeline = get_default_pipeline()
    message = dedent("""\
    Current pipeline is:
        (F)ace Detection:     {}
        (L)andmark Detection: {}
        (Fe)ature Extraction: {}\n
    Available templates are:""".format(*current_pipeline))
    print(message)
    tmp = '        (F){} (L){} (Fe){}'
    for name_template in config['PipelineTemplates'].keys():
        pipeline = get_pipeline(name_template)
        print('    {}'.format(name_template))
        print(tmp.format(*pipeline))


def get_default_pipeline():
    """Get the default pipeline as list of model names."""
    config = get_templates_config()
    default_config = config['CurrentTemplate']
    _fd = default_config['face_detection']
    _ld = default_config['landmark_detection']
    _fe = default_config['feature_extraction']
    return [_fd, _ld, _fe]

def set_default_pipeline(name_pipeline):
    """Set the current pipeline."""
    _fd, _ld, _fe = get_pipeline(name_pipeline)
    config = get_templates_config()
    default_config = config['CurrentTemplate']
    default_config['face_detection'] = _fd
    default_config['landmark_detection'] = _ld
    default_config['feature_extraction'] = _fe
    set_templates_config(config)

def get_pipeline(name_pipeline):
    """Get list of model names as pipeline."""
    config = get_templates_config()
    str_value = config['PipelineTemplates'][name_pipeline]
    list_value = str_value.split(',')
    return list_value

def del_pipeline(name_pipeline):
    """Delete a pipeline."""
    config = get_templates_config()
    assert name_pipeline in config['PipelineTemplates'].keys(), 'Pipeline does not exist.'
    del config['PipelineTemplates'][name_pipeline]
    set_templates_config(config)

def save_pipeline(list_pipeline, name_pipeline):
    """Name a pipeline and save it."""
    assert len(list_pipeline) == 3, 'Number of models should be 3.'
    # assert if available
    types = ['face_detection', 'landmark_detection', 'feature_extraction']
    for type_model, name_model in zip(types, list_pipeline):
        assert_model_availability(type_model, name_model)
    config = get_templates_config()
    config['PipelineTemplates'][name_pipeline] = ','.join(list_pipeline)
    set_templates_config(config)

# Classification
def save_classifier(clf, name_clf):
    available_names = [x.split('_-_')[0] for x in get_available_classifiers()]
    if name_clf in available_names:
        print('Classifier named `{}` is overwritten.'.format(name_clf))
    name_fe = get_default_model('feature_extraction')
    name_fe = name_fe.split('_', 1)[0]
    path_save = get_classifier_path(name_clf, name_fe)
    dump(clf, str(path_save))
    print('Classifier named `{}` succesfully trained and saved.'.format(name_clf))

def get_classifier(name_clf):
    available_classifiers = get_available_classifiers()
    available_names = [x.split('_-_')[0] for x in available_classifiers]
    if name_clf not in available_names:
        print('Classifier with name `{}` does not exist.'.format(name_clf))
        print('Available classifiers are:')
        print_classifiers_info()
        sys.exit()
    name_fe_default = get_default_model('feature_extraction').split('_', 1)[0]
    index = available_names.index(name_clf)
    name_fe = available_classifiers[index].split('_-_')[1]
    if name_fe != name_fe_default:
        message = ('Error: Current feature extractor `{}` is not compatible with'
                   ' the feature extractor `{}` used to train classifier `{}`.')
        sys.exit(message.format(name_fe_default, name_fe, name_clf))
    posix_path_clf = get_classifier_path(name_clf, name_fe)
    clf = load(str(posix_path_clf))
    return clf

def get_classifier_path(name_clf, name_fe):
    """Get posix path to a classifier."""
    name_save = name_clf + '_-_' + name_fe
    path_classifier = pkg_resources.resource_filename(
        'facelib.facerec', '_dataset/classifier/' + name_save)
    posix_path = Path(path_classifier).with_suffix('.joblib')
    posix_path.parent.mkdir(parents=True, exist_ok=True)
    return posix_path

def get_available_classifiers():
    """Get list of posix paths to classifiers."""
    path_dir_classifier = pkg_resources.resource_filename(
        'facelib.facerec', '_dataset/classifier')
    posix_path = Path(path_dir_classifier)
    return [p.stem for p in posix_path.glob('*.joblib')]

def print_classifiers_info():
    for name in get_available_classifiers():
        name_clf, name_fe = name.split('_-_')
        posix_path_clf = get_classifier_path(name_clf, name_fe)
        clf = load(str(posix_path_clf))
        classes = clf.classes_
        print(dedent("""
            Classifier name: {}
                Feature extractor used is {}
                Classes: {}""").format(name_clf, name_fe, ', '.join(classes)))

# Training
def get_num_augment():
    """Get min number of samples per image to train."""
    config = get_settings_config()
    num_augment = int(config['Train']['num_augment'])
    return num_augment
