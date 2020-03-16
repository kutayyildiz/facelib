import platform
import subprocess
import sys

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