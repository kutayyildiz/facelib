"""A face recognition module with various models.

See:
https://github.com/kutayyildiz/facelib
"""

from setuptools import setup, find_packages 
from os import path
here = path.abspath(path.dirname(__file__))
# 
# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='facelib',  # Required
    version='1.0',  # Required
    description="""Face Recognition (train/test/deploy)(tensorflow/tflite/keras/edgetpu)""",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/kutayyildiz/facelib',
    author='Kutay YILDIZ',
    author_email='kkutayyildiz@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',

        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Information Technology',
        'Topic :: Software Development :: Build Tools',

        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3',
    ],
    package_dir={'facelib': 'facelib'},
    keywords='face,recognition,detection,tensorflow,lite,keras,loss,layer,edgetpu',
    packages=find_packages(),
    python_requires='>=3.5, <3.8',
    install_requires=[
        'opencv-python',
        'numpy',
        'scikit-image'
    ],
    extras_require={
        'dev': ['tensorflow']
    },
    project_urls={
        'Source': 'https://github.com/kutayyildiz/facelib'
    },
    include_package_data=True,
)