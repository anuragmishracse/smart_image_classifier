from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='smic',

    version='1.0.1',

    description='Image Classification library built on top of Keras. Identifies the best set of hyperparameters and trains a classification model accordingly, hence, smart.',
    long_description=long_description,

    url='https://github.com/anuragmishracse/smart_image_classifier',

    author='Anurag Mishra',
    author_email='anuragmishracse@gmail.com',

    license='MIT',

    classifiers=[

        'Development Status :: 3 - Alpha',

        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',

        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],

    keywords='image-classification deep-learning keras',

    packages=find_packages(exclude=['contrib', 'docs', 'tests']),

    install_requires=['numpy', 'keras', 'opencv-python', 'tqdm', 'h5py', 'pandas', 'tensorflow'],

)