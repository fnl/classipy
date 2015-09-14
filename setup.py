#!/usr/bin/env python
from setuptools import setup

try:
    with open('README.rst') as file:
        long_description = file.read()
except IOError:
    long_description = "missing"


setup(
    name='classipy',
    version='1.1.1',
    url='https://github.com/fnl/classipy',
    author='Florian Leitner',
    author_email='florian.leitner@gmail.com',
    description='a command-line based text classification tool',
    keywords='text classification machine learning information retrieval',
    license='AGPLv3',
    packages=['classy'],
    install_requires=['scikit-learn', 'scipy', 'numpy', 'segtok'],
    long_description=long_description,
    entry_points={
        'console_scripts': [
            'classipy = classy:main',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: GNU Affero General Public License v3',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries',
        'Topic :: Text Processing',
        'Topic :: Text Processing :: Linguistic',
    ],
)
