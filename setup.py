# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# --------------------------------------------------------------------------------
# Copyright (c) 2017-2019, Daniele Zambon
# All rights reserved.
# Licence: BSD-3-Clause
# --------------------------------------------------------------------------------
# Author: Daniele Zambon 
# Affiliation: Università della Svizzera italiana
# eMail: daniele.zambon@usi.ch
# Last Update: 16/11/2019
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
from setuptools import setup
import os

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='cdg',
    version='2.1',
    author='Daniele Zambon',
    author_email='daniele.zambon@usi.ch',
    description=('Change Detection in a sequence of Graphs.'),
    license='BSD-3-Clause',
    long_description=read('README.md'),
    packages=['cdg'],
    install_requires=['scipy', 'numpy', 'matplotlib', 'tqdm', 'joblib'],
    url='https://github.com/dzambon/cdg'
)
