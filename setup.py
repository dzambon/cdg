# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# --------------------------------------------------------------------------------
# Copyright (c) 2017-2018, Daniele Zambon
# All rights reserved.
# Licence: BSD-3-Clause
# --------------------------------------------------------------------------------
# Author: Daniele Zambon 
# Affiliation: Università della Svizzera italiana
# eMail: daniele.zambon@usi.ch
# Last Update: 19/05/2018
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
from setuptools import setup
import os

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='cdg',
    version='2.0',
    author='Daniele Zambon',
    author_email='daniele.zambon@usi.ch',
    description=('Concept Drift and Anomaly Detection in a Sequence of Graphs.'),
    license='BSD-3-Clause',
    long_description=read('README.md'),
    packages=['cdg'],
    install_requires=['scipy', 'numpy', 'matplotlib', 'tqdm'],
    url='https://github.com/dan-zam/cdg'
)
