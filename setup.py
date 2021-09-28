# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 22:34:57 2021

@author: penglu
"""

from __future__ import absolute_import
from setuptools import setup, find_packages

setup(name='IMC_Denoise',
      packages=find_packages(),
      install_requires=[
          "os",
          "numpy",
          "scipy",
          "glob",
          "matplotlib",
          "jupyter",
          "sklearn",
          "tensorflow-gpu==2.2.0",
          "keras==2.3.1",
          "tifffile",
          "python_version==3.6"
        ]
      )