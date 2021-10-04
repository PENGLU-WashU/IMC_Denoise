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
          "numpy",
          "scipy",
          "matplotlib",
          "jupyter",
          "sklearn",
          "tensorflow==2.2.0",
          "keras==2.3.1",
          "tifffile"
        ]
      )
