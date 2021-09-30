# -*- coding: utf-8 -*-

"""
References:
[1] Anscombe, F.J., "The transformation of Poisson, binomial and negative-binomial data", Biometrika, vol. 35, no. 3/4, pp. 246-254, Dec. 1948.

"""

import numpy as np

def Anscombe_forward(Img_in):
    
    """
    Forward Anscombe transformation

    """
    return 2 * np.sqrt(Img_in + 3/8)

def Anscombe_inverse_direct(Img_in):
    
    """
    Direct inverse Anscombe transformation

    """
    return (Img_in/2)**2 - 3/8
