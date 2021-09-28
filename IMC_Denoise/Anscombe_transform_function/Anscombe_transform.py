# -*- coding: utf-8 -*-

"""
References:
[1] M. Makitalo and A. Foi, "On the inversion of the Anscombe transformation in low-count Poisson image denoising", 
Proc. Int. Workshop on Local and Non-Local Approx. in Image Process., LNLA 2009, Tuusula, Finland, pp. 26-32, August 2009. doi:10.1109/LNLA.2009.5278406
[2] M. Makitalo and A. Foi, "Optimal inversion of the Anscombe transformation in low-count Poisson image denoising", 
IEEE Trans. Image Process., vol. 20, no. 1, pp. 99-109, January 2011. doi:10.1109/TIP.2010.2056693
[3] Anscombe, F.J., "The transformation of Poisson, binomial and negative-binomial data", Biometrika, vol. 35, no. 3/4, pp. 246-254, Dec. 1948.

"""

import numpy as np
from scipy.interpolate import interp1d
import scipy.io as sio
from ..Anscombe_transform_function import place_holder

vector_path = place_holder.__file__
vector_path = vector_path.replace('place_holder.py','\\')

def load_Anscombe_inv_exact_params():
    
    """
    Load parameters for exact unbiased inverse Anscombe transformation

    """
    Anscombe_vectors = sio.loadmat(vector_path + 'Anscombe_vectors.mat')
    Efz = np.array(Anscombe_vectors['Efz'])
    Ez = np.array(Anscombe_vectors['Ez'])
    Efz = Efz[:, 0]
    Ez = Ez[:, 0]

    return Efz, Ez

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



def Anscombe_inverse_exact_unbiased(Img_in):
    
    """
    Exact unbiased inverse Anscombe transformation

    """
    Efz, Ez = load_Anscombe_inv_exact_params()
    Img_in[Img_in < 0.0] = 0.0
    # Img_in = Img_in * 2
    asymptotic = (Img_in/2)**2-1/8
    exact_inverse = interp1d(Efz,Ez,kind = 'linear', fill_value = 'extrapolate')(Img_in)
    outside_exact_inverse_domain = Img_in > np.max(Efz)
    exact_inverse[outside_exact_inverse_domain] = asymptotic[outside_exact_inverse_domain]
    outside_exact_inverse_domain = Img_in < 2*np.sqrt(3/8)
    exact_inverse[outside_exact_inverse_domain] = 0
    
    return exact_inverse
