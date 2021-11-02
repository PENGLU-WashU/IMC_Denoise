# -*- coding: utf-8 -*-

import numpy as np
import scipy.signal as sps
from ..Anscombe_transform.Anscombe_transform_functions import Anscombe_forward, Anscombe_inverse_direct

class DIMR():
    
    """
    The 'DIMR' function enables differential intensity-based map restoration 
        algorithm to effectively remove hot pixels in raw IMC images. 
        
    """
    def __init__(self, n_neighbours = 4, n_lambda = 5, window_size = 3):
        
        """
    
        Parameters
        ----------
        X : list(array)
            The Anscombe-transformed raw IMC image.
        n_neighbours : scalar(int): 1--8
            The first n smallest value to form a new histogram. The default is 4.
        n_lambda : scalar: 3--5
            The value which is larger than mean + n_lambda * std will be removed. The default is 5.
        window_size: scalar(int)
            Slide window size. Must be an odd. The default is 3. 

        """
        assert window_size % 2 == 1 and isinstance(window_size, int), "window_size must be an odd!"
        assert n_neighbours <= window_size**2 - 1 and n_neighbours >= 1 and isinstance(n_neighbours, int), \
            "n_neighbours must be an integer which is between 0 and " + str(window_size**2 - 1) + '!'
        assert n_lambda > 0, "n_lambda must be larger than 0!"
        
        self.n_neighbours = n_neighbours
        self.n_lambda = n_lambda
        self.window_size = window_size

    def predict(self, X):
        
        """
        The 'DIMR' function enables differential intensity-based map restoration 
        algorithm to effectively remove hot pixels in raw IMC images. 
    
        Parameters
        ----------
        X : list(array)
            The Anscombe-transformed raw IMC image.
    
        Returns
        -------
        DIMR_output : The same data type as X
            Hot pixel-removed IMC image.
    
        """
        if X.ndim != 2:
            raise Exception("For DIMR, the input must be a 2d image!")
        
        # image size
        n_rows, n_cols = np.shape(X)
        n_size = n_rows*n_cols
        
        # calculate the image differential map
        Img_pad = np.pad(X,((1,1),(1,1)),'edge')
        Img_pad = np.expand_dims(Img_pad, axis = -1)
        d1 = np.subtract(Img_pad[1:-1, 1:-1], Img_pad[1:-1, 0:-2])
        d2 = np.subtract(Img_pad[1:-1, 1:-1], Img_pad[1:-1, 2:])
        d3 = np.subtract(Img_pad[1:-1, 1:-1], Img_pad[0:-2, 1:-1])
        d4 = np.subtract(Img_pad[1:-1, 1:-1], Img_pad[2:, 1:-1])
        d5 = np.subtract(Img_pad[1:-1, 1:-1], Img_pad[0:-2, 0:-2])
        d6 = np.subtract(Img_pad[1:-1, 1:-1], Img_pad[2:, 0:-2])
        d7 = np.subtract(Img_pad[1:-1, 1:-1], Img_pad[0:-2, 2:])
        d8 = np.subtract(Img_pad[1:-1, 1:-1], Img_pad[2:, 2:])
        
        # concatenate all the differential maps
        d_sum = np.concatenate((d1, d2, d3, d4, d5, d6, d7, d8), axis = -1)
        d_sum = np.reshape(d_sum, (1, n_size, 8), order = 'F')
        
        # remove the very low intensity pixels which cannot be outliers
        idx_remain = X>=2*np.sqrt(1+3/8)
        idx_remain = np.reshape(idx_remain, (1, n_size), order = 'F')
        d_sum = d_sum[:, idx_remain[0,:], :]
                
        # compare the maps and form a new histogram
        d_sum_abs = np.abs(np.subtract(d_sum, np.mean(d_sum, axis = (0, 1))))
        indice_sorted = np.argsort(d_sum_abs, axis = -1)
        d_sum_sorted = np.take_along_axis(d_sum, indice_sorted, axis = -1)
        d_sum_new = d_sum_sorted[:, :, 0:self.n_neighbours]
        d_mat = np.sum(d_sum_new, axis = -1)
        
        # detect outliers based on the statistics of the new histogram
        idx1 = d_mat > np.mean(d_mat) + self.n_lambda * np.std(d_mat)
    
        # remove the hot pixels
        idx2 = np.zeros((n_size), dtype = bool)
        idx2[idx_remain[0, :]] = idx1[0, :]
        idx2 = np.reshape(idx2, (n_rows, n_cols), order = 'F') 
        DIMR_output = X
        Img_med = sps.medfilt2d(X, kernel_size = 3)
        DIMR_output[idx2] = Img_med[idx2]
                  
        return DIMR_output
    
    def predict_augment(self, X):
        
        """
        The 'DIMR' function enables differential intensity-based map restoration 
        algorithm to effectively remove hot pixels in raw IMC images. 
    
        Parameters
        ----------
        X : list(array)
            The Anscombe-transformed raw IMC image.
    
        Returns
        -------
        DIMR_output : The same data type as X
            Hot pixel-removed IMC image.
    
        """
        if X.ndim != 2:
            raise Exception("For DIMR, the input must be a 2d image!")
            
        # image size
        n_rows, n_cols = np.shape(X)
        n_size = n_rows*n_cols
        
        pad_size = (self.window_size - 1)//2
        
        # calculate the image differential map
        Img_pad = np.pad(X, ((pad_size, pad_size), (pad_size, pad_size)), 'edge')
        Rows_pad, Cols_pad = np.shape(Img_pad)
        
        Differential_maps = np.zeros((np.shape(X)[0], np.shape(X)[1], self.window_size**2))
        for ii in range(np.shape(Differential_maps)[2]):
            shift_row = ii // self.window_size - pad_size
            shift_col = ii % self.window_size - pad_size
            Differential_maps[:,:,ii] = np.subtract(Img_pad[pad_size:-pad_size, pad_size:-pad_size], \
                                                    Img_pad[pad_size+shift_row:Rows_pad-pad_size+shift_row, pad_size+shift_col:Cols_pad-pad_size+shift_col])
        
        # concatenate all the differential maps
        d_sum = np.concatenate((Differential_maps[:,:,0:(self.window_size**2-1)//2], Differential_maps[:,:,(self.window_size**2+1)//2:]), axis = -1)
        d_sum = np.reshape(d_sum, (1, n_size, self.window_size**2-1), order = 'F')
        
        # remove the very low intensity pixels which cannot be outliers
        idx_remain = X >= 2*np.sqrt(1 + 3/8)
        idx_remain = np.reshape(idx_remain, (1, n_size), order = 'F')
        d_sum = d_sum[:, idx_remain[0,:], :]
                
        # compare the maps and form a new histogram
        d_sum_abs = np.abs(np.subtract(d_sum, np.mean(d_sum, axis = (0, 1))))
        indice_sorted = np.argsort(d_sum_abs, axis = -1)
        d_sum_sorted = np.take_along_axis(d_sum, indice_sorted, axis = -1)
        d_sum_new = d_sum_sorted[:, :, 0:self.n_neighbours]
        d_mat = np.sum(d_sum_new, axis = -1)
        
        # detect outliers based on the statistics of the new histogram
        idx1 = d_mat > np.mean(d_mat) + self.n_lambda * np.std(d_mat)
    
        # remove the hot pixels
        idx2 = np.zeros((n_size), dtype = bool)
        idx2[idx_remain[0, :]] = idx1[0,:]
        idx2 = np.reshape(idx2, (n_rows, n_cols), order = 'F') 
        DIMR_output = X
        Img_med = sps.medfilt2d(X, kernel_size = self.window_size)
        DIMR_output[idx2] = Img_med[idx2]
                  
        return DIMR_output
    
    def perform_DIMR(self, X):
        
        """
        Perform DIMR algorithm to remove hot pixels for single raw image.

        Parameters
        ----------
        X : float or int
            Loaded raw images.

        Returns
        -------
        X_denoised : float
            Hot pixel removed images.

        """
        X_Anscombe_transformed = Anscombe_forward(X)
        X_DIMR = self.predict_augment(X_Anscombe_transformed)
        X_denoised = Anscombe_inverse_direct(X_DIMR)
        return X_denoised
    
    def perform_DIMR_batch(self, X):
        
        """
        Perform DIMR algorithm to remove hot pixels for a batch of raw images

        Parameters
        ----------
        X : float or int
            Loaded raw images.

        Returns
        -------
        X_denoised : float
            Hot pixel removed images.

        """
        if X.ndim != 3:
            raise Exception("For DIMR batch processing, the input must be a 3d array ([N, R, C])!")
        
        X_Anscombe_transformed = Anscombe_forward(X)
        X_DIMR = np.zeros(np.shape(X_Anscombe_transformed))
        for ii in range(np.shape(X_Anscombe_transformed)[0]):
            X_DIMR[ii, :, :] = self.predict_augment(X_Anscombe_transformed[ii, :, :])
        
        X_denoised = Anscombe_inverse_direct(X_DIMR)
        return X_denoised
