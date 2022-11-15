# -*- coding: utf-8 -*-

import numpy as np
import scipy.signal as sps
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from ..DIMR_utils.KDE_functions import KernelDensityEstimation
from ..Anscombe_transform.Anscombe_transform_functions import Anscombe_forward, Anscombe_inverse_direct

class DIMR():
    
    """
    The 'DIMR' function enables differential intensity-based map restoration 
        algorithm to effectively remove hot pixels in raw IMC images. 
        
    """
    def __init__(self, n_neighbours = 4, n_iter = 3, window_size = 3, binWidth = 1, is_moving_mean_filter = True, mmf_window_size = 3):
        
        """
    
        Parameters
        ----------
        X : list(array)
            The Anscombe-transformed raw IMC image.
        n_neighbours : scalar(int): 1--8
            The first n smallest value to form a new histogram. The default is 4.
        n_iter : scalar: The default is 3.
            Iteration number for DIMR
        window_size: scalar(int)
            Slide window size. Must be an odd. The default is 3. 
        bin_width: scalar(float)
            Bin width in the kernel density estimation. The default is 1 for adequate sampling.
        is_moving_mean_filter: (bool)
            Whether a moving mean filter is applied. The default is True.
        mmf_window_size: scalar(int)
            The window size of the moving mean filter. The default is 3.

        """
        assert window_size % 2 == 1 and isinstance(window_size, int), "window_size must be an odd!"
        assert n_neighbours <= window_size**2 - 1 and n_neighbours >= 1 and isinstance(n_neighbours, int), \
            "n_neighbours must be an integer which is between 0 and " + str(window_size**2 - 1) + '!'
        assert n_iter > 0, "n_iter must be larger than 0!"
        
        self.n_neighbours = n_neighbours
        self.n_iter = n_iter
        self.window_size = window_size
        self.binWidth = binWidth
        self.is_moving_mean_filter = is_moving_mean_filter
        self.mmf_window_size = mmf_window_size

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
            
        DIMR_output = np.copy(X)
        binWidth = self.binWidth
        
        # image size
        n_rows, n_cols = np.shape(X)
        n_size = n_rows*n_cols
        
        for jj in range(self.n_iter):
            
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
            idx_remain = X>=2*np.sqrt(4+3/8)
            idx_remain = np.reshape(idx_remain, (1, n_size), order = 'F')
            d_sum = d_sum[:, idx_remain[0,:], :]
                
            # compare the maps and form a new histogram
            d_sum_abs = np.abs(np.subtract(d_sum, np.median(d_sum, axis = (0, 1))))
            indice_sorted = np.argsort(d_sum_abs, axis = -1)
            d_sum_sorted = np.take_along_axis(d_sum, indice_sorted, axis = -1)
            d_sum_new = d_sum_sorted[:, :, 0:self.n_neighbours]
            d_mat = np.sum(d_sum_new, axis = -1)
            
            if np.prod(np.shape(d_mat)) == 0:
                break
        
            nPoint = int(np.ceil((np.max(d_mat)-np.min(d_mat))/binWidth))
            if nPoint <= self.n_neighbours/binWidth*(2*np.sqrt(4.375)-2*np.sqrt(0.375)):
                break
            
            nBandWidth = 1.06*np.std(d_mat)*nPoint**(-0.2)
            
            if d_mat.ndim == 1:
                d_mat = np.expand_dims(d_mat, axis = -1)
            if np.shape(d_mat)[0] < np.shape(d_mat)[1]:
                d_mat = d_mat.transpose()
                
            ff, xx1 = KernelDensityEstimation(d_mat, Bandwidth = nBandWidth, NumPoints = nPoint)
            ff = ff[0,:]
            xx1 = xx1[0,:]
            
            if self.is_moving_mean_filter:
                ff = uniform_filter1d(ff, size = self.mmf_window_size)
            ff_smoothed = np.prod(np.shape(d_mat)) * ff * binWidth
            peaks_loc, _ = find_peaks(ff_smoothed, height = np.mean(ff_smoothed))
            
            if np.prod(np.shape(peaks_loc)) == 0:
                break
            ii_max = peaks_loc[0]
            
            diff_ff = np.diff(ff_smoothed, n = 1)
            
            length_diff_ff = np.prod(np.shape(diff_ff))
            
            for ii in range(ii_max, length_diff_ff):
                if diff_ff[ii] >= diff_ff[ii-1] and diff_ff[ii] <= 0 and diff_ff[ii] >= -2: # Condition 1
                    break
                elif diff_ff[ii-1] >= diff_ff[ii-2] and diff_ff[ii-1] <= 0 and diff_ff[ii] >= 0: # Condition 1
                    break
                elif diff_ff[ii-1] >= diff_ff[ii-2] and diff_ff[ii-1] >= diff_ff[ii] and diff_ff[ii-1] < -2: # Condition 2
                    break
             
            if ii < length_diff_ff:
                idx1 = d_mat > xx1[ii]
                if np.sum(idx1) == 0:
                    break
                
            # remove the hot pixels
            idx2 = np.zeros((n_size), dtype = bool)
            idx2[idx_remain[0, :]] = np.transpose(idx1[:, 0])
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
            
        DIMR_output = np.copy(X)
        binWidth = self.binWidth
            
        # image size
        n_rows, n_cols = np.shape(X)
        n_size = n_rows*n_cols
        
        pad_size = (self.window_size - 1)//2
        
        for jj in range(self.n_iter):
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
            idx_remain = X >= 2*np.sqrt(4 + 3/8)
            idx_remain = np.reshape(idx_remain, (1, n_size), order = 'F')
            d_sum = d_sum[:, idx_remain[0,:], :]
                    
            # compare the maps and form a new histogram
            d_sum_abs = np.abs(np.subtract(d_sum, np.median(d_sum, axis = (0, 1))))
            indice_sorted = np.argsort(d_sum_abs, axis = -1)
            d_sum_sorted = np.take_along_axis(np.subtract(d_sum, np.median(d_sum, axis = (0, 1))), indice_sorted, axis = -1)
            d_sum_new = d_sum_sorted[:, :, 0:self.n_neighbours]
            d_mat = np.sum(d_sum_new, axis = -1)
            
            if np.prod(np.shape(d_mat)) == 0:
                break
        
            nPoint = np.ceil((np.max(d_mat)-np.min(d_mat))/binWidth)
            # few points without possibility of hot pixels.
            if nPoint <= self.n_neighbours/binWidth*(2*np.sqrt(4.375)-2*np.sqrt(0.375)): 
                break
            
            nBandWidth = 1.06*np.std(d_mat)*nPoint**(-0.2)
            
            if d_mat.ndim == 1:
                d_mat = np.expand_dims(d_mat, axis = -1)
            if np.shape(d_mat)[0] < np.shape(d_mat)[1]:
                d_mat = d_mat.transpose()
                
            ff, xx1 = KernelDensityEstimation(d_mat, Bandwidth = nBandWidth, NumPoints = nPoint)
            ff = ff[0,:]
            xx1 = xx1[0,:]
            
            if self.is_moving_mean_filter:
                ff = uniform_filter1d(ff, size = self.mmf_window_size)
                
            ff_smoothed = np.prod(np.shape(d_mat)) * ff * binWidth
            peaks_loc, _ = find_peaks(ff_smoothed, height = np.mean(ff_smoothed))
            
            if np.prod(np.shape(peaks_loc)) == 0:
                break
            ii_max = peaks_loc[0]
            
            diff_ff = np.diff(ff_smoothed, n = 1)
            
            length_diff_ff = np.prod(np.shape(diff_ff))
            
            for ii in range(ii_max, length_diff_ff):
                if diff_ff[ii] >= diff_ff[ii-1] and diff_ff[ii] <= 0 and diff_ff[ii] >= -2: # Condition 1
                    break
                elif diff_ff[ii-1] >= diff_ff[ii-2] and diff_ff[ii-1] <= 0 and diff_ff[ii] >= 0: # Condition 1
                    break
                elif diff_ff[ii-1] >= diff_ff[ii-2] and diff_ff[ii-1] >= diff_ff[ii] and diff_ff[ii-1] < -2: # Condition 2
                    break
             
            if ii < length_diff_ff:
                idx1 = d_mat > xx1[ii]
                if np.sum(idx1) == 0:
                    break
                
            # remove the hot pixels
            idx2 = np.zeros((n_size), dtype = bool)
            idx2[idx_remain[0, :]] = np.transpose(idx1[:, 0])
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
