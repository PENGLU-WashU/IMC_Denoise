# -*- coding: utf-8 -*-

import numpy as np
import tifffile as tp
import os
from os import listdir
from os.path import isfile, join
from glob import glob
from ..IMC_Denoise_main.DIMR import DIMR
from ..Anscombe_transform.Anscombe_transform_functions import Anscombe_forward

class DeepSNF_DataGenerator():
    
    """
    Data generation class, load the raw images from directory and generate training data.
    
    """
    def __init__(self, patch_row_size = 64, patch_col_size = 64, row_step = 60, col_step = 60, 
                 ratio_thresh = 0.95, channel_name = None, is_augment = True, 
                 n_neighbours = 4, n_lambda = 5, window_size = 3):
        
        """
        Initialize class parameters.
        
        The raw image structure should be as follows.
        
            |---Raw_image_directory
            |---|---Tissue1
            |---|---|---Marker1_img.tiff
            |---|---|---Marker2_img.tiff
                         ...
            |---|---|---Marker_n_img.tiff
            |---|---Tissue2
            |---|---|---Marker1_img.tiff
            |---|---|---Marker2_img.tiff
                         ...
            |---|---|---Marker_n_img.tiff
                         ...
            |---|---Tissue_m
            |---|---|---Marker1_img.tiff
            |---|---|---Marker2_img.tiff
                         ...
            |---|---|---Marker_n_img.tiff

        Parameters
        ----------
        patch_row_size : int, optional
            The row size of generated patch. The default is 64.
        patch_col_size : int, optional
            The column size of generated patch. The default is 64.
        row_step : int, optional
            Row step length when generating training patches from imgs. The default is 60.
        col_step : int, optional
            Column step length when generating training patches from imgs. The default is 60.
        ratio_thresh : float, optional
            The threshold of the sparsity of the generated patch. If larger than this threshold,
            the corresponding patch will be omitted. The default is 0.9.
        channel_name : string, optional
            The channel you want to generate a dataset. The default is None.
        is_augment : bool, optional
            DESCRIPTION. The default is True.
        n_neighbours : int, optional
            See DIMR. The default is 4.
        n_lambda : float, optional
            See DIMR. The default is 5.

        """
        if not isinstance(patch_row_size, int) or not isinstance(patch_col_size, int) \
            or not isinstance(row_step, int) or not isinstance(col_step, int):
            raise ValueError('patch_row_size, patch_col_size, row_step and col_step must be int!')
            
        if not isinstance(is_augment, bool):
            raise ValueError('is_augment must be bool!')
            
        assert ratio_thresh >= 0 and ratio_thresh <= 1, "thresh value must be between 0 and 1."
        if patch_col_size%16 != 0 or patch_row_size%16 != 0:
            print('patch size must be divisible by 16!')
            return
        
        self.patch_row_size = patch_row_size
        self.patch_col_size = patch_col_size
        self.row_step = row_step
        self.col_step = col_step
        self.ratio_thresh = ratio_thresh
        self.is_augment = is_augment
        self.n_neighbours = n_neighbours
        self.n_lambda = n_lambda
        self.window_size = window_size
        
        if channel_name is None:
            raise ValueError('Please provide the channel name!')
        self.channel_name = channel_name
        
    def load_single_img(self, filename):
        
        """
        Loading single image from directory.

        Parameters
        ----------
        filename : The image file name, must end with .tiff.
            DESCRIPTION.

        Returns
        -------
        Img_in : int or float
            Loaded image data.

        """
        if filename.endswith('.tiff'):
            Img_in = tp.imread(filename).astype('float32')
        else:
            raise ValueError('Raw file should end with tiff!')
        if Img_in.ndim != 2:
            raise ValueError('Single image should be 2d!')
        return Img_in
    
    def generate_patches(self, Img_collect):
        
        """
        Generate patches from loaded images.
        Parameters
        ----------
        Img_collect : Loaded images
            
        Returns
        -------
        patch_collect : float
            Generated patches for training.

        """
        patch_collect = []
        dimr = DIMR(n_neighbours = self.n_neighbours, n_lambda = self.n_lambda, window_size = self.window_size)
        for Img in Img_collect:
            Img_Anscombe = Anscombe_forward(Img)
            Img_DIMR = np.array(dimr.predict_augment(Img_Anscombe))
            Rows, Cols = np.shape(Img_DIMR)
            Row_range = list(range(self.patch_row_size//2, Rows - self.patch_row_size//2, self.row_step))
            Col_range = list(range(self.patch_col_size//2, Cols - self.patch_col_size//2, self.col_step))
            if Row_range[-1] < Rows - self.patch_row_size//2 - 1:
                Row_range.append(Rows - self.patch_row_size//2 - 1)
            if Col_range[-1] < Cols - self.patch_col_size//2 - 1:
                Col_range.append(Cols - self.patch_col_size//2 - 1)
                
            patch_collect_sub = self.__extract_patches_from_img__(Img_DIMR, Row_range, Col_range)
            patch_collect.append(patch_collect_sub)
        
        patch_collect = np.concatenate(patch_collect, axis = 0)
        del Img_collect
        
        if self.is_augment:
            patch_collect = self.__augment_patches__(patch_collect)
            print('The generated patches augmented.')
        
        np.random.shuffle(patch_collect)
        print('The generated patches shuffled.')
        
        return patch_collect
    
    def load_imgs_from_directory(self, load_directory):
        
        """
        Load images from a directory

        """
        Img_collect = []
        img_folders = glob(join(load_directory, "*", ""))
    
        print('Image data loaded from ...\n')
        for sub_img_folder in img_folders:
            Img_list = [f for f in listdir(sub_img_folder) if isfile(join(sub_img_folder, f)) & f.endswith(".tiff")]
            for Img_file in Img_list:
                if self.channel_name.lower() in Img_file.lower():
                    Img_read = self.load_single_img(sub_img_folder + Img_file)
                    print(sub_img_folder + Img_file)
                    Img_collect.append(Img_read)
                    break
        
        print('\n' + 'Image data loaded completed!')
        if not Img_collect:
            print('\033[91m' + "No such channels! Please check the channel name again!" + '\033[0m')
            return
                
        return Img_collect
    
    def __extract_patches_from_img__(self, Img, Rows_range, Cols_range):
        
        """
        Extract patches from a single image and then save them into a list.

        """
        patch_collect = []
        for ii in Rows_range:
                for jj in Cols_range:
                    sub_Img = Img[ii-self.patch_row_size//2:ii+self.patch_row_size//2, jj-self.patch_col_size//2:jj+self.patch_col_size//2]
                    if np.sum(sub_Img < 2*np.sqrt(1.375)) / np.prod(np.shape(sub_Img)) < self.ratio_thresh:
                        patch_collect.append(sub_Img)
        return patch_collect
        
    
    def __augment_patches__(self, generated_patches):
        """
        Augment generated data.

        """
        patch_augmented = np.concatenate((generated_patches,
                                          np.rot90(generated_patches, k=1, axes=(1, 2)),
                                          np.rot90(generated_patches, k=2, axes=(1, 2)),
                                          np.rot90(generated_patches, k=3, axes=(1, 2))), axis = 0)
        patch_augmented = np.concatenate((patch_augmented, np.flip(patch_augmented, axis = -2)), axis = 0)
        return patch_augmented
        
            
    def generate_patches_from_directory(self, load_directory):
        
        """
        Generate training set from a specific directory

        Parameters
        ----------
        load_directory structure:
        
            |---Raw_image_directory
            |---|---Tissue1
            |---|---|---Marker1_img.tiff
            |---|---|---Marker2_img.tiff
                         ...
            |---|---|---Marker_n_img.tiff
            |---|---Tissue2
            |---|---|---Marker1_img.tiff
            |---|---|---Marker2_img.tiff
                         ...
            |---|---|---Marker_n_img.tiff
                         ...
            |---|---Tissue_m
            |---|---|---Marker1_img.tiff
            |---|---|---Marker2_img.tiff
                         ...
            |---|---|---Marker_n_img.tiff
            
            
        Returns
        -------
        Generated training set

        """
        Img_collect = self.load_imgs_from_directory(load_directory)
                
        return self.generate_patches(Img_collect)
    
    def save_patches(self, generated_patches, save_directory = None):
        
        """
        Save the training set into a directory.

        """
        if save_directory is None:
            save_directory = os.path.abspath(os.getcwd()) + '\\Generated_training_set'
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        np.savez(save_directory + '\\training_set_' + self.channel_name + '.npz', patches = generated_patches)
        print('The shape of the generated training set is ' + str(generated_patches.shape) + '.')
        return True
        
def load_training_patches(filename, save_directory = None):
    
    """
    Load the saved training set from disk.

    """
    if save_directory is None:
        save_directory = os.path.abspath(os.getcwd()) + '\\Generated_training_set'
    elif not os.path.exists(save_directory):
        raise ValueError('No such dataset!')
    if not filename.endswith('.npz'):
        print('The generated training set should be .npz format!')
        return
    generated_data = np.load(save_directory + '\\' + filename)
    return generated_data['patches']
