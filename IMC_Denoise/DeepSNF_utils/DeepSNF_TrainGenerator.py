# -*- coding: utf-8 -*-

"""
The codes for generating training and validation data in this file are modified from Noise2Void: https://github.com/juglab/n2v.
We assemble and implement the modified codes in our IMC_Denoise package.

Reference:
[1] Krull, Alexander, Tim-Oliver Buchholz, and Florian Jug. "Noise2void-learning denoising from single noisy images." 
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.

"""

import numpy as np
from keras.utils import Sequence

def get_subpatch(patch, coord, local_sub_patch_radius):
    start = np.maximum(0, np.array(coord) - local_sub_patch_radius)
    end = start + local_sub_patch_radius*2 + 1
    shift = np.minimum(0, patch.shape - end)

    start += shift
    end += shift
    slices = [ slice(s, e) for s, e in zip(start, end)]

    return patch[tuple(slices)]

def pm_uniform_withCP(local_sub_patch_radius):
    """
    The dafault pixel manipulation method. Works well for IMC denoising.

    """
    def random_neighbor_withCP_uniform(patch, coords):
        vals = []
        for coord in zip(*coords):
            sub_patch = get_subpatch(patch, coord, local_sub_patch_radius)
            rand_coords = [np.random.randint(0, s) for s in sub_patch.shape[0:2]]
            vals.append(sub_patch[tuple(rand_coords)])
        return vals
    return random_neighbor_withCP_uniform

class DeepSNF_Training_DataGenerator(Sequence):
    
    """
    The DeepSNF_Training_DataGenerator manipulates random pixels of the input patches by a stratfied strategy. The 
    manipulated patches are then set as the input of the network and the raw patches as the output, respectively. 
    
    Parameters
    ----------
    X          : array(float)
                 The noisy input data.
    Y          : array(float)
                 X plus a masking channel.
    batch_size : int
                 Number of samples per batch.
    perc_pix    : int, optional
                 Number of pixels to manipulate. The default is 0.2.
    shape      : tuple(int), optional
                 Shape of the randomly extracted patches. The default is (64, 64).
    value_manipulator : function, optional
                        The manipulator used for the pixel replacement.
                        The default is pm_uniform_withCP(5).
    """

    def __init__(self, X, batch_size, perc_pix=0.2, shape=(64, 64), value_manipulation=pm_uniform_withCP(5)):
        
        self.X = X
        if np.ndim(self.X)==3:
            self.X = np.expand_dims(self.X, axis = -1)
        
        self.batch_size = batch_size
        self.rnd_idx = np.random.permutation(self.X.shape[0])
        self.shape = shape
        self.value_manipulation = value_manipulation

        num_pix = int(np.product(shape)/100.0 * perc_pix)
        assert num_pix >= 1, "No pixel is masked. perc_pix should be at least {}.".format(100.0/np.product(shape))
        print("Each training patch with shape of {} will mask {} pixels.".format(shape, num_pix))

        self.box_size = np.round(np.sqrt(100/perc_pix)).astype(np.int)
        self.rand_float = self.__rand_float_coords2D__(self.box_size)
        
        # X Y zeros
        self.X_Batches = np.zeros((self.X.shape[0], *self.shape, 1), dtype=np.float32)
        self.Y_Batches = np.zeros((self.X.shape[0], *self.shape, 2), dtype=np.float32)

    def __len__(self):
        return int(np.ceil(self.X.shape[0] / float(self.batch_size)))

    def on_epoch_end(self):
        self.rnd_idx = np.random.permutation(self.X.shape[0])
        self.X_Batches *= 0
        self.Y_Batches *= 0

    def __getitem__(self, i):
        idx = slice(i * self.batch_size, (i + 1) * self.batch_size)
        idx = self.rnd_idx[idx]
        self.X_Batches[idx,:,:,:] = np.copy(self.X[idx,:,:,:])
        self.Y_Batches[idx,:,:,0] = np.copy(self.X[idx,:,:,0])
        
        for j in idx:
            coords = self.__get_stratified_coords2D__(self.rand_float, box_size=self.box_size, shape=self.shape)
            self.X_Batches[(j,) + coords + (0,)] = self.value_manipulation(self.X_Batches[j, ..., 0], coords)
            self.Y_Batches[(j,) + coords + (1,)] = 1

        return self.X_Batches[idx], self.Y_Batches[idx]

    @staticmethod
    def __get_stratified_coords2D__(coord_gen, box_size, shape):
        box_count_y = int(np.ceil(shape[0] / box_size))
        box_count_x = int(np.ceil(shape[1] / box_size))
        x_coords = []
        y_coords = []
        for i in range(box_count_y):
            for j in range(box_count_x):
                y, x = next(coord_gen)
                y = int(i * box_size + y)
                x = int(j * box_size + x)
                if (y < shape[0] and x < shape[1]):
                    y_coords.append(y)
                    x_coords.append(x)
        return (y_coords, x_coords)  

    @staticmethod
    def __rand_float_coords2D__(boxsize):
        while True:
            yield (np.random.rand() * boxsize, np.random.rand() * boxsize)

def manipulate_val_data(X_val, perc_pix=0.2, shape=(64, 64), value_manipulation=pm_uniform_withCP(5)):
    
    """
    Manipulate pixels to generate validation data.

    """
   
    box_size = np.round(np.sqrt(100/perc_pix)).astype(np.int)
    get_stratified_coords = DeepSNF_Training_DataGenerator.__get_stratified_coords2D__
    rand_float = DeepSNF_Training_DataGenerator.__rand_float_coords2D__(box_size)

    if np.ndim(X_val)==3:
        X_val = np.expand_dims(X_val, axis = -1)
    Y_val = np.concatenate((X_val, np.zeros(np.shape(X_val))), axis=-1)

    for j in range(X_val.shape[0]):
        coords = get_stratified_coords(rand_float, box_size=box_size, shape=np.array(X_val.shape)[1:-1])
        X_val[(j,) + coords + (0,)] = value_manipulation(X_val[j, ..., 0], coords)
        Y_val[(j,) + coords + (1,)] = 1
    
    return X_val, Y_val
