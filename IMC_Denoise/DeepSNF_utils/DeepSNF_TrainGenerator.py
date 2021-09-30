# -*- coding: utf-8 -*-

"""
Reference:
[1] Krull, Alexander, Tim-Oliver Buchholz, and Florian Jug. "Noise2void-learning denoising from single noisy images." 
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.

We modified the original Noise2Void code so that it can fit in our IMC-Denoise framework. 
The license has been included in the folder.

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
    def random_neighbor_withCP_uniform(patch, coords, dims):
        vals = []
        for coord in zip(*coords):
            sub_patch = get_subpatch(patch, coord,local_sub_patch_radius)
            rand_coords = [np.random.randint(0, s) for s in sub_patch.shape[0:dims]]
            vals.append(sub_patch[tuple(rand_coords)])
        return vals
    return random_neighbor_withCP_uniform

class DeepSNF_Training_DataGenerator(Sequence):
    """
    The DeepSNF_Training_DataGenerator extracts random sub-patches from the given data and manipulates 'num_pix' pixels in the
    input.
    Parameters
    ----------
    X          : array(floats)
                 The noisy input data.
    Y          : array(floats)
                 The same as X plus a masking channel.
    batch_size : int
                 Number of samples per batch.
    num_pix    : int, optional(default=1)
                 Number of pixels to manipulate.
    shape      : tuple(int), optional(default=(64, 64))
                 Shape of the randomly extracted patches.
    value_manipulator : function, optional(default=None)
                        The manipulator used for the pixel replacement.
    """

    def __init__(self, X,Y, batch_size, perc_pix=0.2, shape=(64, 64), value_manipulation=pm_uniform_withCP(5)):
        self.X1 = X[:,:,:,0]
        self.X_rest = X[:,:,:,1:]   
        
        if np.ndim(self.X1)==3:
            self.X1 = np.expand_dims(self.X1,axis = -1)
        if np.ndim(self.X_rest)==3:
            self.X_rest = np.expand_dims(self.X_rest,axis = -1)
        self.Y = Y
        
        self.batch_size = batch_size
        self.perm = np.random.permutation(len(self.X1))
        self.shape = shape
        self.value_manipulation = value_manipulation
        self.range = np.array(self.X1.shape[1:-1]) - np.array(self.shape)
        self.dims = len(shape)

        num_pix = int(np.product(shape)/100.0 * perc_pix)
        assert num_pix >= 1, "Number of blind-spot pixels is below one. At least {}% of pixels should be replaced.".format(100.0/np.product(shape))
        print("{} blind-spots will be generated per training patch of size {}.".format(num_pix, shape))

        if self.dims == 2:
            self.patch_sampler = self.__subpatch_sampling2D__
            self.patch_sampler_rest = self.__subpatch_sampling2D_rest__
            self.box_size = np.round(np.sqrt(100/perc_pix)).astype(np.int)
            self.get_stratified_coords = self.__get_stratified_coords2D__
            self.rand_float = self.__rand_float_coords2D__(self.box_size)
        else:
            raise Exception('Dimensionality not supported.')
        
        # X Y zeros
        self.X_Batches1 = np.zeros((self.X1.shape[0], *self.shape, 1), dtype=np.float32)
        self.X_rest_Batches = np.zeros((self.X_rest.shape[0], *self.shape, np.shape(self.X_rest)[-1]), dtype=np.float32)
        self.Y_Batches1 = np.zeros((self.Y.shape[0], *self.shape, 1), dtype=np.float32)
        self.Y_Batches2 = np.zeros((self.Y.shape[0], *self.shape, 1), dtype=np.float32)

    def __len__(self):
        return int(np.ceil(len(self.X1) / float(self.batch_size)))

    def on_epoch_end(self):
        self.perm = np.random.permutation(len(self.X1))
        self.X_Batches1 *= 0
        self.X_rest_Batches *= 0
        self.Y_Batches1 *= 0
        self.Y_Batches2 *= 0

    def __getitem__(self, i):
        idx = slice(i * self.batch_size, (i + 1) * self.batch_size)
        idx = self.perm[idx]
        self.patch_sampler_rest(self.X1, self.X_Batches1, indices=idx, range=self.range, shape=self.shape)
        self.patch_sampler_rest(self.X_rest, self.X_rest_Batches, indices=idx, range=self.range, shape=self.shape)
        self.patch_sampler_rest(self.Y[:,:,:,0], self.Y_Batches1, indices=idx, range=self.range, shape=self.shape)
        
        for j in idx:
            coords = self.get_stratified_coords(self.rand_float, box_size=self.box_size,
                                                shape=self.shape)

            indexing = (j,) + coords + (0,)
            x_val = self.value_manipulation(self.X_Batches1[j, ..., 0], coords, self.dims)
            
            self.Y_Batches2[indexing] = 1
            self.X_Batches1[indexing] = x_val
                    
        mask_Batches = np.concatenate((self.X_Batches1[idx], self.X_rest_Batches[idx]), axis = -1)
        label_Batches = np.concatenate((self.Y_Batches1[idx], self.Y_Batches2[idx]),axis = -1)

        return mask_Batches, label_Batches

    @staticmethod
    def __subpatch_sampling2D__(X, X_Batches, indices, range, shape):
        for j in indices:
            y_start = np.random.randint(0, range[0] + 1)
            x_start = np.random.randint(0, range[1] + 1)
            X_Batches[j] = np.expand_dims(np.copy(X[j, y_start:y_start + shape[0], x_start:x_start + shape[1]]),axis = -1)
    
    @staticmethod
    def __subpatch_sampling2D_rest__(X_rest, X_rest_Batches, indices, range, shape):
        if np.ndim(X_rest)==3:
            X_rest = np.expand_dims(X_rest,axis=-1)
        if np.ndim(X_rest_Batches)==3:
            X_rest_Batches = np.expand_dims(X_rest_Batches,axis=-1)
        for j in indices:
            y_start = np.random.randint(0, range[0] + 1)
            x_start = np.random.randint(0, range[1] + 1)
            X_rest_Batches[j] = np.copy(X_rest[j, y_start:y_start + shape[0], x_start:x_start + shape[1],:])

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

def manipulate_val_data(X_val,Y_val, perc_pix=0.2, shape=(64, 64), value_manipulation=pm_uniform_withCP(5)):
    dims = len(shape)
    if dims == 2:
        box_size = np.round(np.sqrt(100/perc_pix)).astype(np.int)
        get_stratified_coords = N2V_DataWrapper.__get_stratified_coords2D__
        rand_float = N2V_DataWrapper.__rand_float_coords2D__(box_size)

    X_val1 = X_val[:,:,:,0]
    if np.ndim(X_val1)==3:
        X_val1 = np.expand_dims(X_val1,axis = -1)
    X_val_rest = X_val[:,:,:,1:]
    if np.ndim(X_val_rest)==3:
        X_val_rest = np.expand_dims(X_val_rest,axis = -1)
    Y_val[:,:,:,1] *= 0

    for j in range(X_val1.shape[0]):
        coords = get_stratified_coords(rand_float, box_size=box_size,
                                            shape=np.array(X_val1.shape)[1:-1])
        indexing = (j,) + coords + (0,)
        indexing_mask = (j,) + coords + (1,)
        x_val = value_manipulation(X_val1[j, ..., 0], coords, dims)

        Y_val[indexing_mask] = 1
        X_val1[indexing] = x_val
    
    mask_val = np.concatenate((X_val1, X_val_rest), axis = -1)
    X_val = mask_val
    
    return X_val, Y_val
