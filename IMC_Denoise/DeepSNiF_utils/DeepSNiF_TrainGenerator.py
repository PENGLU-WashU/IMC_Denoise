# -*- coding: utf-8 -*-

"""
Several functions in this file (get_subpatch, pm_uniform_withCP, __get_stratified_coords2D__ and __rand_float_coords2D__)
are inherited or modified from Noise2Void: https://github.com/juglab/n2v. 

Reference:
[1] Krull, Alexander, Tim-Oliver Buchholz, and Florian Jug. "Noise2void-learning denoising from single noisy images." 
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.
__________________________________________________________________________________

Licencse for Noise2Void

BSD 3-Clause License

Copyright (c) 2019, Tim-Oliver Buchholz, Alexander Krull
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

import numpy as np
from tensorflow.keras.utils import Sequence

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

def pm_uniform_withoutCP(local_sub_patch_radius):
    def random_neighbor_withoutCP_uniform(patch, coords):
        vals = []
        for coord in zip(*coords):
            sub_patch = get_subpatch(patch, coord,local_sub_patch_radius)
            rand_coords = [np.random.randint(0, s) for s in sub_patch.shape[0:2]]
            while np.any(rand_coords == coord):
                rand_coords = [np.random.randint(0, s) for s in sub_patch.shape[0:2]]
            vals.append(sub_patch[tuple(rand_coords)])
        return vals
    return random_neighbor_withoutCP_uniform

class DeepSNiF_Training_DataGenerator(Sequence):
    
    """
    The DeepSNiF_Training_DataGenerator manipulates random pixels of the input patches by a stratfied strategy. The 
    manipulated patches are then set as the input of the network and the raw patches as the output, respectively. 
    
    Parameters
    ----------
    X          : array(float)
                 The noisy input data.
    Y          : array(float)
                 X plus a masking channel.
    batch_size : int
                 Number of samples per batch.
    pix_perc    : int, optional
                 Number of pixels to manipulate. The default is 0.2.
    shape      : tuple(int), optional
                 Shape of the randomly extracted patches. The default is (64, 64).
    value_manipulator : function, optional
                        The manipulator used for the pixel replacement.
                        The default is pm_uniform_withoutCP(5).
    """

    def __init__(self, X, batch_size, pix_perc = 0.2, shape = (64, 64), pix_masking_func = pm_uniform_withoutCP(5)):
        
        self.X = X
        if np.ndim(self.X)==3:
            self.X = np.expand_dims(self.X, axis = -1)
        
        self.batch_size = batch_size
        self.shape = shape
        self.pix_masking_func = pix_masking_func
        self.idx_list = list(range(self.X.shape[0]))

        pix_num = int(np.product(shape)/100.0 * pix_perc)
        if pix_num < 1:
            raise Exception("No pixel is masked. pix_perc should be at least {}.".format(100.0/np.product(shape)))
        else:
            print("Each training patch with shape of {} will mask {} pixels.".format(shape, pix_num))
        
        self.box_size = int(np.round(np.sqrt(100/pix_perc)))

    def __len__(self):
        return int(np.ceil(self.X.shape[0] / float(self.batch_size)))

    def on_epoch_end(self):
        np.random.shuffle(self.X)

    def __getitem__(self, ii):
        idx = self.idx_list[ii*self.batch_size:(ii + 1)*self.batch_size]
        X_Train_Batches = np.copy(self.X[idx])
        Y_Train_Batches = np.copy(self.X[idx])
        Y_Train_Batches = np.concatenate((Y_Train_Batches, np.zeros(np.shape(Y_Train_Batches))), axis=-1)
        
        for jj in range(len(idx)):
            masked_coords = self.__get_stratified_coords2D__(self.__rand_float_coords2D__(self.box_size),\
                                                             box_size=self.box_size, shape=self.shape)
            X_Train_Batches[(jj,) + masked_coords + (0,)] = self.pix_masking_func(X_Train_Batches[jj, ..., 0], masked_coords)
            Y_Train_Batches[(jj,) + masked_coords + (1,)] = 1

        return X_Train_Batches, Y_Train_Batches

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

def DeepSNiF_Validation_DataGenerator(X, pix_perc = 0.2, shape = (64, 64), pix_masking_func = pm_uniform_withoutCP(5)):
    
    """
    Manipulate pixels to generate validation data.

    """
   
    box_size = int(np.round(np.sqrt(100/pix_perc)))
   
    if np.ndim(X)==3:
        X = np.expand_dims(X, axis = -1)
    Y = np.concatenate((np.copy(X), np.zeros(np.shape(X))), axis=-1)

    for ii in range(X.shape[0]):
        masked_coords = DeepSNiF_Training_DataGenerator.__get_stratified_coords2D__(DeepSNiF_Training_DataGenerator.__rand_float_coords2D__(box_size), \
                                                                            box_size = box_size, shape = np.array(X.shape)[1:-1])
        X[(ii,) + masked_coords + (0,)] = pix_masking_func(X[ii, ..., 0], masked_coords)
        Y[(ii,) + masked_coords + (1,)] = 1
    
    return X, Y
