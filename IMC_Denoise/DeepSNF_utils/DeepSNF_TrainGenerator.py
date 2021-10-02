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
    pix_perc    : int, optional
                 Number of pixels to manipulate. The default is 0.2.
    shape      : tuple(int), optional
                 Shape of the randomly extracted patches. The default is (64, 64).
    value_manipulator : function, optional
                        The manipulator used for the pixel replacement.
                        The default is pm_uniform_withCP(5).
    """

    def __init__(self, X, batch_size, pix_perc = 0.2, shape = (64, 64), pix_masking = pm_uniform_withCP(5)):
        
        self.X = X
        if np.ndim(self.X)==3:
            self.X = np.expand_dims(self.X, axis = -1)
        
        self.batch_size = batch_size
        self.rnd_idx = np.random.permutation(self.X.shape[0])
        self.shape = shape
        self.pix_masking = pix_masking

        num_pix = int(np.product(shape)/100.0 * pix_perc)
        if num_pix >= 1:
            print("No pixel is masked. pix_perc should be at least {}.".format(100.0/np.product(shape)))
            return
        else:
            print("Each training patch with shape of {} will mask {} pixels.".format(shape, num_pix))

        self.box_size = int(np.round(np.sqrt(100/pix_perc)))
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
            self.X_Batches[(j,) + coords + (0,)] = self.pix_masking(self.X_Batches[j, ..., 0], coords)
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

def DeepSNF_Validation_DataGenerator(X_val, pix_perc = 0.2, shape = (64, 64), pix_masking = pm_uniform_withCP(5)):
    
    """
    Manipulate pixels to generate validation data.

    """
   
    box_size = int(np.round(np.sqrt(100/pix_perc)))
   
    if np.ndim(X_val)==3:
        X_val = np.expand_dims(X_val, axis = -1)
    Y_val = np.concatenate((X_val, np.zeros(np.shape(X_val))), axis=-1)

    for j in range(X_val.shape[0]):
        coords = DeepSNF_Training_DataGenerator.__get_stratified_coords2D__(DeepSNF_Training_DataGenerator.__rand_float_coords2D__(box_size), \
                                                                            box_size=box_size, shape=np.array(X_val.shape)[1:-1])
        X_val[(j,) + coords + (0,)] = pix_masking(X_val[j, ..., 0], coords)
        Y_val[(j,) + coords + (1,)] = 1
    
    return X_val, Y_val
