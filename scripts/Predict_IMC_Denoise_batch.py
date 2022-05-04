# -*- coding: utf-8 -*-
# Import Libraries and model
import numpy as np
import time
import scipy.io as sio
import gc

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


import argparse
from os import listdir
from os.path import isfile, join
from glob import glob
import tifffile as tp
from keras import optimizers
from keras.models import Model
from keras.layers import Input
from IMC_Denoise.IMC_Denoise_main.DeepSNF_model import DeepSNF_net
from IMC_Denoise.IMC_Denoise_main.loss_functions import create_I_divergence, create_mse
from IMC_Denoise.IMC_Denoise_main.DIMR import DIMR
from IMC_Denoise.Anscombe_transform.Anscombe_transform_functions import Anscombe_forward, Anscombe_inverse_exact_unbiased

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--channel_name", help = "the denoised channel name", type = str)
parser.add_argument("--load_directory", help = "the folder of the raw IMC images", type = str)
parser.add_argument("--save_directory", help = "the folder to save the denoised IMC images", type = str)
parser.add_argument("--loss_func", help = "the folder to save the denoised IMC images", type = str, default = "I_divergence")
parser.add_argument("--weights_name", help = "trained network weights. hdf5 format", type = str)
parser.add_argument("--weights_save_directory", help = "directory of trained network weights", type = str, default = None)
parser.add_argument("--n_neighbours", help = "DIMR algorithm parameter", default = 4, type = int)
parser.add_argument("--n_iter", help = "DIMR algorithm parameter", default = 3, type = int)
parser.add_argument("--slide_window_size", help = "DIMR algorithm parameter", default=3, type = int)
                    
args = parser.parse_args()
print(args)

start = time.time()
weights_dir = args.weights_save_directory
if weights_dir is None:
    weights_dir = os.path.abspath(os.getcwd()) + "\\trained_weights\\" 
else:
    weights_dir = weights_dir + '\\'
print('The file containing the trained weights is {}.'.format(weights_dir + args.weights_name))

myrange = np.load(weights_dir + args.weights_name.replace('.hdf5', '_range_val.npz'))
myrange = myrange['range_val']
print('The range is %f' % myrange)
    
input_ = Input (shape = (None, None, 1))
act_ = DeepSNF_net(input_, 'Pred_', loss_func = args.loss_func)
model = Model (inputs= input_, outputs=act_)  

opt = optimizers.Adam(lr=1e-3)
model.compile(optimizer=opt, loss = create_I_divergence(lambda_HF = 0))

# Load the trained weights
model.load_weights(weights_dir + args.weights_name)

myDIMR = DIMR(n_neighbours = args.n_neighbours, n_iter = args.n_iter, window_size = args.slide_window_size)

img_folders = glob(join(args.load_directory, "*", ""))
for sub_img_folder in img_folders:
    Img_list = [f for f in listdir(sub_img_folder) if isfile(join(sub_img_folder, f)) & (f.endswith(".tiff") or f.endswith(".tif"))]
    for Img_file in Img_list:
        if args.channel_name.lower() in Img_file.lower():
            Img_read = tp.imread(sub_img_folder + Img_file).astype('float32')
            
            Img_DIMR = myDIMR.perform_DIMR(Img_read)
            # Img_DIMR = Img_read
            
            if args.loss_func == 'mse_relu':
                Img_DIMR = Anscombe_forward(Img_DIMR)
                Img_DIMR = np.divide(Img_DIMR - 2*np.sqrt(3/8), myrange)
            else:
                Img_DIMR = np.divide(Img_DIMR, myrange)
                
            # Img_DIMR[Img_DIMR>1] = 1
            Rows, Cols = np.shape(Img_DIMR)
            Rows_new = int((Rows//16+1)*16)
            Cols_new = int((Cols//16+1)*16)
            
            Rows_diff = Rows_new - Rows
            Cols_diff = Cols_new - Cols
            
            if Rows_diff%2 == 0:
                Rows_diff1 = Rows_diff2 = int(Rows_diff/2)
            else:
                Rows_diff1 = int(Rows_diff/2)
                Rows_diff2 = Rows_diff1+1
                
            if Cols_diff%2 == 0:
                Cols_diff1 = Cols_diff2 = int(Cols_diff/2)
            else:
                Cols_diff1 = int(Cols_diff/2)
                Cols_diff2 = Cols_diff1+1
            
            Img_DIMR = np.expand_dims(Img_DIMR,axis=-1)
            Img_DIMR = np.pad(Img_DIMR,((Rows_diff1,Rows_diff2),(Cols_diff1,Cols_diff2),(0,0)),'edge')
            Img_DIMR = np.expand_dims(Img_DIMR,axis=-1)
            Img_DIMR = Img_DIMR.transpose(2,0,1,3)
            

            Img_denoised = model.predict(Img_DIMR)
            _ = gc.collect()
            
            Img_denoised = Img_denoised[0,Rows_diff1:(-Rows_diff2),Cols_diff1:(-Cols_diff2),0]
            
            if args.loss_func == 'mse_relu':
                Img_denoised = Img_denoised * myrange + 2*np.sqrt(3/8)
                Img_denoised = Anscombe_inverse_exact_unbiased(Img_denoised)
            else:
                Img_denoised = Img_denoised * myrange
                
            Img_denoised[Img_denoised<0] = 0
            sub_save_directory = args.save_directory + sub_img_folder[len(args.load_directory):]
            if not os.path.exists(sub_save_directory):
                os.makedirs(sub_save_directory)
            tp.imsave(sub_save_directory + Img_file, Img_denoised.astype('float32'))
            
            print(sub_save_directory + Img_file + ' saved!')
            break
 
end = time.time()
print(end - start)
