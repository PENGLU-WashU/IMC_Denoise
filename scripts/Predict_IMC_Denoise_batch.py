# -*- coding: utf-8 -*-

import numpy as np
import time
import gc
import os

import argparse
from os import listdir
from os.path import isfile, join, abspath, exists
from glob import glob
import tifffile as tp
from keras import optimizers
from keras.models import Model
from keras.layers import Input
from IMC_Denoise.IMC_Denoise_main.DeepSNF_model import DeepSNF_net
from IMC_Denoise.IMC_Denoise_main.loss_functions import create_I_divergence, create_mse
from IMC_Denoise.IMC_Denoise_main.DIMR import DIMR
from IMC_Denoise.Anscombe_transform.Anscombe_transform_functions import Anscombe_forward, Anscombe_inverse_exact_unbiased

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--channel_name", help = "the denoised channel name", type = str)
parser.add_argument("--load_directory", help = "the folder of the raw IMC images", type = str)
parser.add_argument("--save_directory", help = "the folder to save the denoised IMC images", type = str)
parser.add_argument("--loss_func", help = "the folder to save the denoised IMC images", type = str, default = "I_divergence")
parser.add_argument("--weights_name", help = "trained network weights. hdf5 format", type = str)
parser.add_argument("--weights_save_directory", help = "directory of trained network weights", type = str, default = None)
parser.add_argument("--batch_size", help = "batch size in prediction", type = int, default = 1)
parser.add_argument("--DIMR", help = "using DIMR?", default = True, type = str2bool)
parser.add_argument("--n_neighbours", help = "DIMR algorithm parameter", default = 4, type = int)
parser.add_argument("--n_iter", help = "DIMR algorithm parameter", default = 3, type = int)
parser.add_argument("--slide_window_size", help = "DIMR algorithm parameter", default=3, type = int)
parser.add_argument("--GPU", help = "using GPU?", default = True, type = str2bool)
args = parser.parse_args()
print(args)

if not args.GPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# define a class to save image information
class single_img_info:
    def __init__(self, Img = None, Sub_folder = None, Img_name = None, Pad_dims = None):
        self.Img = Img
        self.Sub_folder = Sub_folder
        self.Img_name = Img_name
        self.Pad_dims = Pad_dims

def split_border(length):
    half_length = int(length/2)
    if length%2 == 0:
        return half_length, half_length
    else:
        return half_length, half_length + 1

start = time.time()

Max_row_num = 0
Max_col_num = 0
Image_collect = []
Img_folders = glob(join(args.load_directory, "*", ""))
myDIMR = DIMR(n_neighbours = args.n_neighbours, n_iter = args.n_iter, window_size = args.slide_window_size)
for Sub_img_folder in Img_folders:
    Img_list = [f for f in listdir(Sub_img_folder) if isfile(join(Sub_img_folder, f)) & (f.endswith(".tiff") or f.endswith(".tif"))]
    for Img_file in Img_list:
        if args.channel_name.lower() in Img_file.lower():
            Img_read = tp.imread(Sub_img_folder + Img_file).astype('float32')
            if args.DIMR:
                Img_read = myDIMR.perform_DIMR(Img_read)
            Image_collect.append(single_img_info(Img_read, Sub_img_folder, Img_file))
            Rows, Cols = np.shape(Img_read)
            Max_row_num = max(Max_row_num, Rows)
            Max_col_num = max(Max_col_num, Cols)
            break
            
Max_row_num = int((Max_row_num//16+1)*16)
Max_col_num = int((Max_col_num//16+1)*16)

print('Loading model...')
weights_dir = args.weights_save_directory
if weights_dir is None:
    weights_dir = abspath('trained_weights')
trained_weights = join(weights_dir, args.weights_name))
print('The file containing the trained weights is {}.'.format(trained_weights)

myrange = np.load(join(weights_dir, args.weights_name.replace('.hdf5', '_range_val.npz')))
myrange = myrange['range_val']
print('The range is %f.' % myrange)

input_ = Input (shape = (None, None, 1))
act_ = DeepSNF_net(input_, 'Pred_', loss_func = args.loss_func)
model = Model (inputs= input_, outputs=act_)
model.compile(optimizer = optimizers.Adam(lr=1e-3), loss = create_I_divergence(lambda_HF = 0))
model.load_weights(trained_weights)
print('Model loaded!')

Img_num = len(Image_collect)
All_img_read = np.zeros((Img_num, Max_row_num, Max_col_num, 1))
for ii in range(Img_num):
    Cur_img = Image_collect[ii].Img
    Image_collect[ii].Img = None
    Rows, Cols = np.shape(Cur_img)
    
    if args.loss_func == 'mse_relu':
        Cur_img = Anscombe_forward(Cur_img)
        Cur_img = np.divide(Cur_img - 2*np.sqrt(3/8), myrange)
    else:
        Cur_img = np.divide(Cur_img, myrange)
    
    Rows_diff = Max_row_num - Rows
    Cols_diff = Max_col_num - Cols
    Rows_diff1, Rows_diff2 = split_border(Rows_diff)
    Cols_diff1, Cols_diff2 = split_border(Cols_diff)
        
    All_img_read[ii][:,:,0] = np.pad(Cur_img,((Rows_diff1,Rows_diff2),(Cols_diff1,Cols_diff2)),'edge')
    Image_collect[ii].Pad_dims = [Rows_diff1, Rows_diff2, Cols_diff1, Cols_diff2]

All_img_denoised = model.predict(All_img_read, batch_size = args.batch_size)
_ = gc.collect()

for ii in range(Img_num):
    Img_denoised = All_img_denoised[ii][:,:,0]
    Pad_dims = Image_collect[ii].Pad_dims
    Sub_img_folder = Image_collect[ii].Sub_folder 
    Img_name = Image_collect[ii].Img_name 

    Img_denoised = Img_denoised[Pad_dims[0]:(-Pad_dims[1]),Pad_dims[2]:(-Pad_dims[3])]

    if args.loss_func == 'mse_relu':
        Img_denoised = Img_denoised * myrange + 2*np.sqrt(3/8)
        Img_denoised = Anscombe_inverse_exact_unbiased(Img_denoised)
    else:
        Img_denoised = Img_denoised * myrange
    
    Img_denoised[Img_denoised<0] = 0
    sub_save_directory = join(args.save_directory, Sub_img_folder[len(args.load_directory):])
    if not exists(sub_save_directory):
        os.makedirs(sub_save_directory)
    tp.imsave(join(sub_save_directory, Img_name), Img_denoised.astype('float32'))

    print(sub_save_directory + Img_name + ' saved!')
 
end = time.time()
print(end - start)
