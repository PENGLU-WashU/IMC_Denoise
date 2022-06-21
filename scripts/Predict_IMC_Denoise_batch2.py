# -*- coding: utf-8 -*-

import numpy as np
import time
import scipy.io as sio
import gc
import os

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
    def __init__(self, img = None, sub_folder = None, img_name = None, pad_dims = None):
        self.img = img
        self.sub_folder = sub_folder
        self.img_name = img_name
        self.pad_dims = pad_dims

start = time.time()

myDIMR = DIMR(n_neighbours = args.n_neighbours, n_iter = args.n_iter, window_size = args.slide_window_size)

max_row_num = 0
max_col_num = 0
image_collect = []
img_folders = glob(join(args.load_directory, "*", ""))
for sub_img_folder in img_folders:
    Img_list = [f for f in listdir(sub_img_folder) if isfile(join(sub_img_folder, f)) & (f.endswith(".tiff") or f.endswith(".tif"))]
    for Img_file in Img_list:
        if args.channel_name.lower() in Img_file.lower():
            Img_read = tp.imread(sub_img_folder + Img_file).astype('float32')
            Img_DIMR = myDIMR.perform_DIMR(Img_read)

            image_collect.append(single_img_info(Img_DIMR, sub_img_folder, Img_file))
                
            Rows, Cols = np.shape(Img_DIMR)
            max_row_num = max(max_row_num, Rows)
            max_col_num = max(max_col_num, Cols)
            break
            
max_row_num = int((max_row_num//16+1)*16)
max_col_num = int((max_col_num//16+1)*16)

print('Loading model...')
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
model.compile(optimizer = optimizers.Adam(lr=1e-3), loss = create_I_divergence(lambda_HF = 0))
model.load_weights(weights_dir + args.weights_name)
print('Model loaded.')

all_img = []
for cur_image_collect in image_collect:
    cur_img = cur_image_collect.img
    Rows, Cols = np.shape(cur_img)
    
    if args.loss_func == 'mse_relu':
        cur_img = Anscombe_forward(cur_img)
        cur_img = np.divide(cur_img - 2*np.sqrt(3/8), myrange)
    else:
        cur_img = np.divide(cur_img, myrange)
    
    Rows_diff = max_row_num - Rows
    Cols_diff = max_col_num - Cols
    
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
        
    all_img.append(np.pad(cur_img,((Rows_diff1,Rows_diff2),(Cols_diff1,Cols_diff2)),'edge'))
    cur_image_collect.pad_dims = [Rows_diff1, Rows_diff2, Cols_diff1, Cols_diff2]

all_img_denoised = model.predict(np.expand_dims(np.array(all_img),axis=-1), batch_size = args.batch_size)
_ = gc.collect()

for ii in range(np.shape(all_img_denoised)[0]):
    img_denoised = all_img_denoised[ii][:,:,0]
    pad_dims = image_collect[ii].pad_dims
    sub_img_folder = image_collect[ii].sub_folder 
    img_name = image_collect[ii].img_name 

    img_denoised = img_denoised[pad_dims[0]:(-pad_dims[1]),pad_dims[2]:(-pad_dims[3])]

    if args.loss_func == 'mse_relu':
        img_denoised = img_denoised * myrange + 2*np.sqrt(3/8)
        img_denoised = Anscombe_inverse_exact_unbiased(img_denoised)
    else:
        img_denoised = img_denoised * myrange
    
    img_denoised[img_denoised<0] = 0
    sub_save_directory = args.save_directory + sub_img_folder[len(args.load_directory):]
    if not os.path.exists(sub_save_directory):
        os.makedirs(sub_save_directory)
    tp.imsave(sub_save_directory + img_name, img_denoised.astype('float32'))

    print(sub_save_directory + img_name + ' saved!')
 
end = time.time()
print(end - start)

