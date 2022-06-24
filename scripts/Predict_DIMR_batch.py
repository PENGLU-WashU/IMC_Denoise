# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 11:16:00 2022

@author: penglu
"""


# -*- coding: utf-8 -*-
# Import Libraries and model
import numpy as np
import time

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


import argparse
from os import listdir
from os.path import isfile, join, exists
from glob import glob
import tifffile as tp
from IMC_Denoise.IMC_Denoise_main.DIMR import DIMR

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--channel_name", help = "the denoised channel name", type = str)
parser.add_argument("--load_directory", help = "the folder of the raw IMC images", type = str)
parser.add_argument("--save_directory", help = "the folder to save the denoised IMC images", type = str)
parser.add_argument("--n_neighbours", help = "DIMR algorithm parameter", default = 4, type = int)
parser.add_argument("--n_iter", help = "DIMR algorithm parameter", default = 3, type = int)
parser.add_argument("--slide_window_size", help = "DIMR algorithm parameter", default=3, type = int)
                    
args = parser.parse_args()
print(args)

start = time.time()

myDIMR = DIMR(n_neighbours = args.n_neighbours, n_iter = args.n_iter, window_size = args.slide_window_size)

img_folders = glob(join(args.load_directory, "*", ""))
for sub_img_folder in img_folders:
    Img_list = [f for f in listdir(sub_img_folder) if isfile(join(sub_img_folder, f)) & f.endswith(".tiff")]
    for Img_file in Img_list:
        if args.channel_name.lower() in Img_file.lower():
            Img_read = tp.imread(sub_img_folder + Img_file).astype('float32')
            
            Img_DIMR = myDIMR.perform_DIMR(Img_read)
                
            Img_DIMR[Img_DIMR<0] = 0
            sub_save_directory = join(args.save_directory, sub_img_folder[len(args.load_directory):])
            if not exists(sub_save_directory):
                os.makedirs(sub_save_directory)
            tp.imsave(join(sub_save_directory, Img_file), Img_DIMR.astype('float32'))
            
            print(sub_save_directory + Img_file + ' saved!')
            break
 
end = time.time()
print(end - start)
