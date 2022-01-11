# -*- coding: utf-8 -*-
"""
python scripts/Predict_IMC_Denoise_script.py --Raw_img_name 'D:\IMC analysis\Raw_IMC_dataset\H1527528\141Pr-CD38_Pr141.tiff' 
                                             --Denoised_img_name 'D:\IMC analysis\Denoised_IMC_dataset\141Pr-CD38_Pr141.tiff' 
                                             --weights_name "weights_141Pr-CD38.hdf5"   
                                             --n_neighbours '4' --n_iter '3' --slide_window_size '3' 
"""
import argparse
import os
from os import listdir
from os.path import isfile, join
from glob import glob
import tifffile as tp
# from keras import backend as K
from IMC_Denoise.IMC_Denoise_main.DeepSNF import DeepSNF

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--channel_name", help = "the denoised channel name", type = str)
parser.add_argument("--load_directory", help = "the folder of the raw IMC images", type = str)
parser.add_argument("--save_directory", help = "the folder to save the denoised IMC images", type = str)
parser.add_argument("--weights_name", help = "trained network weights. hdf5 format", type = str)
parser.add_argument("--weights_save_directory", help = "location where 'weights_name' saved. If the \
                    value is None, the files will be loaded from the default directory.", default = None, type = str)
parser.add_argument("--n_neighbours", help = "DIMR algorithm parameter", default = 4, type = int)
parser.add_argument("--n_iter", help = "DIMR algorithm parameter", default = 3, type = int)
parser.add_argument("--slide_window_size", help = "DIMR algorithm parameter", default=3, type = int)
                    
args = parser.parse_args()
print(args)

img_folders = glob(join(args.load_directory, "*", ""))
for sub_img_folder in img_folders:
    Img_list = [f for f in listdir(sub_img_folder) if isfile(join(sub_img_folder, f)) & f.endswith(".tiff")]
    for Img_file in Img_list:
        if args.channel_name.lower() in Img_file.lower():
            Img_read = tp.imread(sub_img_folder + Img_file).astype('float32')
            
            deepsnf = DeepSNF(weights_name = args.weights_name,
                  weights_dir = args.weights_save_directory,
                  is_load_weights = True) # in prediction, this parameter should be set as true so that the trained weights can be loaded.  
            Img_denoised = deepsnf.perform_IMC_Denoise(Img_read, n_neighbours = args.n_neighbours, n_iter = args.n_iter, window_size = args.slide_window_size)
            
            sub_save_directory = args.save_directory + sub_img_folder[len(args.load_directory):]
            if not os.path.exists(sub_save_directory):
                os.makedirs(sub_save_directory)
            tp.imsave(sub_save_directory + Img_file, Img_denoised.astype('float32'))
            
            print(sub_save_directory + Img_file + ' saved!')
            break
    # K.clear_session()
    # time.sleep(0.2)
# deepsnf = DeepSNF(weights_name = args.weights_name,
#                   weights_dir = args.weights_save_directory,
#                   is_load_weights = True) # in prediction, this parameter should be set as true so that the trained weights can be loaded.  

# deepsnf.perform_IMC_Denoise_from_directory(args.channel_name, args.load_directory, args.save_directory, n_neighbours = args.n_neighbours, n_iter = args.n_iter, window_size = args.slide_window_size)
# print('The denoised image has been saved!')
