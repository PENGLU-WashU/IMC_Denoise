# -*- coding: utf-8 -*-
"""
python scripts/Data_generation_DeepSNF_script.py --channel_name '141Pr' 
                                                 --Raw_directory "D:\python_file_folder\IMC_learning\IMC_Denoise\Raw_IMC_for_training"
                                                 --n_neighbours '4' --n_lambda '5' --slide_window_size '3'
"""
         
from IMC_Denoise.DeepSNF_utils.DeepSNF_DataGenerator import DeepSNF_DataGenerator
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--channel_name", help = "Channel used to generate training set, e.g. 141Pr", type = str)
parser.add_argument("--patch_row_size", help = "The row size of generated patch.", default = 64, type = int)
parser.add_argument("--patch_col_size", help = "The column size of generated patch.", default = 64, type = int)
parser.add_argument("--row_step", help = "Row step length when generating training patches from imgs.", default = 60, type = int)
parser.add_argument("--col_step", help = "Column step length when generating training patches from imgs.", default = 60, type = int)
parser.add_argument("--is_augment", help = "Augment data", default = True, type = bool)
parser.add_argument("--ratio_thresh", help = "The threshold of the sparsity of the generated patch. If larger than this threshold, \
            the corresponding patch will be omitted. The default is 0.95.", default = 0.95, type = float)
parser.add_argument("--Raw_directory", help = "The directory which contained raw IMC images used to generate training set", type = str)
parser.add_argument("--Save_directory", help = "The directory which saves the generated training set. If None, the generated \
                    training set will be saved in the default directory", default = None, type = str)
parser.add_argument("--n_neighbours", help = "DIMR algorithm parameter", default = 4, type = int)
parser.add_argument("--n_lambda", help = "DIMR algorithm parameter", default = 5, type = float)
parser.add_argument("--slide_window_size", help = "DIMR algorithm parameter", default = 3, type = int)


args = parser.parse_args()
print(args)
print(args.slide_window_size+1)
DataGenerator = DeepSNF_DataGenerator(channel_name = args.channel_name, is_augment = args.is_augment,\
                                      patch_row_size = args.patch_row_size, patch_col_size = args.patch_col_size, \
                                      row_step = args.row_step, col_step = args.col_step, \
                                      ratio_thresh = args.ratio_thresh, n_neighbours = args.n_neighbours, \
                                      n_lambda = args.n_lambda, window_size = args.slide_window_size)
generated_patches = DataGenerator.generate_patches_from_directory(load_directory = args.Raw_directory)
if DataGenerator.save_patches(generated_patches,  save_directory = args.Save_directory):  
    print('Data generated successfully for ' + str(args.channel_name) + '.')
