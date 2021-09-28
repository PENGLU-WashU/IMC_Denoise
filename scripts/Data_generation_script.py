# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 22:34:44 2021

@author: penglu
"""
# python Data_generation_script.py --marker_name 'CD38' --Raw_directory "D:\python_file_folder\IMC_learning\IMC_Denoise\Raw_IMC_for_training" 
         
from IMC_Denoise.DataGenerator.DeepSNF_DataGenerator import DeepSNF_DataGenerator
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--marker_name", help = "marker used to generate training set")
parser.add_argument("--Raw_directory", help = "The directory which contained raw IMC images used to generate training set")
parser.add_argument("--Save_directory", help = "The directory which saves the generated training set. If None, the generated \
                    training set will be saved in the default directory", default = None)
parser.add_argument("--n_neighbours", help = "DIMR algorithm parameter", default = 4, type = int)
parser.add_argument("--n_lambda", help = "DIMR algorithm parameter", default = 5)
parser.add_argument("--slide_window_size", help = "DIMR algorithm parameter", default = 3)

args = parser.parse_args()
print(args)

DataGenerator = DeepSNF_DataGenerator(marker_name = args.marker_name, n_neighbours = args.n_neighbours, \
                                      n_lambda = args.n_lambda, window_size = args.slide_window_size)
generated_patches = DataGenerator.generate_patches_from_directory(load_directory = args.Raw_directory)
if DataGenerator.save_patches(generated_patches,  save_directory = args.Save_directory):  
    print('Data generated successfully for ' + str(args.marker_name) + '.')