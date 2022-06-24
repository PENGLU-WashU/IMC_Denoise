# -*- coding: utf-8 -*-
import argparse
import sys
import numpy as np
import os
from os import listdir
from os.path import isfile, join, exists

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--load_directory", help = "the folder of the generated IMC training sets", type = str)
parser.add_argument("--save_directory", help = "the folder to save the combined IMC training set", type = str)
parser.add_argument("--saved_training_set_name", help = "the name of the saved IMC training set", default = 'combined_training_set.npz', type = str)
                    
args = parser.parse_args()
print(args)

if not args.saved_training_set_name.endswith(".npz"):
    print('The saved training set name must end with .npz!')
    quit()
    
if not exists(args.save_directory):
    os.makedirs(args.save_directory)

dataset_list = [f for f in listdir(args.load_directory) if isfile(join(args.load_directory, f)) & f.endswith(".npz")]
for ii in range(len(dataset_list)):
    data_file = dataset_list[ii]
    print('Loading {}'.format(data_file))
    cur_dataset = np.load(args.load_directory + '\\' + data_file)
    cur_dataset = cur_dataset['patches']
    if ii == 0:
        combined_dataset = cur_dataset
    else:
        combined_dataset = np.concatenate((combined_dataset, cur_dataset), axis = 0)
        
np.random.shuffle(combined_dataset)
saved_name = join(args.save_directory, args.saved_training_set_name)
np.savez(saved_name, patches = combined_dataset)
print('The generated training set with shape of {} is saved as {}.'.format(combined_dataset.shape, saved_name))

    
 
