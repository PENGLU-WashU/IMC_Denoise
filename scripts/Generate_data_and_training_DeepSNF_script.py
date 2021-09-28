# -*- coding: utf-8 -*-
"""
python scripts/Generate_data_and_training_DeepSNF_script.py --marker_name 'CD38' 
                                                            --weights_name 'weights_CD38.hdf5'
                                                            --Raw_directory "Raw_IMC_for_training" 
                                                            --train_epoches '50' 
                                                            --train_batch_size '128'
                                             
"""
from IMC_Denoise.IMC_Denoise_main.DeepSNF import DeepSNF
from IMC_Denoise.DataGenerator.DeepSNF_DataGenerator import DeepSNF_DataGenerator
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--marker_name", help = "marker used to generate training set")
parser.add_argument("--Raw_directory", help = "The directory which contained raw IMC images used to generate training set")
parser.add_argument("--n_neighbours", help = "DIMR algorithm parameter", default = 4, type = int)
parser.add_argument("--n_lambda", help = "DIMR algorithm parameter", default = 5)
parser.add_argument("--slide_window_size", help = "DIMR algorithm parameter", default = 3, type = int)
                    
parser.add_argument("--weights_name", help = "trained network weights. hdf5 format")
parser.add_argument("--loss_name", help = "training and validation losses saved here, either .mat or .npz format. \
                    If not defined, the losses will not be saved.", default = None)
parser.add_argument("--weights_save_directory", help = "location where 'weights_name' and 'loss_name' saved. If the \
                    value is None, the files will be saved in the current file folder.", default = None)
parser.add_argument("--train_epoches", help = "training_epoches", default = 100, type = int)
parser.add_argument("--train_initial_lr", help = "initial learning rate", default = 5e-4)
parser.add_argument("--train_batch_size", help = "batch size", default=256, type = int)
parser.add_argument("--pixel_mask_percent", help = "percentage of the masked pixels in each patch", default = 0.2)
parser.add_argument("--val_set_percent", help = "percentage of validation set", default = 0.15)
parser.add_argument("--loss_function", help = "loss function used, bce or mse", default = "bce")

args = parser.parse_args()
print(args)

DataGenerator = DeepSNF_DataGenerator(marker_name = args.marker_name, n_neighbours = args.n_neighbours, \
                                      n_lambda = args.n_lambda, window_size = args.slide_window_size)
generated_patches = DataGenerator.generate_patches_from_directory(load_directory = args.Raw_directory)

print('The shape of the generated training set is ' + str(generated_patches.shape) + '.')
print('###########################################################')

deepsnf = DeepSNF(train_epoches = args.train_epoches, 
                  train_learning_rate = args.train_initial_lr,
                  train_batch_size = args.train_batch_size,
                  mask_perc_pix = args.pixel_mask_percent,
                  val_perc = args.val_set_percent,
                  loss_func = args.loss_function,
                  weights_name = args.weights_name,
                  loss_name = args.loss_name,
                  weights_dir = args.weights_save_directory)

deepsnf.train(generated_patches)