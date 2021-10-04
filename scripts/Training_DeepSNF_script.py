# -*- coding: utf-8 -*-
"""
python scripts/Training_DeepSNF_script.py --train_set_name 'training_set_141Pr.npz' --weights_name 'weights_141Pr-CD38.hdf5' --train_epoches '50' --train_batch_size '128'

"""

from IMC_Denoise.IMC_Denoise_main.DeepSNF import DeepSNF
from IMC_Denoise.DeepSNF_utils.DeepSNF_DataGenerator import load_training_patches
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--train_set_name", help = "the pre-generated training set. npz format", type = str)
parser.add_argument("--train_data_directory", help = "training data directory. If None, the \
                    training set will be loaded from the default directory", default = None, type = str)
parser.add_argument("--weights_name", help = "trained network weights. hdf5 format", type = str)
parser.add_argument("--loss_name", help = "training and validation losses saved here, either .mat or .npz format. \
                    If not defined, the losses will not be saved.", default = None, type = str)
parser.add_argument("--weights_save_directory", help = "location where 'weights_name' and 'loss_name' saved. If the \
                    value is None, the files will be saved in the current file folder.", default = None, type = str)
parser.add_argument("--train_epoches", help = "training_epoches", default = 100, type = int)
parser.add_argument("--train_initial_lr", help = "initial learning rate", default = 5e-4, type = float)
parser.add_argument("--train_batch_size", help = "batch size", default = 256, type = int)
parser.add_argument("--pixel_mask_percent", help = "percentage of the masked pixels in each patch", default = 0.2, type = float)
parser.add_argument("--val_set_percent", help = "percentage of validation set", default = 0.15, type = float)
parser.add_argument("--loss_function", help = "loss function used, bce or mse", default = "bce", type = str)
parser.add_argument("--is_load_weights", help = "If True, the pre-trained will be loaded, which is fit for \
                    prediction or transfer learning", default = False, type = bool)
parser.add_argument("--amp_max_rate", help = "the max_val of the channel is amp_max_rate*max(images, 0.99999 maximum truncated). \
                    The default is 1.1. It should work in most cases. \
                    When the maximum of the predicted image is much higher, the value may be set higher during \
                    training. But the values which is out of the range of the training set may not be predicted \
                    well. Therefore, the selection of a good training set is important.", default = 1.1, type = float)   

args = parser.parse_args()
print(args)

print('###########################################################')
print("loading training data...")
train_data = load_training_patches(filename = args.train_set_name, save_directory = args.train_data_directory)
print("data loading completed!")
print('###########################################################')

deepsnf = DeepSNF(train_epoches = args.train_epoches, 
                  train_learning_rate = args.train_initial_lr,
                  train_batch_size = args.train_batch_size,
                  mask_perc_pix = args.pixel_mask_percent,
                  val_perc = args.val_set_percent,
                  loss_func = args.loss_function,
                  weights_name = args.weights_name,
                  loss_name = args.loss_name,
                  weights_dir = args.weights_save_directory,
                  is_load_weights = args.is_load_weights,
                  amp_max_rate = self.amp_max_rate)

deepsnf.train(train_data)
