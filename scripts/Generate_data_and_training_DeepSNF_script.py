# -*- coding: utf-8 -*-
"""
python scripts/Generate_data_and_training_DeepSNF_script.py --channel_name '141Pr' 
                                                            --weights_name 'weights_141Pr-CD38.hdf5'
                                                            --Raw_directory "Raw_IMC_for_training" 
                                                            --train_epoches '50' 
                                                            --train_batch_size '128'
                                                            --n_neighbours '4' --n_iter '3' --slide_window_size '3'
                                             
"""
from IMC_Denoise.IMC_Denoise_main.DeepSNF import DeepSNF
from IMC_Denoise.DeepSNF_utils.DeepSNF_DataGenerator import DeepSNF_DataGenerator
import argparse

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

parser.add_argument("--channel_name", help = "channel used to generate training set, e.g. 141Pr", type = str)
parser.add_argument("--is_augment", help = "Augment data?", default = True, type = str2bool)
parser.add_argument("--ratio_thresh", help = "The threshold of the sparsity of the generated patch. If larger than this threshold, \
            the corresponding patch will be omitted. The default is 0.8.", default = 0.8, type = float)
parser.add_argument("--patch_row_size", help = "The row size of generated patch.", default = 64, type = int)
parser.add_argument("--patch_col_size", help = "The column size of generated patch.", default = 64, type = int)
parser.add_argument("--row_step", help = "Row step length when generating training patches from imgs.", default = 60, type = int)
parser.add_argument("--col_step", help = "Column step length when generating training patches from imgs.", default = 60, type = int)
parser.add_argument("--Raw_directory", help = "The directory which contained raw IMC images used to generate training set", type = str)
parser.add_argument("--n_neighbours", help = "DIMR algorithm parameter", default = 4, type = int)
parser.add_argument("--n_iter", help = "DIMR algorithm parameter", default = 3, type = int)
parser.add_argument("--slide_window_size", help = "DIMR algorithm parameter", default = 3, type = int)
                    
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
                    prediction or transfer learning", default = False, type = str2bool)
parser.add_argument("--truncated_max_rate", help = "the max_val of the channel is 1.1*max(images, truncated_max_rate*maximum truncated). \
                    The default is 0.99999. It should work in most cases. \
                    When the maximum of the predicted image is much higher, the value may be set higher during \
                    training. But the values which is out of the range of the training set may not be predicted \
                    well. Therefore, the selection of a good training set is important.", default = 0.99999, type = float)    
parser.add_argument("--lambda_HF", help = "The parameter for Hessian regularization", default = 0, type = float)

args = parser.parse_args()
print(args)

DataGenerator = DeepSNF_DataGenerator(channel_name = args.channel_name,  is_augment = args.is_augment,\
                                      patch_row_size = args.patch_row_size, patch_col_size = args.patch_col_size, \
                                      row_step = args.row_step, col_step = args.col_step, \
                                      ratio_thresh = args.ratio_thresh, n_neighbours = args.n_neighbours, \
                                      n_iter = args.n_iter, window_size = args.slide_window_size)
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
                  weights_dir = args.weights_save_directory,
                  is_load_weights = args.is_load_weights,
                  truncated_max_rate = args.truncated_max_rate,
                  lambda_HF = args.lambda_HF)

deepsnf.train(generated_patches)
