# -*- coding: utf-8 -*-
"""
python scripts/Training_DeepSNF_script.py --train_set_name 'training_set_141Pr.npz' --weights_name 'weights_141Pr-CD38.hdf5' --train_epoches '50' --train_batch_size '128'

"""

from IMC_Denoise.IMC_Denoise_main.DeepSNF import DeepSNF
from IMC_Denoise.DeepSNF_utils.DeepSNF_DataGenerator import load_training_patches
import argparse

seed_value= 1

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
# tf.random.set_seed(seed_value)
# for later versions: 
tf.compat.v1.set_random_seed(seed_value)

# 5. Configure a new global `tensorflow` session
from keras import backend as K
# session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
# K.set_session(sess)
# for later versions:
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

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

parser.add_argument("--train_set_name", help = "the pre-generated training set. npz format", type = str)
parser.add_argument("--train_data_directory", help = "training data directory. If None, the \
                    training set will be loaded from the default directory", default = None, type = str)
parser.add_argument("--weights_name", help = "trained network weights. hdf5 format", type = str)
parser.add_argument("--loss_name", help = "training and validation losses saved here, either .mat or .npz format. \
                    If not defined, the losses will not be saved.", default = None, type = str)
parser.add_argument("--weights_save_directory", help = "location where 'weights_name' and 'loss_name' saved. If the \
                    value is None, the files will be saved in the current file folder.", default = None, type = str)
parser.add_argument("--train_epoches", help = "training_epoches", default = 100, type = int)
parser.add_argument("--train_initial_lr", help = "initial learning rate", default = 1e-3, type = float)
parser.add_argument("--lr_decay_rate", help = "decay learning rate", default = 0.5, type = float)
parser.add_argument("--train_batch_size", help = "batch size", default = 256, type = int)
parser.add_argument("--pixel_mask_percent", help = "percentage of the masked pixels in each patch", default = 0.2, type = float)
parser.add_argument("--val_set_percent", help = "percentage of validation set", default = 0.15, type = float)
parser.add_argument("--loss_function", help = "loss function used, I_divergence, mse or mse_relu", default = "I_divergence", type = str)
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

print('###########################################################')
print("loading training data...")
train_data = load_training_patches(filename = args.train_set_name, save_directory = args.train_data_directory)
print("data loading completed!")
print('###########################################################')

deepsnf = DeepSNF(train_epoches = args.train_epoches, 
                  train_learning_rate = args.train_initial_lr,
                  lr_decay_rate = args.lr_decay_rate,
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

deepsnf.train(train_data)
