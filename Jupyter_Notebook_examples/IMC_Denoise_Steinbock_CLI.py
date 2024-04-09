### This script is to integrate DMIR / DeepSNIF analysis from the IMC_denoise package into the steinbock (https://bodenmillergroup.github.io/steinbock/) workflow
### Step of steinbock pipeline: after the conversion of mcd into .tiff files 

### Edited from the example DeepSNiF train/run jupyter notebook script in IMC_denoise
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tp
from IMC_Denoise.IMC_Denoise_main.DIMR import DIMR
from IMC_Denoise.IMC_Denoise_main.DeepSNiF import DeepSNiF
from IMC_Denoise.DeepSNiF_utils.DeepSNiF_DataGenerator import DeepSNiF_DataGenerator
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--train", type=str, help="Path to the training directory")
parser.add_argument("-i", "--input", type=str, help="Path to the input directory where raw multichannel tiffs are located")
parser.add_argument("-p", "--pre_denoised", type=str, default=None, help="Optional: Path to the pre-denoised directory where original raw files will be backed up")
parser.add_argument("-c", "--channels", type=str, default=None, help="Comma-separated list of channels to denoise (e.g., '1,2,3'). If not provided, all channels will be used.")
parser.add_argument("-b", "--batch_size", type=int, default=128, help="Batch size for training of DeepSNIF, default is 128")
parser.add_argument("-e", "--epochs", type=int, default=100, help="Number of Epochs for training, default is 100")
parser.add_argument("-l", "--learning_rate", type=int, default=1e-3, help="Initial learning rate, default ist 1e-3")
args = parser.parse_args()

if 'generated_patches' in globals():
    del generated_patches

train_directory = args.train 
Raw_directory = args.input 
pre_denoised_directory = args.pre_denoised

if not os.path.exists(train_directory) or not os.path.exists(Raw_directory):
    raise FileNotFoundError("Training or raw directory does not exist.")

if pre_denoised_directory:
    if not os.path.exists(pre_denoised_directory):
        os.makedirs(pre_denoised_directory)
    for filename in os.listdir(Raw_directory):
        shutil.copy(os.path.join(Raw_directory, filename), os.path.join(pre_denoised_directory, filename))
    print(f"Original raw files have been backed up to {pre_denoised_directory}")

output_directory = Raw_directory

if args.channels:
    channel_names = [int(ch.strip()) for ch in args.channels.split(',')]
else:
    # Dynamically determine all channels if none are specified
    print('No channels specified! Will attempt to denoise all channels in the provided .tiffs')
    first_tiff_path = os.path.join(Raw_directory, os.listdir(Raw_directory)[0])
    with tp.TiffFile(first_tiff_path) as tif:
        channel_number = len(tif.pages)
        channel_names = list(range(channel_number))  
### Step 0.1: edit IMC_denoise so that it will ingest multi-channle tiffs directly from the directory, instead of requiring the directory structure of the original IMC_denoise package
# Edits in this file: ...\IMC_Denoise\DeepSNIF_utils\DeepSniF_DataGenerator.py
#display()

## Step 0.2: set up a function that integrates all the training functions of IMC_denoise
# Just done here so that the code for iterating through the channels is simpler
# DO note the -- run_type = 'multi_channel_tiff' --  attribute in the DataGenerator call: this is what allows the ingestion of multi-channel .tiffs by the program 
# Without specifying run_type (by specifying runtype = 'single_channel_tiff'), IMC Denoise should default to reading the official, single-channel .tiff format
def DeepSNIF_train(channel_name, n_neighbours = 4, n_iter = 3, window_size = 3, train_epoches = 100, train_initial_lr = 1e-3, 
                  train_batch_size = 128, pixel_mask_percent = 0.2, val_set_percent = 0.15, loss_function = "I_divergence",
                  weights_name = None, loss_name = None, weights_save_directory = None, lambda_HF = 3e-6):
    '''
    This function merges the training functions of IMC_denoise for simplicity of the script here, particularly with most/all default setting for hyperparamters.
    If you want more control / want to adjust many of the hyperparameters, it may be better to split this function into the original pieces like original IMC_Denoise_Train_and_Predict notebook. 
    This function is also set to work with multi-channel .tiffs, not single-channel, but that could be changed by editing the runtype = 'multi_channel_tiff' argument in the DataGenerator call. 
    '''
    DataGenerator = DeepSNiF_DataGenerator(run_type = 'multi_channel_tiff', channel_name = channel_name, ratio_thresh = 0.8,
                                           patch_row_size = 64, patch_col_size = 64, row_step = 60, col_step = 60,
                                           n_neighbours = n_neighbours, n_iter = n_iter, window_size = window_size)
    generated_patches = DataGenerator.generate_patches_from_directory(load_directory = train_directory)
    print('The shape of the generated training set for channel ' + str(channel_name)  + ' is ' + str(generated_patches.shape) + '.')
    is_load_weights = False # Use the trained model directly. Will not read from saved one.
    deepsnif = DeepSNiF(train_epoches = args.epochs, 
                    train_learning_rate = args.learning_rate,
                    train_batch_size = args.batch_size,
                    mask_perc_pix = pixel_mask_percent,
                    val_perc = val_set_percent,
                    loss_func = loss_function,
                    weights_name = weights_name,
                    loss_name = loss_name,
                    weights_dir = weights_save_directory, 
                    is_load_weights = is_load_weights,
                    lambda_HF = lambda_HF)
    train_loss, val_loss = deepsnif.train(generated_patches)
    return(n_neighbours, n_iter, window_size, train_loss, val_loss, deepsnif)


# Step 1: iterate through the channels, training  and then running for each image
for i in channel_names:
    n_neighbours, n_iter, window_size, train_loss, val_loss, deepsnif = DeepSNIF_train(i)
    for img in os.listdir(Raw_directory):
        img_path = os.path.join(Raw_directory, img)
        if not img_path.lower().endswith('.tif'):
            continue  # Skip non-TIFF files
        Img_raw = tp.TiffFile(img_path).pages[i].asarray()
        Img_DIMR_DeepSNiF = deepsnif.perform_IMC_Denoise(Img_raw, n_neighbours=n_neighbours, n_iter=n_iter, window_size=window_size)
        numpy_tiff = tp.imread(img_path)
        numpy_tiff[i] = Img_DIMR_DeepSNiF
        tp.imwrite(os.path.join(output_directory, img), numpy_tiff, photometric='minisblack')

# Steinbock should now be able to seemlessly work with the denoised files


