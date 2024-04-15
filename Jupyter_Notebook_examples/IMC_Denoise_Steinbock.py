### IMC_Denoise_integrated_with_steinbock_output.ipynb
import os
import shutil
import numpy as np
import tifffile as tp
from IMC_Denoise.IMC_Denoise_main.DIMR import DIMR
from IMC_Denoise.IMC_Denoise_main.DeepSNiF import DeepSNiF
from IMC_Denoise.DeepSNiF_utils.DeepSNiF_DataGenerator import DeepSNiF_DataGenerator
import argparse
import tempfile
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--train", type=str, help="Path to the training directory")
parser.add_argument("-i", "--input", type=str, help="Path to the input directory where raw multichannel tiffs are located")
parser.add_argument("-o", "--output", type=str, help="Path to the directory where denoised images will be saved. If not specified, denoised images will overwrite the original images in the input directory.")
parser.add_argument("-c", "--channels", type=str, default=None, help="Comma-separated list of channels to denoise (e.g., '1,2,3'). If not provided, all channels will be used.")
parser.add_argument("-b", "--batch_size", type=int, default=128, help="Batch size for training of DeepSNIF, default is 128")
parser.add_argument("-e", "--epochs", type=int, default=100, help="Number of Epochs for training, default is 100")
parser.add_argument("-l", "--learning_rate", type=float, default=1e-3, help="Initial learning rate, default ist 1e-3")
args = parser.parse_args()

train_directory = args.train 
Raw_directory = args.input
output_directory = args.output

# Directory checks
if not os.path.isdir(args.train):
    print(f"Error: The specified training directory '{args.train}' does not exist.")
    sys.exit(1)

if not os.path.isdir(args.input):
    print(f"Error: The specified input directory '{args.input}' does not exist.")
    sys.exit(1)

if output_directory:
    if not os.path.isdir(output_directory):
        print(f"Error: The specified output directory '{output_directory}' does not exist.")
        sys.exit(1)

if 'generated_patches' in globals():
    del generated_patches

if args.channels:
    channel_names = [int(ch.strip()) for ch in args.channels.split(',')]
else:
    channel_names = []

# Check if the output directory is the same as the raw directory
if Raw_directory == output_directory:
    # Create a temporary directory to store intermediate outputs
    temp_output_dir = tempfile.mkdtemp()
else:
    temp_output_dir = output_directory
     
## Step 0.1: set up a function that integrates all the training functions of IMC_denoise
# Just done here so that the code for iterating through the channels is simpler
# DO note the -- run_type = 'multi_channel_tiff' --  attribute in the DataGenerator call: this is what allows the ingestion of multi-channel .tiffs
# Without specifying run_type, IMC Denoise should default to reading the official, single-channel .tiff format
def DeepSNiF_train(channel_name, n_neighbours = 4, n_iter = 3, window_size = 3, train_epoches = 25, train_initial_lr = 1e-3, 
                  train_batch_size = 128, pixel_mask_percent = 0.2, val_set_percent = 0.15, loss_function = "I_divergence",
                  weights_name = None, loss_name = None, weights_save_directory = None, lambda_HF = 3e-6):
    '''
    This function merges the training functions of IMC_denoise for simplicity of the script here, particularly with most/all default setting for hyperparamters.
    If you want more control / want to adjust many of the hyperparameters, it may be better to split this function into its original pieces. 
    This function is also set to work only with multi-channel .tiffs, but that can be edited in the DataGenerator call.
    '''
    # Generate a unique weights name for each channel
    weights_name = f'IMC_Model_Channel_{channel_name}.hdf5'
    
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
 # This code block makes it so that if no channel names are specified, all channels are run! Depending on how many channels / how long the training -- that would take a very long time
if len(channel_names) < 1:
    print('No channels specified! Attempting to denoise all channels in provided .tiffs...')
    try:
        with tp.TiffFile(os.path.join(Raw_directory, os.listdir(Raw_directory)[0])) as tif:
            channel_names = list(range(len(tif.pages)))
    except Exception as e:
        print(f"Error reading TIFF file to determine channels: {e}")
        exit(1)

# Train a model for each channel and store it
trained_models = {}
for i in channel_names:
    n_neighbours, n_iter, window_size, train_loss, val_loss, deepsnif = DeepSNiF_train(i)
    trained_models[i] = deepsnif  # Store the trained model for later use

# Iterate through each image in the Raw_directory
for img in os.listdir(Raw_directory):
    img_path = os.path.join(Raw_directory, img)
    # Load the entire multi-channel image once
    numpy_tiff = tp.imread(img_path)

    # Process each specified channel using the pre-trained models
    for i in channel_names:
        deepsnif = trained_models[i]  # Retrieve the trained model for this channel
        Img_raw = numpy_tiff[i]  # Work directly with the loaded numpy array
        Img_DIMR_DeepSNiF = deepsnif.perform_IMC_Denoise(Img_raw, n_neighbours=n_neighbours, n_iter=n_iter, window_size=window_size)
        numpy_tiff[i] = Img_DIMR_DeepSNiF  # Update the channel in the numpy array

    # Write the fully denoised image to the output directory after all channels have been processed
    tp.imwrite(os.path.join(temp_output_dir, img), numpy_tiff, photometric='minisblack')

# If a temporary directory was used, move the processed files back to the original directory
if Raw_directory == output_directory:
    for img in os.listdir(temp_output_dir):
        shutil.move(os.path.join(temp_output_dir, img), os.path.join(Raw_directory, img))
    # Remove the temporary directory
    shutil.rmtree(temp_output_dir)
# Steinbock should now be able to seemlessly work with the denoised files
