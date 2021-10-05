# -*- coding: utf-8 -*-
"""
python scipts/Predict_DIMR_script.py --Raw_img_name 'D:\IMC analysis\Raw_IMC_dataset\H1527528\141Pr-CD38_Pr141.tiff' 
                                     --Denoised_img_name 'D:\IMC analysis\Denoised_IMC_dataset\141Pr-CD38_Pr141.tiff' 
                                     --n_neighbours '4' --n_lambda '5' --slide_window_size '3' 
"""

import tifffile as tp
from IMC_Denoise.IMC_Denoise_main.DIMR import DIMR
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--Raw_img_name", help = "the raw IMC image. tiff format", type = str)
parser.add_argument("--Denoised_img_name", help = "the denoised IMC image. tiff format", type = str)
parser.add_argument("--n_neighbours", help = "DIMR algorithm parameter", default = 4, type = int)
parser.add_argument("--n_lambda", help = "DIMR algorithm parameter", default = 5,  type = float)
parser.add_argument("--slide_window_size", help = "DIMR algorithm parameter", default=3, type = int)
                    
args = parser.parse_args()
print(args)

dimr = DIMR(n_neighbours = args.n_neighbours,
                  n_lambda = args.n_lambda,
                  window_size = args.slide_window_size) 

Img_raw = tp.imread(args.Raw_img_name)
Img_DIMR = dimr.perform_DIMR(Img_raw)

tp.imsave(args.Denoised_img_name, Img_DIMR.astype('float32'))
print('The denoised image has been saved!')