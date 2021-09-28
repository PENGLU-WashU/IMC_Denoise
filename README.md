# IMC-Denoise: a content aware denoising pipeline to enhance Imaging Mass Cytometry

## Contents

- [Introduction to the project](#introduction-to-the-project)
- [Directory structure of IMC-Denoise](#directory-structure-of-imc-denoise)
- [Customize environment for IMC-Denoise](#customize-environment-for-imc-denoise)
  - [Our IMC-Denoise environment](#our-imc-denoise-environment)
  - [Installation](#installation)
- [Implement IMC-Denoise](#implement-imc-denoise)
  - [Directory structure of raw IMC images](#directory-structure-of-raw-imc-images) 
  - [Download example data](#download-example-data)
  - [IMC-Denoise tutorials with Jupyter Notebook](#imc-denoise-tutorials-with-jupyter-notebook)
  - [Implement IMC-Denoise with scripts](#implement-imc-denoise-with-scripts)
- [License](#license)
- [Contact](#contact)
- [References](#references)

<!-- Introduction to the project -->
## Introduction to the project

<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/IMC_paper_fig-1.png" alt="Logo" width="600" align = "right">
  </a>
</p>

Imaging Mass Cytometry (IMC) is an emerging multiplexed imaging technology for analyzing complex microenvironments that has the ability to detect the spatial distribution of at least 40 cell markers. However, this new modality has unique image data processing requirements, particularly when applying this
technology to patient tissue specimens. In these cases, signal-to-noise ratio (SNR) for particular markers can be low despite optimization of staining conditions, and the presence of pixel intensity artifacts can deteriorate image quality and the performance of downstream analysis. Here we demonstrate a contentaware
pipeline, IMCDenoise, to restore IMC images. Specifically, we deploy a differential intensity mapbased restoration (DIMR) algorithm for removing hot pixels and a selfsupervised deep learning algorithm for filtering photon shot noise (DeepSNF). IMCDenoise outperforms existing methods for adaptive hot pixel removal without loss of resolution and delivers significant SNR improvement to a diverse set of IMC channels and datasets, including a technically challenging unique human bone marrow IMC dataset. Moreover, with cellscale analysis on this bone marrow data, our approach reduces noise variability in modeling of
intercell communications, enhances cell phenotyping including T cell subsetspecific biological interpretations.

<!-- Directory structure -->
## Directory structure of IMC-Denoise
```
IMC_Denoise
|---IMC_Denoise
|---|---IMC-Denoise_main
|---|---|---DeepSNF.py
|---|---|---DeepSNF_model.py
|---|---|---DIMR.py
|---|---|---loss_functions.py
|---|---DataGenerator
|---|---|---DeepSNF_DataGenerator.py
|---|---Anscombe_transform_function
|---|---|---Anscombe_transform.py
|---|---|---Anscombe_vectors.mat
|---|---|---place_holder.py
|---|---N2V_utils
|---|---|---N2V_util.py
|---|---|---N2V_DataWrapper.py
|---scripts
|---|---Data_generation_script.py
|---|---Training_script.py
|---|---Generate_data_and_training.py
|---|---Predict_script.py
|---Jupyter_Notebook_examples
|---|---IMC_Denoise_Train_and_Predict.ipynb
|---|---IMC_Denoise_Train.ipynb
|---|---IMC_Denoise_Predict.ipynb
```
- **IMC_Denoise** implements DIMR and DeepSNF algorithms to remove hot pixels and filter shot noise in IMC images, respectively.
- **scripts** and **Jupyter Notebooks** include several examples to implement IMC_Denoise algorithms.

## Customize environment for IMC-Denoise
### Our IMC-Denoise environment
- Windows 10 64bit
- Python 3.6
- Tensorflow 2.2.0
- Keras 2.3.1
- NVIDIA GPU (24 GB Memory) + CUDA (smaller memory also works)

### Installation
- Create a virtual environment and install tensorflow-gpu, keras and jupyter.
```
$ conda create -n 'IMC_Denoise' python=3.6
$ conda activate IMC_Denoise
$ pip install tensorflow-gpu==2.2.0 keras==2.3.1
$ pip install jupyter
```
- Download the source code and install the package
```
$ git clone git://github.com/LUPENG7803111/IMC_Denoise
$ cd IMC_Denoise
$ pip install -e .
```

## Implement IMC-Denoise
### Directory structure of raw IMC images
In order to generate training set for DeepSNF, the directory structure of raw IMC images must be arranged as follows.
```
|---Raw_image_directory
|---|---Tissue1_sub_directory
|---|---|---Marker1_img.tiff
|---|---|---Marker2_img.tiff
             ...
|---|---|---Marker_n_img.tiff
|---|---Tissue2_sub_directory
|---|---|---Marker1_img.tiff
|---|---|---Marker2_img.tiff
             ...
|---|---|---Marker_n_img.tiff
             ...
|---|---Tissue_m_sub_directory
|---|---|---Marker1_img.tiff
|---|---|---Marker2_img.tiff
             ...
|---|---|---Marker_n_img.tiff
```
### Download example data

### IMC-Denoise tutorials with Jupyter Notebook
- [DeepSNF: generate data and training](https://github.com/LUPENG7803111/IMC_Denoise/Jupyter_Notebook_examples/IMC_Denoise_Train.ipynb)
- [IMC-Denoise: remove hot pixels with DIMR and filter shot noise with the pre-trained model of DeepSNF](https://github.com/LUPENG7803111/IMC_Denoise/Jupyter_Notebook_examples/IMC_Denoise_Predict.ipynb)
- [IMC-Denoise: remove hot pixels with DIMR and filter shot noise with the onsite training of DeepSNF](https://github.com/LUPENG7803111/IMC_Denoise/Jupyter_Notebook_examples/IMC_Denoise_Train_and_Predict.ipynb)

### Implement IMC-Denoise with scripts
- Generate training set of a specific marker channel for DeepSNF. The generated training data will be saved in a sub-directory "Generated_training_set" of the current folder other than setting a customized folder. Here we take CD38 channel as an example.
```
python scripts/Data_generation_script.py --marker_name 'CD38' 
                                         --Raw_directory "Raw_IMC_for_training" 
```
- Train a DeepSNF network. The generated training set will be loaded from a default folder other than choosing a customized folder. The trained weights will be saved in a sub-directory "trained_weights" of the current folder other than setting a customized folder. Hyper-parameters can be adjusted.
```
python scripts/Training_script.py --train_set_name 'training_set_CD38.npz' 
                                  --weights_name 'weights_CD38.hdf5' 
                                  --train_epoches '50' 
                                  --train_batch_size '128'
```
- Generate training set for a specific marker channel and then train a DeepSNF network. In this process, the generated training set will not be saved in a directory.
```
python scripts/Generate_data_and_training.py --marker_name 'CD38' 
                                             --weights_name 'weights_CD38.hdf5'
                                             --Raw_directory "Raw_IMC_for_training" 
                                             --train_epoches '50' 
                                             --train_batch_size '128'
```                                             
- Implement IMC-Denoise including DIMR and DeepSNF for a single IMC image. The trained weights will be loaded from the default directory other than choosing a customized folder. 
```
python scripts/Predict_script.py --Raw_img_name 'Raw_IMC_dataset\H1527528\141Pr-CD38_Pr141.tiff' 
                                 --Denoised_img_name 'Denoised_IMC_dataset\141Pr-CD38_Pr141.tiff' 
                                 --weights_name "weights_CD38.hdf5" 
```
<!-- LICENSE -->
## License

<!-- CONTACT -->
## Contact

Peng Lu - [@penglu10](https://twitter.com/penglu10) - penglu@wustl.edu
<br/>Project Link: [https://github.com/LUPENG7803111/IMC_Denoise](https://github.com/LUPENG7803111/IMC_Denoise)
<br/>Lab Website: [Thorek Lab WashU](https://sites.wustl.edu/thoreklab/)

## References


