# IMC-Denoise: a content aware denoising pipeline to enhance Imaging Mass Cytometry

## Contents

- [Introduction to the project](#introduction-to-the-project)
- [Directory structure](#directory-structure)
- [Customize environment for IMC-Denoise](#customize-environment-for-imc-denoise)
  - [Our IMC-Denoise environment](#our-imc-denoise-environment)
  - [Installation](#installation)
 

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
## Directory structure
```
|---IMC_Denoise
|---|---IMC_Denoise_main
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
|---Jupyter Notebooks
|---|---IMC_Denoise_Train_and_Predict.ipynb
|---|---IMC_Denoise_Train.ipynb
|---|---IMC_Denoise_Predict.ipynb
```
- **IMC_Denoise** implements DIMR and DeepSNF algorithms to remove hot pixels and filter shot noise in IMC images, respectively.
- **scripts** and **Jupyter Notebooks** include several examples to implement IMC_Denoise algorithms.

<!-- GETTING STARTED -->
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

## IMC-Denoise tutorials with Jupyter Notebook
- [DeepSNF: Generate data and training](https://github.com/LUPENG7803111/IMC_Denoise/Jupyter Notebooks/IMC_Denoise_Train.ipynb)
## Roadmap

See the [open issues](https://github.com/othneildrew/Best-README-Template/issues) for a list of proposed features (and known issues).



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

<!-- CONTACT -->
## Contact

Peng Lu - [@your_twitter](https://twitter.com/penglu10) - penglu@wustl.edu

Project Link: [https://github.com/LUPENG7803111/IMC_Denoise](https://github.com/LUPENG7803111/IMC_Denoise)

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements


