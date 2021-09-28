# IMC-Denoise: a content aware denoising pipeline to enhance Imaging Mass Cytometry

## Contents

- [Introduction to the project](#Introduction to the project)


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
## Tensorflow and Keras code
### Environment
- Windows 10 64bit
- Python 3.6
- Tensorflow 2.2.0
- Keras 2.3.1
- NVIDIA GPU (24 GB Memory) + CUDA (smaller memory also works)

### Installation
- Create a virtual environment and install Keras and tensorflow-gpu.
```
$ conda create -n 'IMC_Denoise' python=3.6
$ conda activate IMC_Denoise
$ pip install tensorflow-gpu==2.2.0 keras==2.3.1
$ pip install jupyter
```
- Install other dependencies
```
$ pip install tifffile scipy 
```
### Download the source code
```
$ git clone git://github.com/LUPENG7803111/IMC_Denoise
$ cd IMC_Denoise
```

<!-- ROADMAP -->
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

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com

Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Img Shields](https://shields.io)
* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Pages](https://pages.github.com)
* [Animate.css](https://daneden.github.io/animate.css)
* [Loaders.css](https://connoratherton.com/loaders)
* [Slick Carousel](https://kenwheeler.github.io/slick)
* [Smooth Scroll](https://github.com/cferdinandi/smooth-scroll)
* [Sticky Kit](http://leafo.net/sticky-kit)
* [JVectorMap](http://jvectormap.com)
* [Font Awesome](https://fontawesome.com)





<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
