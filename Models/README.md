## Quick start

This repository is build upon Python v2.7 and Pytorch v1.0.1. The code may also work with Pytorch v0.4.1 but has not been tested.

### Installation

1. Clone this repository. In [Google Drive](https://drive.google.com/drive/folders/1U-Xp7w31jJJ3IN98ZfEHyXVqCQNXlG3s?usp=sharing), download `param_mesh.mat` and put it into `configs` directory. Then download `phase1_wpdc_vdc_v2.pth.tar` and `shape_predictor_68_face_landmarks.dat`, and put them into `models` directory.

	```
	>> git clone git@github.com:garyzhao/FRGAN.git
	>> cd FRGAN
	```

2. We recommend installing Python v2.7 from [Anaconda](https://www.anaconda.com/), installing Pytorch (>= 1.0.1) following guide on the [official instructions](https://pytorch.org/) according to your specific CUDA version. In addition, you need to install dependencies below.

	```
	>> pip install -r requirements.txt
	```

3. Build the C++ extension for computing normal maps from the 3D face model.

	```
	>> cd dfa
	>> python setup.py build_ext --inplace
	>> cd -
	```
