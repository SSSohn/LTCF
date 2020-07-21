## Installation

1. Clone this repository. In [Google Drive](https://drive.google.com/drive/folders/1Nw2OwdJU3K_IfR6Gexw-5raCpegKsFwv?usp=sharing), download the pre-trained models and place them in the `Models/Pre-Trained Models/` directory.

	```
	>> git clone https://github.com/SSSohn/LTCF.git
	>> cd LTCF
	```

2. We recommend installing Python v3.6.5 from [Anaconda](https://www.anaconda.com/), installing Pytorch (>= 1.1.0) following guide on the [official instructions](https://pytorch.org/) according to your specific CUDA version.

## Training


## Testing
In order to test the pre-trained models located in `Models/Pre-Trained Models/` on one of the Testing sets, move the contents of the Testing set into `Models/Testing/` and run `Models/RunTesting.py`. This will convert the image files into numpy files and predict the output using the specified model. The output images are saved in `Models/Output/`.
