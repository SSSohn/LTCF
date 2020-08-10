# Long-Term Crowd Flow Dataset

This repository contains a dataset of 87,430 simulated crowd flows used in [Laying the Foundations of Deep Long-Term Crowd Flow Prediction]() by Samuel S. Sohn, Honglu Zhou, Seonghyeon Moon, Sejong Yoon, Vladimir Pavlovic, and Mubbasir Kapadia. If you find this dataset useful in your research, please consider citing:

```
@inproceedings{sohnECCV20crowdflow,
  author    = {Sohn, Samuel S. and Zhou, Honglu and Moon, Seonghyeon and Yoon, Sejong and Pavlovic, Vladimir and Kapadia, Mubbasir},
  title     = {Laying the Foundations of Deep Long-Term Crowd Flow Prediction},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2020}
}
```

## Installation

1. Clone this repository. In [Google Drive](https://drive.google.com/drive/folders/1Nw2OwdJU3K_IfR6Gexw-5raCpegKsFwv?usp=sharing), download the pre-trained models and place them in the `Models/Pre-Trained Models/` directory.

	```
	>> git clone https://github.com/SSSohn/LTCF.git
	>> cd LTCF
	```

2. We recommend installing Python v3.6.5 from [Anaconda](https://www.anaconda.com/), installing PyTorch (>= 1.1.0) following guide on the [official instructions](https://pytorch.org/) according to your specific CUDA version.

## Training

(TBA)


## Testing
In order to test the pre-trained models located in `Models/Pre-Trained Models/` on one of the Testing sets, move the contents of the Testing set into `Models/Testing/` and run `Models/RunTesting.py`. This will convert the image files into numpy files and predict the output using the specified model. The output images are saved in `Models/Output/`.


## Folder Structure
```
+---Proxy Crowd Flow
|   +---CR 1.00
|   |   +---Testing
|   |   \---Training
|   +---CR 1.25
|   |   +---Testing
|   |   \---Training
|   +---CR 1.50
|   |   +---Testing
|   |   \---Training
|   +---CR 1.75
|   |   +---Testing
|   |   \---Training
|   \---CR 2.00
|       +---Testing
|       \---Training
\---Simulated Crowd Flow
    +---Multi-Goal Non-Uniform Agents
    |   +---Testing
    |   \---Training
    +---Non-Axis-Aligned Real Floorplans
    |   \---Testing
    \---Single-Goal Uniform Agents
        +---Testing
        \---Training
```
In each of the `Training` and `Testing` folders, there are 10 sub-folders, each corresponding to a component in the Framework image below:
```
+---A
+---A'
+---Cx'
+---Cy'
+---E
+---E'
+---G
+---G'
+---Y  (Ground Truth)
\---Y' (Compressed Ground Truth)
```

## Dataset Sample
| A | G | E | Cx' | Cy' | A' | G' | E' | Y | Y' |
|---|---|---|---|---|---|---|---|---|---|
| <p align="center"><img src="Proxy Crowd Flow/CR 1.50/Testing/A/0.png" width="100%" alt="" /></p> | <p align="center"><img src="Proxy Crowd Flow/CR 1.50/Testing/G/0.png" width="100%" alt="" /></p> | <p align="center"><img src="Proxy Crowd Flow/CR 1.50/Testing/E/0.png" width="100%" alt="" /></p> | <p align="center"><img src="Proxy Crowd Flow/CR 1.50/Testing/Cx'/0.png" width="100%" alt="" /></p> | <p align="center"><img src="Proxy Crowd Flow/CR 1.50/Testing/Cy'/0.png" width="100%" alt="" /></p> | <p align="center"><img src="Proxy Crowd Flow/CR 1.50/Testing/A'/0.png" width="100%" alt="" /></p> | <p align="center"><img src="Proxy Crowd Flow/CR 1.50/Testing/G'/0.png" width="100%" alt="" /></p> | <p align="center"><img src="Proxy Crowd Flow/CR 1.50/Testing/E'/0.png" width="100%" alt="" /></p> | <p align="center"><img src="Proxy Crowd Flow/CR 1.50/Testing/Y/0.png" width="100%" alt="" /></p> | <p align="center"><img src="Proxy Crowd Flow/CR 1.50/Testing/Y'/0.png" width="100%" alt="" /></p> | 

## Framework
<p align="center"><img src="Framework Image.png" width="100%" alt="" /></p>
