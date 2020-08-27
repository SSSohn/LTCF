# -*- coding: UTF-8 -*-
import sys
import torch
import torchvision
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from os import path
import os

dataPath = 'Data/'
savePath = 'DataNumpy/'

trainingCount = 150
testingCount = 10
counts = { 'Training' : trainingCount, 'Testing' : testingCount }

image_ext = '.png'
image_size = 112

for t in counts:
	for i in range(counts[t]):
		print(i)

		tempDataPath = dataPath + t + '/'
		path_compCoordX = tempDataPath + 'Cx\'/'
		path_compCoordY = tempDataPath + 'Cy\'/'
		path_compObs = tempDataPath + 'E\'/'
		path_compOcc = tempDataPath + 'A\'/'
		path_compDist = tempDataPath + 'G\'/'
		path_compGT = tempDataPath + 'Y\'/'

		image_open_path_A = path_compOcc + str(i) + image_ext
		image_open_path_E = path_compObs + str(i) + image_ext
		image_open_path_G = path_compDist + str(i) + image_ext
		image_open_path_Cx = path_compCoordX + str(i) + image_ext
		image_open_path_Cy = path_compCoordY + str(i) + image_ext
		image_open_path_Y = path_compGT + str(i) + image_ext

		if not path.exists(image_open_path_A):
			break

		image_A = Image.open(image_open_path_A)
		image_A = np.array(image_A, dtype=float)
		image_A = image_A / 255.0

		image_E = Image.open(image_open_path_E)
		image_E = np.array(image_E, dtype=float)
		image_E = image_E / 255.0

		image_G = Image.open(image_open_path_G)
		image_G = np.array(image_G, dtype=float)
		image_G = image_G / 255.0

		image_Cx = Image.open(image_open_path_Cx)
		image_Cx = np.array(image_Cx, dtype=float)
		image_Cx = image_Cx / 255.0

		image_Cy = Image.open(image_open_path_Cy)
		image_Cy = np.array(image_Cy, dtype=float)
		image_Cy = image_Cy / 255.0

		image_Y = Image.open(image_open_path_Y)
		image_Y = np.array(image_Y, dtype=float)
		image_Y = image_Y / 255.0

		image_input = np.zeros(image_size * image_size * 5).reshape(image_size, image_size, 5)
		for Dx in range(image_size):
			for Dy in range(image_size):
				image_input[Dx, Dy, 0] = image_A[Dx, Dy]
				image_input[Dx, Dy, 1] = image_E[Dx, Dy]
				image_input[Dx, Dy, 2] = image_G[Dx, Dy]
				image_input[Dx, Dy, 3] = image_Cx[Dx, Dy]
				image_input[Dx, Dy, 4] = image_Cy[Dx, Dy]
		image_input = image_input.transpose()
		np.save(savePath + t + '/Input/' + str(i) + '.npy', image_input)
		
		image_output = np.zeros(image_size * image_size * 1).reshape(image_size, image_size, 1)
		for Dx in range(image_size):
			for Dy in range(image_size):
				image_output[Dx, Dy, 0] = image_Y[Dx, Dy]
		image_output = image_output.transpose()
		np.save(savePath + t + '/Output/' + str(i) + '.npy', image_output)

	pathsFilePath = savePath + t + '/paths.txt'
	if path.exists(pathsFilePath):
		os.remove(pathsFilePath)
	f = open(pathsFilePath, 'a+')
	for i in range(counts[t]):
		inpath = savePath + t + '/Input/' + str(i) + '.npy'
		outpath = savePath + t + '/Output/' + str(i) + '.npy'
		f.write(inpath + ' ' + outpath + '\n')
	f.close()
