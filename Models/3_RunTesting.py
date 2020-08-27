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
import os.path
from os import path

modelName = "PanSF"
trainingCount = 10

input_nbr = 5
output_nbr = 1

# ----------------------Define SegNet-----------------------------
class SegNet(nn.Module):
    def __init__(self, input_nbr, output_nbr):
        super(SegNet, self).__init__()

        batchNorm_momentum = 0.1

        self.conv21 = nn.Conv2d(input_nbr, 128, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(128, momentum=batchNorm_momentum)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(128, momentum=batchNorm_momentum)

        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(256, momentum=batchNorm_momentum)

        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)

        self.conv51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)

        self.conv53d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv52d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv51d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)

        self.conv43d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv42d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv41d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)

        self.conv33d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv32d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv31d = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(128, momentum=batchNorm_momentum)

        self.conv22d = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(128, momentum=batchNorm_momentum)
        self.conv21d = nn.Conv2d(128, output_nbr, kernel_size=3, padding=1)

        self.sigmoid0d = nn.Sigmoid()

    def forward(self, x):
        # Stage 2
        x21 = F.relu(self.bn21(self.conv21(x)))
        x22 = F.relu(self.bn22(self.conv22(x21)))
        x2p, id2 = F.max_pool2d(x22, kernel_size=2, stride=2, return_indices=True)

        # Stage 3
        x31 = F.relu(self.bn31(self.conv31(x2p)))
        x32 = F.relu(self.bn32(self.conv32(x31)))
        x33 = F.relu(self.bn33(self.conv33(x32)))
        x3p, id3 = F.max_pool2d(x33, kernel_size=2, stride=2, return_indices=True)

        # Stage 4
        x41 = F.relu(self.bn41(self.conv41(x3p)))
        x42 = F.relu(self.bn42(self.conv42(x41)))
        x43 = F.relu(self.bn43(self.conv43(x42)))
        x4p, id4 = F.max_pool2d(x43, kernel_size=2, stride=2, return_indices=True)

        # Stage 5
        x51 = F.relu(self.bn51(self.conv51(x4p)))
        x52 = F.relu(self.bn52(self.conv52(x51)))
        x53 = F.relu(self.bn53(self.conv53(x52)))
        x5p, id5 = F.max_pool2d(x53, kernel_size=2, stride=2, return_indices=True)

        # Stage 5d
        x5d = F.max_unpool2d(x5p, id5, kernel_size=2, stride=2)
        x53d = F.relu(self.bn53d(self.conv53d(x5d)))
        x52d = F.relu(self.bn52d(self.conv52d(x53d)))
        x51d = F.relu(self.bn51d(self.conv51d(x52d)))

        # Stage 4d
        x4d = F.max_unpool2d(x51d, id4, kernel_size=2, stride=2)
        x43d = F.relu(self.bn43d(self.conv43d(x4d)))
        x42d = F.relu(self.bn42d(self.conv42d(x43d)))
        x41d = F.relu(self.bn41d(self.conv41d(x42d)))

        # Stage 3d
        x3d = F.max_unpool2d(x41d, id3, kernel_size=2, stride=2)
        x33d = F.relu(self.bn33d(self.conv33d(x3d)))
        x32d = F.relu(self.bn32d(self.conv32d(x33d)))
        x31d = F.relu(self.bn31d(self.conv31d(x32d)))

        # Stage 2d
        x2d = F.max_unpool2d(x31d, id2, kernel_size=2, stride=2)
        x22d = F.relu(self.bn22d(self.conv22d(x2d)))
        x21d = self.conv21d(x22d)

        # Stage 0d
        x0d = self.sigmoid0d(x21d)

        return x0d

modelPath = "Pre-Trained Models/"
model = torch.load(modelPath + modelName + ".pt")

modelFlag = isinstance(model,dict)
if modelFlag:
    model = SegNet(input_nbr, output_nbr)
    model.load_state_dict(torch.load(modelPath + modelName +".pt"))

model.eval()

dataPath = 'Data/Testing/'
savePath = 'Output/'
path_compCoordX = dataPath + 'Cx\'/'
path_compCoordY = dataPath + 'Cy\'/'
path_compObs = dataPath + 'E\'/'
path_compOcc = dataPath + 'A\'/'
path_dist = dataPath + 'G\'/'

image_ext = '.png'
image_size = 112

for i in range(trainingCount):
    print(i)

    output_npy_path = savePath + str(i) + image_ext

    image_open_path_A = path_compOcc + str(i) + image_ext
    image_open_path_E = path_compObs + str(i) + image_ext
    image_open_path_G = path_dist + str(i) + image_ext
    image_open_path_Cx = path_compCoordX + str(i) + image_ext
    image_open_path_Cy = path_compCoordY + str(i) + image_ext

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

    image_input = np.zeros(image_size * image_size * 5).reshape(image_size, image_size, 5)
    for Dx in range(image_size):
        for Dy in range(image_size):
            image_input[Dx, Dy, 0] = image_A[Dx, Dy]
            image_input[Dx, Dy, 1] = image_E[Dx, Dy]
            image_input[Dx, Dy, 2] = image_G[Dx, Dy]
            image_input[Dx, Dy, 3] = image_Cx[Dx, Dy]
            image_input[Dx, Dy, 4] = image_Cy[Dx, Dy]
    image_input = image_input.transpose()

    data = image_input
    data = torch.tensor(data, dtype=torch.float64)
    data = Variable(data, requires_grad=True)
    data = data.unsqueeze(0)  # .cuda().type(torch.FloatTensor)
    data = data.type(torch.FloatTensor)
    
    if modelFlag:
        output = model(data)
    else:
        output = model(data.cuda())
    output_image = output.cpu().data.numpy()

    output_image = output_image[0, :, :, :].transpose()
    output_image = np.resize(output_image, (output_image.shape[0], output_image.shape[1]))

    result = Image.fromarray((output_image * 255).astype(np.uint8))
    result.save(output_npy_path)
