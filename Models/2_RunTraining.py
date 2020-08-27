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

modelName = "New Model"

input_nbr = 5
output_nbr = 1
iterations = 200
batchsize = 4

# ----------------------Prepare Dataset-----------------------------
def default_loader(path):
    return np.load(path)

class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], words[1]))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        layout_path, target_path = self.imgs[index]
        layout = self.loader(layout_path)
        target = self.loader(target_path)
        return torch.from_numpy(layout.astype(float)), torch.from_numpy(target.astype(float))

    def __len__(self):
        return len(self.imgs)

train_data = MyDataset(txt='DataNumpy/Training/paths.txt', transform=None)
test_data = MyDataset(txt='DataNumpy/Testing/paths.txt', transform=None)
train_loader = DataLoader(dataset=train_data, batch_size=batchsize, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batchsize)

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

model = SegNet(input_nbr, output_nbr)

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

if torch.cuda.is_available():
    USE_CUDA = 1
    model.cuda()
else:
    USE_CUDA = 0
    model.float()

learningRate = 0.01
momentumValue = 0.9
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate, momentum=momentumValue)

def train(epoch):
    model.train()
    lr = learningRate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if (USE_CUDA):
        loss_func = nn.L1Loss().cuda()
    else:
        loss_func = nn.L1Loss()

    total_loss = 0.
    i = 0
    mean_std = np.zeros(len(train_loader), dtype=float)

    for batch_x, batch_y in train_loader:
        input_batch = batch_x.float()
        output_batch = batch_y.float()
        if USE_CUDA:
            input_batch, output_batch = Variable(input_batch.cuda()), Variable(output_batch.cuda())
        else:
            input_batch, output_batch = Variable(input_batch), Variable(output_batch)

        output = model(input_batch)
        loss = loss_func(output, output_batch)
        mean_std[i] = loss
        total_loss += loss.cpu().data.numpy()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("training epoch[%d] iter[%d/%d] loss: %f" % (epoch + 1, i + 1, len(train_loader), loss.cpu().data.numpy()))
        i = i + 1

    return np.mean(mean_std), np.std(mean_std), output

def test(epoch):
    model.eval()
	
    if(USE_CUDA):
        loss_func = nn.L1Loss().cuda()
    else:
        loss_func = nn.L1Loss()
    total_loss = 0.

    i = 0
    mean_std = np.zeros(len(test_loader), dtype=float)
    for batch_x, batch_y in test_loader:
        input_batch = batch_x.float()
        output_batch = batch_y.float()
        if USE_CUDA:
            input_batch, output_batch = Variable(input_batch.cuda()), Variable(output_batch.cuda())
        else:
            input_batch, output_batch = Variable(input_batch), Variable(output_batch)

        output = model(input_batch)
        loss = loss_func(output,output_batch)
        mean_std[i] = loss
        print("testing epoch[%d] iter[%d/%d] loss: %f" % (epoch + 1, i + 1, len(test_loader), loss.cpu().data.numpy()))
        i = i + 1

    return np.mean(mean_std), np.std(mean_std),  output

for epoch in range(iterations):
    train_loss, train_std, train_output = train(epoch)

    print("epoch %d - Total Training Loss: %.6f" % (epoch + 1, train_loss))
    print("epoch %d - Training Loss SD : %.6f\n" % (epoch + 1, train_std))

    test_loss, test_std, test_output = test(epoch)
    print("epoch %d - Total Testing Loss: %.6f\n" % (epoch + 1, test_loss))

torch.save(model, modelPath + modelName + ".pt")