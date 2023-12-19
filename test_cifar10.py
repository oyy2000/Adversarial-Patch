import os
import json
import sys

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
from tqdm import tqdm


net = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)

CHECKPOINT_PATH = './saved_models/'

# CIFAR-10 related constants
NUM_CLASSES = 10
LABEL_NAMES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
MODELS = ['resnet20', 'resnet32', 'vgg16_bn', 'vgg19_bn'] #, 'densenet121', 'densenet161']

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 8
BATCH_SIZE=128

# Mean and Std from CIFAR-10
NORM_MEAN = np.array([0.491, 0.482, 0.446])
NORM_STD = np.array([0.247, 0.243, 0.261])
# load cifar-10

def place_patch(img, patch, random=False):
    for i in range(img.shape[0]):
        if random:
            h_offset = np.random.randint(0,img.shape[2]-patch.shape[1]-1)
            w_offset = np.random.randint(0,img.shape[3]-patch.shape[2]-1)
        else:
            h_offset = int(0.1 * (img.shape[2] - patch.shape[1]))
            w_offset = int(0.1 * (img.shape[3] - patch.shape[2]))
        TENSOR_MEANS, TENSOR_STD = torch.FloatTensor(NORM_MEAN)[:,None,None], torch.FloatTensor(NORM_STD)[:,None,None]
        img[i,:,h_offset:h_offset+patch.shape[1],w_offset:w_offset+patch.shape[2]] = (torch.tanh(patch) + 1 - 2 * TENSOR_MEANS) / (2 * TENSOR_STD)
    return img

# No resizing and center crop necessary as images are already preprocessed.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN,
                         std=NORM_STD)
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=NUM_WORKERS)

exmp_batch, label_batch = next(iter(testloader))
exmp_batch, label_batch = exmp_batch.to(device), label_batch.to(device)
# predict the original image
with torch.no_grad():
    net.eval()
    net.to(device)
    pred = net(exmp_batch)
    pred = pred.argmax(dim=1)
    print (pred)
    print (label_batch)
    # place patch on the exmp_batch
    # predict the patched image
    patch = torch.load(os.path.join(CHECKPOINT_PATH, 'resnet20/16_7.pt'))
    place_patch(exmp_batch, patch, random=True)
    pred = net(exmp_batch)
    pred = pred.argmax(dim=1)
    print (pred)
