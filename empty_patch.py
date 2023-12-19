# Generate random patches
import os
import json
import sys

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


def train_patch_untargeted(patch_size):
    if not isinstance(patch_size, tuple):
        patch_size = (patch_size, patch_size)
    patch = nn.Parameter(torch.zeros(3, patch_size[0], patch_size[1]), requires_grad=True)
    return patch

for patch_size in [3, 5, 7, 9, 16]:
    patch = train_patch_untargeted(patch_size)
    torch.save(patch, 'empty_patch_{}.pt'.format(patch_size))