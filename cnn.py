import torch.nn as nn
import os
from typing import Dict, Tuple

import PIL
import torchvision
from PIL.Image import Image
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision
import random
# import torch.nn.init as weight_init


# Cnn class that represents the cnn part of the network.
# Code is based on https://arxiv.org/ftp/arxiv/papers/1608/1608.02833.pdf
class Cnn(nn.Module):
    def __init__(self):
        super().__init__()
        base_filters = 96
        input_channels = 3

        architecture = []

        # mean = 0.0
        # std = 0.02
        architecture += [nn.Dropout(p=0.2, inplace=False)]
        architecture += [nn.Conv2d(input_channels, base_filters,
                               kernel_size=5, stride=1, padding=2)]
        architecture += [nn.ReLU()]

        architecture += [nn.Conv2d(base_filters, base_filters,
                               kernel_size=3, stride=2, padding=0)]
        architecture += [nn.ReLU()]
        architecture += [nn.Dropout(p=0.5, inplace=False)]

        architecture += [nn.Conv2d(base_filters, base_filters * 2,
                           kernel_size=5, stride=1, padding=1)]
        architecture += [nn.ReLU()]

        architecture += [nn.Conv2d(base_filters * 2, base_filters * 2,
                                  kernel_size=3, stride=2, padding=0)]
        architecture += [nn.ReLU()]
        architecture += [nn.Dropout(p=0.5, inplace=False)]

        architecture += [nn.Conv2d(base_filters * 2, base_filters * 2,
                                  kernel_size=3, stride=1, padding=1)]
        architecture += [nn.ReLU()]

        architecture += [nn.Conv2d(base_filters * 2, base_filters * 2,
                                  kernel_size=1, stride=1, padding=0)]
        architecture += [nn.ReLU()]


        # weight_init.normal_(sublayer_1.weight, mean=mean, std=std) # Not sure how they initialize their weights
        # weight_init.normal_(sublayer_2.weight, mean=mean, std=std)

        self.model = nn.Sequential(*architecture)

    def forward(self, x):
        """Standard forward."""
        return self.model(x)

