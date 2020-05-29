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
        filter_size = 3
        padding_width = 2
        base_filters = 32
        total_layers = 3
        input_channels = 3
        dropout_values = [0.1, 0.1, 0.4]
        stride_size = 2
        conv_stride_size = 1

        architecture = []

        # mean = 0.0
        # std = 0.02

        for n in range(total_layers):
            if n == 0:
                sublayer_1 = nn.Conv2d(input_channels, base_filters,
                                       kernel_size=filter_size, stride=conv_stride_size, padding=padding_width)
            else:
                sublayer_1 = nn.Conv2d(base_filters * 2**n, base_filters * 2**n,
                                       kernel_size=filter_size, stride=conv_stride_size, padding=padding_width)

            sublayer_2 = nn.Conv2d(base_filters * 2**n, base_filters * 2**(n + 1),
                               kernel_size=filter_size, stride=conv_stride_size, padding=padding_width)

            # weight_init.normal_(sublayer_1.weight, mean=mean, std=std) # Not sure how they initialize their weights
            # weight_init.normal_(sublayer_2.weight, mean=mean, std=std)

            architecture += [sublayer_1]
            architecture += [sublayer_2]
            architecture += [nn.MaxPool2d(kernel_size=3, stride=(stride_size, stride_size), padding=0)]
            architecture += [nn.Dropout(dropout_values[n])]

        self.model = nn.Sequential(*architecture)

    def forward(self, x):
        """Standard forward."""
        return self.model(x)


def main():
    cnn = Cnn()
    print(cnn)
    path_to_picture = 'test_resources/icon-developer-icon.png'
    input = PIL.Image.open(path_to_picture).convert('RGB')
    input = torchvision.transforms.ToTensor()(input)
    input = input.unsqueeze(0)
    result = cnn.forward(input)
    print(result)
    print(result.shape)


if __name__ == "__main__":
    main()
