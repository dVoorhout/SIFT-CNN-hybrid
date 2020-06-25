import torch.nn as nn
import torch
import torchvision
import numpy as np
import math
from PIL import Image, ImageFilter
from torchvision.transforms import Compose, CenterCrop, Resize, ToPILImage, ToTensor
from cnn import Cnn
from sift import Sift
import torch.autograd as autograd
from torch.autograd import Variable
import time

IMG_SIZE = 33
SUB_IMG_SIZE = int(math.sqrt(IMG_SIZE ** 2 / 9))   # Size of sub image for dense SIFT.


class Classifier(nn.Module):
    def __init__(self, num_sift_features, num_classes, sift):
        super(Classifier, self).__init__()

        # define the cnn model:
        self.cnn = Cnn()
        self.sift = sift

        classifier_model = []
        classifier_model += [nn.Linear(192 * 5 * 5 + 1152, 2976)]
        classifier_model += [nn.Dropout(0.2)]
        classifier_model += [nn.ReLU()]
        classifier_model += [nn.Linear(2976, 1000)]
        classifier_model += [nn.Dropout(0.2)]
        classifier_model += [nn.ReLU()]
        classifier_model += [nn.Linear(1000, 10)]
        # classifier_model += [nn.Dropout(0.2)]
        # classifier_model += [nn.ReLU()]
        # classifier_model += [nn.Linear(864, 432)]
        # classifier_model += [nn.Dropout(0.2)]
        # classifier_model += [nn.ReLU()]
        # classifier_model += [nn.Linear(432, 10)]

        # classifier_model += [nn.Linear(192 * 6 * 6 + 1152, 192 * 6 * 6 + 1152)]
        # classifier_model += [nn.ReLU()]
        # classifier_model += [nn.Dropout(0.2)]
        # classifier_model += [nn.Linear(192 * 6 * 6 + 1152, 1000)]
        # classifier_model += [nn.ReLU()]
        # classifier_model += [nn.Dropout(0.2)]
        # classifier_model += [nn.Linear(1000, 10)]
        #classifier_model += [nn.Softmax(dim=1)]

        self.classifier = nn.Sequential(*classifier_model)

    def forward(self, image):
        cnn_images, sift_images = preprocess(image, image.shape[0])

        # Get CNN feature maps
        cnn_features = self.cnn(cnn_images)

        # Reshape CNN features for the Dense layer
        cnn_features = torch.reshape(cnn_features, (image.shape[0], 192 * 5 * 5))

        # Sequentially calculate SIFT tensor
        dense_sift_features_tensor = torch.zeros(image.shape[0], 1152)
        for i, img in enumerate(sift_images):
            img = img.astype(np.uint8)
            dense_sift_features = self.sift.perform_dense_sift(img)
            dense_sift_features_tensor[i] = torch.tensor(dense_sift_features)

        # Concatenate CNN and SIFT features
        features = torch.cat([cnn_features, dense_sift_features_tensor.cuda()], dim=1)

        # Create classification of all features
        return self.classifier(features)


def preprocess(images, batch_size):
    # images are already tensors so we just use that for the CNN
    cnn_images = images

    # Process images for sift
    transform = Compose([ToPILImage()])
    sift_images = np.zeros((batch_size, 3, 33, 33))

    # We sequentially process each image in the batch to prepare it for SIFT feature extraction
    for i, img in enumerate(images[0]):
        sift_image = transform(img)

        width, height = sift_image.size
        crop_size = width if width <= height else height

        # Blurring for SIFT.
        sift_image = sift_image.filter(ImageFilter.BLUR)

        # Crop to be square, resize to desired size.
        crop = Compose([
            CenterCrop(crop_size),
            Resize(IMG_SIZE)
        ])

        # Add the processed image to our numpy array
        sift_images[i] = crop(sift_image)

    # We return the Cuda Variable version of the initial tensor for the cnn as we couldn't do that before; we had
    # to first process it for SIFT
    return Variable(cnn_images.cuda()), sift_images


