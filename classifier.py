import torch.nn as nn
import torch
import torchvision
import numpy as np
import math
from PIL import Image, ImageFilter
from torchvision.transforms import Compose, CenterCrop, Resize, ToPILImage, ToTensor
from cnn import Cnn
from sift import Sift
from torch.autograd import Variable
import time

IMG_SIZE = 33
SUB_IMG_SIZE = int(math.sqrt(IMG_SIZE ** 2 / 9))   # Size of sub image for dense SIFT.


class Classifier(nn.Module):
    def __init__(self, num_sift_features, num_classes):
        super(Classifier, self).__init__()

        # define the cnn model:
        self.cnn = Cnn()

        classifier_model = []
        classifier_model += [nn.Linear(192 * 6 * 6, 3456)]
        classifier_model += [nn.Linear(3456, 864)]
        classifier_model += [nn.Linear(864, 10)]
        #classifier_model += [nn.Softmax(dim=1)]

        self.classifier = nn.Sequential(*classifier_model)

    def forward(self, image, sift, batch_size):
        cnn_images, sift_images = preprocess(image, batch_size)
        cnn_features = self.cnn(cnn_images)
        #print(cnn_features.shape)
        #print(sift_images.shape)
        cnn_features = torch.reshape(cnn_features, (batch_size, 192 * 6 * 6))

        dense_sift_features_tensor = torch.zeros(batch_size, 1152)
        for i, img in enumerate(sift_images):
            img = img.astype(np.uint8)
            #print(img.shape)
            dense_sift_features = sift.perform_dense_sift(img)
            dense_sift_features_tensor[i] = torch.tensor(dense_sift_features)

        #print(dense_sift_features_tensor.shape)
        features = torch.cat([cnn_features, dense_sift_features_tensor.cuda()], dim=1)
        return self.classifier(cnn_features)


def preprocess(images, batch_size):
    # image = Image.open(path_to_image)

    # Process image for cnn
    #cnn_image = torchvision.transforms.ToTensor()(image.convert('RGB'))
    #cnn_image = image.unsqueeze(0)
    cnn_images = images

    # Process image for sift
    transform = Compose([ToPILImage()])
    sift_images = np.zeros((batch_size, 3, 33, 33))
    for i, img in enumerate(images[0]):
        sift_image = transform(img)
        #sift_images = torch.Tensor.cpu(image).detach().numpy()[:,:,:,:]
        #sift_image.show()

        width, height = sift_image.size
        crop_size = width if width <= height else height

        # Blurring for SIFT.
        sift_image = sift_image.filter(ImageFilter.BLUR)

        # Crop to be square, resize to desired size.
        crop = Compose([
            CenterCrop(crop_size),
            Resize(IMG_SIZE)
        ])

        sift_images[i] = crop(sift_image)
    #print(sift_images.shape)
    return Variable(cnn_images.cuda()), sift_images


