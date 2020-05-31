import torch.nn as nn
import torch
import torchvision
import numpy as np
import math
from PIL import Image, ImageFilter
from torchvision.transforms import Compose, CenterCrop, Resize
from cnn import Cnn
from sift import Sift

IMG_SIZE = 33
SUB_IMG_SIZE = int(math.sqrt(IMG_SIZE ** 2 / 9))   # Size of sub image for dense SIFT.


class Classifier(nn.Module):
    def __init__(self, num_sift_features, num_classes):
        super(Classifier, self).__init__()

        # define the cnn model:
        self.cnn = Cnn()

        classifier_model = []
        classifier_model += [nn.Linear(192 * 6 * 6 + num_sift_features, num_classes)]
        classifier_model += [nn.Softmax(dim=0)]

        self.classifier = nn.Sequential(*classifier_model)

    def forward(self, cnn_image, sift_image, sift):
        cnn_features = self.cnn(cnn_image)
        cnn_features_flat = torch.flatten(cnn_features)
        dense_sift_features = sift.perform_dense_sift(sift_image)
        dense_sift_features_tensor = torch.flatten(torch.tensor(dense_sift_features))
        features = torch.cat([cnn_features_flat, dense_sift_features_tensor])
        return self.classifier(features)


def preprocess(path_to_image):
    image = Image.open(path_to_image)

    # Process image for cnn
    cnn_image = torchvision.transforms.ToTensor()(image.convert('RGB'))
    cnn_image = cnn_image.unsqueeze(0)

    # Process image for sift
    width, height = image.size
    crop_size = width if width <= height else height

    # Blurring for SIFT.
    sift_image = image.filter(ImageFilter.BLUR)

    # Crop to be square, resize to desired size.
    crop = Compose([
        CenterCrop(crop_size),
        Resize(IMG_SIZE)
    ])

    sift_image = crop(sift_image)
    return cnn_image, np.array(sift_image)


def main():
    cnn_image, sift_image = preprocess('test_resources/iconfinder_Rotation_131696.png')
    sift = Sift()

    classifier = Classifier(1152, 100)
    output = classifier(cnn_image, sift_image, sift)
    print(output)


if __name__ == "__main__":
    main()
