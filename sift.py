import math
import cv2 as cv
import numpy as np

from PIL import Image, ImageFilter
from torchvision.transforms import Compose, CenterCrop, Resize

IMG_SIZE = 33
SUB_IMG_SIZE = int(math.sqrt(IMG_SIZE ** 2 / 9))   # Size of sub image for dense SIFT.


class Sift:

    def __init__(self):
        # Ensure the SIFT detector creates no more than 9 key points.
        self.sift = cv.xfeatures2d.SIFT_create(9)

    def perform_dense_sift(self, img_array):
        """
        Detect dense SIFT features in image.

        :param img_array: the input image as a numpy array.
        :return: array of 2048 features.
        """

        # Define our key points as the middle points of 9 equal areas in the image.
        key_points = [cv.KeyPoint(x + SUB_IMG_SIZE / 2.0, y + SUB_IMG_SIZE / 2.0, SUB_IMG_SIZE)
                      for y in range(0, img_array.shape[0], SUB_IMG_SIZE)
                      for x in range(0, img_array.shape[1], SUB_IMG_SIZE)]

        # Draw result
        # cv_img = cv.cvtColor(img_array, cv.COLOR_RGBA2BGR)
        # img = cv.drawKeypoints(cv_img, key_points, None, -1, cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv.imshow("Dense key points", img)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        # Compute descriptors from given key points.
        descriptors = self.sift.compute(img_array, key_points)

        # Flatten to single feature container.
        features = descriptors[1].reshape([1, 1152])
        return features

    def perform_sift(self, img_array):
        """
        Detect top 9 SIFT features in image.

        :param img_array: the input image as a numpy array.
        :return: array of 2048 features.
        """

        # Detect top 9 SIFT features
        key_points, descriptors = self.sift.detectAndCompute(img_array, None)

        # Draw result
        # cv_img = cv.cvtColor(img_array, cv.COLOR_RGBA2BGR)
        # img = cv.drawKeypoints(cv_img, key_points, None)
        # cv.imshow("Top key points", img)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        # Flatten to single feature container.
        features = descriptors.reshape([1, 1152])
        return features


