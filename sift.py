import cv2 as cv


# Wrapper class for using the OpenCV SIFT operation
class Sift:

    def __init__(self):
        self.sift = cv.xfeatures2d.SIFT_create()

    def detect_features(self, img_name):
        """
        Detect SIFT features in image after blurring and converting to grayscale.

        :param img_name: the input image name in single quotes.
        :return: the image with SIFT features drawn on.
        """
        img = cv.imread(img_name, cv.IMREAD_GRAYSCALE)
        med = cv.medianBlur(img, 5)

        key_points, descriptors = self.sift.detectAndCompute(med, None)
        img = cv.drawKeypoints(med, key_points, None)

        cv.imshow("Image", img)
        cv.waitKey(0)
        cv.destroyAllWindows()

        return img


def main():
    sift = Sift()
    sift.detect_features('disparity.png')


if __name__ == "__main__":
    main()
