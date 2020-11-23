import cv2
from numpy import ones, uint8, array
import matplotlib.pyplot as plt

kernel_list = [
    ones((3, 3), uint8),
    ones((5, 5), uint8),
    array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], uint8),
    array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], uint8),
    array([[0, 1, 0], [1, -1, 1], [0, 1, 0]], uint8),
    array([[0, -1, -1], [1, 1, -1], [0, 1, 0]], uint8),
    array([[-1, -1, 0], [-1, 1, 0], [-1, -1, 0]], uint8),
    array([[0, 1, 0], [0, 1, 1], [0, 0, -1]], uint8),
    array([[-1, 1, -1], [1, 1, 0], [-1, 0, 0]], uint8)
]


def to_binary(image, thr):
    """Convert a grayscale image to a binary image."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.blur(image, (3, 3))
    return cv2.threshold(image, thr, 255, cv2.THRESH_BINARY_INV)[1]


def hit_miss(image, kernel):
    """Apply hit-or-miss transform on a binary image."""
    image = cv2.morphologyEx(image, cv2.MORPH_HITMISS, kernel)
    return image


if __name__ == "__main__":
    img = cv2.imread('captcha.png')
    cv2.imshow('img',img)
    img = to_binary(img, 80)
    plt.subplot(5, 3, 2)
    plt.imshow(img, cmap='gray')
    i = 4

    for kernel in kernel_list:
        res = hit_miss(img, kernel)
        plt.subplot(5, 3, i)
        i += 1
        plt.imshow(res, cmap='gray')
    plt.show()
