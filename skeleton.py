import cv2
import numpy as np
import matplotlib.pyplot as plt


def skeleton(img):
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)

    ret, img = cv2.threshold(img, 120, 255, 0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while (not done):
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True
    return skel


img = cv2.imread('assets/fingerprint.jpg', 0)
plt.subplot(221)
plt.title('Original image')
plt.imshow(img, cmap='gray')

bl = cv2.GaussianBlur(img, (3,3),0)
plt.subplot(222)
plt.title('Blurred image')
plt.imshow(bl, cmap='gray')

skel = skeleton(img)
plt.subplot(223)
plt.imshow(skel, cmap='gray')

skel = skeleton(bl)
plt.subplot(224)
plt.imshow(skel, cmap='gray')

plt.show()
