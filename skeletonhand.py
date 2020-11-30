import cv2
import numpy as np
import matplotlib.pyplot as plt


def skeleton(img):
    img = cv2.GaussianBlur(img, (3, 3), 0)
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)

    ret, img = cv2.threshold(img, 100, 255, 0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False
    print(size)
    print(img.shape)
    while (True):
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        #the image has completely eroded
        if (cv2.countNonZero(img)) == 0:
            break
    return skel


img = cv2.imread('assets/hand.jpg', 0)
plt.subplot(221)
plt.title('Original image')
plt.imshow(img, cmap='gray')

bl = cv2.GaussianBlur(img, (3, 3), 0)
plt.subplot(222)
plt.title('After Gaussian blur')
plt.imshow(bl, cmap='gray')

skel = skeleton(img)
plt.subplot(223)
plt.imshow(skel, cmap='gray')

skel = skeleton(bl)
plt.subplot(224)
plt.imshow(skel, cmap='gray')

plt.show()
