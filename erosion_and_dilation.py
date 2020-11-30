import cv2
import numpy as np
import matplotlib.pyplot as plt

path = 'assets/coin.jpg'
img = cv2.imread(path)

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (5, 5), 0)
img1 = cv2.threshold(img, 0, 255,
                     cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

kernel = np.ones((3, 3), np.uint8)
erosion = cv2.erode(img1, kernel, iterations=2)
dilation = cv2.dilate(img1, kernel, iterations=3)

cv2.imshow("gray", img)
cv2.imshow("binary", img1)
cv2.imshow("binary2", img1)
cv2.imshow("erosion", erosion)
cv2.imshow("dilate", dilation)
cv2.imwrite("assets/before_erosion.jpg", img1)
cv2.imwrite("assets/erosion.jpg", erosion)
cv2.imwrite("assets/dilation.jpg", dilation)

cv2.waitKey(0)
cv2.destroyAllWindows()
