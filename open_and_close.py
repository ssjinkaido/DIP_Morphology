import cv2
import numpy as np
import matplotlib.pyplot as plt

path = 'assets/balls.jpg'
img = cv2.imread(path)
#img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
img1 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
img2 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
kernel = np.ones((3,3),np.uint8)
erosion = cv2.erode(img1,kernel,iterations=3)
dilate = cv2.dilate(erosion,kernel,iterations=3)

dilate1 = cv2.dilate(img2,kernel,iterations=3)
erosion1 = cv2.erode(dilate1,kernel,iterations=3)

cv2.imshow("gray",img)
cv2.imshow("binary",img1)
cv2.imshow("erosion",erosion)
cv2.imwrite("assets/open.jpg",dilate)
cv2.imshow("dilate",dilate1)
cv2.imwrite("assets/close.jpg",erosion1)

cv2.imshow("erosion1",erosion1)
cv2.imshow("dilate1",dilate1)
cv2.imwrite("assets/close.jpg",dilate)


cv2.waitKey(0)
cv2.destroyAllWindows()
