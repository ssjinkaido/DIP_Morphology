import cv2
import numpy as np
import matplotlib.pyplot as plt

path = 'number.jpg'
img = cv2.imread(path)
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
ret,img1 = cv2.threshold(img, 135, 255, cv2.THRESH_BINARY)

kernel = np.ones((3,3),np.uint8)
erosion = cv2.erode(img1,kernel,iterations=3)
dilate = cv2.dilate(img1,kernel,iterations=1)

erosion1 = cv2.erode(img,kernel,iterations=3)
dilate1 = cv2.dilate(img,kernel,iterations=1)

cv2.imshow("gray",img)
cv2.imshow("binary",img1)
cv2.imshow("erosion",erosion)
cv2.imshow("dilate",dilate)

cv2.imshow("erosion1",erosion1)
cv2.imshow("dilate1",dilate1)

cv2.waitKey(0)
cv2.destroyAllWindows()
