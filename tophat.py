import cv2
import numpy as np
from skimage.morphology import disk
# Getting the kernel to be used in Top-Hat
kernel = np.ones((3,3),np.uint8)
# Reading the image named 'input.jpg'
input_image = cv2.imread("assets/rice.jpg")
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
input_image1=input_image
# Applying the Top-Hat operation
output= cv2.adaptiveThreshold (input_image, 255.0, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, -20.0)
cv2.imshow("Adaptive Thresholding", output)
kernel=disk(5)
# filterSize =(11, 11)
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
#                                    filterSize)

tophatt_img = cv2.morphologyEx(input_image,
                              cv2.MORPH_TOPHAT,
                              kernel)
bottomhatt_img = cv2.morphologyEx(input_image,
                              cv2.MORPH_BLACKHAT,
                              kernel)
final=input_image+tophatt_img-bottomhatt_img
cv2.imshow("original", input_image)
cv2.imshow("tophatt", tophatt_img)
cv2.imshow("bottom_hatt",bottomhatt_img)
cv2.imshow("final",final)
cv2.imwrite("assets/original_rice.jpg", input_image)
cv2.imwrite("assets/adaptive_threshold_rice.jpg",output)
cv2.imwrite("assets/tophat_rice.jpg",tophatt_img)
cv2.imwrite("assets/bottomhat_rice.jpg",bottomhatt_img)
cv2.imwrite("assets/final_rice.jpg",final)


cv2.waitKey(0)
cv2.destroyAllWindows()
