import cv2
import numpy as np

path = 'alphabet.png'
img = cv2.imread(path)
# img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
img2 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
kernel = np.ones((3, 3), np.uint8)
open=cv2.morphologyEx(img1,cv2.MORPH_OPEN,kernel)
close=cv2.morphologyEx(img2,cv2.MORPH_CLOSE,kernel)
cv2.imshow("gray", img)
cv2.imshow("binary", img1)
cv2.imwrite("assets/open.jpg", open)
cv2.imwrite("assets/close.jpg", close)
cv2.imshow("open", open)
cv2.imshow("close", close)
cv2.waitKey(0)
cv2.destroyAllWindows()
