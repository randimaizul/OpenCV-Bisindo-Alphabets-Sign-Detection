import cv2
import numpy as np
from matplotlib import pyplot as plt
from header import image_enhanchment

img = cv2.imread("train_data/1.png")
# cv2.imshow('origin',img)

#1 treshold
ret2,thresh = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
# cv2.imshow("threshold",thresh)

#2 gray
gray_image = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
# cv2.imshow("threshold2",gray_image)

#3 blur
#blur = cv2.GaussianBlur(gray_image,(11,11),0)
#cv2.imshow("threshold2",blur)

#4 adaptiveThreshold Mean
th2 = cv2.adaptiveThreshold(gray_image,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
# cv2.imshow("th2_ADAPTIVE_THRESH_MEAN_C",th2)
cv2.imwrite("train_data/ADAPTIVE_THRESH_MEAN_C.png", th2)

#5 adaptiveThreshold GAUSSIAN
# th3 = cv2.adaptiveThreshold(gray_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
# cv2.imshow("th3_ADAPTIVE_THRESH_GAUSSIAN_C",th3)
# cv2.imwrite("train_data/THRESH_GAUSSIAN.png", th3)

kernel = np.ones((3,3),np.uint8)
#dilasi = cv2.erode(img,kernel,iterations = 1)
#erosi = cv2.morphologyEx(thresh,cv2.MORPH_OPEN, kernel)

#6 Dilasi
# dilasi = cv2.dilate(th2,kernel,iterations = 1)
# cv2.imshow("Dilasi",dilasi)
# cv2.imwrite("train_data/dilasi_3.png", dilasi)

#7 erosi
erosi = cv2.erode(th2,kernel,iterations = 1)
# cv2.imshow("Erosi",erosi)
cv2.imwrite("train_data/erosi_3.png", erosi)

# ================ MORFOLOGI =============================
#6 Closing
# erosi_close = cv2.morphologyEx(th3,cv2.MORPH_CLOSE, kernel)
# cv2.imshow("Morfo Closing",erosi_close)
# cv2.imwrite("train_data/morfo_closing_3.png", erosi_close)

#7 opening
# erosi_open = cv2.morphologyEx(th3,cv2.MORPH_OPEN, kernel)
# cv2.imshow("Morfo Open",erosi_open)
# cv2.imwrite("train_data/morfo_opening_3.png", erosi_open)
# ================ MORFOLOGI =============================
 
 

imagem = cv2.bitwise_not(th2)
    
cv2.imshow("negative",imagem)
 
 
plt.subplot(131),plt.imshow(img,cmap = 'gray')
plt.title('Citra Awal'), plt.xticks([]), plt.yticks([])

plt.subplot(132),plt.imshow(th2,cmap = 'gray')
plt.title('ADAPTIVE_GAUSSIAN'), plt.xticks([]), plt.yticks([])

plt.subplot(133),plt.imshow(erosi,cmap = 'gray')
plt.title('Erosi'), plt.xticks([]), plt.yticks([])



plt.show()

