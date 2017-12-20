import numpy as np
import cv2
from header import image_enhanchment
from matplotlib import pyplot as plt
s = "C"
for i in range(1,11):
    filename = "train_data/_original_/rya/" + str(s) + "/" + str(i) + ".png"
    img = cv2.imread(filename)
    #cv2.imshow('origin',img)

    ret2,thresh = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
    #cv2.imshow("threshold",thresh)
    gray_image = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

    th2 = cv2.adaptiveThreshold(gray_image,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
    th3 = cv2.adaptiveThreshold(gray_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    #cv2.medianBlur(th3,3,th3)
    #cv2.imshow("th2",th2)
    #cv2.imshow("th3",th3)

    #filename2 = "train_data/Nth2.png"
    filename1 = "train_data/" + str(s) + "/" + str(i) + ".png"
    #cv2.imwrite(filename2, th2)
    cv2.imwrite(filename1, th3)
    print(i)

    cv2.waitKey(0)
    cv2.destroyAllWindows()