import numpy as np
import cv2
from header import image_enhanchment
from matplotlib import pyplot as plt
import string

alphabet  = list(string.ascii_uppercase)

#looping alphabet
#for j in range(0,25):  #B original cuman 10 data, dan lompat pada looping J (tidak ada data original)
for j in range(10,25):

    s = alphabet[j]
    #looping data train 10 kali
    for i in range(1,26):
        filename = "train_data/_original_/rya/" + str(s) + "/" + str(i) + ".png"
        img = cv2.imread(filename)
        #cv2.imshow('origin',img)

        ret2,thresh = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
        #cv2.imshow("threshold",thresh)
        gray_image = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

        #th2 = cv2.adaptiveThreshold(gray_image,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
        th3 = cv2.adaptiveThreshold(gray_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

        #MORFOLOGI
        kernel = np.ones((3,3),np.uint8)
        morfo_open = cv2.morphologyEx(th3,cv2.MORPH_OPEN, kernel)
        morfo_close = cv2.morphologyEx(th3,cv2.MORPH_CLOSE, kernel)    

        #file segmentasi
        filename_segmentasi = "train_data_morfologi/" + str(s) + "/segmentasi/" + str(i) + ".png"
        cv2.imwrite(filename_segmentasi, th3)
        #file opening
        filename_opening = "train_data_morfologi/" + str(s) + "/opening/" + str(i) + ".png"
        cv2.imwrite(filename_opening, morfo_open)
        #file closing
        filename_close = "train_data_morfologi/" + str(s) + "/closing/" + str(i) + ".png"
        cv2.imwrite(filename_close, morfo_close)
        print(i)

        cv2.waitKey(0)
        cv2.destroyAllWindows()