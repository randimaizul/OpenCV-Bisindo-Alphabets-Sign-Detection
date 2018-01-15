import numpy as np
import cv2

s = "A"
for i in range(1,21):
    filename = "train_data/" + str(s) + "/" + str(i) + ".png"
    img = cv2.imread(filename)

    ret2,thresh = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
    gray_image = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    th3 = cv2.adaptiveThreshold(gray_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

    kernel = np.ones((5, 5), np.uint8)
    morp = cv2.morphologyEx(th3, cv2.MORPH_OPEN, kernel)

    filename1 = "dataset/" + str(s) + "." + str(i - 1) + ".jpg"
    cv2.imwrite(filename1, img)
    print(filename1)