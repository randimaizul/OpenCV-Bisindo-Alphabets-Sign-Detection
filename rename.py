import numpy as np
import cv2
j=9
for i in range(67,90):
    if i != 74:
        k = 0
        for j in range(j,j+5):
            filename = "test/" + str(j) + ".jpg"
            img = cv2.imread(filename)

            filename1 = "test/" + chr(i) + "." + str(k) + ".jpg"
            k = k + 1
            cv2.imwrite(filename1, img)
            print(filename)

        j = j+1