import numpy as np
import cv2

class HandMask:
    def __init__(self,x0,y0,x1,y1):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

    def getSource(self,frame):
        mask = frame[self.x0:self.y0, self.x1:self.y1].copy()
        #mask = cv2.GaussianBlur(mask, (5, 5), 0)
        #mask = cv2.fastNlMeansDenoisingColored(mask, None, 10, 10, 7, 21) #untuk mereduce noise
        #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        #laplacian = cv2.Laplacian(mask, cv2.CV_64F)
        #sobelx = cv2.Sobel(mask, cv2.CV_64F, 1, 0, ksize=7)
        #sobely = cv2.Sobel(mask, cv2.CV_64F, 0, 1, ksize=7)

        #img = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

        return mask