import numpy as np
import cv2

class HandMask:
    def __init__(self,x0,y0,x1,y1):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

    def getSource(self,frame):
        mask = frame[self.x0:self.y0, self.x1:self.y1]
        cv2.imshow('mask', mask)
        cv2.moveWindow('mask', 650, 0)