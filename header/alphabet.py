import numpy as np
import cv2

class Alphabet:
    def printAlphabet(val,img):
        cv2.putText(img, val, (400, 250), cv2.FONT_HERSHEY_SIMPLEX, 8, (0, 255, 0), 2, cv2.LINE_AA)