from header import hand_mask
from header import image_enhanchment
from header import alphabet
import numpy as np
import cv2

def segmentasi(img):
    ret2, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
    gray_image = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    th3 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return th3

filename = "train_data/dengan norm-contrast.png"
img = cv2.imread(filename)
# cv2.imshow('withoutNorm', img)
# cv2.moveWindow('withoutNorm', 0, 0)
# cv2.imshow('withoutNormSegmentation', segmentasi(img))
# cv2.moveWindow('withoutSegmentation', 0, 300)

img1 = cv2.GaussianBlur(img, (5, 5), 0)
# cv2.imshow('justBlur', img1)
# cv2.moveWindow('justBlur', 300, 0)
# cv2.imshow('justBlurSegmentation', segmentasi(img1))
# cv2.moveWindow('justBlurSegmentation', 300, 325)

img3 = image_enhanchment.Enhancement.getNormContrast(img)
# cv2.imshow('justNorm', img3)
# cv2.moveWindow('justNorm', 600, 0)
# cv2.imshow('justNormSegmentation', segmentasi(img3))
# cv2.moveWindow('justNormSegmentation', 600, 325)

img2 = image_enhanchment.Enhancement.getNormContrast(img1)
# cv2.imshow('withNorm', img2)
# cv2.moveWindow('withNorm', 900, 0)
# cv2.imshow('withNormSegmentation', segmentasi(img2))
# cv2.moveWindow('withNormSegmentation', 900, 325)

#img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
ret,thresh1 = cv2.threshold(img2,127,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(img2,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(img2,127,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(img2,127,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(img2,127,255,cv2.THRESH_TOZERO_INV)
# cv2.imshow('THRESH_BINARY', thresh1)
# cv2.imshow('THRESH_BINARY_INV', thresh2)
cv2.imshow('THRESH_TRUNC', thresh3)
# cv2.imshow('THRESH_TOZERO', thresh4)
# cv2.imshow('THRESH_TOZERO_INV', thresh5)
gray_image = cv2.cvtColor(thresh3, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray_image', gray_image)

laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=7)
sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=7)

sobel = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
th2 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
th3 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

canny = cv2.Canny(gray_image,100,200)

cv2.imshow('laplacian', laplacian)
cv2.imshow('sobel', sobel)
cv2.imshow('ADAPTIVE_THRESH_MEAN_C', th2)
cv2.imshow('ADAPTIVE_THRESH_GAUSSIAN_C', th3)
cv2.imshow('canny', canny)
cv2.waitKey(0)