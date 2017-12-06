import numpy as np
import cv2

cap = cv2.VideoCapture(0)
#fgbg = cv2.createBackgroundSubtractorMOG2() #to remove background

while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edge = cv2.Laplacian(blur,cv2.CV_64F)
    #fgmask = fgbg.apply(edge) #remove background but the image will be black and white

    # Display the resulting frame
    cv2.imshow('frame',edge)
    k = cv2.waitKey(10)
    if k == 27:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()