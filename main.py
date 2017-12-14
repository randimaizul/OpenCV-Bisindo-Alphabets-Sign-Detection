from header import hand_mask
from header import image_enhanchment
from header import alphabet
import numpy as np
import cv2

#deklarasi ukuran frame tangan
x0,y0,x1,y1 = 10,300,10,300
mask = hand_mask.HandMask(x0,y0,x1,y1)

#capture video
#cv2.VideoCapture(device), device is id of the opened video capturing device (i.e. a camera index). If there is a single camera connected, just pass 0.
cap = cv2.VideoCapture(0)
#pengambilan gambar
i = 0
while(cap.isOpened()):
    # Capture frame-by-frame (return_value, frame)
    ret, frame = cap.read()
    # flip agar miror(tidak membingungkan)
    cv2.flip(frame, 1, frame)

    # perbaiki contrast dan blur
    frame = image_enhanchment.Enhancement.getNormContrast(frame)
    # mengambil mask frame untuk tangan
    hand = mask.getSource(frame)
    cv2.imshow('mask', hand)
    cv2.moveWindow('mask', 650, 0)
    #membuat kotakan ijo di frame
    cv2.rectangle(frame, (x0,x1),(y0,y1), (0, 255, 0), 3)

    # print alphabet-nya
    alphabet.Alphabet.printAlphabet("A",frame)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    cv2.moveWindow('frame', 0, 0)
    cv2.resizeWindow('frame',640,480)

    interrupt = cv2.waitKey(10)

    # Quit by pressing 'q'
    if interrupt & 0xFF == ord('q'):
        break
    # Capture hand by pressing 'c'
    elif interrupt & 0xFF == ord('p'):
        i += 1
        filename = "train_data/" + str(i) + ".png"
        print(filename)
        cv2.imwrite(filename, hand)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
