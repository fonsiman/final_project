import numpy as np
import cv2
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)
i=0

while True:

    ret, frame = cap.read()

    tuplas_crop = (int(.6 * frame.shape[1]), 20), (frame.shape[1]-20, int(0.6 * frame.shape[0]))

    #cv2.rectangle(frame, tuplas_crop[0], tuplas_crop[1], (255, 0, 0), 2)

    #reading the image
    image = frame
    #converting image to grayscale format
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #apply thresholding
    ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #get a kernel
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations = 2)
    #extract the background from image
    sure_bg = cv2.dilate(opening,kernel,iterations = 3)

    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret,sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_bg)

    ret,markers = cv2.connectedComponents(sure_fg)

    markers = markers+1

    markers[unknown==255] = 0

    markers = cv2.watershed(image,markers)
    image[markers==-1] = [255,0,0]

    cv2.imshow("Image", sure_fg)


    # cv2.imshow("Image", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
