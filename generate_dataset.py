import cv2
import numpy as np

cap = cv2.VideoCapture(0)
i=0

while True:

    ret, frame = cap.read()

    tuplas_crop = (int(.6 * frame.shape[1]), 20), (frame.shape[1]-20, int(0.6 * frame.shape[0]))

    cv2.rectangle(frame, tuplas_crop[0], tuplas_crop[1], (255, 0, 0), 2)

    if cv2.waitKey(1) & 0xFF == ord('t'):

        crop_img = frame[tuplas_crop[0][1]:tuplas_crop[1][1], tuplas_crop[0][0]:tuplas_crop[1][0]]
        gray_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        YCrCb_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2YCrCb)

        writeStatus =  cv2.imwrite( "images/gray/Gray_Image"+str(i)+".jpg", gray_crop )
        if writeStatus is True:
            print("Guardado gray: " + str(i))
        else:
            print("Problema con gray")


        writeStatus2 =  cv2.imwrite( "images/YCrCb/YCrCb_Image"+str(i)+".jpg", YCrCb_crop )
        if writeStatus2 is True:
            print("Guardado YCrCb: " + str(i))
        else:
            print("Problema con YCrCb")

        i+=1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.imshow("Image", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
