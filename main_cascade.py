import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    cascade = cv2.CascadeClassifier('haarcascades_opencv/haarcascade_fullbody.xml')
    handRect = cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=1, minSize=(1,1))

    print(handRect)
    if len(handRect) >= 1:
        for rect in handRect:
            cv2.rectangle(frame,
                      (rect[0],
                       rect[1]),
                      (rect[0]+rect[2],
                       rect[1]+rect[3]),
                      (0, 0, 255),
                      thickness=2)

    cv2.imshow("Image", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


