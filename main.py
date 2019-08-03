import cv2
import time
from random import randint

threshold = 0.8

class_names = {0: 'Palma',
              1: 'Thumb', 2: 'Victoria', 3: 'Índice', 4: 'Rock&Roll', 5: 'OK'}

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
direccion = randint(0, 3)
width = frame.shape[1] // 2
height = frame.shape[0] // 2

while True:

    '''    img_processed = processImage(frame)
    
    prediction = int(model.predict_classes((np.array(img_processed))))
    
    print(prediction)'''

    if direccion == 0 or direccion == 3:
        width += 5
    else:
        width -= 5
    if direccion == 0 or direccion == 1:
        height += 5
    else:
        height -= 5

    ret, frame = cap.read()
    cv2.circle(frame, (width, height), 10, (255, 255, 255), 16)

    cv2.imshow('object_detection', frame)

    # Rebotes en las paredes
    if height == (frame.shape[0] - 10) and direccion == 0:
        direccion = 3
    elif height == (frame.shape[0] - 10):
        direccion = 2

    if height == (10) and direccion == 3:
        direccion = 0
    elif height == (10):
        direccion = 1

    if width == (frame.shape[1] - 10):
        ret, frame = cap.read()
        cv2.line(frame, (width*2-50, 0), (width*2-10, height*2), (0, 0, 255), 20)
        cv2.imshow('object_detection', frame)
        time.sleep(2)
        direccion = randint(0, 3)
        width = frame.shape[1]  // 2
        height = frame.shape[0]  // 2

    if width == (10):
        ret, frame = cap.read()
        cv2.line(frame, (0, 0), (0, frame.shape[0]), (0, 0, 255), 20)
        cv2.imshow('object_detection', frame)
        time.sleep(2)
        direccion = randint(0, 3)
        width = frame.shape[1]  // 2
        height = frame.shape[0]  // 2

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
