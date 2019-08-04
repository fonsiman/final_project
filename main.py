import cv2
import numpy as np
import time
from random import randint
from keras.models import load_model

def pinball():
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
        cv2.circle(frame, (width, height), 10, (0, 0, 255), 16)

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

threshold = 0.8

class_names = {0: 'down', 1: 'palm', 2: 'L', 3: 'fist', 4: 'fist_moved', 5: 'thumb', 6: "index",
               7: "ok", 8: "palm_moved", 9: "c"}

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
direccion = randint(0, 3)
width = frame.shape[1] // 2
height = frame.shape[0] // 2

model = load_model('models/second_model.h5')
#model = cv2.dnn.readNetFromTensorflow('models/second_model.h5')

def predict_rgb_image(img):
    return model.predict(img)

def process_img(img):
    X = []
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Converts into the corret colorspace (GRAY)
    img = cv2.resize(img, (320, 120)) # Reduce image size so training can be faster
    X.append(img)

    X = np.array(X, dtype="uint8")
    X = X.reshape(1, 120, 320, 1)

    return X

while True:

    ret, frame = cap.read()

    predictions = predict_rgb_image(process_img(frame))

    best_pred = np.argmax(predictions, axis=1)

    if (predictions[0][best_pred] > .8):
        print(predictions[0][best_pred], class_names[best_pred[0]])
    cv2.imshow('object_detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
