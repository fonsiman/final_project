import cv2
import numpy as np
import time
from random import randint
from keras.models import load_model
from git import Repo
import webbrowser

PATH_OF_GIT_REPO = r'/home/alfonso/ironhack/final_project/.git'  # make sure .git folder is properly configured
COMMIT_MESSAGE = 'Guardado automático por gestos'

def git_push():
    try:

        repo = Repo(PATH_OF_GIT_REPO)  # if repo is CWD just do '.'
        repo.index.add(["main.py"])
        repo.index.commit(COMMIT_MESSAGE)
        origin = repo.remote('origin')
        origin.push()

    except:
        print('Some error occured while pushing the code')

threshold = 0.8

class_names = {0: 'fist', 1: 'ok', 2: 'peace', 3: 'C', 4: 'nothing'}

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
direccion = randint(0, 3)
width = frame.shape[1] // 2
height = frame.shape[0] // 2

model = load_model('models/model5000.h5')
#model = cv2.dnn.readNetFromTensorflow('models/second_model.h5')

cont = {"nothing": 0}

def predict_rgb_image(img):
    prediction = model.predict(img)
    return prediction

def process_img(img):
    X = []
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Converts into the corret colorspace (GRAY)
    img = cv2.resize(img, (320, 120)) # Reduce image size so training can be faster
    X.append(img)

    X = np.array(X, dtype="uint8")
    X = X.reshape(1, 120, 320, 1)

    return X

def process_image(img):



    connectivity = 4
    flags = connectivity
    flags |= cv2.FLOODFILL_FIXED_RANGE
    tolerancia = 80
    width = img.shape[1]
    height = img.shape[0]


    new_image = cv2.floodFill(img, None, (height//2, width//2), (255), (tolerancia,) * 3, (tolerancia,) * 3, flags)

    new_image=cv2.threshold(new_image[1],254,255,cv2.THRESH_BINARY)

    new_image = cv2.resize(new_image[1][:,:,0], (200,200)) # Reduce image size so training can be faster

    '''writeStatus =  cv2.imwrite( "images/Camera_Image.jpg", new_image )
    if writeStatus is True:
        print("Guardado ok: ")
    else:
        print("Problema con ok")'''

    X=[]

    X.append(new_image)

    X = np.array(X, dtype="uint8")
    X = X.reshape(1, 200, 200, 1)
    X=X/255
    return X

while True:

    ret, frame = cap.read()

    tuplas_roi = ((20, 20), (int(.4 * frame.shape[1]), int(0.6 * frame.shape[0]))) # Tupla con los vértices del ROI

    cv2.rectangle(frame, tuplas_roi[0], tuplas_roi[1], (255, 0, 0), 2) # Mostramos el rectángulo del ROI en la pantalla

    crop_img = frame[tuplas_roi[0][1]:tuplas_roi[1][1], tuplas_roi[0][0]:tuplas_roi[1][0]] # Recortamos el contenido del rectángulo para pasarlo por el modelo

    prediction = predict_rgb_image(process_image(crop_img))

    best_prediction = class_names[np.argmax(prediction)]
    score = np.amax(prediction)

    gesto_actual = list(cont.keys())[0]

    if gesto_actual == best_prediction:
        cont[gesto_actual] += 1
    else:
        cont = {
            best_prediction: 0
        }
    print(cont)
    if gesto_actual == "ok" and list(cont.values())[0] == 30:
        webbrowser.open('http://www.google.com')

    if gesto_actual == "fist" and list(cont.values())[0] == 30:
        git_push()

    cv2.putText(frame,
        best_prediction + ' ' + str(round(score, 2)),
        (20, 30 + int(0.6 * frame.shape[0])),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
        cv2.LINE_AA)

    cv2.imshow("Image", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
