import cv2
import numpy as np
from keras.models import load_model
from git import Repo
import webbrowser
import os

PATH_OF_GIT_REPO = r'/home/alfonso/ironhack/final_project/.git'  # make sure .git folder is properly configured
COMMIT_MESSAGE = 'Guardado automático por gestos'
class_names = {0: 'fist', 1: 'ok', 2: 'peace', 3: 'C', 4: 'nothing'}

cont = {"nothing": 0}
spotify = False
play = False
model = load_model('models/model-generator.h5')

def git_push():
    try:

        repo = Repo(PATH_OF_GIT_REPO)  # if repo is CWD just do '.'
        repo.index.add(["main.py"])
        repo.index.commit(COMMIT_MESSAGE)
        origin = repo.remote('origin')
        origin.push()

    except:
        print('Some error occured while pushing the code')

def predict_rgb_image(img):
    prediction = model.predict(img)
    return prediction


def process_image(img):

    connectivity = 4
    flags = connectivity
    flags |= cv2.FLOODFILL_FIXED_RANGE
    tolerancia = 40
    width = img.shape[1]
    height = img.shape[0]


    new_image = cv2.floodFill(img, None, (height//2, width//2), (255, 255, 255), (tolerancia,) * 3, (tolerancia,) * 3, flags)

    new_image=cv2.threshold(new_image[1],254,255,cv2.THRESH_BINARY)

    new_image = cv2.resize(new_image[1][:,:,0], (200,200)) # Reduce image size so training can be faster

    X=[]

    X.append(new_image)

    X = np.array(X, dtype="uint8")
    X = X.reshape(1, 200, 200, 1)
    X=X/255
    return X


cap = cv2.VideoCapture(0)
ret, frame = cap.read()
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame.shape[1],frame.shape[0]))

width = frame.shape[1] // 2
height = frame.shape[0] // 2

while True:

    ret, frame = cap.read()

    tuplas_roi = ((20, 20), (int(.4 * frame.shape[1]), int(0.6 * frame.shape[0]))) # Tupla con los vértices del ROI

    cv2.rectangle(frame, tuplas_roi[0], tuplas_roi[1], (255, 0, 0), 2) # Mostramos el rectángulo del ROI en la pantalla

    crop_img = frame[tuplas_roi[0][1]:tuplas_roi[1][1], tuplas_roi[0][0]:tuplas_roi[1][0]] # Recortamos el contenido del rectángulo para pasarlo por el modelo

    crop_img_sep = crop_img.copy()

    prediction = predict_rgb_image(process_image(crop_img_sep))

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
    if spotify is False:
        if gesto_actual == "ok" and list(cont.values())[0] == 30:
            webbrowser.open('http://www.google.com')

        if gesto_actual == "fist" and list(cont.values())[0] == 30:
            git_push()

        if gesto_actual == "peace" and list(cont.values())[0] == 30:
            spotify = True
            cont = {
                best_prediction: 0
            }
        if gesto_actual == "C" and list(cont.values())[0] == 50:
            break

    else:
        cv2.putText(frame,
        "MODO SPOTIFY ACTIVO",
        (20 , int(frame.shape[0]) -50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
        cv2.LINE_AA)
        if gesto_actual == "peace" and list(cont.values())[0] == 30 and play is False:
            cont = {
                best_prediction: 0
            }
            os.system("sp play")
            play = True
        elif gesto_actual == "peace" and list(cont.values())[0] == 30:
            os.system("sp pause")
            play = False
        elif gesto_actual == "fist" and list(cont.values())[0] == 30:
            os.system("sp next")
        elif gesto_actual == "ok" and list(cont.values())[0] == 30:
            os.system("sp prev")
        elif gesto_actual == "C" and list(cont.values())[0] == 30:
            os.system("sp pause")
            spotify = False

    cv2.putText(frame,
        best_prediction + ' ' + str(round(score, 2)),
        (20, 30 + int(0.6 * frame.shape[0])),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
        cv2.LINE_AA)

    # ret, crop_img_sep= cv2.threshold(crop_img_sep, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    cv2.imshow("Image", frame)
    cv2.imshow("Image mask", crop_img_sep)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
