import os
import cv2
import numpy as np
from keras.utils import to_categorical
from imutils import rotate_bound
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

myFolder = "images"
imagepaths = set()

for root, dirs, files in os.walk(myFolder):
    for fileName in files:
        imagepaths.add( os.path.join( root[len(myFolder):], fileName ))

def process_image(path):
    img = cv2.imread(myFolder + path)
    img = cv2.resize(img, (100, 100))
    img = np.array(img)
    return img

X = [] # Image data
y = [] # Labels# Loops through imagepaths to load images and labels into arrays

for path in imagepaths:
    X.append(process_image(path))

    # Processing label in image path
    category = path.split("/")[1]
    label = int(category[0]) # We need to convert 10_down to 00_down, or else it crashes
    y.append(label)


X = np.array(X, dtype = 'int8')

X = np.stack((X,)*3, axis=-1)

y = np.array(y)
y = to_categorical(y)




print("Images loaded: ", len(X))
print("Labels loaded: ", len(y))

X_train, X_aux, y_train, y_aux = train_test_split(X, y, test_size=.3, random_state=42, stratify=y)

X_test, X_val, y_test, y_val = train_test_split(X_aux, y_aux, test_size=.5, random_state=42, stratify=y_aux)

#from keras.models import Sequential
#from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten, Dropout
from keras.models import Model
from keras.applications import VGG16

# Construction of model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(100,100,3))

x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu', name='fc1')(x)
x = Dense(128, activation='relu', name='fc2')(x)
x = Dense(128, activation='relu', name='fc3')(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu', name='fc4')(x)


predictions = Dense(4, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train[:,:,:,0], y_train, epochs=5, batch_size=64, verbose=2, validation_data=(X_val[:,:,:,0], y_val))
