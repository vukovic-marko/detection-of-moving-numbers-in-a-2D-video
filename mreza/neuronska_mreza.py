import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout
from pathlib import Path
import cv2
import numpy as np

model = None
saved_model = Path('mreza/model.h5')

def kreiraj_mrezu(model):
    image_size = 784 # zbog mnist skupa podataka, koristi se 28*28 = 784 ulaza za NM
    num_classes = 10 # svaka od mogucih vrednosti: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9

    model = Sequential()

    model.add(Dense(512, activation='relu', input_shape=(image_size,)))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()

    return model

def obuci_mrezu(model):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_temp_train = []
    for image in x_train:
        x_temp_train.append(prilagodi_sliku(image))            
    x_train = np.array(x_temp_train, ndmin=1)

    x_temp_test = []
    for image in x_test:
        x_temp_test.append(prilagodi_sliku(image))            
    x_test = np.array(x_temp_test, ndmin=1)
    
    x_train = x_train.reshape((x_train.shape[0], 28 * 28))
    x_train = x_train.astype('float32') / 255
    x_test = x_test.reshape((x_test.shape[0], 28 * 28))
    x_test = x_test.astype('float32') / 255

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    model.fit(x_train, y_train,
          batch_size=128,
          epochs=10,
          verbose=1,
          validation_data=(x_test, y_test))

    loss, accuracy  = model.evaluate(x_test, y_test, verbose=False)

    print(f'Test loss: {loss:.3}')
    print(f'Test accuracy: {accuracy:.3}')

    model.save('mreza/model.h5')

    return model

def predict(img):
    return model.predict(img)

def prilagodi_sliku(image):
    mask = image != 0
    grid_mask = np.ix_(mask.any(1), mask.any(0))
    image = image[grid_mask]
    image = cv2.resize(image,(28,28))  
    return image

if saved_model.is_file() is False:
    model = kreiraj_mrezu(model)
    obuci_mrezu(model)
model = load_model('mreza/model.h5')