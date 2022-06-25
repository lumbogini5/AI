from pyexpat import model
import matplotlib.pyplot as plt
import numpy as np
import cv2
import cv2 as cv
import time
from datetime import datetime
import sys

from cProfile import label
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Activation, Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.optimizers import Adam
from keras.metrics import categorical_crossentropy

from tensorflow.keras.utils import load_img, img_to_array


classer = {'phat':0, 'quyen':1}
model = load_model('nhandangmat2.h5')

cap = cv2.VideoCapture(0)

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%d/%m/%Y, %H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

def main():
    while True:
        success, image = cap.read()
        cv2.imshow("image",image)
        cv2.waitKey(1)
        cv2.imwrite('example.png', image)
        cv2.waitKey(1)
        picture = load_img('example.png', target_size =(224,224))
        plt.imshow(picture)
        picture = img_to_array(picture)
        picture =picture.reshape(1,224,224,3)
        picture = picture.astype('float32')
        picture = picture/255
        result = np.argmax(model.predict(picture),axis= 1)
        name = [k for k, v in classer.items() if v == result]

        if result <0.5:
            name= 'unknow'

        print( 'Name is:', name)
        markAttendance(name)

           
#face_cascade = 'haarcascade_frontalface_default.xml'    

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

frame = cv2.VideoCapture(0)

while True:
    
    ret, capture = frame.read()
    gray = cv2.cvtColor(capture, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale( gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE) #cv2.cv.CV_HAAR_SCALE_IMAGE

    for (x, y, w, h) in faces:
        cv2.rectangle(capture, (x, y), (x+w, y+h), (0, 255, 0), 2)
        main()

    #cv2.imshow('Video', capture)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
