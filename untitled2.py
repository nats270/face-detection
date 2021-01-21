# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 15:57:01 2020

@author: Natasha
"""

import keras
import cv2
import numpy as np
import urllib
from keras.models import load_model

classifier=cv2.CascadeClassifier(r"C:/Users/Natasha/Desktop/project1/proj/haarcascade_frontalface_default.xml")

#URL="http://192.168.43.1:8080/shot.jpg"
model = load_model('Nov_aiml1.h5')

def get_pre_label(pred):
    labels = ['Arya Uncle', 'Ashok Uncle', 'Kaku' ,'NKP', 'Nilu' ,'natasha']
    return labels[pred]
def preprocessing(img):
    img = cv2.resize(img,(100,100))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img.reshape(1,100,100,1)
    return img


ret = True
cap=cv2.VideoCapture(0)
while ret:
    #img_url = urllib.request.urlopen(URL)
    #image = np.array(bytearray(img_url.read()),np.uint8)
    #frame = cv2.imdecode(image,-1)
    _,frame=cap.read()
    
    faces = classifier.detectMultiScale(frame,1.5,5)
    for x,y,w,h in faces:
            face_image = frame[y:y+h,x:x+w].copy()
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
            cv2.putText(frame,get_pre_label
                        (model.predict_classes(preprocessing(face_image))[0]),(x,y),
                        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,2,(0,255,255),2)
    cv2.imshow('capturing',frame)
            
    if cv2.waitKey(1) == 27:
        break
    
cap.release()
cv2.destroyAllWindows()
