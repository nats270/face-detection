# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 15:56:04 2020

@author: Natasha
"""



import cv2
import urllib
import numpy as np

classifier=cv2.CascadeClassifier(r"C:/Users/Natasha/Desktop/project1/proj/haarcascade_frontalface_default.xml")

url="http://192.168.43.1:8080/shot.jpg"

data=[]

while len(data)<100:
    print("connecting to url")
    image_from_url=urllib.request.urlopen(url)
    frame=np.array(bytearray(image_from_url.read()),np.uint8)
    frame=cv2.imdecode(frame,-1)
    face_frame=frame.copy()
    
    faces=classifier.detectMultiScale(frame,1.3,5)
    if len(faces)>0:
        for x,y,w,h in faces:
            face_frame=frame[y:y+h,x:x+w].copy()
            cv2.imshow("Face",face_frame)
            if len(data)<=100:
                print(len(data)+1,"/100")
                data.append(face_frame)
            else:
                break
    cv2.imshow("preview window",frame)
    key=cv2.waitKey(25)
    
    if key==27:
        break
    
#store data
if len(data)==100:
    name=input("Enter user name:")
    for i in range(100):
        cv2.imwrite("images/"+name+"_"+str(i+1)+'.jpg',data[i])
    print("completed")
else:
    print("insufficient data")
    
    

