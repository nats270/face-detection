# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 15:24:31 2020

@author: Natasha
"""


import os
import cv2
import numpy as np
import pickle

data_dr=os.path.join(os.getcwd(),'pickle_images_labels')

img_dr=os.path.join(os.getcwd(),'images')

def preprocessing(image):
    image=cv2.resize(image,(100,100))
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    return image

images=[] #--> all pickle images
labels=[] #--> all names for images

for i in os.listdir(img_dr):
    image=cv2.imread(os.path.join(img_dr,i))
    image=preprocessing(image)
    images.append(image)
    labels.append(i.split('_')[0])
    
images=np.array(images)
labels=np.array(labels)

with open(os.path.join(data_dr,'images.p'),'wb') as f:
    pickle.dump(images,f)
    
with open(os.path.join(data_dr,'labels.p'),'wb') as f:
    pickle.dump(labels,f)
    

