# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 14:51:40 2020

@author: Ucif
"""

import face_recognition
import cv2
import os
import pickle

dataset =r"E:\P_PROJECTS\PI_CAM_RECOG\dataset"

#list the sub folders within the dataset which is named after the individual whose images it
#contains
path = os.listdir(dataset)

#create a dictionary of name and the corresponding images of the individual to be recognised

imageList = {name:os.listdir(os.path.join(dataset,name)) for name in path}

#create a list to hold the names of people
nameList = []
#create a dictionary to hold the name of people and  a list of encoded images
nameEncoding = {}

#loop over the names in the dataset 
for name in imageList:
    #create a list to hold the encoded images 
    encodingList = []
    #get the path to all the images for individual names
    print("[INFO] encoding images of {} ".format(name))
    for img in imageList[name]:
        imgPath = os.path.join(dataset,name,img)
        image = cv2.imread(imgPath)
        #convert the bgr colour to rgb required by dlib
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #form the bounding box for each face in the image        
        boxes = face_recognition.face_locations(rgb, model="hog")
        #create the encoding for the images in the bounding boxes
         
        encodings = face_recognition.face_encodings(rgb, boxes)
        #add the encoding of faces in a list 
        encodingList.append(encodings)
        
    #store the name as key and a list of encoded faces for the person as value for all persons    
    nameEncoding[name] = encodingList   
    
    
#print(nameEncoding)

pfile = open("face_encodings.pickle", "wb")
pickle.dump(nameEncoding,pfile)
pfile.close()


#print(imageList)
