# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 23:10:43 2020

@author: Ucif
"""

import face_recognition
import cv2
import pickle


image = r"E:\P_PROJECTS\BING_IMAGE_API\DATASET\barack obama\00000004.jpg"
#load the dictionary of encoded images of people in the dataset by running my_encode.py
encoding = pickle.load(open("face_encodings.pickle", "rb"))

#read image 
img = cv2.imread(image)
#convert the image from brg to rgb
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#obtain the cordinates of the bounding box arond the face
boxes = face_recognition.face_locations(rgb, model="hog")
#create the embeddings for the image in the boundng box
encoded = face_recognition.face_encodings(rgb, boxes)
#create an empty dictionary to hold the count for compared names
counts = {}
#create an empty list to hold the names of people detected in a single image
detected =[]

#loop over the encoded images in case it contains multiple faces
print("[INFO] finding matches for detected images")
for face in encoded:
    #loop over the dictionary of encoded images and compare the image to be recognised
    for name in encoding:
        #compare the encoded image to all the images of specific peoples in the dictionary
        matches = [face_recognition.compare_faces(x,face) for x in encoding[name]]
        #count the total number of matches for each person 
        count = len([x for x in matches if x[0]])
        #insert the name of person in dataset as key and count of matches as values in 
        #dictionary counts
        counts[name] = count
        
    #find the name of the person with maximum matches        
    match = max(counts, key=counts.get)
    match = "unknown" if max(counts.values())<=0 else match
    #add the name of matched person to the list of detected people in an image and "unknown" if
    #not found
    
   
    detected.append(match) 
    print("[INFO]  {} detected in the image".format(match))
 
for (top, right, bottom, left), named in zip(boxes, detected):
	#draw a rectange on the predicted face name on the image
    #make sure to use the image obtained by cv.imread
    cv2.rectangle(img, (left, top), (right, bottom), (10, 10, 255), 2)
    text_top = bottom + 15
    #write the name of the detected person on the image
    cv2.putText(img, named, (left, text_top), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 220, 50), 2)
 # display the image with detected faces and their names
cv2.imshow("face detector", img)
k = cv2.waitKey(0)
#keeps the display window until ESC os spacebar is pressed
while( k==32 or 27):
    cv2.destroyAllWindows()
    break

       