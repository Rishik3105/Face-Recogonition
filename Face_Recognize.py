import numpy as np
import cv2 as cv
haar_cascade=cv.CascadeClassifier('D:\Python\opeancv\haar_face.xml')
face_recognizer=cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')
people=['Ben Afflek','Elton John','Jerry Seinfield','Madonna','Mindy Kaling']
#img=cv.imread(r'D:\Python\photos&video_opeancv\Train\Ben Afflek\5.jpg')
img_1=cv.VideoCapture(0)
img=img_1.read()
#gray_img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#cv.imshow('gray_image',gray_img)
#DETECT THE FACE IN THE IMAGE
faces_rect=haar_cascade.detectMultiScale(img,1.1,4)
for (x,y,w,h) in faces_rect:
    faces_roi=gray_img[y:y+h,x:x+w]
    label,confidence=face_recognizer.predict(faces_roi)
    print(f'labels = {people[label]} with a confidence of {confidence}')
    cv.putText(img,str(people[label]),(20,20),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),2)
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    cv.imshow('Detected_Face',img)
    cv.waitKey(0)
   