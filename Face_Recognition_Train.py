import os
import cv2 as cv
import numpy as np
# ONE WAY TO TAKE TRAIN FOLDERS 
people=['Ben Afflek','Elton John','Jerry Seinfield','Madonna','Mindy Kaling']
#OR
# ANOTHER WAY TO TAKE TRAIN FLDERS
#p=[]#creating an empty lsit
#for i in os.listdir(r'D:\Python\photos&video_opeancv\Train'): #Looping over the folder 
 #  p.append(i)
#print(i)
DIR=r'D:\Python\photos&video_opeancv\Train'
haar_cascade=cv.CascadeClassifier('D:\Python\opeancv\haar_face.xml')
features=[]
labels=[]
def create_train():
    for person in people:
        path=os.path.join(DIR,person)
        label=people.index(person)
        for img in os.listdir(path):
            img_path=os.path.join(path,img)
            img_array=cv.imread(img_path)
            gray_img=cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)
            faces_rect=haar_cascade.detectMultiScale(gray_img,scaleFactor=1.1,minNeighbors=4)
            for (x,y,w,h) in faces_rect:
                faces_roi=gray_img[y:y+h,x:x+w]
                features.append(faces_roi)
                labels.append(label)
create_train()
print(f'length of features = {len(features)}')
print(f'length of labels = {len(labels)}')
features=np.array(features,dtype='object')
labels=np.array(labels)
face_recognizer=cv.face.LBPHFaceRecognizer_create()
#Training the recognizer on the features and labels 
face_recognizer.train(features,labels)
face_recognizer.save('face_trained.yml')
np.save('features.npy',features)
np.save('labels.npy',labels)
#face_recognizer('face_trained.yml')