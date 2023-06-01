  import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC


path=r'C:\Users\sujat\OneDrive\Desktop\backup\Face-Recognition-Attendance-System-main\IMAGE_FILES'
pathtest=r'C:\Users\sujat\OneDrive\Desktop\backup\Face-Recognition-Attendance-System-main\IMAGEtest_FILES'


images=[]
className=[]
myList=os.listdir(path)
print(myList)
for cl in myList:
    curImg=cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    className.append(os.path.splitext(cl)[0])
print(className)


imagestest=[]
classNametest=[]
myListtest=os.listdir(pathtest)
print(myListtest)
for cl in myListtest:
    curImg=cv2.imread(f'{pathtest}/{cl}')
    imagestest.append(curImg)
    classNametest.append(os.path.splitext(cl)[0])
print(classNametest)


def findEncodings(images):
    encodeList=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def findEncodingstest(imagestest):
    encodeListtest=[]
    for img in imagestest:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encodetest = face_recognition.face_encodings(img)[0]
        encodeListtest.append(encodetest)
    return encodeListtest

encodeListKnown=findEncodings(images)
print('Encoding Complete')

encodeListKnowntest=findEncodingstest(imagestest)

# Create a model object
model = SVC(kernel='linear', probability=True)


# Train the model using the training data
model.fit(encodeListKnown, className)


# train_preds = model.predict(encodeListKnown)
accuracy=model.score(encodeListKnowntest,classNametest)
print(accuracy*100)
# train_acc = accuracy_score(className, train_preds)

