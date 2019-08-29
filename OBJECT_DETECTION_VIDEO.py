"""OBJECT DETECTION IN VIDEO FILE"""

import cv2
import numpy as np

video = cv2.VideoCapture('peoplenyc.mp4')
a = 1
path = 'C:\\Users\\kuk\\OBJECT_DETECTION_VIDEO\\haar_cascade'
while True:
    a = a+1 
    check, frame = video.read()
    if check == False:
        break
#    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#    gray_frame = cv2.resize(gray_frame, (250,250))
    frame = cv2.resize(frame, (250,250))
    
    face_cascade = cv2.CascadeClassifier(path +"\\haarcascade_frontalcatface.xml")
    fullbody_cascade = cv2.CascadeClassifier(path + '\\haarcascade_fullbody.xml')

    faces = face_cascade.detectMultiScale(frame, 1.05, 5)
    fullbody = fullbody_cascade.detectMultiScale(frame, 1.05, 5)
    
    for x,y,w,h in faces:
        rec_img = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0) ,0)
        face_text = cv2.putText(frame, "FACE",(x, y+10),cv2.FONT_HERSHEY_COMPLEX_SMALL,.7,(0,255,0))
        
    for x,y,w,h in fullbody:
        rec_img = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0) ,0)
        body_text = cv2.putText(frame, "full body",(x, y+10),cv2.FONT_HERSHEY_COMPLEX_SMALL,.7,(0,255,0))
        
  
    cv2.imshow('vid', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(a)
video.release

cv2.destroyAllWindows()




