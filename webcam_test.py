'''
Created by Gerson Urban

Use code created by Adrian Rosebrock.
pyimagesearch.com/2018/06/11/how-to-build-a-custom-face-recognition-dataset/

'''
import numpy as np
import cv2
import imutils
from imutils.video import VideoStream

cap = cv2.VideoCapture(0)
# load OpenCV's Haar cascade for face detection from disk
#detector = cv2.CascadeClassifier(args["cascade"])
path = "/home/pi/Documents/opencv/data/haarcascades/"
#cv2.cascadeClassifier.load("haarcascade_frontalface_default.xml")
detector = cv2.CascadeClassifier(path+"haarcascade_frontalface_default.xml")

while(True):
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=400)
    
    #Detect faces in the grayscale frame
    rects = detector.detectMultiScale(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1,
        minNeighbors=5, minSize=(30,30)
        )
    # loop over the face detections and draw then on the frame
    for (x,y,w,h) in rects:
        cv2.rectangle(frame, (x,y),(x+w,y+h), (255,0,0),2)
    
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xff==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()