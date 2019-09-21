'''
code based in pyimagesearch.com/2018/06/25/raspberry-pi-face-recognition/
'''
import face_recognition
import imutils
import pickle
import time
import cv2
import numpy as np

#doing with json instead pickle
#import json
# load the known faces and embeddings along with OpenCV's
# Haar cascade for face detection


data = pickle.loads(open("encodings.pkl", "rb").read())
#with open('encodings.json') as json_file:
#    data = json.load(json_file)
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# initialize video stream
cap = cv2.VideoCapture(0)

while(True):
    # grab the frame and resize to 500px
    ret, frame = cap.read()
    orig = frame.copy()
    frame = imutils.resize(frame, width=500)
    
    # convert the input frame from BGR to grayscale (for face detection)
    # and from BGR to RGB for facerecognition
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # detect faces in grayscale frame
    rects = detector.detectMultiScale(gray, scaleFactor=1.1,
                                      minNeighbors=5, minSize=(30,30))
    
    # OpenCV returns boundboxes but is need reorder
    boxes = [(y,x+w,y+h,x)for(x,y,w,h) in rects]
    
    # compute the facial embeddings for each face bounding box
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []
    
    # loop over the facial embeddings
    for encoding in encodings:
        #print(encoding)
        #attempt to match each face in the input image to out known encodings
        #matches = face_recognition.compare_faces("encodings.pkl",encoding)
        #matches = face_recognition.compare_faces(data,encoding)
        #min_a = 10
        #min_n = ''
        matches = []
        for i, vec in enumerate(data['encodings']):
            a = np.linalg.norm(vec - encoding)
            #if a < min_a:
            #    min_a = a
            #    min_n = data['names'][i]
            if a <= 0.5:
                matches.append(True)
            else:
                matches.append(False)
            
            
        #matches = face_recognition.compare_faces("encodings.json",encoding)
        #matches = [False]
        name = "Unknown"
        
        if True in matches:
            
            matched_ids = [i for (i,b) in enumerate(matches) if b]
            counts = {}
            
            # loop over the matched indexes and maintain a count for
            # each recognized face
            for i in matched_ids:
                name = data['names'][i]
                counts[name] = counts.get(name,0)+1
                
            # determine the recogized face with the largest number of votes
            name = max(counts, key=counts.get)
            
        # update the list of names
        names.append(name)
        
    # loop over the recognized faces
    for ((top, right, bottom, left), name) in zip(boxes, names):
        #draw the predicted face name on the image
        cv2.rectangle(frame, (left, top), (right, bottom),
                      (0,255,0),2)
        y = top - 15 if top -15 >15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 255, 0), 2)
        
    # display the image to our screen
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    # if the 'q' key was pressed, break from the loop
    if key == ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()     
    
    
    
    
