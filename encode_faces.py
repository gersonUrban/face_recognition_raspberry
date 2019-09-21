from imutils import paths
import face_recognition
#import argparse
import pickle
import cv2
import os

# grab the paths to the input images in our dataset
image_paths = list(paths.list_images("dataset/"))

# initialize the list of known encodings and known names
known_encodings = []
known_names = []

# loop over the images
for i, image_path in enumerate(image_paths):
    # extract the person name from the image path
    print("[INFO] processing image {}/{}".format(i+1,len(image_paths)))
    name = image_path.split(os.path.sep)[-2]
    
    # load the input image and convert it from BGR (OpenCV) to RGB (dlib)
    image = cv2.imread(image_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # detect the (x,y)-coord of the bouding boxes
    # corresponding to each face in the input image
    boxes = face_recognition.face_locations(rgb, model="hog")
    
    # compute the facial embedding for the face
    encodings = face_recognition.face_encodings(rgb,boxes)
    
    # loop over the encodings
    for encoding in encodings:
        # add each encoding + name to our set of known names and encodings
        known_encodings.append(encoding)
        known_names.append(name)
        
        
# dump the facil encodings + names to disk
print("[INFO] serializing encodings...")
data = {"encodings":known_encodings, "names":known_names}
print(data)
#doing with json
#import json
#with open('encodings.json','w') as outfile:
#    json.dump(data, outfile)
    
#print(data['encodings'][0])
# pickle have problem to read, than do it with json
pickle.dump(data, open( "encodings.pkl", "wb" ))
#f = open("encodings.pkl", "wb")
#f.write(pickle.dumps(data))
#f.close()
        
