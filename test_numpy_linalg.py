import numpy as np


face_encodings = np.array([0.222, 0.456])
face_to_compare = np.array([1.09732, -1.9284391])
print(face_encodings - face_to_compare)
a = np.linalg.norm(face_encodings - face_to_compare)
print(a)
print(list(face_encodings))

face_to_compare = np.array([0.222, 0.5])
a = np.linalg.norm(face_encodings - face_to_compare)
print(a)