import os
import cv2 as cv
import numpy as np


people = [] # add list of people

DIR = r'/resources/train' # add appropriate path


features = []
labels = []


haar_cascade = cv.CascadeClassifier('haar_face.xml') # add appropriate path

allowed_extensions = ['.jpg', '.jpeg', '.png']

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img_name in os.listdir(path):

            if not any(img_name.lower().endswith(ext) for ext in allowed_extensions):
                print(f"Skipping {img_name} as it is not an allowed image type.")
                continue
            
            img_path = os.path.join(path, img_name)
            img_array = cv.imread(img_path)
            

            if img_array is None:
                print(f"Warning: Couldn't read the image {img_path}")
                continue

            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            if len(faces_rect) == 0:
                print(f"Warning: No faces detected in {img_path}")
                continue

            
            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                faces_roi = cv.resize(faces_roi, (100, 100))  
                features.append(faces_roi)
                labels.append(label)

create_train()

if len(features) == 0 or len(labels) == 0:
    print("No valid data was collected for training.")
else:
    print(f'Collected {len(features)} features and {len(labels)} labels.')

    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    face_recognizer.train(np.array(features), np.array(labels))
    
    face_recognizer.save('face_trained.yml')
    np.save('features.npy', np.array(features))
    np.save('labels.npy', np.array(labels))
