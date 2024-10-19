import os
import cv2 as cv


haar_cascade = cv.CascadeClassifier('haar_face.xml') # add appropriate file path


people = [] # add list of people

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read(r'face_trained.yml') # add appropriate file path


img_path = r'resources/val/xyz' # add appropriate file path
img = cv.imread(img_path)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person', gray)

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

for (x, y, w, h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+w]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with a confidence of {confidence}')


    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)


    cv.putText(img, f'{people[label]}', (x, y - 10), cv.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), thickness=2)

cv.imshow("Person", img)

cv.waitKey(0)
cv.destroyAllWindows()
