
# coding: utf-8
"""
================
NHận diện khuôn mặt với OpenCV
================
"""

import cv2
import os
import pickle
import glob

# read subject
# subjects = []
with open("trained_model/subject.pkl", 'rb') as f:
    subjects = pickle.load(f)

# Load face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("trained_model/trained_model.xml")


# function to detect face using OpenCV


def detect_face(img):
    # convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # load OpenCV face detector use LBP  more accurate but slow Haar classifier
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

    # detect multiscale images
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=1)  # result is a list of faces

    # if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None

    (x, y, w, h) = faces[0]  # assumption that there will be only one face,

    # return only the face part of the image
    return gray[y:y+w, x:x+h], faces[0]

# ### Prediction

def predict(test_img):
    # make a copy of the image
    img = test_img.copy()
    # detect face from the image
    face, rect = detect_face(img)
    if face is None:
        print(" Can not detect face from file!")
        return None
    # predict the image using face recognizer
    label, confidence = face_recognizer.predict(face)
    print(label, confidence)

    return confidence


if __name__ == '__main__':
    testfiles = glob.glob("data/s*/[123].pgm")
    testfiles.sort()
    # load test images
    test_imgs = [cv2.imread(path) for path in testfiles]
    
    success = 0
    failure = 0
    # perform a prediction
    for img in test_imgs:
        conf = predict(img)
        if conf is not None:
            success += 1
        else:
            failure += 1
    
    precision = success / len(testfiles)
    print("result: ", precision)
