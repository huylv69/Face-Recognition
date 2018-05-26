
# coding: utf-8
"""
================
NHận diện khuôn mặt với OpenCV
================
"""

import cv2
import os
import numpy as np
import pickle

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
    face_cascade = cv2.CascadeClassifier(
        'opencv-files/haarcascade_frontalface_alt.xml')

    # detect multiscale images
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=1,minSize=(20,20))  # result is a list of faces

    # if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None

    best_face = None
    biggest_face = 0
    for face in faces:
        (x,y,w,h) = face
        if(w*h > biggest_face):
            best_face = face
            biggest_face = w*h

    (x, y, w, h) = best_face  # assumption that there will be only one face,

    # return only the face part of the image
    return gray[y:y+w, x:x+h], best_face

# ### Prediction
# function to draw rectangle on image according to given (x, y) coordinates and given width and heigh


def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
#   cv2.rectangle(img, topLeftPoint, bottomRightPoint, rgbColor, lineWidth)` to draw rectangle.

# function to draw text on give image starting from  passed (x, y) coordinates.


def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
#   cv2.putText(img, text, startPoint, font, fontSize, rgbColor, lineWidth)` to draw text on image.


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
    # get name of respective label returned by face recognizer
    label_text = subjects[label]

    # draw a rectangle around face detected
    draw_rectangle(img, rect)
    # draw name of predicted person
    draw_text(img, label_text, rect[0], rect[1]-5)
    return img


if __name__ == '__main__':
    print("Chọn nguồn để nhận dạng:")
    print("1. Từ ảnh đầu vào")
    print("2. Từ camera")
    try:
        mode = int(input('>>: '))
    except ValueError:
        print("Not a number")
    if mode == 1:
        path = raw_input('Nhập đường dẫn ảnh: ')
        assert os.path.exists(path), "Không tồn tại file, " + str(path)
        # load test images
        test_img = cv2.imread(path)
        # perform a prediction
        predicted_img = predict(test_img)
        if(predicted_img is None):
            pass
        else:
            print("Prediction complete")
            # display images
            cv2.imshow("Result", predicted_img)
            cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif mode == 2:
        cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier(
            'opencv-files/haarcascade_frontalface_alt.xml')
        while 1:
            ret, img = cap.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                label, conf = face_recognizer.predict(gray[y:y+h, x:x+w])
                print(label, conf)
                # if label < len(subjects):
                # draw_text(img, subjects[label], rect[0], rect[1]-5)
                if conf < 100:
                    cv2.putText(
                        img, subjects[label], (x, y+h), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                else:
                    cv2.putText(
                        img, subjects[0], (x, y+h), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                    # cv2.PutText(img,subjects[label],(x,y+h),font,255)
            cv2.imshow('img', img)
            if cv2.waitKey(1) == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
