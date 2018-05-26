
# coding: utf-8

# Face Recognition with OpenCV
import cv2
import os
import numpy as np
import pickle


def detect_face(img):
    # convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # load OpenCV face detector use LBP  more accurate but slow Haar classifier
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

    # detect multiscale images
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=1, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)  # result is a list of faces

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


def prepare_training_data(data_folder_path):

    # ------STEP-1--------
    # get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)
    subjects = ["unknown"]
    # list to hold all subject faces
    faces = []
    # list to hold labels for all subjects
    labels = []
    i = 0
    # let's go through each directory and read images within it
    for dir_name in dirs:
        i = i+1
        #  add subject
        subjects.append(dir_name)

        # build path of directory containin images for current subject subject
        # sample subject_dir_path = "training-data/s1"
        subject_dir_path = data_folder_path + "/" + dir_name

        # get the images names that are inside the given subject directory
        subject_images_names = os.listdir(subject_dir_path)

        # detect face and add face to list of faces
        holder = 0
        print("done: ", dir_name)
        for image_name in subject_images_names:

            holder += 1
            if holder == 1 or holder == 2 or holder == 3:
                continue

            # build image path
            image_path = subject_dir_path + "/" + image_name

            # read image
            image = cv2.imread(image_path)

            # display an image window to show the image
            # cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
            # cv2.waitKey(1000)

            # detect face
            face, rect = detect_face(image)

            # ignore faces that are not detected
            if face is not None:
                # add face to list of faces
                faces.append(face)
                # add label for this face
                labels.append(i)
                # cv2.imshow("face", face)
                # cv2.waitKey(1000)

    with open("trained_model/subject.pkl", 'wb') as f:
        pickle.dump(subjects, f)
        print(subjects)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    return faces, labels


print("Preparing data...")
faces, labels = prepare_training_data("data")
# print faces, labels ,np.array(labels)

print("Data prepared")

# print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

print("Training model...")

# create our LBPH face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# use EigenFaceRecognizer
#face_recognizer = cv2.face.EigenFaceRecognizer_create()

# use FisherFaceRecognizer
#face_recognizer = cv2.face.FisherFaceRecognizer_create()


# train our face recognizer of our training faces
# OpenCV expects labels vector to be a `numpy` array.
face_recognizer.train(faces, np.array(labels))
face_recognizer.write("trained_model/trained_model.xml")
print("Training done!")
