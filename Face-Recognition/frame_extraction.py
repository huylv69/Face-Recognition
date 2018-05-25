import numpy as np
import cv2


#function to detect face using OpenCV
def detect_face(img):
    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #load OpenCV face detector use LBP  more accurate but slow Haar classifier
    face_cascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_default.xml')

    # detect multiscale images
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5) #result is a list of faces
    
    #if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None
    
    (x, y, w, h) = faces[0] # assumption that there will be only one face,
    
    #return only the face part of the image
    return gray[y:y+w, x:x+h], faces[0]

def predict(test_img):
    #make a copy of the image 
    img = test_img.copy()
    #detect face from the image
    face, rect = detect_face(img)
    if face is None:
        print (" Can not detect face from file!")
        return None
    #predict the image using face recognizer 
    label, confidence = face_recognizer.predict(face)
    # label_text = subjects[label]
    print (label , confidence)
    return confidence

cap = cv2.VideoCapture(0)
images = []
print ("Capturing image.....")
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    face, rect = detect_face(frame)
    if face is None:
        pass
    else :
        images.append(frame)
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print ("Capture done.")
# When everything done, release the capture
cap.release()
cv2.destroyWindow('frame')

# Load face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("trained_model/trained_model.xml")

print ("Choosing the best frame...")
print len(images)
best_image = images[0]
best_score = predict(best_image)
for image in images:
    if (predict(image)<best_score):
        best_score = predict(image)
        best_image = image
cv2.imshow('Best frame',best_image)
cv2.imwrite('best_frame.png',best_image)
cv2.waitKey(1)
cv2.destroyAllWindows()