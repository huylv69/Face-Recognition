import cv2
import sys

cascPath = "data\haarcascade_frontalface_alt.xml"
details = False
if len(sys.argv) >= 2:
    details = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)
count = 0

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if details:
        if len(faces) > 0:
            coord = faces[0]
            cropped = frame[coord[1]:coord[1]+coord[3], coord[0]:coord[0]+coord[2]]
            cv2.imwrite("data/details/frame%d.jpg" % count, cropped)
            count += 1

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()