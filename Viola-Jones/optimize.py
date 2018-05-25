import cv2
import sys
import glob

cascPath = "haarcascade_frontalface_alt.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

# The files are in PGM format, and can conveniently be viewed on UNIX (TM) systems using the 'xv' program. The size of each image is 92x112 pixels, with 256 grey levels per pixel. The images are organised in 40 directories (one for each subject), which have names of the form sX, where X indicates the subject number (between 1 and 40). In each of these directories, there are ten different images of that subject, which have names of the form Y.pgm, where Y is the image number for that subject (between 1 and 10).
filenames = glob.glob('data/s*/*.pgm')
images = [cv2.imread(f) for f in filenames]

result = open("result.txt", "w")
best = 0
best_params = []

for tenScale in range(11, 20):
    for minNb in range(1, 10):
        for size in range(20, 40):
                count = 0
                for img in images:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    ###
                    # scaleFactor – Parameter specifying how much the image size is reduced at each image scale.
                    # minNeighbors – Parameter specifying how many neighbors each candidate rectangle should have to retain it.
                    # flags – Parameter with the same meaning for an old cascade as in the function cvHaarDetectObjects. It is not used for a new cascade.
                    # minSize – Minimum possible object size. Objects smaller than that are ignored.
                    # maxSize – Maximum possible object size. Objects larger than that are ignored.
                    ###

                    # scaleFactor: (1, 2.0] - 10
                    # minNeighbors: [3, 10) - 7
                    # minsize: x, y - [20, 40) - 20
                    # = 15 * 7 * 20
                    scale = tenScale / 10
                    faces = faceCascade.detectMultiScale(
                        gray,
                        scaleFactor=scale,
                        minNeighbors=minNb,
                        minSize=(size, size),
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )

                    if len(faces) == 1:
                        count += 1
                accurate = count / 400
                if accurate > best:
                    best = accurate
                    best_params = [tenScale/10, minNb, size, accurate]
                print(f"{tenScale/10} {minNb} {size} {accurate}\n")
                result.write(f"{tenScale/10} {minNb} {size} {accurate}\n")

print("best scale: ", best_params[0])
print("best minNeighbors: ", best_params[1])
print("best minSize: ", best_params[2])
print("best accurate: ", best_params[3])
result.close()
# When everything is done, wait for stop
cv2.waitKey(0)
cv2.destroyAllWindows()
