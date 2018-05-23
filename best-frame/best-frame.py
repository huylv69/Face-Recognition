import cv2
import sys
import glob
import math

def quantify_hue(hue):
    result_value = 7

    if hue > 155 or hue <= 10:
        result_value = 0
    elif hue <= 20:
        result_value = 1
    elif hue <= 37:
        result_value = 2
    elif hue <= 77:
        result_value = 3
    elif hue <= 95:
        result_value = 4
    elif hue <= 135:
        result_value = 5
    elif hue <= 147:
        result_value = 6
    else:
        result_value = 7
    
    return result_value

def quantify_sat(sat):
    result_value = 2

    if sat <= 51:
        result_value = 0
    elif sat <= 178:
        result_value = 1
    else:
        result_value = 2

    return result_value

def quantify_val(val):
    result_value = 2

    if val <= 51:
        result_value = 0
    elif val <= 178:
        result_value = 1
    else:
        result_value = 2

    return result_value

def euclid_distance(h, g):
    sum = 0
    for i in range(72):
        sum += pow(g[i] - h[i], 2)
    
    distance = math.sqrt(sum)
    return distance

## MAIN

filenames = glob.glob("data/details/*.jpg")
filenames.sort()
images = []
names = []
for img in filenames:
    images.append(cv2.imread(img))
    names.append(img)

count = 0
histograms = []
for img in images:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    height, width, channel = hsv.shape
    histo_value = []
    for i in range(72):
        histo_value.append(0)
    for i in range(height):
        for j in range(width):
            px = hsv[i, j]
            hue = px[0]  # [0, 179]
            sat = px[1]  # [0, 255]
            val = px[2]  # [0, 255]

            hue = quantify_hue(hue)
            sat = quantify_sat(sat)
            val = quantify_val(val)

            g_value = 9*hue + 3*sat + val # [0, 71]
            histo_value[g_value] += 1
    
    # convert histogram to probality value
    for i in range(72):
        histo_value[i] = histo_value[i] / (height*width)
    
    histograms.append(histo_value)

    print ('done image ', count, ' - ', names[count])
    count += 1

# calculate average histogram
average = []
for i in range(72):
    length = len(histograms)
    color_i = 0
    for j in range(length):
        color_i += histograms[j][i]
    
    color_i = color_i / length
    average.append(color_i)

print ("average: ", average)

# find best-frame
min_distance_index = 0
min_distance = euclid_distance(histograms[min_distance_index], average)
for i in range(len(histograms)):
    d = euclid_distance(histograms[i], average)
    if d < min_distance:
        min_distance = d
        min_distance_index = i

print("best: ", min_distance)
print("images: ", min_distance_index)
print("name: ", names[min_distance_index])

cv2.imshow("image", cv2.cvtColor(images[min_distance_index], cv2.COLOR_HSV2RGB))
cv2.waitKey(0)
cv2.destroyAllWindows()
