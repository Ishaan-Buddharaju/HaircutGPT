import numpy as np 
import cv2 
import dlib 
from sklearn.cluster import KMeans 
import math
from math import degrees
from dotenv import load_dotenv
import os

load_dotenv()


#paths
image_path = os.getenv('IMAGE_PATH')
face_cascade_path = os.getenv('FACE_CASCADE_PATH')
predictor_path = os.getenv('PREDICTOR_PATH')

#make haar cascade for detecting face and smile
faceCascade = cv2.CascadeClassifier(face_cascade_path)

#facial landmark predictor
predictor = dlib.shape_predictor(predictor_path)

image = cv2.imread(image_path)
image = cv2.resize(image, (500, 500)) 
original = image.copy()

#convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Gaussian blur with a 3 x 3 kernel to help remove high frequency noise
gauss = cv2.GaussianBlur(gray, (3, 3), 0)


faces = faceCascade.detectMultiScale(
    gauss,
    scaleFactor=1.05,
    minNeighbors=5,
    minSize=(100, 100),
    flags=cv2.CASCADE_SCALE_IMAGE
    )
print("found {0} faces!".format(len(faces)))

for (x, y, w, h) in faces:
    #rect around face
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #convert opencv rect coords to Dlib rectangle
    dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
    detected_landmarks = predictor(image, dlib_rect).parts()
    landmarks = np.matrix([[p.x, p.y] for p in detected_landmarks])
    
results = original.copy()

for (x, y, w, h) in faces:
    cv2.rectangle(results, (x, y), (x + w, y + h), (0, 255, 0), 2)
    temp = original.copy()
    #getting area of interest from image (25% of face)
    forehead = temp[y:y + int(0.25 * h), x:x + w]
    rows, cols, bands = forehead.shape
    X = forehead.reshape(rows * cols, bands)

    #seperate hair from face
    kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=0)
    y_kmeans = kmeans.fit_predict(X)

    for i in range(rows):
        for j in range(cols):
            if y_kmeans[i * cols + j] == 0:
                forehead[i][j] = [255, 255, 255]
            else:
                forehead[i][j] = [0, 0, 0]

    forehead_mid = [int(cols / 2), int(rows / 2)]
    lef = 0
    rig = 0

    pixel_value = forehead[forehead_mid[1], forehead_mid[0]]

    for i in range(0, forehead_mid[0]):
        if forehead_mid[0] - i < 0:  # Check for out-of-bounds
            break
        if forehead[forehead_mid[1], forehead_mid[0] - i].all() != pixel_value.all():
            lef = forehead_mid[0] - i
            break

    for i in range(0, cols - forehead_mid[0]):
        if forehead_mid[0] + i >= cols: 
            break
        if forehead[forehead_mid[1], forehead_mid[0] + i].all() != pixel_value.all():
            rig = forehead_mid[0] + i
            break

    line1 = rig - lef
    cv2.line(results, (x + lef, y), (x + rig, y), color=(0, 255, 0), thickness=2)
    cv2.putText(results, 'Line 1', (x + lef, y - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
    cv2.circle(results, (x + lef, y), 5, color=(255, 0, 0), thickness=-1)
    cv2.circle(results, (x + rig, y), 5, color=(255, 0, 0), thickness=-1)

    linepointleft = (landmarks[1, 0], landmarks[1, 1])
    linepointright = (landmarks[15, 0], landmarks[15, 1])
    line2 = np.linalg.norm(np.subtract(linepointright, linepointleft))
    cv2.line(results, linepointleft, linepointright, color=(0, 255, 0), thickness=2)
    cv2.putText(results, 'Line 2', (int(linepointleft[0]), int(linepointleft[1]) - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
    cv2.circle(results, linepointleft, 5, color=(255, 0, 0), thickness=-1)
    cv2.circle(results, linepointright, 5, color=(255, 0, 0), thickness=-1)

    linepointleft = (landmarks[3, 0], landmarks[3, 1])
    linepointright = (landmarks[13, 0], landmarks[13, 1])
    line3 = np.linalg.norm(np.subtract(linepointright, linepointleft))
    cv2.line(results, linepointleft, linepointright, color=(0, 255, 0), thickness=2)
    cv2.putText(results, 'Line 3', (int(linepointleft[0]), int(linepointleft[1]) - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
    cv2.circle(results, linepointleft, 5, color=(255, 0, 0), thickness=-1)
    cv2.circle(results, linepointright, 5, color=(255, 0, 0), thickness=-1)

    linepointbottom = (landmarks[8, 0], landmarks[8, 1])
    linepointtop = (landmarks[8, 0], y)
    line4 = abs(landmarks[8, 1] - y)
    cv2.line(results, linepointtop, linepointbottom, color=(0, 255, 0), thickness=2)
    cv2.putText(results, 'Line 4', (int(linepointbottom[0]), int(linepointbottom[1]) - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
    cv2.circle(results, linepointtop, 5, color=(255, 0, 0), thickness=-1)
    cv2.circle(results, linepointbottom, 5, color=(255, 0, 0), thickness=-1)

    similarity = np.std([line1, line2, line3])
    ovalsimilarity = np.std([line2, line4])

    #calc angle for jawline
    ax, ay = landmarks[3, 0], landmarks[3, 1]
    bx, by = landmarks[4, 0], landmarks[4, 1]
    cx, cy = landmarks[5, 0], landmarks[5, 1]
    dx, dy = landmarks[6, 0], landmarks[6, 1]
    alpha0 = math.atan2(cy - ay, cx - ax)
    alpha1 = math.atan2(dy - by, dx - bx)
    alpha = alpha1 - alpha0
    angle = abs(degrees(alpha))
    angle = 180 - angle
    
    if similarity < 10:
        if angle < 160:
            print('Squared shape') #Jawlines are more angular
            faceShape = 'Squared shape'
        else:
            print('Round shape') #Jawlines are not that angular
            faceShape = 'Round shape'
        break
    if line3 > line1:
        if angle < 160:
            print('Triangle shape') #Forehead is wider
            faceShape = 'Triangle shape'
        else:
            print('Triangle shape') #Jawlines are more angular
            faceShape = 'Triangle shape'
        break
    if ovalsimilarity < 10:
        print('Diamond shape') #Line 2 & Line 4 are similar and Line 2 is slightly larger
        faceShape = 'Diamond shape'
        break
    if line4 > line2:
        if angle < 160:
            print('Rectangular shape') #Face length is largest and jawlines are angular
            faceShape = 'Rectangular shape'
        else:
            print('Oblong shape') #Face length is largest and jawlines are not angular')
            faceShape = 'Oblong Shape'
        break
    else:
      print("Something is wrong :( ! Make sure to contact me and explain the issue!")



#show the result image
cv2.imshow('Face Shape Detection', results)
cv2.waitKey(0)
cv2.destroyAllWindows()





