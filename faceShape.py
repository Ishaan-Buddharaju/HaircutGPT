#importing the libraries
import numpy as np #for mathematical calculations
import cv2 #for face detection and other image operations
import dlib #for detection of facial landmarks ex:nose,jawline,eyes
from sklearn.cluster import KMeans #for clustering
import math
from math import degrees
from dotenv import load_dotenv
import os

load_dotenv()


# Paths
image_path = os.getenv('IMAGE_PATH')
face_cascade_path = os.getenv('FACE_CASCADE_PATH')
predictor_path = os.getenv('PREDICTOR_PATH')

class FaceShape: 

    def __init__(self, image_path, face_cascade_path, predictor_path):
        self.image_path = image_path
        self.face_cascade_path = face_cascade_path
        self.predictor_path = predictor_path
        self.faceCascade = cv2.CascadeClassifier(face_cascade_path)
        self.predictor = dlib.shape_predictor(predictor_path)
        self.image = cv2.imread(image_path)
        self.image = cv2.resize(self.image, (500, 500))
        self.original = self.image.copy()
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.gauss = cv2.GaussianBlur(self.gray, (3, 3), 0)
        self.faces = self.faceCascade.detectMultiScale(
            self.gauss,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE
            )
        self.num_faces = len(self.faces)
        

    
    def classify_face_shape(self):
        for (x, y, w, h) in self.faces:
            #draw a rectangle around the faces
            cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #converting the opencv rectangle coordinates to Dlib rectangle
            dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            #detecting landmarks
            detected_landmarks = self.predictor(self.image, dlib_rect).parts()
            #converting to np matrix
            landmarks = np.matrix([[p.x, p.y] for p in detected_landmarks])
    
        #making another copy for showing final results
        results = self.original.copy()

        for (x, y, w, h) in self.faces:
            #draw a rectangle around the faces
            cv2.rectangle(results, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #making temporary copy
            temp = self.original.copy()
            #getting area of interest from image i.e., forehead (25% of face)
            forehead = temp[y:y + int(0.25 * h), x:x + w]
            rows, cols, bands = forehead.shape
            X = forehead.reshape(rows * cols, bands)

            #kmeans clustering with 2 clusters (for hair and skin)
            kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=0)
            y_kmeans = kmeans.fit_predict(X)

            for i in range(rows):
                for j in range(cols):
                    if y_kmeans[i * cols + j] == 0:
                        forehead[i][j] = [255, 255, 255]
                    else:
                        forehead[i][j] = [0, 0, 0]

            # Initialize the midpoint of the forehead
            forehead_mid = [int(cols / 2), int(rows / 2)]  # midpoint of forehead
            lef = 0
            rig = 0

            # Get the pixel value at the midpoint
            pixel_value = forehead[forehead_mid[1], forehead_mid[0]]

            # Loop for detecting the left edge
            for i in range(0, forehead_mid[0]):
                if forehead_mid[0] - i < 0:  # Check for out-of-bounds
                    break
                if forehead[forehead_mid[1], forehead_mid[0] - i].all() != pixel_value.all():
                    lef = forehead_mid[0] - i
                    break

            # Loop for detecting the right edge
            for i in range(0, cols - forehead_mid[0]):
                if forehead_mid[0] + i >= cols:  # Check for out-of-bounds
                    break
                if forehead[forehead_mid[1], forehead_mid[0] + i].all() != pixel_value.all():
                    rig = forehead_mid[0] + i
                    break

            # Draw line 1 on forehead with circles
            line1 = rig - lef
            cv2.line(results, (x + lef, y), (x + rig, y), color=(0, 255, 0), thickness=2)
            cv2.putText(results, 'Line 1', (x + lef, y - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
            cv2.circle(results, (x + lef, y), 5, color=(255, 0, 0), thickness=-1)
            cv2.circle(results, (x + rig, y), 5, color=(255, 0, 0), thickness=-1)

            # Draw line 2 with circles
            linepointleft = (landmarks[1, 0], landmarks[1, 1])
            linepointright = (landmarks[15, 0], landmarks[15, 1])
            line2 = np.linalg.norm(np.subtract(linepointright, linepointleft))
            cv2.line(results, linepointleft, linepointright, color=(0, 255, 0), thickness=2)
            cv2.putText(results, 'Line 2', (int(linepointleft[0]), int(linepointleft[1]) - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
            cv2.circle(results, linepointleft, 5, color=(255, 0, 0), thickness=-1)
            cv2.circle(results, linepointright, 5, color=(255, 0, 0), thickness=-1)

            # Draw line 3 with circles
            linepointleft = (landmarks[3, 0], landmarks[3, 1])
            linepointright = (landmarks[13, 0], landmarks[13, 1])
            line3 = np.linalg.norm(np.subtract(linepointright, linepointleft))
            cv2.line(results, linepointleft, linepointright, color=(0, 255, 0), thickness=2)
            cv2.putText(results, 'Line 3', (int(linepointleft[0]), int(linepointleft[1]) - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
            cv2.circle(results, linepointleft, 5, color=(255, 0, 0), thickness=-1)
            cv2.circle(results, linepointright, 5, color=(255, 0, 0), thickness=-1)

            # Draw line 4 with circles
            linepointbottom = (landmarks[8, 0], landmarks[8, 1])
            linepointtop = (landmarks[8, 0], y)
            line4 = abs(landmarks[8, 1] - y)
            cv2.line(results, linepointtop, linepointbottom, color=(0, 255, 0), thickness=2)
            cv2.putText(results, 'Line 4', (int(linepointbottom[0]), int(linepointbottom[1]) - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
            cv2.circle(results, linepointtop, 5, color=(255, 0, 0), thickness=-1)
            cv2.circle(results, linepointbottom, 5, color=(255, 0, 0), thickness=-1)

            similarity = np.std([line1, line2, line3])
            ovalsimilarity = np.std([line2, line4])

            # Calculate angle for jawline
            ax, ay = landmarks[3, 0], landmarks[3, 1]
            bx, by = landmarks[4, 0], landmarks[4, 1]
            cx, cy = landmarks[5, 0], landmarks[5, 1]
            dx, dy = landmarks[6, 0], landmarks[6, 1]
            alpha0 = math.atan2(cy - ay, cx - ax)
            alpha1 = math.atan2(dy - by, dx - bx)
            alpha = alpha1 - alpha0
            angle = abs(degrees(alpha))
            angle = 180 - angle
            
        # Determine face shape based on calculated metricsfor i inrange(1):
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
        #returns string faceShape and resulting image 
        #results can be shown with cv2.imshow('Face Shape', results)
        return faceShape, results     