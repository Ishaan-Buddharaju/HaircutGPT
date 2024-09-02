import numpy as np
import cv2
import dlib
from sklearn.cluster import KMeans
from math import degrees, atan2

class FaceShapeDetector:
    def __init__(self, image_path, face_cascade_path, predictor_path):
        self.image_path = image_path
        self.face_cascade_path = face_cascade_path
        self.predictor_path = predictor_path
        self.face_cascade = cv2.CascadeClassifier(self.face_cascade_path)
        self.predictor = dlib.shape_predictor(self.predictor_path)

    def load_and_preprocess_image(self):
        image = cv2.imread(self.image_path)
        image = cv2.resize(image, (500, 500))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gauss = cv2.GaussianBlur(gray, (3, 3), 0)
        return image, gauss

    def detect_faces(self, gauss):
        faces = self.face_cascade.detectMultiScale(
            gauss,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return faces

    def get_landmarks(self, image, face):
        x, y, w, h = face
        dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        detected_landmarks = self.predictor(image, dlib_rect).parts()
        landmarks = np.matrix([[p.x, p.y] for p in detected_landmarks])
        return landmarks

    def apply_kmeans(self, forehead):
        rows, cols, bands = forehead.shape
        X = forehead.reshape(rows * cols, bands)
        kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=0)
        y_kmeans = kmeans.fit_predict(X)
        return y_kmeans, rows, cols

    def detect_forehead_edges(self, forehead, y_kmeans, cols, midpoint):
        lef, rig = 0, 0
        pixel_value = forehead[midpoint[1], midpoint[0]]
        for i in range(0, midpoint[0]):
            if midpoint[0] - i < 0:
                break
            if forehead[midpoint[1], midpoint[0] - i].all() != pixel_value.all():
                lef = midpoint[0] - i
                break
        for i in range(0, cols - midpoint[0]):
            if midpoint[0] + i >= cols:
                break
            if forehead[midpoint[1], midpoint[0] + i].all() != pixel_value.all():
                rig = midpoint[0] + i
                break
        return lef, rig

    def draw_lines_and_calculate_metrics(self, results, face, landmarks, lef, rig):
        x, y, w, h = face
        line1 = rig - lef
        cv2.line(results, (x + lef, y), (x + rig, y), color=(0, 255, 0), thickness=2)
        linepointleft = (landmarks[1, 0], landmarks[1, 1])
        linepointright = (landmarks[15, 0], landmarks[15, 1])
        line2 = np.linalg.norm(np.subtract(linepointright, linepointleft))
        cv2.line(results, linepointleft, linepointright, color=(0, 255, 0), thickness=2)
        linepointleft = (landmarks[3, 0], landmarks[3, 1])
        linepointright = (landmarks[13, 0], landmarks[13, 1])
        line3 = np.linalg.norm(np.subtract(linepointright, linepointleft))
        cv2.line(results, linepointleft, linepointright, color=(0, 255, 0), thickness=2)
        linepointbottom = (landmarks[8, 0], landmarks[8, 1])
        linepointtop = (landmarks[8, 0], y)
        line4 = abs(landmarks[8, 1] - y)
        cv2.line(results, linepointtop, linepointbottom, color=(0, 255, 0), thickness=2)

        return line1, line2, line3, line4

    def calculate_similarity(self, line1, line2, line3, line4):
        similarity = np.std([line1, line2, line3])
        ovalsimilarity = np.std([line2, line4])
        return similarity, ovalsimilarity

    def calculate_jawline_angle(self, landmarks):
        ax, ay = landmarks[3, 0], landmarks[3, 1]
        bx, by = landmarks[4, 0], landmarks[4, 1]
        cx, cy = landmarks[5, 0], landmarks[5, 1]
        dx, dy = landmarks[6, 0], landmarks[6, 1]
        alpha0 = atan2(cy - ay, cx - ax)
        alpha1 = atan2(dy - by, dx - bx)
        alpha = alpha1 - alpha0
        angle = abs(degrees(alpha))
        return 180 - angle

    def calculate_face_shape(self, similarity, ovalsimilarity, line1, line2, line3, line4, angle):
        if similarity < 10:
            return 'Squared shape' if angle < 160 else 'Round shape'
        if line3 > line1:
            return 'Triangle shape'
        if ovalsimilarity < 10:
            return 'Diamond shape'
        if line4 > line2:
            return 'Rectangular shape' if angle < 160 else 'Oblong shape'
        return "Error: Face shape could not be determined!"

    def detect_face_shape(self):
        image, gauss = self.load_and_preprocess_image()
        faces = self.detect_faces(gauss)
        print(f"Found {len(faces)} faces!")

        if len(faces) == 0:
            return "No faces detected."

        results = image.copy()
        face_shape = "Unknown"

        for face in faces:
            x, y, w, h = face
            cv2.rectangle(results, (x, y), (x + w, y + h), (0, 255, 0), 2)
            landmarks = self.get_landmarks(image, face)
            forehead = image[y:y + int(0.25 * h), x:x + w]
            y_kmeans, rows, cols = self.apply_kmeans(forehead)
            midpoint = [int(cols / 2), int(rows / 2)]
            lef, rig = self.detect_forehead_edges(forehead, y_kmeans, cols, midpoint)
            line1, line2, line3, line4 = self.draw_lines_and_calculate_metrics(results, face, landmarks, lef, rig)
            similarity, ovalsimilarity = self.calculate_similarity(line1, line2, line3, line4)
            angle = self.calculate_jawline_angle(landmarks)
            face_shape = self.calculate_face_shape(similarity, ovalsimilarity, line1, line2, line3, line4, angle)
        cv2.imshow('Face Shape Detection', results)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return face_shape


# Paths
image_path = "/Users/dhirennarne/workspace/FaceShape/i7.jpg"
face_cascade_path = "/Users/dhirennarne/workspace/FaceShape/lbpcascade_frontalcatface.xml"
predictor_path = "/Users/dhirennarne/workspace/FaceShape/shape_predictor_68_face_landmarks.dat"

# Initialize and run face shape detection
detector = FaceShapeDetector(image_path, face_cascade_path, predictor_path)
face_shape = detector.detect_face_shape()
print(f"Detected Face Shape: {face_shape}")
