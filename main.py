from faceShape import FaceShape
from dotenv import load_dotenv
import os
from cv2 import imshow, waitKey, destroyAllWindows
load_dotenv()


def main():
    # Paths
    image_path = os.getenv('IMAGE_PATH')
    face_cascade_path = os.getenv('FACE_CASCADE_PATH')
    predictor_path = os.getenv('PREDICTOR_PATH')

    face_to_classify = FaceShape(image_path, face_cascade_path, predictor_path)
    print(face_to_classify.faceShape)
    imshow('Face Measurements: press 0 to close', face_to_classify.measured_image)
    waitKey(0)
    destroyAllWindows()   


if __name__ == "__main__":
    main()