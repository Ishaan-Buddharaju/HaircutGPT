from FaceShape import FaceShape
from dotenv import load_dotenv
import os
load_dotenv()
import pandas as pd
import cv2

def main():
    face_cascade_path = os.getenv('FACE_CASCADE_PATH')
    predictor_path = os.getenv('PREDICTOR_PATH')

    accuracy_df = pd.DataFrame(columns=['Person', 'Image_Path', 'Num_Detected_Faces', 'Classified_Face_Shape', 'True_Face_Shape'])

    base_dir = "training-data"


    for person_folder in os.listdir(base_dir):
        person_folder_path = os.path.join(base_dir, person_folder)

        if os.path.isdir(person_folder_path):
    
            for image_file in os.listdir(person_folder_path):
                image_path = os.path.join(person_folder_path, image_file)
                
                if os.path.isfile(image_path):
                    try:

                        face = FaceShape(image_path, face_cascade_path, predictor_path)
                        new_row = pd.DataFrame([{
                            'Person': person_folder, 
                            'Image_Path': image_path, 
                            'Num_Detected_Faces': face.num_faces, 
                            'Classified_Face_Shape': face.faceShape, 
                            'True_Face_Shape': None
                        }])
                        accuracy_df = pd.concat([accuracy_df, new_row], ignore_index=True)

                    except UnboundLocalError as e:
                        print(f"Error processing image: {image_path}")
                    except cv2.error as e:
                        print(f"cv2 error image: {image_path}")
                    
    accuracy_df.to_csv("accuracy_data.csv", index = True)

    print("Accuracy Test Complete")

    

if __name__ == "__main__":
    main()