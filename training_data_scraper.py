import os
import shutil
import requests
import io
import cv2
import pandas as pd
from selenium import webdriver
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

driver_path = os.getenv('driver_path')

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name) 

def scroll_down(wd):
    for t in range(500):
        wd.execute_script("window.scrollBy(0,10000)")

def get_images(wd):
    scroll_down(wd)
    image_links = []
    extensions = [ "jpg", "jpeg", "png"]        #create list of picture extensions to save later   
    html = wd.page_source.split('["')           #grabs html source of page
    for content in html:
        content_suffix = content.split('"')[0].split('.')[-1]                #grabs the suffix of the html to be compared to image extensions
        if content.startswith('http') and content_suffix in extensions:      #checks if content is img
            image_links.append(content.split('"')[0])                             #adds image link to list "image_links"

    return image_links

def count_faces(image, face_cascade):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=6, minSize= [30,30])
    return len(faces)

def download_image(folder_name, url, file_name):   #function to download an image provided the directory, img url, and desired file name 
    try:
        image_content = requests.get(url).content
        image_file = io.BytesIO(image_content)    #takes content and converts to bytes
        image = Image.open(image_file)              
        file_path = os.path.join(folder_name, file_name)   
        image.save(file_path, "JPEG")               #saves image at path as JPG
        print("Success")

        return (file_path)
    except Exception as e:
        print('FAILED -', e)
        return False

            
if __name__ == '__main__':

    wd = webdriver.Chrome(driver_path)       #initialize webdriver
    celebs = pd.read_csv("celebrities.csv")['Celebrity']  #read in celeb csv and column with celeb names
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")  #read face cascade once to be used to count faces
    for name in celebs:            
        name = name.replace(" ", "+")    #replace spaces with "+" for search format
        search_url = f"https://www.google.co.in/search?q={name}&source=lnms&tbm=isch"
        wd.get(search_url)
        image_links = get_images(wd)        #list of image links
        create_folder(f"{name}")            #create folder of celeb name to download images into
        create_folder(f"{name}/{name}-omitted")    #create folder for omitted images (improper downloads or more than one face)
        for idx, link in enumerate(image_links):     #iterates through list of links and downloads images
            downloaded_path = download_image(f"{name}", link, f'{name}_{idx}.jpg')
            if downloaded_path is not False:
                if count_faces(cv2.imread(downloaded_path), face_cascade) != 1:
                    shutil.move(downloaded_path, f"{name}/{name}-omitted")  #move the unwanted image into the omitted folder
                    print(f"moved {name}_{idx}")

    print("\n\n\nDone!")
    wd.quit()

