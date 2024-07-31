from selenium import webdriver
import os
import requests
import io
from PIL import Image
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

celebs = pd.read_csv("celebrities.csv")
driver_path = os.getenv('driver_path')
wd = webdriver.Chrome(driver_path)          #initialize webdriver as wd


def scroll_down(wd):
    wd.execute_script("window.scrollBy(0,10000)")

def get_images(wd):
    images = []
    extensions = [ "jpg", "jpeg", "png"]        #create list of picture extensions to save later   
    html = wd.page_source.split('["')           #grabs html source of page
    for content in html:
        content_suffix = content.split('"')[0].split('.')[-1]                #grabs the suffix of the html to be compared to image extensions
        if content.startswith('http') and content_suffix in extensions:      #checks if content is img
            images.append(content.split('"')[0])                             #adds image link to list "images"

    return images


def download_image(download_path, url, file_name):   
	try:
		image_content = requests.get(url).content   
		image_file = io.BytesIO(image_content)      #takes content and converts to bytes
		image = Image.open(image_file)              #opens the image from bytes   
		file_path = download_path + file_name

		with open(file_path, "wb") as f:
			image.save(f, "JPEG")           #saves image at the filepath as a jpg

		print("Success")
	except Exception as e:
		print('FAILED -', e)

