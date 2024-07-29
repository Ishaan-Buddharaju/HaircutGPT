import os
from dotenv import load_dotenv
load_dotenv()

import requests
import io
from PIL import Image
from selenium import webdriver

driver_path = os.getenv('driver_path')
path = driver_path
wd = webdriver.Chrome(path)


def download_img(download_path, url, fname):       #function to download images provided download path, image url, and file name
    try:
        image_content = requests.get(url).content     
        image_file = io.BytesIO(image_content)       
        image =  Image.open(image_file)
        file_path = download_path + fname

        with open(file_path, 'wb') as f:
            image.save(f, 'JPEG')

    except Exception as e:
        print('Failed Download -', e)


def get_images(wd, delay, max_imgs, url):  #function to get images
    def scroll(wd):
        wd.execute_script('window.scrollTo(0, document.body.scrollHeight);')
        time.sleep(delay)

    wd.get(url)

wd.quit()