from selenium import webdriver
import os
import requests
import io
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

driver_path = os.getenv('driver_path')
searchterm = 'Zendaya' #input your search item here
url = "https://www.google.co.in/search?q="+searchterm+"&source=lnms&tbm=isch"
browser = webdriver.Chrome(driver_path) #insert path to chromedriver inside parentheses
browser.get(url)
img_count = 0
extensions = { "jpg", "jpeg", "png", "gif" }


if not os.path.exists(searchterm):
    os.mkdir(searchterm)

for t in range(500):  
    browser.execute_script("window.scrollBy(0,10000)")  #scroll down to load images
    
html = browser.page_source.split('["')      #grabs html source of page
images = []
for i in html:
    if i.startswith('http') and i.split('"')[0].split('.')[-1] in extensions:   #checks if img by splitting and comparing if its one of the extensions
        images.append(i.split('"')[0])   #adds image link to list "images"


def download_image(download_path, url, file_name):   #function to download an image provided the directory, img url, and desired file name
	try:
		image_content = requests.get(url).content
		image_file = io.BytesIO(image_content)    #takes content and converts to bytes
		image = Image.open(image_file)              
		file_path = download_path + file_name

		with open(file_path, "wb") as f:
			image.save(f, "JPEG")           #saves image at the filepath as a jpg

		print("Success")
	except Exception as e:
		print('FAILED -', e)

for idx, link in enumerate(images):
      download_image("", link, f'Zendaya{idx}.jpg')


print("Success :)")

browser.quit()