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
browser = webdriver.Chrome("/Users/ishaan/Documents/Code/HaircutCV/chromedriver") #insert path to chromedriver inside parentheses
browser.get(url)
img_count = 0
extensions = { "jpg", "jpeg", "png", "gif" }
if not os.path.exists(searchterm):
    os.mkdir(searchterm)

for _ in range(500):
    browser.execute_script("window.scrollBy(0,10000)")
    
html = browser.page_source.split('["')
images = []
for i in html:
    if i.startswith('http') and i.split('"')[0].split('.')[-1] in extensions:
        images.append(i.split('"')[0])


def download_image(download_path, url, file_name):
	try:
		image_content = requests.get(url).content
		image_file = io.BytesIO(image_content)
		image = Image.open(image_file)
		file_path = download_path + file_name

		with open(file_path, "wb") as f:
			image.save(f, "JPEG")

		print("Success")
	except Exception as e:
		print('FAILED -', e)

for idx, link in enumerate(images):
      download_image("", link, f'Zendaya{idx}.jpg')


print("Success :)")

browser.quit()