from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager

# Set up the Chrome WebDriver with options for headless mode (no GUI), no sandbox, and disabling shared memory usage
options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

# Initialize the WebDriver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Perform a Google image search with request query (input desired search term in search_query)
search_query = "Zendaya"
driver.get(f"https://www.google.com/search?q={search_query}&tbm=isch")

# Wait for images to load with a maximum wait time of 20 seconds
wait = WebDriverWait(driver, 20)
try:
    image_elements = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "img.rg_i.Q4LuWd")))
except TimeoutException: # If images do not load within the specified time, print a timeout message and the page source for debugging
    print("Timed out waiting for images to load.") 
    print(driver.page_source)  
    driver.quit()
    exit()

# Get the first five image URLs
image_urls = []
for i in range(min(5, len(image_elements))):
    src = image_elements[i].get_attribute("src") # Get the 'src' attribute of the image element
    if not src: # If 'src' is not available, get 'data-src' attribute
        src = image_elements[i].get_attribute("data-src")
    image_urls.append(src) # Add the URL to the list

# Print the URLs of the first five images
for i, url in enumerate(image_urls):
    print(f"Image {i+1}: {url}")

# Close the WebDriver
driver.quit()
