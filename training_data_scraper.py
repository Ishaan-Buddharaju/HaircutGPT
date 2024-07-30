from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager

# Set up the Chrome WebDriver
options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Perform a Google image search with request query
search_query = "Zendaya"
driver.get(f"https://www.google.com/search?q={search_query}&tbm=isch")

# Wait for images to load
wait = WebDriverWait(driver, 20)
try:
    image_elements = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "img.rg_i.Q4LuWd")))
except TimeoutException:
    print("Timed out waiting for images to load.") 
    print(driver.page_source)  # Print page source for debugging
    driver.quit()
    exit()

# Get the first five image URLs
image_urls = []
for i in range(min(5, len(image_elements))):
    src = image_elements[i].get_attribute("src")
    if not src:
        src = image_elements[i].get_attribute("data-src")
    image_urls.append(src)

# Print the URLs
for i, url in enumerate(image_urls):
    print(f"Image {i+1}: {url}")

# Close the WebDriver
driver.quit()
