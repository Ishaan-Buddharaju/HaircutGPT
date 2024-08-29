import openai
import os
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY2') 

'''
The function recommends a haircut using ChatGPT given the 
face shape identified by the algorithm
'''
def recommendHaircut(faceShape):

    #Below is the instructions for our ChatGPT Assistant
    message = [{"role": "system", "content": "You are a helpful haircut recommendation assistant." +  
             " You take in a user's face shape and gender and recommend 3 haircuts in the form of list." + 
             " Only write the name of the haircut followed by a short description of the haircut." + 
             " The list should be in the following form" + 
             " 1. Recommendation 1: Description 2. Recommendation 2: Description 3. Recommendation 3: Description"}]

    #append the faceShape into the the list to be passed into GPT
    faceshapeModelInput = message.append({"role": "user", "content": faceShape})
    chat = openai.chat.completions.create(model = 'gpt-3.5-turbo', messages = faceshapeModelInput)
    reply = chat.choices[0].message.content
    return reply

'''
Once we get a recommended haircut we plug it into this function
with a predetermined prompt to generate an image using DALLE-3.

The function outputs the image in the form of a url.
'''

def generateExampleImage(haircutRecommendation):
    image = openai.images.generate(prompt = haircutRecommendation, n = 2, size = "1024x1024")
    return image.data[0].url

