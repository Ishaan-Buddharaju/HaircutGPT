import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv() 
api_key = os.getenv("HAIRCUT_API_KEY")
client = OpenAI(api_key=api_key)

def get_hairstyle_recommendations(face_shape, desired_hair_length, gender):

    prompt = f"""
    {face_shape}, {desired_hair_length}, {gender}
    """

    #OpenAI API request
    response = client.chat.completions.create(model="gpt-4",  # or use "gpt-3.5-turbo" depending on your access
    messages=[
        {"role": "system", "content": """You are a helpful assistant that takes in an input of a human face shape. 
             With this information, you recommend 3 different hairstyles that fit the face shape. 
             Along with each face shape you will provide a list of pros and cons that arise with each hairstyle."""},
        {"role": "user", "content": prompt}
    ])

    return response.choices[0].message.content.strip()

#example usage
if __name__ == "__main__":
    face_shape = "Round"  #this would be inputted from face picture analysis
    gender = "Female"
    desired_hair_length = "Long"
    recommendations = get_hairstyle_recommendations(face_shape, desired_hair_length, gender)
    print("Recommendations for face shape '{}':".format(face_shape))
    print(recommendations)
