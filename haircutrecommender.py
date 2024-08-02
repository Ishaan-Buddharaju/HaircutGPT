import os
from dotenv import load_dotenv
import openai

def get_hairstyle_recommendations(face_shape, desired_hair_length, gender):
    # Load environment variables from .env file
    load_dotenv()
    api_key = os.getenv("HAIRCUT_API_KEY")
    openai.api_key = api_key
    
    # Construct the prompt for the model
    prompt = f"""
    {face_shape}, {desired_hair_length}, {gender}
    """
    
    # Making a request to the OpenAI API using the new interface
    response = openai.ChatCompletion.create(
        model="gpt-4",  # or use "gpt-3.5-turbo" depending on your access
        messages=[
            {"role": "system", "content": "You are a helpful assistant that takes in an input of a human face shape. With this information, you recommend 3 different hairstyles that fit the face shape. Along with each face shape you will provide a list of pros and cons that arise with each hairstyle."},
            {"role": "user", "content": prompt}
        ]
    )
    
    # Extract and return the response
    return response['choices'][0]['message']['content'].strip()

# Example usage
if __name__ == "__main__":
    face_shape = "Oblong"  # This would be inputted from model's analysis of face picture
    gender = "Male"
    desired_hair_length = "Long"
    recommendations = get_hairstyle_recommendations(face_shape, desired_hair_length, gender)
    print("Recommendations for face shape '{}':".format(face_shape))
    print(recommendations)
