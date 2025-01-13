# Face Shape Identifier

A Python-based tool that identifies the shape of a human face from an input image and provides recommendations for suitable hairstyles based on the detected face shape. The program also overlays visual measurements on the image to display the calculations.

---

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Functionality](#functionality)

---

## Features
- Detects facial landmarks (e.g., jawline, forehead, etc.) using a pre-trained Dlib model.
- Measures facial proportions and classifies face shapes into:
  - Round
  - Square
  - Oval
  - Triangle
  - Diamond
  - Rectangular
  - Oblong
- Overlays visual annotations on the image for clarity.
- Provides hairstyle recommendations tailored to face shape, gender, and desired hair length using OpenAI's GPT-4 model.

---
## Requirements

To use this, the following Python libraries must be installed. You can install all of them using the commands provided below:

1. **OpenCV**:  
   OpenCV is used for face detection and image annotation.  
   ```bash
   pip install opencv-python
2. **Dlib**:  
   Dlib provides facial landmark detection using a pre-trained shape predictor model.  
   ```bash
   pip install dlib
3. **Scikit-learn**:  
   Scikit-learn is used for classification and clustering of face shapes  
   ```bash
   pip install scikit-learn
4. **NumPy**:  
   NumPy is used for mathematical calculations, including measurements of facial proportions.
   ```bash
   pip install numpy
4. **OpenAI API**:  
   This allows integration with the OpenAI GPT-4 API for hairstyle recommendations.
   ```bash
   pip install openai
5. **Python-dotenv**:  
   Python-dotenv is used to manage environment variables (e.g., API keys, file paths).
   ```bash
   pip install python-dotenv

---
## Functionality

### Face Shape:  
Face Shape is identified with drawn facial landmarks on the user's face that use face ratios to identify the face shape.  

<img width="549" alt="Screenshot 2025-01-13 at 11 10 42 AM" src="https://github.com/user-attachments/assets/83b051b9-23ff-4722-ac8e-d51862f06490" />

**Figure 1**: Preview Image of Face Shape Identifier on Actor Adam Driver. He Was Identified as a Triangle Face Shape.

### Hairstyle Recommendation:

**Once the face shape is identified, the tool provides personalized hairstyle recommendations based on:**  

Detected face shape.  

Desired hair length (short, medium, or long).  

Gender preferences.

*Example:* A triangle face shape is characterized by a narrow forehead and wider jawline. The goal is to balance these proportions by adding width and volume at the top while minimizing emphasis on the jawline. This means good hairstyles would include **Textured Layers**, **Side-Parted Styles**, or **Fringes with Volume** as these hairstyles soften the pronounced forehead area and adds balance to the face.





   




