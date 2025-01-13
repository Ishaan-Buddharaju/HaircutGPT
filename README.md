# Face Shape Identifier

A Python-based tool that identifies the shape of a human face from an input image and provides recommendations for suitable hairstyles based on the detected face shape. The program also overlays visual measurements on the image to display the calculations.

---

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Outputs](#outputs)
- [Hairstyle Recommendations](#hairstyle-recommendations)

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
   pip3 install opencv-python
2. **Dlib**:  
   Dlib provides facial landmark detection using a pre-trained shape predictor model.  
   ```bash
   pip3 install dlib
3. **Scikit-learn**:  
   Scikit-learn is used for classification and clustering of face shapes  
   ```bash
   pip3 install scikit-learn
4. **NumPy**:  
   NumPy is used for mathematical calculations, including measurements of facial proportions.
   ```bash
   pip3 install numpy
4. **OpenAI API**:  
   This allows integration with the OpenAI GPT-4 API for hairstyle recommendations.
   ```bash
   pip3 install openai
5. **Python-dotenv**:  
   Python-dotenv is used to manage environment variables (e.g., API keys, file paths).
   ```bash
   pip3 install python-dotenv



   




