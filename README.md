
 # Emotion Detection System with OpenCV and FER
Overview
This project implements an Emotion Detection System using OpenCV for face detection and the FER (Facial Expression Recognition) library for emotion analysis. It captures video from the webcam, detects faces, and classifies emotions in real time. The detected emotions are logged, and statistics are displayed during the session.

Features
Face Detection: Detects faces using Haar Cascade Classifier.
Emotion Recognition: Identifies emotions like happiness, sadness, anger, surprise, and more using the FER library.
Live Statistics: Displays real-time emotion statistics on the video feed.
Logging: Logs detected emotions with timestamps to a text file.

Technologies Used
OpenCV: For video capture, face detection, and visualization.
FER Library: For emotion recognition.
Python: Backend for controlling the camera, performing detections, and managing the system logic.

Project Components
Haar Cascade Classifier: Pre-trained classifier for face detection.
FER (Facial Expression Recognition): Deep learning-based emotion detection.
Webcam Video Feed: Real-time video input for processing.

Setup Instructions
Prerequisites
Install the required Python libraries:
bash
Copy code
pip install opencv-python opencv-python-headless fer
Running the Project
Clone the repository and navigate to the project directory.

Run the Python script:

bash
Copy code
python emotion_detection.py
The system will try to access your webcam and start detecting faces and emotions. Emotions detected will be displayed on the screen, along with statistics.

To stop the program, press the 'q' key.

Code Explanation
Face Detection: Uses OpenCV's Haar Cascade Classifier to detect faces.
Emotion Detection: For each detected face, FER's deep learning model predicts the dominant emotion.
Cooldown Mechanism: Ensures that emotion detection is performed periodically (every 1 second) to avoid redundant detections.
Emotion Logging: All detected emotions, along with their scores, are logged to emotion_log.txt.
Statistics Display: Real-time emotion counts are displayed on the video feed.
