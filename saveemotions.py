import cv2
import time
from fer import FER

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the FER emotion detector
emotion_detector = FER(mtcnn=True)

# Try different camera indices (0, 1, 2, ...) and different backends (CAP_DSHOW, CAP_V4L2, etc.)
cap = None
for i in range(5):
    print(f"Trying camera index {i}")
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        print(f"Camera index {i} opened successfully")
        break
    else:
        print(f"Camera index {i} failed to open")

if not cap or not cap.isOpened():
    print("Error: Could not open any camera.")
    exit()

# Cooldown period in seconds for emotion detection
cooldown_period = 1
last_detection_time = time.time()

# Emotion counts
emotion_counts = {
    'happy': 0,
    'sad': 0,
    'neutral': 0,
    'angry': 0,
    'surprise': 0,
    'fear': 0
}

# Open the file in append mode
with open("emotion_log.txt", "a") as log_file:
    try:
        running = True
        while running:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not capture frame.")
                break

            # Perform face detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Draw rectangles around detected faces and perform emotion detection if cooldown period has passed
            current_time = time.time()
            if current_time - last_detection_time >= cooldown_period:
                for (x, y, w, h) in faces:
                    face = frame[y:y+h, x:x+w]
                    emotion, score = emotion_detector.top_emotion(face)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    if emotion:
                        text = f"{emotion}: {score:.2f}"
                        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                        print(f"Detected emotion: {text}")
                        # Log the detected emotion to the file
                        log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {text}\n")
                        # Increment the emotion count
                        if emotion in emotion_counts:
                            emotion_counts[emotion] += 1
                last_detection_time = current_time

            # Display the frame with face detection and emotion results
            stats_text = (
                f"Emotion Statistics:\n"
                f"Happy: {emotion_counts['happy']}, Sad: {emotion_counts['sad']}, Angry: {emotion_counts['angry']}, "
                f"Neutral: {emotion_counts['neutral']}, Surprise: {emotion_counts['surprise']}, Fear: {emotion_counts['fear']}\n"
            )
            cv2.putText(frame, stats_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.namedWindow('Emotion Detection', cv2.WINDOW_NORMAL)  # Create a resizable window
            cv2.imshow('Emotion Detection', frame)

            # Check for quit key press
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            #running = False
            if cv2.waitKey(1) & 0xFF == ord('q'):
              break
            
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Release the camera and close all OpenCV windows
        cap.release()
        cv2.destroyAllWindows()
