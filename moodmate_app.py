# moodmate_app.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from recommend_music import recommend_songs

# Load the trained model
model = load_model('emotion_model.h5')

# Emotion labels (as per FER-2013 dataset)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def detect_emotion(image_path):
    face_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    face_img = cv2.resize(face_img, (48, 48))
    face_img = face_img.reshape(1, 48, 48, 1) / 255.0
    prediction = model.predict(face_img)
    emotion = emotion_labels[np.argmax(prediction)]
    return emotion

if __name__ == "__main__":
    img_path = input("Enter image path: ")
    emotion = detect_emotion(img_path)
    print(f"\nDetected Emotion: {emotion}\n")
    print("🎵 Recommended Songs:")
    print(recommend_songs(emotion))
