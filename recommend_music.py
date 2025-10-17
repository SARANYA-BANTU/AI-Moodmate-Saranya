# recommend_music.py
import pandas as pd

# Load cleaned music dataset
music_df = pd.read_csv('cleaned_emotion_music_data.csv')

def recommend_songs(emotion):
    """
    Recommends songs based on detected emotion.
    """
    emotion = emotion.lower()
    emotion_map = {
        'happy': 'Happy',
        'sad': 'Sad',
        'angry': 'Energetic',
        'fear': 'Calm',
        'surprise': 'Excited',
        'disgust': 'Relaxed',
        'neutral': 'Chill'
    }

    mood = emotion_map.get(emotion, 'Calm')
    recommended = music_df[music_df['mood'].str.contains(mood, case=False, na=False)]

    if recommended.empty:
        return "No songs found for this emotion."
    else:
        return recommended[['track_name', 'artist_name']].head(5)
