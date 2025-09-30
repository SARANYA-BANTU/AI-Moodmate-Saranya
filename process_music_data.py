import pandas as pd

# This dictionary creates our "emotion-music label mapping"
# We are linking emotions to specific music genres found in the new dataset.
emotion_genre_map = {
    'happy': ['dance pop', 'pop', 'latin', 'canadian hip hop'],
    'sad': ['acoustic', 'sad', 'piano'],
    'calm': ['chill', 'r&b', 'trap'],
    'angry': ['rock', 'metal', 'electropop']
}

print("Loading the new music dataset (top50.csv)...")
# Load the new dataset from the CSV file
# We use 'latin1' encoding because of special characters in the data.
df = pd.read_csv('top50.csv', encoding='latin1')

# Rename columns to be easier to work with (the original names have dots)
df.rename(columns={'Track.Name': 'song_name', 'Artist.Name': 'artist', 'Genre': 'genre'}, inplace=True)

print("Filtering and organizing music features...")
# Create an empty list to hold our cleaned data
processed_songs = []

# Loop through each emotion in our map
for emotion, genres in emotion_genre_map.items():
    # Find all songs that belong to the genres for the current emotion
    for genre in genres:
        # Find rows where the 'genre' column matches exactly
        emotion_df = df[df['genre'] == genre]
        
        # For each song found, grab the important details
        for index, row in emotion_df.iterrows():
            processed_songs.append({
                'emotion': emotion,
                'artist': row['artist'],
                'song_name': row['song_name'],
                'genre': row['genre']
            })

# Convert our list of songs into a new pandas DataFrame
final_df = pd.DataFrame(processed_songs)

# Remove any duplicate songs
final_df.drop_duplicates(subset=['artist', 'song_name'], inplace=True)

# Save the clean, mapped data to the same output file
output_filename = 'cleaned_emotion_music_data.csv'
final_df.to_csv(output_filename, index=False)

print(f"Success! Preprocessed music data saved to {output_filename}")
print("Here is a sample of your cleaned data:")
print(final_df.head())