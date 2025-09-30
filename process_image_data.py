import os
import shutil

# Define the path to the downloaded training data
# This assumes the 'train' folder is inside your 'MoodMate' project folder
source_train_folder = 'train' 

# This is where we will save our samples
output_sample_folder = 'fer2013_samples'

# The emotions are now the names of the subfolders (e.g., 'happy', 'sad')
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# How many sample images to copy for each emotion
max_samples_per_emotion = 5

print("Checking for dataset folders...")

if not os.path.isdir(source_train_folder):
    print(f"Error: The '{source_train_folder}' folder was not found.")
    print("Please make sure you have extracted the dataset and the 'train' folder is in your project directory.")
else:
    # Create the main folder to store our samples
    os.makedirs(output_sample_folder, exist_ok=True)
    print(f"Created sample folder: '{output_sample_folder}'")

    # Loop through each emotion
    for emotion in emotions:
        print(f"Processing '{emotion}' images...")

        # Create a subfolder for this emotion in our samples directory
        emotion_output_path = os.path.join(output_sample_folder, emotion)
        os.makedirs(emotion_output_path, exist_ok=True)

        # Define the source folder for this emotion's images
        emotion_source_path = os.path.join(source_train_folder, emotion)

        # Get a list of all the image filenames for this emotion
        image_files = os.listdir(emotion_source_path)

        # Copy the first few images as samples
        for i in range(min(len(image_files), max_samples_per_emotion)):
            source_file = os.path.join(emotion_source_path, image_files[i])
            destination_file = os.path.join(emotion_output_path, image_files[i])
            shutil.copyfile(source_file, destination_file)

    print("\nSuccess! Copied sample images to the 'fer2013_samples' folder.")