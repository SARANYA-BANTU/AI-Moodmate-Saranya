# train_emotion_model.py
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

base_dir = r"F:\riki\projects\MoodMate"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

# Image data generator for augmentation
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir, target_size=(48, 48), batch_size=64, color_mode='grayscale', class_mode='categorical'
)
test_data = test_datagen.flow_from_directory(
    test_dir, target_size=(48, 48), batch_size=64, color_mode='grayscale', class_mode='categorical'
)

# CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 emotion classes
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Save best model automatically
checkpoint = ModelCheckpoint('emotion_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)

# Train model
model.fit(train_data, validation_data=test_data, epochs=25, callbacks=[checkpoint])

print("✅ Model trained and saved as emotion_model.h5")
