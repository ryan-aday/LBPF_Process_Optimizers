import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Paths to the image directories
train_data_dir = 'path/to/training_data'
test_data_dir = 'path/to/test_data'  # Optional, can split images from video for testing
image_size = (128, 128)  # Resize images to a fixed size
batch_size = 32

# Step 1: Set up the data generator for training and validation
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # 80/20 train-validation split
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# Step 2: Define a simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Step 3: Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 4: Train the model
epochs = 10
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=epochs
)

# Step 5: Save the model
model.save('print_classification_model.h5')

# Step 6: Load and classify new images
def classify_image(model, image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, image_size)
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    prediction = model.predict(img)[0][0]
    return "Good Print" if prediction > 0.5 else "Bad Print"

# Load the model and classify images
model = tf.keras.models.load_model('print_classification_model.h5')

# Example classification for a directory of images
image_directory = 'output_frames_folder'
for filename in os.listdir(image_directory):
    if filename.endswith('.png'):
        image_path = os.path.join(image_directory, filename)
        classification = classify_image(model, image_path)
        print(f"Image: {filename} - Classification: {classification}")
