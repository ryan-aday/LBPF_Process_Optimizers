import os
import cv2
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths to data
train_data_dir = 'path/to/training_images'
auxiliary_data_dir = 'path/to/auxiliary_data'  # JSON files
image_size = (128, 128)
batch_size = 32

# Helper function to load auxiliary data from JSON files
def load_auxiliary_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    # Process JSON data to match your needs, converting to numpy array
    return np.array([data.get(key, 0) for key in sorted(data.keys())])

# Data generator for image and auxiliary data
class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_dir, auxiliary_dir, batch_size, target_size, shuffle=True):
        self.image_dir = image_dir
        self.auxiliary_dir = auxiliary_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.image_filenames = os.listdir(image_dir)
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.floor(len(self.image_filenames) / self.batch_size))
    
    def __getitem__(self, index):
        batch_filenames = self.image_filenames[index*self.batch_size:(index+1)*self.batch_size]
        images, auxiliary_data, labels = [], [], []
        
        for filename in batch_filenames:
            # Load image
            img_path = os.path.join(self.image_dir, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, self.target_size)
            img = img / 255.0
            images.append(img)
            
            # Load auxiliary data
            aux_file = filename.replace('.png', '.json')  # assuming matching JSON filename
            aux_path = os.path.join(self.auxiliary_dir, aux_file)
            if os.path.exists(aux_path):
                auxiliary_data.append(load_auxiliary_data(aux_path))
            else:
                auxiliary_data.append(np.zeros(10))  # Default if JSON not found
            
            # Dummy labels - replace with actual labels
            label = 1 if 'good' in filename else 0  # Replace this with actual logic
            labels.append(label)
        
        return [np.array(images), np.array(auxiliary_data)], np.array(labels)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.image_filenames)

# Model for image and auxiliary data
def build_model(image_shape, aux_shape):
    # Image processing branch
    image_input = Input(shape=image_shape)
    x = Conv2D(32, (3, 3), activation='relu')(image_input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    
    # Auxiliary data branch
    aux_input = Input(shape=aux_shape)
    y = Dense(64, activation='relu')(aux_input)
    y = Dropout(0.5)(y)
    
    # Combine branches
    combined = Concatenate()([x, y])
    z = Dense(128, activation='relu')(combined)
    z = Dropout(0.5)(z)
    output = Dense(1, activation='sigmoid')(z)
    
    # Define the model
    model = Model(inputs=[image_input, aux_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Set up data generators
train_generator = CustomDataGenerator(
    image_dir=train_data_dir,
    auxiliary_dir=auxiliary_data_dir,
    batch_size=batch_size,
    target_size=image_size,
    shuffle=True
)

# Build and train the model
image_shape = (image_size[0], image_size[1], 3)
aux_shape = (10,)  # Adjust this based on auxiliary data dimensions
model = build_model(image_shape, aux_shape)

# Train the model
epochs = 10
model.fit(
    train_generator,
    epochs=epochs
)

# Save the model
model.save('image_aux_model.h5')
