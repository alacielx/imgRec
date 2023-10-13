import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os

# Define the path to your dataset directory
dataset_directory = r'C:\Users\agarza\OneDrive - Arrow Glass Industries\Documents\GitHub\imgRec\trainingData\doorType\output'

# Define image dimensions and batch size
image_height, image_width = 224, 224
batch_size = 32

# Create an ImageDataGenerator to load and preprocess data
data_generator = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to [0, 1]
    validation_split=0.2,  # Split data into training (80%) and validation (20%)
)

# Load the dataset using the ImageDataGenerator
train_data = data_generator.flow_from_directory(
    dataset_directory,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical',  # Use 'categorical' for multi-class classification
    subset='training',  # Specify 'training' to get the training data
)

validation_data = data_generator.flow_from_directory(
    dataset_directory,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',  # Specify 'validation' to get the validation data
)

# Define the model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(len(train_data.class_indices), activation='softmax')  # Output layer with the number of classes
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
epochs = 10  # Adjust the number of epochs as needed
history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=epochs,
)

# Save the trained model for later use
model.save('my_model.h5')

# Print training and validation accuracy
train_accuracy = history.history['accuracy'][-1]
val_accuracy = history.history['val_accuracy'][-1]
print(f'Training accuracy: {train_accuracy:.2f}')
print(f'Validation accuracy: {val_accuracy:.2f}')
