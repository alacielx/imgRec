import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.preprocessing import image
from matplotlib import pyplot as plt
import os
import fitz
import shutil

def preprocess_image(image):
    # Resize the image to match the input dimensions of your model
    resized_img = cv2.resize(image, (256, 256))
    
    # Normalize the pixel values to the range [0, 1]
    normalized_img = resized_img / 255.0
    
    return normalized_img

def recognizeImages():
    # Directory containing the images to be predicted
    prediction_directory = 'trainingData/doorPanel/test'

    # List the image filenames
    image_filenames = os.listdir(prediction_directory)

    # Create a directory to store predicted images
    predicted_directory = prediction_directory
    os.makedirs(predicted_directory, exist_ok=True)
    for filename in image_filenames:
        image_path = os.path.join(prediction_directory, filename)
        img = cv2.imread(image_path)  # Load the image using OpenCV or any other library
        
        # Preprocess the image (resize, normalize, etc.)
        preprocessed_img = preprocess_image(img)
        
        # Expand dimensions to match model input shape
        preprocessed_img = np.expand_dims(preprocessed_img, axis=0)
        
        # Make predictions
        predictions = model.predict(preprocessed_img)
        
        # Get the predicted class index
        predicted_class_index = np.argmax(predictions)
        
        # Get the predicted class name
        predicted_class_name = class_names[predicted_class_index]
        
        # Create a subdirectory for the predicted class
        predicted_class_directory = os.path.join(predicted_directory, predicted_class_name)
        os.makedirs(predicted_class_directory, exist_ok=True)
        
        # Move the image to the predicted class directory
        shutil.move(image_path, os.path.join(predicted_class_directory, filename))

def trainData(imageDirectory, saveDirectory, imageSize=(256, 256)):
    data: tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(imageDirectory,image_size=imageSize)

    data = data.map(lambda x, y: (x / 255, y))



    batch = data.as_numpy_iterator().next()

    image_height, image_width, num_channels = batch[0].shape[1:]

    class_names = os.listdir(imageDirectory)

    # Now you can use len(class_names) to get the number of classes
    num_classes = len(class_names)

    model = keras.Sequential([
        layers.Input(shape=(image_height, image_width, num_channels)),  # Input layer
        
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),  # Convolutional layer
        layers.MaxPooling2D(pool_size=(2, 2)),  # Pooling layer
        
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        layers.Flatten(),  # Flatten the 2D feature maps to a 1D vector
        
        layers.Dense(128, activation='relu'),  # Fully connected layer
        layers.Dense(num_classes, activation='softmax')  # Output layer
    ])

    model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # Use 'sparse_categorical_crossentropy' if your labels are integers
    metrics=['accuracy']
    )

    model.summary()

    # Split dataset into 80% training and 20% validation
    train_size = int(0.8 * len(data))
    val_size = len(data) - train_size

    train_data = data.take(train_size)
    val_data = data.skip(train_size)

    epochs = 20  # Start with a small number of epochs
    batch_size = 8  # Use a small batch size

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
    )

    # Plot training and validation accuracy
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()

    # Plot training and validation loss
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim([0, 1])
    plt.legend(loc='upper right')
    plt.show()

    
    model.save(saveDirectory)

    test_directory = 'trainingData/doorPanel/test'

    test_data = tf.keras.utils.image_dataset_from_directory(
        test_directory,
        batch_size=8,  # Use the same batch size as in training
        image_size=imageSize,  # Specify image size
        shuffle=False  # No need to shuffle for evaluation
    )

    test_data = test_data.map(lambda x, y: (x / 255, y))  # Normalize pixel values

    test_loss, test_acc = model.evaluate(test_data, verbose=2)
    print('\nTest accuracy:', test_acc)

    return model
    
def predict_image_class(image_array, model):
    img_array = image_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])  # Get the index of the predicted class

    return predicted_class

# Function to classify images within a PDF using fitz
def classify_pdf_page(pdf_path, model, page_number=0):
    doc = fitz.open(pdf_path)
    
    if page_number < doc.page_count:
        page = doc[page_number]
        image = page.get_pixmap()
        
        image

        for img_index, img in enumerate(image_list):
            img_array = np.frombuffer(img.get_pixmap_data(output='jpeg'), dtype=np.uint8)
            
            # Get image dimensions and channels from pixmap
            img_height = img.height
            img_width = img.width
            img_channels = 3  # Assuming RGB channels
            
            # Reshape the flat array into image shape
            img_array = img_array.reshape((img_height, img_width, img_channels))
            
            # Predict image class
            predicted_class = predict_image_class(img_array, model)
            print(f"Page {page_number + 1}, Image {img_index + 1} - Predicted class:", predicted_class)
    
    doc.close()

######################################################################################################

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

imageDirectory = r'C:\Users\agarza\OneDrive - Arrow Glass Industries\Documents\Scripts\Test\imgRec\trainingData\doorPanel\train'

# data: tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(imageDirectory, labels='inferred', label_mode='int', validation_split=0.2, subset='training', seed=1337, image_size=(img_height, img_width), batch_size=batch_size, color_mode='rgb', shuffle=True, seed=1337, interpolation='bilinear')

imageSize = (256, 256)

modelDir = r'C:\Users\agarza\OneDrive - Arrow Glass Industries\Documents\Scripts\Test\imgRec\doorPanelModel'

try:
    model = keras.models.load_model(modelDir)
except:
    print("Could not load model")

# Train new data
if False:
    model= trainData(imageDirectory,imageSize)

pdfPath = r"C:\Users\agarza\OneDrive - Arrow Glass Industries\Documents\Scripts\Test\imgRec\trainingData\doorPanel\test\input"

for file in os.listdir(pdfPath):
    if os.path.splitext(file)[1] == ".pdf":
        file = os.path.join(pdfPath, file)
        classify_pdf_page(file, model)






