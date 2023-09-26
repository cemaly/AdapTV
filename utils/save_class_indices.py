import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import resnet_v2

# Define the directory paths
DATA_DIR = 'data/grayscale'
TRAIN_DIR = DATA_DIR + '/train'
CLASS_INDICES_PATH = 'trained_models/class_indices.json'

# Set the target image size
TARGET_SIZE = (200, 200)

# Initialize the image data generator with the preprocessing function specific to ResNet
train_datagen = ImageDataGenerator(preprocessing_function=resnet_v2.preprocess_input)

# Configure the training data generator to read images from the training directory
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=TARGET_SIZE,
    batch_size=32
)

# Save the class indices (mapping of class names to integers) to a JSON file for future use
with open(CLASS_INDICES_PATH, 'w') as f:
    json.dump(train_generator.class_indices, f)