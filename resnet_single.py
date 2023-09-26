import json
import numpy as np
import sys
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import resnet_v2

# Constants
width, height = 80, 80
target_size = (height, width)
num_channels = 3
input_shape = (height, width, num_channels)
model_name = 'resnet_20_51_80.h5'  # Name generated as per the training script


def load_class_indices():
    with open('trained_models/class_indices.json', 'r') as f:
        class_indices = json.load(f)

    # Swap key-value pairs to get mapping from index to class name
    index_to_class = {v: k for k, v in class_indices.items()}
    return index_to_class

def preprocess_image(img_path):
    # Load the image and resize it
    img = load_img(img_path, target_size=target_size, color_mode="grayscale")

    # Convert image to array
    img_array = img_to_array(img)

    # Convert grayscale to 3-channel by repeating the same grayscale values
    img_array = np.repeat(img_array, 3, axis=-1)  # Use numpy's repeat instead of stack

    # Apply ResNet preprocessing
    img_array = resnet_v2.preprocess_input(img_array)

    # Expand dimensions to make it a single sample
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def predict_image_class(model, img_array):
    # Predict class probabilities
    predictions = model.predict(img_array)

    # Get the class with the maximum probability
    predicted_class = np.argmax(predictions[0])

    return predicted_class


def main():
    # Check if the image path is provided as a command-line argument
    if len(sys.argv) < 2:
        print("Error: Please provide an image path as a command-line argument.")
        sys.exit(1)

    # Load the model
    model = tf.keras.models.load_model('trained_models/' + model_name)

    # Get image path from command line arguments
    img_path = sys.argv[1]

    # Preprocess the image
    img_array = preprocess_image(img_path)

    # Predict class
    pred_class = predict_image_class(model, img_array)

    # Load class index to class name mapping
    index_to_class = load_class_indices()

    print(f"Predicted Class for {img_path}: {index_to_class[pred_class]}")


if __name__ == "__main__":
    main()
