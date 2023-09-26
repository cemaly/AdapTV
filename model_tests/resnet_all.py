import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import resnet_v2

# Constants
width, height = 80, 80
target_size = (height, width)
batch_size = 64  # same as training batch_size, but you can modify if needed

# Load the saved model
model_name = 'resnet_{}_{}_{}.h5'.format(20, 51, 80)  # replace the placeholders with your actual values
model = load_model('trained_models/' + model_name)

# Prepare test data using ImageDataGenerator
test_data_generator = ImageDataGenerator(preprocessing_function=resnet_v2.preprocess_input)

test_generator = test_data_generator.flow_from_directory(
    'data/grayscale/test',  # assuming test data is in this directory
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # No need to shuffle for evaluation
)

# Evaluate the model
loss, accuracy = model.evaluate(test_generator)

print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy*100:.2f}%")
