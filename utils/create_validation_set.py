import os
import random
from shutil import move

# Path to the folder containing the training data
training_data_dir = 'data/new/train'

# Path to the folder where the validation set will be stored
validation_data_dir = 'data/new/validation'

# Create the validation data directory if it doesn't exist
os.makedirs(validation_data_dir, exist_ok=True)

# Set the probability of an image being moved to the validation set (e.g., 20%)
validation_split_percentage = 20

# Go through each subfolder in the training data
for subfolder in os.listdir(training_data_dir):
    # Create the same subfolder in the validation data
    os.makedirs(os.path.join(validation_data_dir, subfolder), exist_ok=True)

    # Get a list of all the image files in the current subfolder
    images = os.listdir(os.path.join(training_data_dir, subfolder))

    # Calculate the number of images to move to the validation set based on the percentage
    num_images_to_move = int(len(images) * (validation_split_percentage / 100))

    # Randomly select images to move to the validation set
    images_to_move = random.sample(images, num_images_to_move)

    # Move the selected images to the validation set
    for img in images_to_move:
        source_path = os.path.join(training_data_dir, subfolder, img)
        destination_path = os.path.join(validation_data_dir, subfolder, img)
        move(source_path, destination_path)

print("Validation set creation completed.")
