import os
from PIL import Image



# Path to the folder containing the training data
data_dir = 'data/new/test'

# Path to the folder where the converted images will be stored
gray_dir = 'data/grayscale/test'

# Go through each subfolder in the training data
for subfolder in os.listdir(data_dir):
    # Create the same subfolder in the validation data
    os.makedirs(os.path.join(gray_dir, subfolder), exist_ok=True)

    for img in os.listdir(os.path.join(data_dir, subfolder)):

        # Open the image
        image = Image.open(os.path.join(data_dir, subfolder, img))

        # Convert to grayscale
        gray_image = image.convert("L")

        # Save the grayscale image
        gray_image.save(os.path.join(gray_dir, subfolder, img))
