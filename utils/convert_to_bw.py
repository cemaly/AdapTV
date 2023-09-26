import os
import cv2


# Path to the folder containing the training data
data_dir = 'data/new/test'

# Path to the folder where the converted images will be stored
gray_dir = 'data/bw/test'

# Go through each subfolder in the training data
for subfolder in os.listdir(data_dir):
    # Create the same subfolder in the validation data
    os.makedirs(os.path.join(gray_dir, subfolder), exist_ok=True)
    # Go through each image in the subfolder
    for img in os.listdir(os.path.join(data_dir, subfolder)):
        image = cv2.imread(os.path.join(data_dir, subfolder, img))
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply Otsu's thresholding method to the grayscale image
        ret, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Save the grayscale image
        cv2.imwrite(os.path.join(gray_dir, subfolder, img), bw)
