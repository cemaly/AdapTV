import os

# Path to the folder containing the training data
data_dir = 'data/new/train'
total = 0

# Go through each subfolder in the training data
for subfolder in os.listdir(data_dir):
    # Get the number of images in the current subfolder
    num_current = len(os.listdir(os.path.join(data_dir, subfolder)))

    # Add the number of images in the current subfolder to the total count
    total += num_current

    # Print the class name (subfolder name) and the number of images in that class
    print(subfolder, ": ", num_current)

# Print the total number of images across all classes
print("Total Num: ", total)
