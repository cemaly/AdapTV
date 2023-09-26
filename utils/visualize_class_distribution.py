import os
import matplotlib.pyplot as plt
import seaborn as sns

# Define the path to the directory containing the image folders
data_dir = f"data/grayscale/train"

# Create a list of the classes in the dataset by getting the folder names
classes = os.listdir(data_dir)

# Create a list of the number of images in each class folder
num_images = [len(os.listdir(os.path.join(data_dir, c))) for c in classes]

# Create a bar plot of the number of images in each class
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=classes, y=num_images, ax=ax, color="blue")
ax.set(xlabel="Class", ylabel="Number of Images")
ax.set_title("Distribution of Images in Dataset", fontweight="bold", fontsize=14)
ax.tick_params(axis="x", labelsize=10, labelrotation=90)
ax.tick_params(axis="y", labelsize=10)
plt.tight_layout()
plt.show()
