import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Define the path to the directory containing the image folders
data_dir = f"data/grayscale/train"

# Get a list of class folders
classes = os.listdir(data_dir)

# Collect image sizes
widths, heights = [], []
for c in classes:
    for image_file in os.listdir(os.path.join(data_dir, c)):
        image_path = os.path.join(data_dir, c, image_file)
        with Image.open(image_path) as img:
            width, height = img.size
            widths.append(width)
            heights.append(height)

# Create a scatter plot for image sizes
sns.set_style("white")
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=widths, y=heights, color="red", ax=ax, s=10)
ax.set(xlabel="Image Width", ylabel="Image Height")
ax.set_xlim(0, 500)
ax.set_ylim(0, 500)
ax.set_title("Distribution of Image Sizes", fontweight="bold", fontsize=14)
ax.tick_params(axis="x", labelsize=10)
ax.tick_params(axis="y", labelsize=10)
plt.tight_layout()
plt.show()
