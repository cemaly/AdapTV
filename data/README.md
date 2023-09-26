# Dataset

This folder contains the dataset for training models for icon classification. Due to GitHub's file size limitations, we have hosted the dataset on Dropbox.

## Accessing the Dataset

To access the dataset, follow these steps:

1. Click on the following Dropbox link to access the dataset folder:
   [Icon Classification Dataset on Dropbox](https://www.dropbox.com/sh/4a36ufkqt29tkvj/AADkO6ay4Nj971ye9FC03OgFa?dl=0)

2. Once you are on the Dropbox page, you can either download the original or grayscale data that are both zipped for easy download.

## Dataset Structure

The dataset is organized into the following structure:

- `original/`: This folder contains the original icon images.
  - `train/`: The training dataset.
    - `class1/`: Folder containing icon images for class 1.
    - `class2/`: Folder containing icon images for class 2.
    - ...
  - `test/`: The testing dataset.
    - `class1/`: Folder containing icon images for class 1.
    - `class2/`: Folder containing icon images for class 2.
    - ...

- `grayscale/`: This folder contains grayscale versions of the icon images.
  - `train/`: The training dataset.
    - `class1/`: Folder containing grayscale icon images for class 1.
    - `class2/`: Folder containing grayscale icon images for class 2.
    - ...
  - `test/`: The testing dataset.
    - `class1/`: Folder containing grayscale icon images for class 1.
    - `class2/`: Folder containing grayscale icon images for class 2.
    - ...

To create a validation set from the training dataset, you can use the [create_validation_set.py script](https://github.com/cemaly/AdapTV/blob/main/utils/create_validation_set.py):

---

**Note:** Reference the paper if you are using this dataset.

