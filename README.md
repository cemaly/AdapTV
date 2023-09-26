# AdapTV+: Enhancing Model-Based Test Adaptation for Smart TVs through Icon Recognition

## Overview

This repository is dedicated to the development of a dataset and a classifier for icon classification in the context of smart TV UI testing. The project originated as part of the AdapTV research work but has been separated into a standalone repository for clarity and accessibility.

### Background

In the fast-evolving world of smart TVs, the need for effective testing methodologies is paramount. The AdapTV project, in collaboration with Arcelik, Europe's fourth-largest home appliance manufacturer, has been exploring model-based testing approaches tailored for smart TVs. The main goal of this project is to address the challenges associated with test adaptation in the smart TV environment.

## Project Details

- **Integration with AdapTV**: This project was an extension of and integrated with AdapTV research but has been isolated into a standalone repository to provide a clear and focused resource for UI icon classification.

- **Project Aim**: It's essential to clarify that the primary aim of this project is not to develop new machine learning algorithms or enhance existing ones. Instead, the focus is on utilizing machine learning to address a practical and critical issue in the context of smart TV testing. Therefore, the approach prioritizes practicality and efficiency without introducing unnecessary computational complexity that could potentially hinder the proposed solution.

## Repository Structure

The repository is organized as follows:

- `data/`: Contains the dataset used for training the icon classification models.

- `images/`: Contains images that were used in the paper.

- `model_tests/`: Provides scripts for testing the top 2 pre-trained models.

- `model_trainings/`: Contains scripts for training the top-2-performing models.

- `scraper/`: Contains scripts for finding and downloading images from Google and Iconfinder.

- `trained_models/`: Houses pre-trained machine learning models for icon classification.

- `utils/`: Includes useful scripts and utilities for various tasks.

- `resnet_single.py`: A script located in the root directory that can be used to classify a given image passed as a command-line argument using the best-performing model.

## Usage

### Icon Classification

- The pre-trained models in the `trained_models/` folder can be used for icon classification tasks.

- To test the best-performing model, you can provide the path of an icon to be predicted as a command-line argument to the [resnet_single.py script](https://github.com/cemaly/AdapTV/blob/main/resnet_single.py).

## References

Please cite the following papers when referencing this work:

1. [*AdapTV+: Enhancing Model-Based Test Adaptation for Smart TVs through Icon Recognition*.](#)

2. [M. Y. Azimi, C. C. Elgun, A. Firat, F. Erata and C. Yilmaz, "AdapTV: A Model-Based Test Adaptation Approach for End-to-End User Interface Testing of Smart TVs," in IEEE Access, vol. 11, pp. 32095-32118, 2023, doi: 10.1109/ACCESS.2023.3262746.](https://ieeexplore.ieee.org/document/10083126)

3. [A. Fırat, M. Y. Azimi, C. Ç. Elgün, F. Erata and C. Yılmaz, "Model-Based Test Adaptation for Smart TVs," 2022 IEEE/ACM International Conference on Automation of Software Test (AST), Pittsburgh, PA, USA, 2022, pp. 52-53, doi: 10.1145/3524481.3527237.](https://ieeexplore.ieee.org/document/9796444)

---

For any inquiries, questions, or contributions, please feel free to open an issue or reach out to the project maintainers.
