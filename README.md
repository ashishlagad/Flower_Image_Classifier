# Flower_Image_Classifier

In this project, an image classifier is trained to identify various flower species. PyTorch, a well-known deep learning framework, is utilized to construct and train a neural network model. Initially, a pre-trained network (DenseNet121) is loaded, followed by defining a new, untrained feed-forward network as a classifier with ReLU activations and dropout. The classifier layers are trained through backpropagation using the pre-trained network to extract features. Tracking the loss and accuracy on the validation set helps in determining the best hyperparameters. Finally, the model is saved as a checkpoint for future use in making predictions.

## Dataset

The dataset used for this project is a subset of the [102 Category Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) created by Maria-Elena Nilsback and Andrew Zisserman. It contains 8189 images of flowers belonging to 102 categories. The images have various sizes and orientations, and some categories are more represented than others.

The dataset is divided into three folders: train, valid, and test. The train folder contains 6552 images for training the model. The valid folder contains 818 images for validating the model during training. The test folder contains 819 images for testing the model after training.

The dataset also includes a file called cat_to_name.json that maps each category code (from 1 to 102) to the corresponding flower name.

## Installation

To run this project, you need to have python 3 installed on your system. You also need to install the following packages:

- PIL
- matplotlib
- torch
- torchvision
- numpy
- json
- argparse

You can install these packages using pip or conda.

