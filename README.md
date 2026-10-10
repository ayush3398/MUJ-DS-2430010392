# MUJ-DS-2430010392
Your instructor requires these details:
- Name: Ayush Pandey
- Registration Number: 2430010392
- Branch: BTECH - CSE (DATA SCIENCE)
- Batch: D
- Project Title: Smart Waste Classification Using Convolutional Neural Networks
- GitHub Username: ayush3398
- Training Program:

- About This Repository

This repository contains my individual work for the Data Science training program, including assignments, Python implementations, notebooks, learning resources, presentations, and project documentation.

The repository will be updated regularly as I progress through the training program.

Repository Structure

MUJ-DS-2430010392/
├── README.md
├── assignments/
├── notebooks/
├── code/
│   └── waste_classification/
├── resources/
├── presentations/
└── capstone/

Featured Project: Smart Waste Classification Using CNN

Overview

This project uses a Convolutional Neural Network (CNN) to classify waste images into five categories: cardboard, glass, metal, paper, and plastic.

The project explores image preprocessing, CNN architecture, model training, regularization, evaluation metrics, and comparisons with traditional machine learning and neural network baselines.

Dataset

Dataset: TrashNet

Source: TrashNet GitHub repository

Classes: Cardboard, Glass, Metal, Paper, Plastic

Images used: 2,387 after removing three exact duplicates

Input size: 128 × 128 RGB images

Data split: 70% training, 15% validation, 15% testing

The experiment uses five selected classes and excludes the original dataset's trash class.

Model Architecture

The custom CNN consists of:

Convolutional layer with 32 filters

ReLU activation and max pooling

Convolutional layer with 64 filters

ReLU activation and max pooling

Convolutional layer with 128 filters

ReLU activation and max pooling

Flatten layer

Fully connected layer with 128 neurons

ReLU activation and dropout

Five-class output layer

The model has approximately 4.29 million trainable parameters.

Technologies Used

Python

PyTorch

Torchvision

NumPy

Pandas

Scikit-learn

Matplotlib

Scikit-image

Experimental Results

The current baseline experiments produced the following test-set results:

Model

Accuracy

Macro Precision

Macro Recall

Macro F1

MLP on flattened pixels

53.20%

54.68%

53.48%

53.23%

HOG + SVM

62.12%

62.59%

60.91%

61.29%

Custom CNN with augmentation and dropout

71.03%

71.56%

70.05%

70.45%

These are the actual results from the current experiment, not target or estimated values. Future experiments and improvements will be documented separately after they have been run and evaluated.

Project Files

File

Purpose

config.py

Configuration and hyperparameters

download_data.py

Dataset setup

dataset_analysis.py

Dataset statistics and analysis

preprocessing.py

Image preprocessing and data loading

model.py

CNN architecture

train.py

Model training, validation, and training curves

evaluate.py

Test evaluation and classification metrics

compare_models.py

Comparison of implemented models

hog_svm.py

HOG feature extraction and SVM baseline

predict.py

Prediction on an input image

report.py

Results reporting

requirements.txt

Python dependencies

Setup and Execution

1. Clone the repository

git clone https://github.com/ayush3398/MUJ-DS-2430010392.git
cd MUJ-DS-2430010392/code/waste_classification

2. Install the dependencies

python -m pip install -r requirements.txt

3. Prepare the dataset

Follow the dataset setup instructions in download_data.py and the project documentation.

4. Train the model

python train.py

To run an individual experiment:

python train.py --exp baseline --epochs 10

5. Evaluate the trained models

python evaluate.py

Refer to the project scripts and configuration for the available experiment options and expected dataset paths.

Learning Outcomes

This project provides practical experience with:

Dataset collection and analysis

Image preprocessing and augmentation

Convolutional neural networks

Forward propagation and backpropagation

Loss functions and Adam optimization

Dropout and overfitting analysis

Training and validation curves

Classification metrics and confusion matrices

Comparison of deep learning and traditional machine learning models

Reproducible experimentation and Git-based version control

Training Progress

This repository will include:

Weekly assignments and coding exercises

Jupyter/Colab notebooks

Project implementations and evaluation results

Learning resources and presentations

Documentation of individual contributions

Relevant issues, commits, and pull requests

Academic Integrity and Reproducibility

Project results are documented according to actual experiments. Model improvements, comparisons, and future work will be recorded as they are implemented and evaluated.

License

Unless a license is added, all rights to the original code remain with their respective owners. The TrashNet dataset is maintained by its original contributors; consult its repository for dataset attribution and usage terms.
