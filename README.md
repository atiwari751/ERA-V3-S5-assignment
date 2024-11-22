# MNIST Classification with CI/CD Pipeline

This project implements a lightweight CNN model for MNIST digit classification with a complete CI/CD pipeline using GitHub Actions.

## Model Architecture

The model is a compact CNN with the following features:
- 2 Convolutional layers with BatchNormalization
- 2 MaxPooling layers
- 2 Fully connected layers
- Dropout for regularization
- Total parameters: < 25,000

Architecture details:
- Conv1: 1 → 4 channels (3x3 kernel)
- Conv2: 4 → 8 channels (3x3 kernel)
- FC1: 392 → 32 neurons
- FC2: 32 → 10 neurons (output)

## Project Structure
