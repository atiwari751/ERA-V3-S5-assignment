# MNIST Classification with CI/CD Pipeline

![ML Pipeline](https://github.com/atiwari751/ERA-V3-S5-assignment/actions/workflows/ml-pipeline.yml/badge.svg)

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

  │
  ├── model/
  │ ├── init.py
  │ └── network.py # Model architecture definition
  │
  ├── tests/
  │ └── test_model.py # Test cases for model
  │
  ├── utils/
  │ └── visualize_augmentations.py # Data augmentation visualization
  │
  ├── .github/
  │ └── workflows/
  │ └── ml-pipeline.yml # CI/CD pipeline configuration
  │
  ├── train.py # Training script
  ├── requirements.txt # Project dependencies
  ├── README.md # Project documentation
  └── .gitignore # Git ignore rules


## Tests

The project includes three main tests:

1. **Parameter Count Test**
   - Ensures model has less than 25,000 parameters
   - Validates model's lightweight nature

2. **Input/Output Shape Test**
   - Verifies model accepts 28x28 images
   - Confirms output of 10 classes

3. **Accuracy Test**
   - Trains model if no trained model exists
   - Tests on MNIST test set
   - Requires >95% accuracy

## Expected Results

- Model Parameter Count: ~10,000 parameters
- Training Time: ~2-3 minutes on CPU
- Test Accuracy: >95% after one epoch
- Memory Usage: <100MB

## Running Locally

1. Create virtual environment:
bash
python -m venv venv
source venv/bin/activate # or venv\Scripts\activate on Windows


2. Install dependencies:
bash
pip install -r requirements.txt


3. Train model:
bash
python train.py


4. Run tests:
bash
python -m pytest tests/test_model.py -v


## CI/CD Pipeline

The GitHub Actions pipeline automatically:
1. Sets up Python environment
2. Installs dependencies
3. Trains the model
4. Runs all tests
5. Saves trained model as artifact

The pipeline ensures code quality and model performance with every push to the repository.