# Multi-Party Computation for ASL Recognition

This project implements a privacy-preserving American Sign Language (ASL) recognition system using Multi-Party Computation (MPC). We provide multiple examples demonstrating MPC concepts, from basic to advanced implementations, leading up to our main ASL recognition system.

## Example 1: Basic Salary Computation with MPC

### Overview
The salary computation example (`MPCSalaryExample.py`) demonstrates fundamental MPC concepts through a simple yet practical scenario. It shows how multiple parties can compute an average salary without revealing individual salaries.

### Dependencies
- crypten
- torch

### Description
- Implements secure multi-party average computation
- Uses secret sharing for secure data distribution
- Demonstrates basic encryption/decryption workflow
- Shows how to perform basic arithmetic operations on encrypted data

### Running the Example
1. Install dependencies
2. Run `MPCSalaryExample.py`
3. The program will output:
   - Secure average computation
   - Verification result (for demonstration purposes)

## Example 2: Breast Cancer Classification with MPC

### Overview
`breastTumour.ipynb` demonstrates advanced MPC concepts by implementing secure machine learning on the Wisconsin Breast Cancer dataset. This example showcases both vertical and horizontal data partitioning approaches.

### Dependencies
- torch
- torchvision
- pandas
- scikit-learn
- matplotlib
- crypten (install with --no-deps flag)

### Key Features
1. Horizontal Data Partitioning:
   - Splits records across different parties
   - Each party has complete feature sets
   - Demonstrates secure model training across distributed data

2. Vertical Data Partitioning:
   - Divides features among parties
   - Each party holds different attributes for all records
   - Shows secure feature aggregation and model training

### Technical Details
- Uses logistic regression for binary classification
- Implements secure gradient computation
- Demonstrates encrypted model parameter updates
- Achieves ~97% accuracy while maintaining data privacy
- Uses CrypTen for secure computations
- Includes precision handling and learning rate decay

## Main Project: ASL Recognition System

### Overview
Our main system combines deep learning with MediaPipe's hand tracking to create an accurate ASL recognition model, preparing for future MPC integration.

### Dependencies
Core Requirements:
- flatbuffers==2.0.0
- mediapipe
- torch
- torchvision
- torchaudio
- opencv-python
- numpy

### System Components

#### 1. Feature Generation (Mediapipe_dataset.py)
- Processes ASL images from Kaggle dataset
- Uses MediaPipe for keypoint extraction
- Generates synthetic features for training
- Outputs: synthetic_features_train.csv, synthetic_features_test.csv

#### 2. Model Training (train_dnn.ipynb)
- Input: synthetic_features_train.csv
- Implements DNN architecture:
  - Input layer: 63 dimensions (21 keypoints × 3 coordinates)
  - Three hidden layers with ReLU activation
  - Output layer: 24 classes (ASL letters)
- Outputs trained model file (.pth)

#### 3. Inference System (Model.ipynb)
- Loads trained DNN model
- Processes input images through MediaPipe
- Extracts hand keypoints
- Performs classification
- Visualizes results with OpenCV

### Using the ASL System

1. Feature Preparation:
   - Source ASL dataset from Kaggle [https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet]
   - Run feature extraction to generate training data
   - Verify feature files are correctly formatted (63 columns)

2. Model Training:
   - Use train_dnn.ipynb with synthetic features
   - Monitor training progress
   - Save the trained model

3. Running Inference:
   - Open Model.ipynb
   - Upload your trained model
   - Input an image containing an ASL gesture
   - System will:
     - Detect hand position
     - Extract keypoints
     - Classify the gesture
     - Display results

### Performance Notes
- Model achieves high accuracy on test set
- MediaPipe provides robust keypoint detection
- Real-time capable on standard hardware
- Handles various lighting conditions and backgrounds

## Future Work

- Fully encrypted MPC model training.
- Scalability improvements for larger datasets and additional parties.
- Integration of advanced cryptographic protocols.

### Troubleshooting
- Ensure correct MediaPipe installation
- Verify CUDA compatibility if using GPU or just use google colab
- Check image format and hand visibility
- Confirm model and feature file paths

### Support
For issues or questions:
- Open an issue on GitHub
- Check existing documentation
- Review MediaPipe and CrypTen documentation

