# Brain Tumor Detection Using CNN

## Overview
This project focuses on detecting brain tumors using Convolutional Neural Networks (CNN). The model is trained using deep learning architectures such as **VGG16, InceptionV3, and ResNet50** to classify MRI images into categories of tumor and non-tumor. The aim is to provide an efficient and accurate diagnosis system that can assist medical professionals.

## Features
- **Deep Learning-Based Detection**: Uses CNN architectures for tumor classification.
- **Multiple Pretrained Models**: Implements VGG16, InceptionV3, and EfficientNetB0.
- **Dataset Handling**: Preprocessing, augmentation, and splitting of MRI datasets.
- **Performance Evaluation**: Accuracy, precision, recall, and confusion matrix analysis.
- **User-Friendly Implementation**: Easy-to-follow Jupyter Notebook with detailed explanations.

## Dataset
The dataset consists of MRI images labeled as **tumor** and **non-tumor**. It is essential to preprocess the images before feeding them into the CNN models. The preprocessing steps include:
- Image resizing
- Normalization
- Data augmentation (flipping, rotation, scaling, etc.)
- Splitting into training and testing sets

## Model Architectures
The following CNN architectures are implemented:

### 1. **VGG16**
- A 16-layer deep CNN known for its simplicity and efficiency.
- Pretrained on ImageNet and fine-tuned for brain tumor classification.

### 2. **InceptionV3**
- A deeper CNN with inception modules, reducing computational cost.
- Enhances feature extraction through multiple filter sizes.

### 3. **EfficientNetB0**
- A lightweight yet powerful CNN designed for optimal performance.
- Balances depth, width, and resolution efficiently.

## Dependencies
Ensure you have the following Python libraries installed:
- TensorFlow/Keras
- NumPy
- OpenCV
- Matplotlib
- Scikit-learn

## Model Training & Evaluation
### 1. **Train the Model**
- Load the dataset.
- Perform preprocessing.
- Train different CNN architectures.

### 2. **Evaluate Performance**
- Calculate accuracy, loss, and confusion matrix.
- Visualize predictions and misclassified cases.

## Results
The trained models achieve high accuracy in classifying brain tumor images. Comparative performance metrics for different CNN architectures are analyzed in the results section.

## Future Improvements
- Integrating additional deep learning models.
- Deploying as a web application for real-time detection.
- Improving dataset size and diversity.

## Contributors
- [Pranav Karwa](https://github.com/PranavKarwa2004) - Project Lead

## License
This project is licensed under the MIT License.

## Acknowledgments
- Kaggle/Dataset Providers
- TensorFlow/Keras community for pre-trained models
- Open-source contributors

---
Feel free to contribute by submitting pull requests or reporting issues!

