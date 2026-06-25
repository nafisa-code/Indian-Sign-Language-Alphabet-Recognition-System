# Indian Sign Language (ISL) Alphabet Recognition System

## Overview

This project presents a real-time Indian Sign Language (ISL) Alphabet Recognition System developed using Deep Learning and Computer Vision techniques. The system captures static hand gestures through a webcam and predicts the corresponding ISL alphabet in real time.

The project uses a MobileNetV2-based Convolutional Neural Network (CNN) with Transfer Learning for gesture classification and MediaPipe for hand detection and Region of Interest (ROI) extraction.

The primary objective of this project is to help reduce communication barriers between hearing-impaired individuals and the general population by converting ISL alphabet gestures into readable text.

---

## Features

* Real-time ISL alphabet recognition (A–Z)
* Webcam-based gesture capture
* MediaPipe hand detection and ROI extraction
* MobileNetV2-based CNN classifier
* Transfer Learning using ImageNet weights
* Image preprocessing and enhancement
* Confidence score display
* Prediction logging and image storage
* Graphical User Interface (GUI) using CustomTkinter

---

## Technologies Used

### Programming Language

* Python

### Deep Learning Frameworks

* TensorFlow
* Keras

### Computer Vision

* OpenCV
* MediaPipe

### GUI Development

* CustomTkinter

### Data Processing

* NumPy

### Visualization

* Matplotlib

---

## System Workflow

1. Capture hand gesture through webcam
2. Detect hand using MediaPipe
3. Extract Region of Interest (ROI)
4. Apply image preprocessing
5. Classify gesture using MobileNetV2 CNN
6. Generate confidence score using Softmax
7. Display predicted alphabet
8. Store prediction logs

---

## Dataset

The dataset consists of Indian Sign Language alphabet gesture images (A–Z) collected from a publicly available Kaggle dataset.

Dataset Split:

* Training: 70%
* Validation: 20%
* Testing: 10%

---

## Image Preprocessing

The following preprocessing techniques were applied:

* Resize (224 × 224)
* Gaussian Blur
* Grayscale Conversion
* CLAHE Enhancement
* Normalization
* MobileNetV2 Preprocessing

Data augmentation techniques used:

* Rotation
* Zoom
* Width Shift
* Height Shift
* Brightness Adjustment

---

## Model Architecture

The classification model is based on MobileNetV2 with Transfer Learning.

Additional layers:

* Global Average Pooling
* Batch Normalization
* Dense Layer
* Dropout Layer
* Softmax Output Layer

Regularization techniques:

* Data Augmentation
* Batch Normalization
* Dropout
* EarlyStopping
* ReduceLROnPlateau

---

## Project Diagrams

This repository includes:

* Flowchart
* System Architecture Diagram
* Use Case Diagram
* Activity Diagram
* Sequence Diagram
* Class Diagram
* State Flow Diagram

---

## Results

Performance achieved:

* Approximately 99% Training Accuracy
* Approximately 99% Validation Accuracy
* 99.90% Test Accuracy
* Test Loss: 0.0292

The model demonstrated stable learning performance with minimal overfitting and effective real-time gesture recognition.

---

## Challenges

Some challenges encountered during development:

* Variations in lighting conditions
* Background noise
* Hand positioning differences
* Visually similar alphabet gestures
* Static gesture limitation

---

## Future Scope

Future enhancements may include:

* Dynamic gesture recognition
* Word-level recognition
* Sentence-level translation
* Speech output generation
* Mobile application deployment
* Multilingual support
* Transformer/LSTM-based sequence models

---

## Research Publication

This project was further extended into a research paper titled:

"A Prototype Deep Learning-Based System for Indian Sign Language Alphabet Gesture Recognition"

The research work focuses on applying Deep Learning and Computer Vision techniques for real-time ISL alphabet recognition using MobileNetV2, MediaPipe, and Transfer Learning.

---

## Author

Nafisa Zaman

Master of Computer Applications (MCA)

Girijananda Chowdhury University, Guwahati, Assam

2026
