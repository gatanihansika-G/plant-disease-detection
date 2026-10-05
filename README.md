# 🌱 Plant Disease Detection Using Machine Learning

## 📌 Project Overview

Plant Disease Detection Using Machine Learning is a deep learning-based web application developed to identify plant diseases from leaf images.

The system uses a Convolutional Neural Network (CNN) to classify an uploaded leaf image into the disease classes included in the trained dataset. The application displays the predicted disease or health condition along with the confidence score, treatment suggestions, and prevention tips.

The main aim of this project is to provide a simple and accessible method for detecting plant diseases from leaf images and supporting early identification.

---

## 🎯 Objectives

- Detect plant diseases from leaf images using machine learning.
- Use a Convolutional Neural Network (CNN) for multiclass image classification.
- Preprocess leaf images before classification.
- Display the predicted disease or health condition.
- Provide the confidence score for the prediction.
- Provide basic treatment suggestions and prevention tips.
- Develop a simple and user-friendly web application.

---

## 🌿 Problem Statement

Plant diseases can negatively affect crop health, quality, and production. Identifying diseases manually can be time-consuming and may require expert knowledge.

Some plant diseases also have similar visual symptoms, making accurate identification difficult.

This project aims to develop a machine learning-based system that analyzes a plant leaf image and predicts the corresponding disease class.

---

## 🧠 Proposed Solution

The proposed system uses a trained Convolutional Neural Network (CNN) to analyze plant leaf images and classify them into the appropriate disease category.

The user uploads a leaf image through the web application. The image is preprocessed and given to the trained CNN model. The model predicts the most likely class and displays the result to the user.

### Workflow

```text
Leaf Image Upload
       ↓
Image Preprocessing
       ↓
Trained CNN Model
       ↓
Disease Classification
       ↓
Prediction Display
       ↓
Treatment & Prevention Information

---

---

## 💻 Technologies Used

- Python
- TensorFlow
- Keras
- CNN
- OpenCV
- PIL
- Streamlit

---

## 📈 Results

- Training Accuracy: **98.10%**
- Validation Accuracy: **95.08%**
- Training Loss: **0.0600**
- Validation Loss: **0.1969**

The model achieved a validation accuracy of **95.08%** over 10 epochs.
