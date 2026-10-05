# 🌱 Plant Disease Detection Using Machine Learning

> **CNN-based plant disease detection system that classifies plant leaf images and provides disease information, confidence score, treatment suggestions, and prevention tips through a Streamlit web application.**

---

## 📌 Project Overview

**Plant Disease Detection Using Machine Learning** is a deep learning-based web application developed to identify plant diseases from leaf images.

The system uses a **Convolutional Neural Network (CNN)** to analyze an uploaded plant leaf image and classify it into one of the disease classes included in the trained dataset.

The application provides the predicted disease or health condition along with the **confidence score, treatment suggestions, and prevention tips**.

The main aim of this project is to provide a simple and accessible method for detecting plant diseases from leaf images and supporting early identification.

---

## 🎯 Objectives

1. Detect plant diseases from plant leaf images using machine learning.
2. Use a Convolutional Neural Network (CNN) for multiclass image classification.
3. Preprocess leaf images before giving them to the trained model.
4. Display the predicted disease or health condition.
5. Display the confidence score of the prediction.
6. Provide basic treatment suggestions and prevention tips.
7. Develop a simple and user-friendly web application for plant disease recognition.

---

## ✨ Main Features

### 🌿 Plant Disease Recognition

- Upload a plant leaf image.
- Preprocess the uploaded image.
- Analyze the image using the trained CNN model.
- Predict the corresponding disease or health condition.
- Display the confidence score.
- Provide treatment suggestions.
- Provide prevention tips.

### 🧠 CNN-Based Classification

The system uses a trained **Convolutional Neural Network** to automatically learn visual patterns from plant leaf images.

The CNN can learn features such as:

- Edges
- Textures
- Spots
- Shapes
- Colour patterns

### 🌐 Web Application

The trained model is integrated into a **Streamlit web application** that provides an easy-to-use interface for image upload and disease prediction.

### 📊 Prediction Display

After processing the uploaded image, the application displays:

- Predicted disease/health condition
- Confidence score
- Treatment information
- Prevention information

---

## 🌾 Supported Crop Categories

The dataset used in this project contains images belonging to the following **9 crop categories**:

1. Apple
2. Cherry
3. Corn
4. Grape
5. Peach
6. Pepper
7. Potato
8. Strawberry
9. Tomato

> **Important:** The model is trained only on the classes included in the dataset. Images belonging to crops or diseases outside the trained classes may not be classified reliably.

---

## 🧠 Machine Learning Model

The project uses a **Convolutional Neural Network (CNN)** for multiclass plant disease classification.

### Model Input

```text
Image Size: 128 × 128 pixels
Channels: 3
Format: RGBf **95.08%** over 10 epochs.
