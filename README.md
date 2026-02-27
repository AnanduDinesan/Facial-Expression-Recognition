# Facial Emotion Recognition

Detect human facial emotions from images using deep learning.

Facial emotion recognition helps computers understand human feelings from visual cues. This project analyzes facial image data and builds a deep learning model to classify emotions into seven categories. It also includes a web interface that allows users to upload an image and view the predicted emotion.

---

## 🔍 Project Overview

Facial Emotion Recognition (FER) enables machines to interpret human emotions, which is useful in areas such as human-computer interaction, healthcare monitoring, and behavioral analysis.  
This project builds a multi-class classification model trained on facial expression images and deploys it through a Flask web application for real-time predictions.

---

## 🧠 Approach

1. **Exploratory Data Analysis (EDA)**  
   - Analyze class distribution and pixel intensity patterns  
   - Visualize sample images from each emotion class  
   - Inspect dataset balance and image dimensions  

2. **Data Preprocessing & Augmentation**  
   - Normalize grayscale images  
   - Apply data augmentation (flip, rotation, zoom)  
   - Handle dataset imbalance using class weights and balanced sampling  

3. **Model Training**  
   - Train a deep CNN / EfficientNetV2 model on FER2013 dataset  
   - Use callbacks like early stopping and learning rate reduction  
   - Evaluate using accuracy, confusion matrix, and classification metrics  

---

## 📁 Dataset

The project uses the **FER2013 dataset**, which contains grayscale facial images labeled into seven emotion categories:

- Angry  
- Disgust  
- Fear  
- Happy  
- Neutral  
- Sad  
- Surprise  

All images are **48×48 pixels** and stored in class-wise folders.

---

## 🚀 Features

This project includes:

- Data preprocessing & augmentation pipeline  
- Deep learning emotion classification model  
- Balanced training using class weighting  
- Flask web app for image upload & prediction  
- Real-time inference using trained model  

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/AnanduDinesan/Facial-Expression-Recognition.git
cd Facial-Expression-Recognition
```

### 2. Create virtual environment

**Windows**

```bash
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux/Mac**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Flask App

```bash
python app.py
```

Upload an image to see predicted emotion.

---

## 📦 Model File

The trained model is stored as:

```
fer_efficientnet_v2_final.keras
```

Make sure this file is in the project root before running the app.

---

## 👤 Author

**Anandu Dinesan**  
MCA Student | Machine Learning Enthusiast

---

## 📌 Future Improvements

- Real-time webcam emotion detection  
- Emotion probability visualization on UI  
- Deployment to cloud platform  
- Training on larger datasets like AffectNet