# 🧠 Handwritten Digit Recognizer
This project is a Python-based machine learning application that recognizes handwritten digits (0-9) using image capture, OpenCV for preprocessing, and a trained ML model for classification.

# 📌 Features
Capture handwritten digits from screen region using pyscreenshot

Preprocess the image using OpenCV (grayscale, blur, resize, threshold)

Create and update a training dataset (.csv file)

Train and save a machine learning model

Predict digits in real-time using screen capture

Display predictions with OpenCV overlay

# 🤖 Model Training
Use your own model training script to train on dataset.csv and save the model using joblib.

from sklearn.linear_model import LogisticRegression
import joblib
...
After training
joblib.dump(model, 'model/numberRecog')

📷 Live Prediction
This script uses screen capture to make real-time predictions.

import joblib, pyscreenshot, cv2
...
Continuously predicts based on screen input
Run it using:

python predict_live.py
Make sure your digit is drawn in the specified screen region (100, 200, 600, 700).

# ✅ Requirements
Python 3.x

OpenCV

pandas

scikit-learn

numpy

pyscreenshot

matplotlib

joblib

# 🧠 Model Used
Logistic Regression (can be replaced with SVM, Random Forest, or a Neural Network)
Input size: 28x28 binary pixel values

