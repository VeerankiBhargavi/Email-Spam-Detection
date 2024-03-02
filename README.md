# Email Spam Detection

This repository contains a Python script for detecting spam emails using machine learning techniques.

## Description

Email spam is a prevalent issue in modern communication. This project aims to build a machine learning model that can classify emails as either spam or ham (non-spam). The model is trained on a dataset of labeled emails, where each email is represented by its text content and a binary label indicating spam or ham.

The project includes the following components:
- Data preprocessing: Cleaning and preparing the email data for model training.
- Feature extraction: Using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to convert the text data into numerical features.
- Model training: Training a logistic regression model using the extracted features.
- Model evaluation: Evaluating the trained model's performance on a separate test set using accuracy, precision, recall, and F1-score metrics.
- Visualization: Visualizing the model's performance with a confusion matrix.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/VeerankiBhargavi/email-spam-detection.git

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt


3. Run the Python script:
   ```bash
   python email_spam_detection.py


  4.Follow the on-screen instructions to input your email message and see the classification result.



**Dataset**

The dataset used in this project (`mail_data.csv`) contains labeled email messages, where each email is labeled as either "spam" or "ham" (non-spam). The dataset is preprocessed and cleaned before model training.

