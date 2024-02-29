import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('mail_data.csv')

# Data preprocessing
df = df.dropna()  # Drop any rows with missing values
df['Category'] = df['Category'].map({'spam': 1, 'ham': 0})  # Convert labels to binary

# Split data into features and labels
X = df['Message']
Y = df['Category']

# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Feature extraction using TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = tfidf_vectorizer.fit_transform(X_train)
X_test_features = tfidf_vectorizer.transform(X_test)

# Model selection and training
model = make_pipeline(StandardScaler(with_mean=False), LogisticRegression())
model.fit(X_train_features, Y_train)

# Model evaluation
train_accuracy = model.score(X_train_features, Y_train)
test_accuracy = model.score(X_test_features, Y_test)
print("Model Performance:")
print("====================")
print(f"Train Accuracy: {train_accuracy:.2%}")
print(f"Test Accuracy: {test_accuracy:.2%}")

# Additional metrics
Y_pred = model.predict(X_test_features)
print("\nAdditional Metrics:")
print("====================")
print("Classification Report:")
print(classification_report(Y_test, Y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(Y_test, Y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, 
            xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Dynamic Input
input_mail = input("Enter your email message: ")
input_data_features = tfidf_vectorizer.transform([input_mail])
prediction = model.predict(input_data_features)
print("\nPrediction:")
print("====================")
if prediction[0] == 1:
    print('This email is classified as Spam.')
else:
    print('This email is classified as Ham.')
