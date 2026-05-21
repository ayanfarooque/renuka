# spam_classifier.py
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load dataset
data_path = Path(__file__).with_name("spamhamdata.xls")
if not data_path.exists():
    raise FileNotFoundError(
        f"Dataset not found at {data_path}. Put spamhamdata.xls in the same folder as this script."
    )

data = pd.read_csv(data_path, delimiter='\t', header=None)
data.columns = ['label', 'text'] # Assigning correct column names
X = data['text']
y = data['label']
# Convert labels to binary (spam=1, ham=0)
y = y.map({'ham': 0, 'spam': 1})

# 2. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Create Pipeline for Naïve Bayes
nb_model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', MultinomialNB())
])

# 4. Create Pipeline for SVM
svm_model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', LinearSVC())
])

# 5. Train Models
nb_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)

# 6. Evaluate Naïve Bayes
nb_preds = nb_model.predict(X_test)
print("=== Naïve Bayes Results ===")
print("Accuracy:", accuracy_score(y_test, nb_preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, nb_preds))
print("Classification Report:\n", classification_report(y_test, nb_preds))

# 7. Evaluate SVM
svm_preds = svm_model.predict(X_test)
print("\n=== SVM Results ===")
print("Accuracy:", accuracy_score(y_test, svm_preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, svm_preds))
print("Classification Report:\n", classification_report(y_test, svm_preds))

# 8. Predict on New Email
sample_email = ["Congratulations! You've won a free lottery. Click now!"]
# Naive Bayes Prediction
nb_prediction = nb_model.predict(sample_email)
print("\nSample Prediction (Naive Bayes):",
      "Spam" if nb_prediction[0] == 1 else "Ham")
# SVM Prediction
svm_prediction = svm_model.predict(sample_email)
print("Sample Prediction (SVM):",
      "Spam" if svm_prediction[0] == 1 else "Ham")