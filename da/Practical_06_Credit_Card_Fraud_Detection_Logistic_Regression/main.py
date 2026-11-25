import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

csv_file = 'creditcard.csv'

if not os.path.exists(csv_file):
    print(f"'{csv_file}' not found locally. Generating sample Credit Card Fraud Dataset...")
    # Synthetic imbalanced data (9 genuine, 1 fraud)
    data = {
        'V1': [-1.35, 1.19, -1.35, -0.96, -1.15, -0.42, 1.22, -0.64, 0.12, -2.31],
        'V2': [-0.07, 0.26, -1.34, -0.18, 0.87, 0.96, 0.14, 1.20, -0.04, 1.95],
        'V3': [2.53, 0.16, 1.77, 1.79, 1.54, 1.14, 0.04, 1.07, 0.88, -1.60],
        'Amount': [149.62, 2.69, 378.66, 123.50, 69.99, 3.67, 4.99, 15.99, 85.00, 239.93],
        'Class': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    print(f"Sample '{csv_file}' created successfully.\n")

df = pd.read_csv(csv_file)

# Input and output
X = df.drop('Class', axis=1)
y = df['Class']

# Split data with stratification (ensuring minority class is represented)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create model with balanced class weight
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

print("Predictions:", pred)
print("Actual:     ", list(y_test))
print("Accuracy:   ", accuracy_score(y_test, pred))
