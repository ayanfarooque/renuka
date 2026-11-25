import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

csv_file = 'finance_data.csv'

if not os.path.exists(csv_file):
    print(f"'{csv_file}' not found locally. Generating sample Finance Default Dataset...")
    data = {
        'age': [25, 45, 35, 51, 23, 40, 60, 48, 33, 29],
        'income': [30000, 85000, 54000, 120000, 20000, 62000, 95000, 80000, 48000, 38000],
        'loan_amount': [10000, 25000, 15000, 30000, 8000, 18000, 20000, 22000, 12000, 14000],
        'employment_status': ['Employed', 'Employed', 'Self-Employed', 'Employed', 'Unemployed', 'Employed', 'Employed', 'Self-Employed', 'Employed', 'Unemployed'],
        'default': [0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    print(f"Sample '{csv_file}' created successfully.\n")

df = pd.read_csv(csv_file)

# Convert categorical data to numeric dummy columns
df = pd.get_dummies(df).dropna()

# Input and target
X = df.drop('default', axis=1)
y = df['default']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

# Evaluation
print("Accuracy Score:", accuracy_score(y_test, pred))

# Visualization
ConfusionMatrixDisplay.from_predictions(y_test, pred, cmap='Blues')
plt.title("Confusion Matrix – Loan Default Prediction")
plt.tight_layout()
plt.show()
