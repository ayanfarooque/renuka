import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

csv_file = 'data.csv'

if not os.path.exists(csv_file):
    print(f"'{csv_file}' not found locally. Generating sample Wisconsin Breast Cancer Dataset...")
    data = {
        'id': [842302, 842517, 84300903, 84348301, 84358402, 843786, 844359, 844582, 844981, 84501001],
        'radius_mean': [17.99, 20.57, 19.69, 11.42, 20.29, 12.45, 18.25, 13.71, 13.00, 12.46],
        'texture_mean': [10.38, 17.77, 21.25, 20.38, 14.34, 15.70, 19.98, 20.83, 21.82, 24.04],
        'perimeter_mean': [122.8, 132.9, 130.0, 77.58, 135.1, 82.57, 119.6, 90.20, 87.50, 83.97],
        'area_mean': [1001.0, 1326.0, 1203.0, 386.1, 1297.0, 477.1, 1040.0, 577.9, 519.8, 475.9],
        'smoothness_mean': [0.1184, 0.0847, 0.1096, 0.1425, 0.1003, 0.1278, 0.0946, 0.1189, 0.1273, 0.1186],
        'diagnosis': ['M', 'M', 'M', 'M', 'M', 'B', 'M', 'B', 'B', 'B'],
        'Unnamed: 32': [float('nan')]*10
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    print(f"Sample '{csv_file}' created successfully.\n")

df = pd.read_csv(csv_file)

# Input and output (safe handling if Unnamed: 32 exists)
drop_cols = ['id', 'diagnosis']
if 'Unnamed: 32' in df.columns:
    drop_cols.append('Unnamed: 32')

X = df.drop(drop_cols, axis=1)
y = df['diagnosis']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create SVM model
model = SVC()
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

print("Actual Test Labels:   ", list(y_test))
print("Predicted Test Labels:", list(pred))
print("\nAccuracy Score:", accuracy_score(y_test, pred))
