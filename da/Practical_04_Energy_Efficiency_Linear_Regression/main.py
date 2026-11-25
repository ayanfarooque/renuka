import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

csv_file = 'ENB2012_data.csv'

if not os.path.exists(csv_file):
    print(f"'{csv_file}' not found locally. Generating sample Energy Efficiency Dataset...")
    data = {
        'X1': [0.98, 0.90, 0.86, 0.82, 0.79, 0.76, 0.74, 0.71, 0.69, 0.66], # Relative Compactness
        'X2': [514.5, 563.5, 588.0, 612.5, 637.0, 661.5, 686.0, 710.5, 735.0, 759.5], # Surface Area
        'X3': [294.0, 318.5, 294.0, 318.5, 343.0, 416.5, 245.0, 269.5, 294.0, 318.5], # Wall Area
        'X4': [110.25, 122.5, 147.0, 147.0, 147.0, 122.5, 220.5, 220.5, 220.5, 220.5], # Roof Area
        'X5': [7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 3.5, 3.5, 3.5, 3.5], # Overall Height
        'X6': [2, 3, 4, 5, 2, 3, 4, 5, 2, 3], # Orientation
        'X7': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # Glazing Area
        'X8': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # Glazing Area Distribution
        'Y1': [15.55, 20.84, 19.87, 23.53, 28.52, 24.77, 10.32, 10.66, 12.18, 14.17], # Heating Load
        'Y2': [21.33, 28.28, 29.63, 31.44, 37.73, 29.79, 13.43, 14.28, 14.28, 16.63]  # Cooling Load
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    print(f"Sample '{csv_file}' created successfully.\n")

df = pd.read_csv(csv_file)

# Input and output
X = df.drop(['Y1', 'Y2'], axis=1)
y = df['Y1']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

print("Actual Test Values:   ", list(y_test))
print("Predicted Test Values:", [round(p, 2) for p in pred])
print("\nModel Coefficients:", model.coef_)
print("Model Intercept:   ", model.intercept_)
print("Mean Squared Error (MSE):", round(mean_squared_error(y_test, pred), 4))
print("R2 Score:                ", round(r2_score(y_test, pred), 4))
