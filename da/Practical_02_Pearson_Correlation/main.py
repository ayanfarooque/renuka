import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

csv_file = 'diabetes.csv'

if not os.path.exists(csv_file):
    print(f"'{csv_file}' not found locally. Generating sample PIMA Indians Diabetes Dataset...")
    data = {
        'Pregnancies': [6, 1, 8, 1, 0, 5, 3, 10],
        'Glucose': [148, 85, 183, 89, 137, 116, 78, 115],
        'BloodPressure': [72, 66, 64, 66, 40, 74, 50, 0],
        'SkinThickness': [35, 29, 0, 23, 35, 0, 32, 0],
        'Insulin': [0, 0, 0, 94, 168, 0, 88, 0],
        'BMI': [33.6, 26.6, 23.3, 28.1, 43.1, 25.6, 31.0, 35.3],
        'DiabetesPedigreeFunction': [0.627, 0.351, 0.672, 0.167, 2.288, 0.201, 0.248, 0.134],
        'Age': [50, 31, 32, 21, 33, 30, 26, 29],
        'Outcome': [1, 0, 1, 0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    print(f"Sample '{csv_file}' created successfully.\n")

df = pd.read_csv(csv_file)

# Calculate Pearson correlation
corr = df.corr(method='pearson')

print("Pearson Correlation Matrix:")
print(corr.round(3))

# Display heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Pearson Correlation Heatmap – PIMA Diabetes Dataset")
plt.tight_layout()
plt.show()
