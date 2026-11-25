import os
import pandas as pd
import numpy as np

# Load dataset (with automatic fallback to realistic sample data if CSV is absent)
csv_file = 'healthcare_dataset.csv'

if not os.path.exists(csv_file):
    print(f"'{csv_file}' not found locally. Generating realistic sample Healthcare Dataset for demonstration...")
    raw_data = {
        'Name': ['John Doe', 'Jane Smith', 'Alice Brown', 'Bob Green', 'John Doe', 'Charlie White', 'Jane Smith', 'David Hall'],
        'Age': [45, np.nan, 29, 62, 45, 50, np.nan, 38],
        'Gender': ['Male', 'Female', 'female', 'M', 'Male', 'Male', 'Female', 'MALE'],
        'Blood Type': ['O+', 'A+', 'B+', 'O-', 'O+', 'AB+', 'A+', 'O+'],
        'Medical Condition': ['Diabetes', 'Hypertension', 'Asthma', 'Arthritis', 'Diabetes', 'Cancer', 'Hypertension', 'Diabetes'],
        'Billing Amount': [12500.50, 23000.00, np.nan, 4100.25, 12500.50, 18900.00, 23000.00, 15400.00]
    }
    df = pd.DataFrame(raw_data)
    df.to_csv(csv_file, index=False)
    print(f"Sample '{csv_file}' created successfully.\n")

df = pd.read_csv(csv_file)

# Display dataset
print("Initial Dataset Head:")
print(df.head())
print("\nDataset Shape:", df.shape)

# Find missing values
print("\nMissing Values per Column:")
print(df.isnull().sum())

# Find duplicate records
print("\nTotal Duplicate Records:", df.duplicated().sum())

# Remove duplicates
df = df.drop_duplicates()
print("Shape after removing duplicates:", df.shape)

# Remove missing values
df = df.dropna()
print("Shape after removing missing values:", df.shape)

# Find inconsistent entries in categorical columns
print("\nUnique entries in categorical columns:")
for col in df.select_dtypes('object'):
    print(f"  {col}: {df[col].unique()}")

print("\nCleaned Dataset Head:")
print(df.head())
