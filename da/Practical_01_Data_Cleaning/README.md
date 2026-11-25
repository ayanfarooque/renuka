# Practical 1: Data Cleaning – Missing Values, Duplicates and Inconsistent Entries

**Level:** Basic Level  
**Subject:** Data Analytics  
**Platform:** Python (Jupyter Notebook / Google Colab / Standalone)

---

## Aim
To download a real-world healthcare dataset and detect missing values, duplicate records and inconsistent entries using Python, and perform essential data preprocessing.

---

## Dataset Information
- **Dataset Name:** Healthcare Dataset
- **Default Filename:** `healthcare_dataset.csv`
- **Kaggle Source:** [https://www.kaggle.com/datasets/prasad22/healthcare](https://www.kaggle.com/datasets/prasad22/healthcare)

---

## Comprehensive Theory

### 1. Introduction to Data Cleaning
Data cleaning (also known as data cleansing or data scrubbing) is the foundational stage of the Data Analytics lifecycle. In real-world data collection, raw data is notoriously dirty—plagued by missing readings, human entry typos, duplicated transmissions, system crashes, and formatting discrepancies. The operational maxim **"Garbage In, Garbage Out" (GIGO)** asserts that even the most sophisticated statistical or machine learning model will yield erroneous and biased predictions if trained on uncleaned, corrupt data.

Data cleaning ensures:
- **Data Integrity and Quality:** Maintaining consistency, validity, and accuracy across all attributes.
- **Improved Model Convergence:** Removing noise and anomalies stabilizes gradient descent and parameter estimation.
- **Reduced Bias:** Handling missing data systematically prevents systematic attrition bias in downstream analyses.

---

### 2. Typology of Data Inconsistencies

#### A. Missing Values
Missing data arises from non-responses in surveys, sensor dropouts, transmission errors, or optional form fields. In statistical theory (Rubin, 1976), missing data mechanisms are classified into:
1. **Missing Completely at Random (MCAR):** The probability of missingness is entirely independent of observed and unobserved data. Removing missing rows (`dropna()`) does not bias parameter estimates, though it reduces statistical power.
2. **Missing at Random (MAR):** Missingness depends systematically on other observed variables but not on the missing value itself (e.g., males being less likely to report depression scores).
3. **Missing Not at Random (MNAR):** Missingness depends directly on the unobserved value itself (e.g., high-income individuals refusing to disclose salary). Simple deletion here induces severe bias.

**Handling Strategies:**
- **Listwise/Pairwise Deletion (`df.dropna()`):** Practical when missing records constitute $< 3-5\%$ of large datasets.
- **Statistical Imputation:** Replacing missing fields with central tendencies:
  - Numerical attributes: Mean (for symmetric distributions) or Median (for skewed distributions with outliers).
  - Categorical attributes: Mode (most frequent category) or explicit `"Unknown"` imputation.
- **Predictive Imputation:** Using K-Nearest Neighbors (KNN) or MICE (Multivariate Imputation by Chained Equations).

#### B. Duplicate Records
Duplicate observations occur due to repeated form submissions, merging multiple database extracts without primary key deduplication, or network retry mechanisms. Duplicates artificially inflate sample sizes, bias statistical significance tests, and cause data leakage when split into training and testing sets.
- `df.duplicated().sum()` identifies redundant identical rows.
- `df.drop_duplicates()` retains the first occurrence and removes redundant copies.

#### C. Inconsistent and Anomalous Entries
Categorical inconsistencies emerge from non-standardized user inputs (e.g., `"Male"`, `"male"`, `"M"`, `" male "`), typographical errors, or mismatched encoding schemes.
- Discrepancies are identified using `df[col].unique()` and `df[col].value_counts()`.
- Rectification involves string stripping (`.str.strip()`), case folding (`.str.lower()` or `.str.capitalize()`), and regular expression replacement mapping (`.replace()`).

---

### 3. Step-by-Step Practical Workflow
1. **Inspection:** Compute dimensions (`df.shape`), inspect schema (`df.info()`), and preview top rows (`df.head()`).
2. **Missing Value Audit:** Compute frequency per feature via `df.isnull().sum()`.
3. **Duplicate Audit:** Sum duplicate flags via `df.duplicated().sum()`.
4. **Pruning & Cleaning:** Execute `df.drop_duplicates()` followed by selective imputation or `df.dropna()`.
5. **Categorical Normalization:** Inspect unique values of object columns via `df.select_dtypes('object')` and apply uniform formatting.

---

### 4. Viva Voce Questions & Answers

**Q1: What is the fundamental difference between data cleaning and data transformation?**  
*Answer:* Data cleaning focuses on identifying and fixing errors, missing data, noise, and duplicates to make data accurate. Data transformation involves altering data format, scale, or structure (e.g., normalization, one-hot encoding, log transforms) to suit analytical modeling.

**Q2: When should you drop rows with missing values versus imputing them?**  
*Answer:* Dropping (`dropna`) is acceptable when the missing values account for a negligible fraction ($< 5\%$) of the dataset and the missingness mechanism is MCAR. If missing data is substantial or follows MAR/MNAR patterns, dropping causes severe information loss and bias; imputation (mean, median, mode, or model-based) is preferred.

**Q3: Why must duplicate records be eliminated before splitting data into train and test sets?**  
*Answer:* If duplicate rows exist, identical copies of an observation may appear simultaneously in both the training set and testing set. This causes severe data leakage, resulting in artificially inflated test accuracy while failing in production.

**Q4: Which Pandas functions detect missing and duplicate entries?**  
*Answer:* `df.isnull().sum()` or `df.isna().sum()` computes missing values per column, and `df.duplicated().sum()` counts duplicate rows.


---

## Step-by-Step Algorithm
1. Import the Pandas library (`import pandas as pd`).
2. Load the healthcare dataset from CSV into a DataFrame using `pd.read_csv()`.
3. Inspect initial records using `df.head()` and structure using `df.shape`.
4. Detect and sum missing values in each column using `df.isnull().sum()`.
5. Check total duplicate rows using `df.duplicated().sum()`.
6. Drop duplicate rows using `df.drop_duplicates()`.
7. Handle missing values by dropping corrupted rows using `df.dropna()`.
8. Filter text columns using `df.select_dtypes('object')` and inspect unique category entries with `df[col].unique()`.
9. Verify that the cleaned DataFrame is free from nulls and duplicates, then display `df.head()`.

---

## Libraries Required
```python
import pandas as pd
```

---

## Python Program
```python
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
```

---

## Expected Output
```text
Initial Dataset Head:
          Name   Age  Gender Blood Type Medical Condition  Billing Amount
0     John Doe  45.0    Male         O+          Diabetes        12500.50
1   Jane Smith   NaN  Female         A+      Hypertension        23000.00
2  Alice Brown  29.0  female         B+            Asthma             NaN
3    Bob Green  62.0       M         O-         Arthritis         4100.25
4     John Doe  45.0    Male         O+          Diabetes        12500.50

Dataset Shape: (8, 6)

Missing Values per Column:
Name                 0
Age                  2
Gender               0
Blood Type           0
Medical Condition    0
Billing Amount       1
dtype: int64

Total Duplicate Records: 2
Shape after removing duplicates: (6, 6)
Shape after removing missing values: (4, 6)

Unique entries in categorical columns:
  Name: ['John Doe' 'Bob Green' 'Charlie White' 'David Hall']
  Gender: ['Male' 'M' 'MALE']
  Blood Type: ['O+' 'O-' 'AB+']
  Medical Condition: ['Diabetes' 'Arthritis' 'Cancer']

Cleaned Dataset Head:
            Name   Age Gender Blood Type Medical Condition  Billing Amount
0       John Doe  45.0   Male         O+          Diabetes        12500.50
3      Bob Green  62.0      M         O-         Arthritis         4100.25
5  Charlie White  50.0   Male        AB+            Cancer        18900.00
7     David Hall  38.0   MALE         O+          Diabetes        15400.00

```

---

## Result
Successfully detected missing values, duplicate records, and inconsistent categorical entries in the healthcare dataset and applied systematic cleaning procedures.
