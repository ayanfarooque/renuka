# Data Analytics – Practical Exam Guide & Lab Manual

**Subject:** Data Analytics  
**Level:** Basic, Moderate, and Complex  
**Language:** Python 3  
**Platform:** Google Colab / Jupyter Notebook / Local Python Environment  

---

## Quick Navigation: Practicals Directory

| No. | Practical Title | Difficulty | Directory & Documentation |
|:---:|---|:---:|---|
| **1** | **Data Cleaning – Missing Values, Duplicates & Anomalies** | Basic | [Practical_01_Data_Cleaning](./Practical_01_Data_Cleaning/README.md) |
| **2** | **Pearson Correlation – PIMA Diabetes Dataset** | Basic | [Practical_02_Pearson_Correlation](./Practical_02_Pearson_Correlation/README.md) |
| **3** | **Principal Component Analysis (PCA) – Mall Customers** | Basic | [Practical_03_Principal_Component_Analysis_PCA](./Practical_03_Principal_Component_Analysis_PCA/README.md) |
| **4** | **Energy Efficiency – Linear Regression** | Moderate | [Practical_04_Energy_Efficiency_Linear_Regression](./Practical_04_Energy_Efficiency_Linear_Regression/README.md) |
| **5** | **Breast Cancer Classification – Support Vector Machine** | Moderate | [Practical_05_Breast_Cancer_Classification_SVM](./Practical_05_Breast_Cancer_Classification_SVM/README.md) |
| **6** | **Credit Card Fraud Detection – Logistic Regression** | Moderate | [Practical_06_Credit_Card_Fraud_Detection_Logistic_Regression](./Practical_06_Credit_Card_Fraud_Detection_Logistic_Regression/README.md) |
| **7** | **Finance Dataset – End-to-End Predictive Analytics** | Complex | [Practical_07_Finance_Predictive_Analytics_Decision_Tree](./Practical_07_Finance_Predictive_Analytics_Decision_Tree/README.md) |
| **8** | **Twitter Sentiment Analysis – Naive Bayes** | Complex | [Practical_08_Twitter_Sentiment_Analysis_Naive_Bayes](./Practical_08_Twitter_Sentiment_Analysis_Naive_Bayes/README.md) |

---

## 1. Important Python Libraries

| Library | Primary Purpose | Key Modules / Functions |
|---|---|---|
| **Pandas** | High-performance tabular data structures and data manipulation | `pd.read_csv()`, `df.dropna()`, `df.drop_duplicates()`, `pd.get_dummies()` |
| **NumPy** | Multi-dimensional array processing and linear algebra routines | `np.mean()`, `np.nan`, `np.dot()`, array indexing |
| **Matplotlib** | Foundational plotting library for 2D graphics and charts | `plt.figure()`, `plt.scatter()`, `plt.title()`, `plt.show()` |
| **Seaborn** | Statistical data visualization built on Matplotlib | `sns.heatmap()`, `sns.pairplot()`, `sns.boxplot()` |
| **Scikit-Learn** | Industry-standard Machine Learning library | Model training (`fit`), prediction (`predict`), preprocessing, metrics |

### Installation Command
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## 2. Most Frequently Used Python Functions in Data Analytics

| Function Call | Practical Purpose |
|---|---|
| `pd.read_csv('filename.csv')` | Reads a comma-separated values file into a Pandas DataFrame. |
| `df.head(n=5)` | Displays the top $n$ records for structural inspection. |
| `df.shape` | Returns a tuple `(rows, columns)` indicating dimensions. |
| `df.info()` | Prints summary of index, column datatypes, and non-null counts. |
| `df.describe()` | Computes summary statistics (count, mean, std, min, quartiles, max). |
| `df.isnull().sum()` | Returns count of missing values per column. |
| `df.duplicated().sum()` | Counts the number of identical duplicate rows. |
| `df.dropna()` | Drops rows containing missing/null values. |
| `df.drop_duplicates()` | Eliminates redundant duplicate rows. |
| `df.select_dtypes('object')` | Filters columns of string/categorical datatype. |
| `df[col].unique()` | Returns an array of distinct unique entries in a column. |
| `df.corr(method='pearson')` | Computes pairwise Pearson correlation coefficients. |
| `pd.get_dummies(df)` | Converts categorical variables into binary one-hot indicator columns. |
| `train_test_split(X, y, test_size, random_state, stratify)` | Partitions data into independent training and testing sets. |
| `model.fit(X_train, y_train)` | Fits machine learning estimator to training data. |
| `model.predict(X_test)` | Generates class or numerical predictions on unseen test samples. |
| `accuracy_score(y_test, pred)` | Computes ratio of correct predictions to total predictions. |
| `mean_squared_error(y_test, pred)` | Computes average squared prediction error for regression. |
| `r2_score(y_test, pred)` | Computes proportion of variance explained by regression model. |

---

## 3. Algorithm Comparison Matrix

| Practical | Algorithm | Learning Paradigm | Primary Objective | Key Loss / Optimization Metric |
|:---:|---|---|---|---|
| **1** | **Data Cleaning** | Data Preprocessing | Detect & fix missing values, duplicates, and inconsistent text | Data completeness & integrity |
| **2** | **Pearson Correlation** | Statistical Analysis | Quantify linear association between continuous variables | Correlation coefficient $r \in [-1, +1]$ |
| **3** | **PCA** | Unsupervised Reduction | Reduce feature count while maximizing variance retention | Eigenvalues & Explained Variance Ratio |
| **4** | **Linear Regression** | Supervised Regression | Predict continuous building Heating Load | Ordinary Least Squares (OLS) / MSE |
| **5** | **Support Vector Machine (SVM)** | Supervised Classification | Maximize separating margin between benign & malignant tumours | Hinge Loss / Maximum Margin ($\|w\|^{-1}$) |
| **6** | **Logistic Regression** | Supervised Classification | Classify genuine vs fraudulent financial transactions | Log-Loss (Binary Cross-Entropy) |
| **7** | **Decision Tree** | Supervised Classification | Predict loan default risk via hierarchical decision rules | Gini Impurity / Information Gain |
| **8** | **Naive Bayes** | Supervised Classification | Classify tweet sentiment using conditional word probabilities | Maximum A Posteriori (MAP) with Laplace Smoothing |

---

## 4. Master Viva Voce Preparation Guide

### Q1. What is the fundamental difference between Supervised and Unsupervised Learning?
*Answer:* **Supervised Learning** trains on labeled datasets where every input sample is paired with a known ground-truth target ($y$); the goal is learning a mapping function to predict targets for novel inputs. **Unsupervised Learning** operates on unlabeled data ($X$ only); the goal is discovering latent geometric structures, clusters, or lower-dimensional representations (e.g., PCA, K-Means).

### Q2. What is the difference between Classification and Regression?
*Answer:* Classification predicts discrete categorical class labels (e.g., Benign vs Malignant, Spam vs Not Spam). Regression predicts continuous quantitative numerical values (e.g., Heating Load, Stock Price, House Value).

### Q3. What is Overfitting and how can it be mitigated?
*Answer:* Overfitting occurs when a machine learning model learns the training data and noise too closely, achieving near-perfect training performance but failing to generalize to unseen test data. It is mitigated by:
- Regularization (L1/L2 penalties in linear models, $C$ parameter in SVM).
- Tree pruning and depth limiting (`max_depth`).
- Increasing training dataset size.
- Cross-validation and feature selection.

### Q4. What is the difference between Precision and Recall?
*Answer:*
- **Precision** $= \frac{TP}{TP + FP}$: Measures what proportion of positive identifications was actually correct. High precision is crucial when the cost of a false alarm (False Positive) is high.
- **Recall** $= \frac{TP}{TP + FN}$: Measures what proportion of actual positives was identified correctly. High recall is critical in medical screening and fraud detection where missing a true positive (False Negative) is catastrophic.

### Q5. What is the significance of train-test split?
*Answer:* Splitting data into disjoint training and testing partitions provides an unbiased evaluation of how effectively the trained model generalizes to novel, out-of-sample data.

---

## 5. Practical Exam Tips & Checklist

### Standard Code Blueprint to Memorize:
```python
# 1. Load Data
import pandas as pd
df = pd.read_csv('dataset.csv')

# 2. Inspect & Preprocess
print(df.isnull().sum())
df = df.dropna()

# 3. Define Features (X) and Target (y)
X = df.drop('target_col', axis=1)
y = df['target_col']

# 4. Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Model Training & Prediction
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

# 6. Evaluation
from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test, pred))
```

### Dataset Filename Checklist for Exam:
1. **Practical 1:** `healthcare_dataset.csv` (drop `id`, check missing values, clean strings).
2. **Practical 2:** `diabetes.csv` (PIMA Indian diabetes; numeric correlation matrix and heatmap).
3. **Practical 3:** `Mall_Customers.csv` (standardize features before applying PCA).
4. **Practical 4:** `ENB2012_data.csv` (predict `Y1` heating load using linear regression).
5. **Practical 5:** `data.csv` (drop `id` and `Unnamed: 32` if present; classify `diagnosis`).
6. **Practical 6:** `creditcard.csv` (highly imbalanced `Class`; use `stratify=y` and `class_weight='balanced'`).
7. **Practical 7:** `finance_data.csv` (use `pd.get_dummies()` for categorical variables).
8. **Practical 8:** `twitter_training.csv` (pass `header=None`, name columns, vectorize with `CountVectorizer`).
