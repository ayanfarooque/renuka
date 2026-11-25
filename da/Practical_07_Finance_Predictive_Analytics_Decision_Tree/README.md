# Practical 7: Finance Dataset – End-to-End Predictive Analytics

**Level:** Complex Level  
**Subject:** Data Analytics  
**Platform:** Python (Jupyter Notebook / Google Colab / Standalone)

---

## Aim
To execute a complete end-to-end predictive analytics pipeline on a financial dataset—including categorical dummy encoding, Decision Tree classification, performance evaluation, and Confusion Matrix visualization.

---

## Dataset Information
- **Dataset Name:** Finance Dataset
- **Default Filename:** `finance_data.csv`
- **Kaggle Source:** [https://www.kaggle.com/datasets/nitindatta/finance-data](https://www.kaggle.com/datasets/nitindatta/finance-data)

---

## Comprehensive Theory

### 1. End-to-End Predictive Analytics Lifecycle
Predictive analytics leverages empirical data, statistical modeling, and machine learning techniques to anticipate future outcomes. In financial risk management (credit scoring, loan default prediction, insurance underwriting), predictive analytics replaces subjective human heuristics with data-driven decision engines.

#### The 8 Core Phases of the Predictive Analytics Pipeline:
1. **Problem Definition:** Define the target outcome (e.g., predicting loan default: `default = 1` or `0`).
2. **Data Ingestion:** Ingesting structured tables from operational relational databases (`pd.read_csv`).
3. **Data Preprocessing:** Imputing missing values, removing anomalies, and pruning duplicate records (`dropna()`).
4. **Feature Engineering & One-Hot Encoding:** Converting categorical attributes into numeric indicator matrices via dummy variables (`pd.get_dummies()`).
5. **Data Partitioning:** Dividing observations into independent training and validation subsets (`train_test_split`).
6. **Model Formulation & Training:** Fitting non-parametric estimators such as Decision Trees (`model.fit()`).
7. **Model Inference & Validation:** Generating predictions on out-of-sample data (`model.predict()`).
8. **Evaluation & Visualization:** Generating Confusion Matrices, ROC curves, and computing classification metrics.

---

### 2. Decision Tree Classification (CART Algorithm)
A Decision Tree is a non-parametric supervised learning algorithm that partitions the feature space into a hierarchy of mutually exclusive homogeneous sub-regions using greedy recursive binary splitting.

#### Mathematical Splitting Criteria:
1. **Gini Impurity (Default in Scikit-Learn CART):**
   Measures the probability of misclassifying a randomly chosen element if it were labeled according to class distribution in the node:
   $$I_G(p) = 1 - \sum_{k=1}^K p_k^2$$
   Where $p_k$ is the proportion of samples belonging to class $k$. Pure nodes have $I_G = 0$.
2. **Shannon Entropy & Information Gain:**
   $$H(S) = -\sum_{k=1}^K p_k \log_2(p_k)$$
   $$\text{Information Gain}(S, A) = H(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} H(S_v)$$
   The split maximizing Information Gain (or minimizing Gini impurity) is selected at each internal node.

---

### 3. Evaluation via Confusion Matrix
A Confusion Matrix summarizes the counts of correct and incorrect predictions broken down by class:

| Actual \ Predicted | Predicted Negative ($0$) | Predicted Positive ($1$) |
|---|---|---|
| **Actual Negative ($0$)** | **True Negative (TN)** | **False Positive (FP)** (Type I Error) |
| **Actual Positive ($1$)** | **False Negative (FN)** (Type II Error) | **True Positive (TP)** |

- **Accuracy:** $\frac{TP + TN}{TP + TN + FP + FN}$
- **Precision:** $\frac{TP}{TP + FP}$ (Quality of positive predictions; crucial when false alarms are costly).
- **Recall (Sensitivity):** $\frac{TP}{TP + FN}$ (Coverage of actual positives; crucial in loan default and medical diagnoses).
- **F1-Score:** Harmonic mean of precision and recall: $\frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$.

---

### 4. Viva Voce Questions & Answers

**Q1: What is the function of `pd.get_dummies()`?**  
*Answer:* It performs **One-Hot Encoding**, converting categorical variables with $k$ discrete categories into $k$ binary indicator ($0$ or $1$) numerical columns, making categorical data digestible by mathematical machine learning algorithms.

**Q2: What is the main vulnerability of unconstrained Decision Trees?**  
*Answer:* Decision Trees are prone to severe **overfitting**. An unpruned tree can grow deep branches that memorize noise and idiosyncrasies of the training data, leading to perfect training accuracy but poor generalization on testing data. This is controlled by hyperparameters like `max_depth` and `min_samples_split`.

**Q3: What is the difference between Type I and Type II errors in loan default prediction?**  
*Answer:* Type I Error (False Positive) occurs when a safe borrower is mistakenly classified as a defaulter (loss of business). Type II Error (False Negative) occurs when an actual defaulter is classified as safe and granted a loan (direct financial loss). Type II errors are typically far more costly in finance.

**Q4: What does pure node mean in Decision Tree terminology?**  
*Answer:* A pure node is a terminal leaf where $100\%$ of the data points belong to a single class (Gini Impurity = 0, Entropy = 0).


---

## Step-by-Step Algorithm
1. Import Pandas, Matplotlib, train_test_split, DecisionTreeClassifier, and ConfusionMatrixDisplay.
2. Load the financial records dataset using `pd.read_csv('finance_data.csv')`.
3. Apply One-Hot Encoding across categorical columns and drop missing values via `pd.get_dummies(df).dropna()`.
4. Separate predictor features (`X`) by dropping target column `'default'`, and isolate target vector `'default'` (`y`).
5. Partition the data into training (80%) and testing (20%) partitions with `train_test_split`.
6. Instantiate a `DecisionTreeClassifier()` model.
7. Train the classifier on training sets via `model.fit(X_train, y_train)`.
8. Generate target class predictions via `model.predict(X_test)`.
9. Compute accuracy using `accuracy_score(y_test, pred)`.
10. Render and display the visual confusion matrix using `ConfusionMatrixDisplay.from_predictions()` and `plt.show()`.

---

## Libraries Required
```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
```

---

## Python Program
```python
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
```

---

## Expected Output
```text
Accuracy Score: 1.0

```

---

## Result
Successfully developed an end-to-end predictive analytics pipeline utilizing One-Hot Encoding, Decision Tree classification, and Confusion Matrix diagnostics.
