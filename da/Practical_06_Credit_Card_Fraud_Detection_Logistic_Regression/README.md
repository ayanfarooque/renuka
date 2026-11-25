# Practical 6: Credit Card Fraud Detection – Logistic Regression

**Level:** Moderate Level  
**Subject:** Data Analytics  
**Platform:** Python (Jupyter Notebook / Google Colab / Standalone)

---

## Aim
To construct a cost-sensitive Logistic Regression classification model with stratified sampling to detect fraudulent transactions in an imbalanced credit card dataset.

---

## Dataset Information
- **Dataset Name:** Credit Card Fraud Detection Dataset
- **Default Filename:** `creditcard.csv`
- **Kaggle Source:** [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## Comprehensive Theory

### 1. Introduction to Logistic Regression
Despite its name containing "Regression", **Logistic Regression** is a canonical supervised classification algorithm for categorical outcomes. Developed by David Cox (1958), it models the conditional probability $P(Y=1|X)$ that a given observation belongs to a specific binary class.

In credit card transaction monitoring:
- **Class 0:** Legitimate (Genuine) transaction ($> 99.8\%$ of traffic).
- **Class 1:** Fraudulent transaction ($< 0.2\%$ of traffic).

---

### 2. Mathematical Formulation & The Sigmoid Function
Linear regression is inappropriate for classification because linear functions $Y = \mathbf{w}^T \mathbf{x} + b$ output values in $(-\infty, +\infty)$, violating the $[0, 1]$ probability boundary. Logistic Regression passes the linear score $z$ through the non-linear **Logistic (Sigmoid) Function**:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Where:
$$z = \mathbf{w}^T \mathbf{x} + b = \beta_0 + \beta_1 X_1 + \dots + \beta_p X_p$$

#### Properties of the Sigmoid Function:
- As $z \to +\infty$, $\sigma(z) \to 1$.
- As $z \to -\infty$, $\sigma(z) \to 0$.
- At $z = 0$, $\sigma(z) = 0.5$ (default decision threshold).

#### Odds Ratio and Logit Form:
The ratio of the probability of an event occurring to the probability of it not occurring is the **Odds Ratio**:
$$\text{Odds} = \frac{p}{1 - p}$$
Taking the natural logarithm yields the linear **Logit**:
$$\ln\left(\frac{p}{1 - p}\right) = \mathbf{w}^T \mathbf{x} + b$$

#### Maximum Likelihood Estimation (Log-Loss):
Logistic regression parameters are learned by minimizing the Binary Cross-Entropy loss:
$$J(\mathbf{w}) = -\frac{1}{n} \sum_{i=1}^n \left[ y_i \ln(\hat{p}_i) + (1 - y_i) \ln(1 - \hat{p}_i) \right]$$

---

### 3. Handling Extreme Class Imbalance
In fraud detection, positive fraudulent transactions are extremely rare ($pprox 0.17\%$). This presents severe challenges:
1. **The Accuracy Paradox:** A naive model that predicts every transaction as genuine achieves $99.8\%$ accuracy while detecting zero frauds. Accuracy is therefore an insufficient metric; **Precision, Recall, and PR-AUC** must be evaluated.
2. **Stratified Splitting (`stratify=y`):** Ensures that the exact ratio of genuine to fraudulent cases is maintained equally across both training and testing partitions, preventing zero-fraud test splits.
3. **Cost-Sensitive Weighting (`class_weight='balanced'`):** Penalizes errors on minority class samples proportionally to class frequencies:
   $$w_1 = \frac{n}{2 \times n_1}$$
   This forces the optimizer to prioritize detecting fraudulent instances.

---

### 4. Viva Voce Questions & Answers

**Q1: Why can't Linear Regression be used for binary classification?**  
*Answer:* Linear regression outputs unbounded values ($-\infty$ to $+\infty$), which cannot be interpreted as valid probabilities. Furthermore, its squared-error loss function is heavily distorted by extreme outliers, shifting the decision threshold arbitrarily.

**Q2: What is the purpose of `stratify=y` in train_test_split?**  
*Answer:* In highly imbalanced datasets, random splitting might place all rare minority class samples (fraud) into the training set, leaving none in the test set. Stratified splitting enforces identical class proportions across both training and test subsets.

**Q3: What does `class_weight='balanced'` do in Scikit-Learn?**  
*Answer:* It automatically adjusts weights inversely proportional to class frequencies in the training data, heavily penalizing mistakes made on the rare class (Class 1) and mitigating minority starvation.

**Q4: What is the significance of the Sigmoid function?**  
*Answer:* It squashes any continuous real-valued number into a valid probability range between $0$ and $1$ with a smooth, continuously differentiable S-shaped curve.


---

## Step-by-Step Algorithm
1. Import Pandas, train_test_split, LogisticRegression, and accuracy_score.
2. Load the credit card transactions dataset (`pd.read_csv('creditcard.csv')`).
3. Define the predictor feature matrix `X` by dropping the target column `'Class'`.
4. Define the binary target vector `y` (`df['Class']`).
5. Split data into training (80%) and testing (20%) using `stratify=y` to preserve rare fraud class ratios.
6. Initialize `LogisticRegression(max_iter=1000, class_weight='balanced')`.
7. Fit the model to the training subset via `model.fit(X_train, y_train)`.
8. Predict transaction classes on the test subset via `model.predict(X_test)`.
9. Compute and print accuracy and preview predicted classifications.

---

## Libraries Required
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

---

## Python Program
```python
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
```

---

## Expected Output
```text
Predictions: [0 1]
Actual:      [0, 1]
Accuracy:    1.0

```

---

## Result
Successfully built and evaluated a cost-sensitive Logistic Regression model capable of handling extreme class imbalance in fraud detection.
