# Practical 5: Breast Cancer Classification – Support Vector Machine (SVM)

**Level:** Moderate Level  
**Subject:** Data Analytics  
**Platform:** Python (Jupyter Notebook / Google Colab / Standalone)

---

## Aim
To implement a Support Vector Machine (SVM) classifier to predict whether a breast tumour is benign or malignant using clinical biopsy diagnostic measurements.

---

## Dataset Information
- **Dataset Name:** Breast Cancer Wisconsin Diagnostic Dataset
- **Default Filename:** `data.csv`
- **Kaggle Source:** [https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

---

## Comprehensive Theory

### 1. Introduction to Support Vector Machines (SVM)
Support Vector Machine (SVM), formulated by Vladimir Vapnik and Alexey Chervonenkis, is a powerful supervised learning model primarily used for robust binary classification. SVM is grounded in **Structural Risk Minimization (SRM)** from statistical learning theory, which seeks to minimize an upper bound on generalization error rather than empirical training error alone.

In oncology diagnostics, the Wisconsin Breast Cancer dataset records morphological cell nuclei features extracted from digitized Fine Needle Aspirate (FNA) images to discriminate between:
- **Benign ($B$):** Non-cancerous, localized, non-invasive tumour.
- **Malignant ($M$):** Cancerous, invasive, high-risk tumour.

---

### 2. The Maximal Margin Hyperplane
Given a linearly separable training set $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$ with $\mathbf{x}_i \in \mathbb{R}^d$ and $y_i \in \{-1, +1\}$:
A linear decision hyperplane is defined by:
$$\mathbf{w} \cdot \mathbf{x} - b = 0$$
Where:
- $\mathbf{w}$: Normal weight vector perpendicular to the hyperplane.
- $b$: Bias offset term.

#### Geometric Margin Optimization:
SVM selects the unique hyperplane that maximizes the **margin**—the physical distance between the decision boundary and the closest data points from either class.
The canonical hyperplanes bounding the margin are:
$$\mathbf{w} \cdot \mathbf{x}_i - b \ge +1 \quad \text{if } y_i = +1$$
$$\mathbf{w} \cdot \mathbf{x}_i - b \le -1 \quad \text{if } y_i = -1$$
Combined as:
$$y_i (\mathbf{w} \cdot \mathbf{x}_i - b) \ge 1 \quad \forall i$$

The total margin width equals $\frac{2}{\|\mathbf{w}\|}$. Maximizing the margin is equivalent to solving the constrained quadratic optimization problem:
$$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 \quad \text{subject to } y_i (\mathbf{w} \cdot \mathbf{x}_i - b) \ge 1$$

---

### 3. Support Vectors & Soft-Margin SVM
- **Support Vectors:** The critical subset of data points that lie exactly on the margin hyperplanes ($y_i (\mathbf{w} \cdot \mathbf{x}_i - b) = 1$). All other training points can be removed or relocated without altering the decision boundary.
- **Soft Margin Formulation (C Parameter):** Real-world medical data contains noise and overlap. Soft-margin SVM introduces slack variables $\xi_i \ge 0$ to permit controlled misclassifications:
  $$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^n \xi_i$$
  The regularization hyperparameter $C$ balances margin maximization against training error minimization:
  - Large $C$: Emphasizes low training error (narrower margin; risk of overfitting).
  - Small $C$: Tolerates more margin violations (wider margin; better generalization).

---

### 4. Non-Linear SVM & The Kernel Trick
When data is not linearly separable in input space $\mathbb{R}^d$, SVM implicitly projects data into a higher-dimensional Hilbert space $\mathcal{H}$ via a non-linear mapping $\phi(\mathbf{x})$ where linear separation is achievable.
By the **Mercer Theorem**, this transformation is computed without explicitly calculating high-dimensional coordinates by utilizing a **Kernel Function** $K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i) \cdot \phi(\mathbf{x}_j)$:
1. **Linear Kernel:** $K(\mathbf{x}, \mathbf{z}) = \mathbf{x}^T \mathbf{z}$
2. **Radial Basis Function (RBF / Gaussian) Kernel:**
   $$K(\mathbf{x}, \mathbf{z}) = \exp(-\gamma \|\mathbf{x} - \mathbf{z}\|^2)$$
3. **Polynomial Kernel:** $K(\mathbf{x}, \mathbf{z}) = (\mathbf{x}^T \mathbf{z} + c)^d$

---

### 5. Viva Voce Questions & Answers

**Q1: What are support vectors and why are they crucial?**  
*Answer:* Support vectors are the data points closest to the separating hyperplane that satisfy $y_i(\mathbf{w} \cdot \mathbf{x}_i - b) = 1$. They alone define the orientation and position of the optimal decision boundary. Removing non-support vector training samples has zero effect on the model.

**Q2: What is the Kernel Trick in SVM?**  
*Answer:* The kernel trick enables SVM to learn non-linear decision boundaries by computing the inner products between data points in an implicit high-dimensional feature space without ever explicitly mapping or constructing points in that space, preventing the curse of dimensionality.

**Q3: What is the role of the hyperparameter $C$ in SVM?**  
*Answer:* $C$ is the penalty parameter for misclassification. A high value of $C$ forces a hard margin, strictly penalizing misclassifications and risking overfitting; a lower value allows a soft margin with some classification errors, promoting wider margins and better generalization.

**Q4: Why must identifier columns like `id` and empty columns like `Unnamed: 32` be dropped?**  
*Answer:* `id` is a non-informative administrative key that introduces spurious correlation; trailing empty columns contain non-numeric artifacts that disrupt numerical linear algebra routines.


---

## Step-by-Step Algorithm
1. Import Pandas, train_test_split, SVC, and accuracy_score.
2. Load the Breast Cancer dataset using `pd.read_csv('data.csv')`.
3. Drop non-predictive attributes (`id`, `Unnamed: 32` if present) and extract the feature matrix `X`.
4. Extract the target diagnosis column `y` (`'B'` vs `'M'`).
5. Split the data into 80% training and 20% testing partitions using `train_test_split`.
6. Instantiate the Support Vector Classifier (`SVC()`).
7. Train the classifier on training sets via `model.fit(X_train, y_train)`.
8. Predict diagnosis labels on test samples via `model.predict(X_test)`.
9. Measure and display prediction accuracy using `accuracy_score(y_test, pred)`.

---

## Libraries Required
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
```

---

## Python Program
```python
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
```

---

## Expected Output
```text
Actual Test Labels:    ['B', 'B']
Predicted Test Labels: ['B', 'B']

Accuracy Score: 1.0

```

---

## Result
Successfully constructed and trained a Support Vector Classifier to discriminate between benign and malignant breast tumours.
