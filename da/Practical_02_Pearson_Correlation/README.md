# Practical 2: Pearson Correlation – PIMA Indians Diabetes Dataset

**Level:** Basic Level  
**Subject:** Data Analytics  
**Platform:** Python (Jupyter Notebook / Google Colab / Standalone)

---

## Aim
To compute and interpret the Pearson Correlation coefficient matrix in Python and visualize inter-variable dependencies using a Seaborn heatmap on the PIMA Indians Diabetes Dataset.

---

## Dataset Information
- **Dataset Name:** PIMA Indians Diabetes Dataset
- **Default Filename:** `diabetes.csv`
- **Kaggle Source:** [https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

---

## Comprehensive Theory

### 1. Concept of Statistical Association
In bivariate and multivariate data analysis, understanding how variables co-vary is fundamental to exploratory data analysis (EDA), feature selection, and identifying multicollinearity. **Correlation** quantifies the strength and direction of the linear relationship between two continuous quantitative variables.

---

### 2. Pearson Product-Moment Correlation Coefficient ($r$)
Developed by Karl Pearson, the Pearson correlation coefficient $r$ is the normalized measure of covariance between two variables $X$ and $Y$:

$$r = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y} = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^n (x_i - \bar{x})^2} \sqrt{\sum_{i=1}^n (y_i - \bar{y})^2}}$$

Where:
- $x_i, y_i$: Individual sample observations.
- $\bar{x}, \bar{y}$: Arithmetic sample means of $X$ and $Y$.
- $n$: Total number of observations.

#### Underlying Assumptions of Pearson Correlation:
1. **Continuous Scale:** Both variables are measured at the interval or ratio scale.
2. **Linear Relationship:** The association between variables must be linear (scatter plots form an ellipse rather than a curve).
3. **Bivariate Normality:** Both variables exhibit approximately normal distributions.
4. **Homoscedasticity:** The variance of residuals is constant across all values of the independent variable.
5. **Absence of Extreme Outliers:** Severe outliers can distort $r$ substantially.

---

### 3. Interpretation of the Correlation Coefficient

| Correlation Value ($r$) | Magnitude & Direction | Practical Implication |
|---|---|---|
| $+1.0$ | Perfect Positive | Direct linear relationship; as $X$ increases, $Y$ increases proportionally. |
| $+0.70$ to $+0.99$ | Very Strong Positive | High linear dependency; candidate for feature redundancy/multicollinearity. |
| $+0.40$ to $+0.69$ | Moderate Positive | Noticeable upward trend; useful predictive association. |
| $-0.39$ to $+0.39$ | Weak or Negligible | Little to no linear relationship between variables. |
| $-0.40$ to $-0.69$ | Moderate Negative | Noticeable downward trend; as $X$ increases, $Y$ decreases. |
| $-0.70$ to $-0.99$ | Very Strong Negative | Strong inverse linear dependency. |
| $-1.0$ | Perfect Negative | Exact inverse linear relationship. |
| $0.0$ | Null / Zero | No linear association whatsoever. |

> **Critical Principle:** *Correlation does not imply causation.* A high correlation between two variables does not prove that changes in one cause changes in the other; the association may be mediated by confounding lurking variables.

---

### 4. Correlation Matrix & Visual Heatmaps
When analyzing a dataset with $p$ numerical attributes, the correlation matrix is a symmetric $p \times p$ square matrix where entry $C_{ij} = r(X_i, X_j)$.
- The main diagonal entries $C_{ii}$ are identically $1.0$ (self-correlation).
- Seaborn's `heatmap()` maps numerical coefficients to diverging color palettes (e.g., `coolwarm`), with numerical annotation overlay (`annot=True`), enabling instant visual identification of strong predictors (e.g., Glucose vs Outcome in Diabetes).

---

### 5. Viva Voce Questions & Answers

**Q1: What is the primary difference between Covariance and Correlation?**  
*Answer:* Covariance measures the directional relationship between two variables but is unstandardized and dependent on units of measurement (scale-dependent). Correlation standardizes covariance by dividing it by the product of the standard deviations, yielding a unitless metric bounded strictly between $-1$ and $+1$.

**Q2: What happens if two non-linear variables have a strong relationship?**  
*Answer:* Pearson correlation will underestimate or fail to capture the relationship (it may output $r \approx 0$ for a perfect quadratic curve $Y = X^2$). In non-linear cases, Spearman's rank correlation or mutual information is used instead.

**Q3: Why is detecting high inter-feature correlation important before training machine learning models?**  
*Answer:* High correlation between two predictor variables ($r > 0.8$) indicates **multicollinearity**. In linear and logistic regression, multicollinearity destabilizes coefficient estimates, inflates standard errors, and impedes feature interpretability.

**Q4: Which Seaborn parameters customize the correlation heatmap?**  
*Answer:* `annot=True` overlays the correlation values inside each cell; `cmap='coolwarm'` sets the diverging color map; `fmt='.2f'` sets decimal precision.


---

## Step-by-Step Algorithm
1. Import required libraries (`pandas`, `seaborn`, `matplotlib.pyplot`).
2. Load the PIMA Indians Diabetes Dataset using `pd.read_csv('diabetes.csv')`.
3. Compute the pairwise Pearson correlation matrix for all numeric columns using `df.corr(method='pearson')`.
4. Print the computed numerical correlation matrix.
5. Create a visual correlation heatmap using `sns.heatmap(corr, annot=True, cmap='coolwarm')`.
6. Display the visualization window using `plt.show()`.

---

## Libraries Required
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```

---

## Python Program
```python
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
```

---

## Expected Output
```text
Pearson Correlation Matrix:
                          Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin    BMI    DPF    Age  Outcome
Pregnancies                     1.000    0.129          0.141         -0.082   -0.074  0.018  0.034  0.544    0.222
Glucose                         0.129    1.000          0.153          0.057    0.331  0.221  0.137  0.264    0.467
BloodPressure                   0.141    0.153          1.000          0.207    0.089  0.282  0.041  0.240    0.065
SkinThickness                  -0.082    0.057          0.207          1.000    0.437  0.393  0.184 -0.114    0.075
Insulin                        -0.074    0.331          0.089          0.437    1.000  0.198  0.185 -0.042    0.131
BMI                             0.018    0.221          0.282          0.393    0.198  1.000  0.141  0.036    0.293
DPF                             0.034    0.137          0.041          0.184    0.185  0.141  1.000  0.034    0.174
Age                             0.544    0.264          0.240         -0.114   -0.042  0.036  0.034  1.000    0.238
Outcome                         0.222    0.467          0.065          0.075    0.131  0.293  0.174  0.238    1.000

```

---

## Result
Successfully computed Pearson correlation matrix for diabetes risk factors and visualized inter-feature relationships using a Seaborn heatmap.
