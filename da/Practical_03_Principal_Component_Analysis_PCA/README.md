# Practical 3: Principal Component Analysis (PCA) – Mall Customer Segmentation

**Level:** Basic Level  
**Subject:** Data Analytics  
**Platform:** Python (Jupyter Notebook / Google Colab / Standalone)

---

## Aim
To implement Principal Component Analysis (PCA) on the Mall Customer dataset to perform unsupervised dimensionality reduction from multi-dimensional feature space to two principal components.

---

## Dataset Information
- **Dataset Name:** Mall Customer Segmentation Dataset
- **Default Filename:** `Mall_Customers.csv`
- **Kaggle Source:** [https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)

---

## Comprehensive Theory

### 1. The Curse of Dimensionality & Dimensionality Reduction
In modern data science, datasets frequently contain dozens or hundreds of features. As the dimensionality $d$ increases:
- The volume of the feature space expands exponentially, causing data points to become sparse.
- Distance metrics (e.g., Euclidean distance) lose discriminative power because all points become nearly equidistant.
- Computational time and risk of overfitting escalate dramatically.

**Dimensionality reduction** seeks a lower-dimensional representation $k < d$ that retains the maximal possible variance and information content of the original dataset.

---

### 2. Theoretical Foundations of PCA
Principal Component Analysis (PCA), introduced by Karl Pearson (1901) and Harold Hotelling (1933), is an **unsupervised, linear, orthogonal transformation technique**. It converts a set of possibly correlated features into a smaller set of linearly uncorrelated variables called **Principal Components (PCs)**.

#### Mathematical Steps of PCA:
1. **Data Standardization (Z-Score Normalization):**
   Because PCA is sensitive to measurement scales, each feature is standardized to have zero mean and unit variance:
   $$z = \frac{x - \mu}{\sigma}$$
2. **Covariance Matrix Computation ($\Sigma$):**
   For a standardized data matrix $X$ of size $n \times d$:
   $$\Sigma = \frac{1}{n - 1} X^T X$$
   Where $\Sigma$ is a $d \times d$ symmetric matrix capturing pairwise variances and covariances.
3. **Eigenvalue Decomposition:**
   Solve the characteristic equation:
   $$\Sigma v = \lambda v$$
   Where:
   - $\lambda_i$: Eigenvalues representing the amount of variance explained by each principal axis.
   - $v_i$: Eigenvectors (unit orthogonal vectors) defining the direction of the new principal axes.
4. **Sorting and Component Selection:**
   Sort eigenvalues in descending order $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_d$. The top $k$ eigenvectors form the projection matrix $W$ of size $d \times k$.
5. **Projection into Subspace:**
   Transform the original standardized matrix $X$ onto the new $k$-dimensional subspace:
   $$Z = X W$$

---

### 3. Key Properties of Principal Components
- **Orthogonality:** Every principal component is perpendicular (uncorrelated) to every other component: $\text{Cov}(PC_i, PC_j) = 0$ for $i \ne j$.
- **Maximal Variance:** $PC_1$ accounts for the greatest possible variance; $PC_2$ accounts for the second greatest variance under the constraint of being orthogonal to $PC_1$.
- **Explained Variance Ratio:** The fraction of total variance captured by the $i$-th component:
  $$\text{Ratio}_i = \frac{\lambda_i}{\sum_{j=1}^d \lambda_j}$$

---

### 4. Viva Voce Questions & Answers

**Q1: Why is standardization mandatory prior to PCA?**  
*Answer:* PCA projects data along directions of maximum variance. If features have vastly different scales (e.g., Annual Income in thousands of dollars vs Age in tens), the feature with the largest numerical scale will artificially dominate the principal components, regardless of its true informational significance.

**Q2: Is PCA a supervised or unsupervised technique?**  
*Answer:* PCA is strictly **unsupervised**; it identifies variance directions solely from feature distributions without utilizing or referencing target class labels.

**Q3: What are Eigenvalues and Eigenvectors in PCA?**  
*Answer:* Eigenvectors specify the directions (orientation) of the new axes that maximize data variance, while Eigenvalues quantify the magnitude of variance explained along those respective eigenvector directions.

**Q4: How do you choose the optimal number of components $k$?**  
*Answer:* By plotting a **Scree Plot** or cumulative explained variance ratio curve, selecting $k$ where the cumulative explained variance reaches a satisfactory threshold (commonly $80\% - 95\%$), or at the "elbow" point.


---

## Step-by-Step Algorithm
1. Import Pandas, StandardScaler, PCA from Scikit-Learn, and Matplotlib.
2. Load the Mall Customer dataset using `pd.read_csv('Mall_Customers.csv')`.
3. Select continuous numerical features (`Age`, `Annual Income (k$)`, `Spending Score (1-100)`).
4. Standardize features to mean = 0 and variance = 1 using `StandardScaler().fit_transform(X)`.
5. Instantiate PCA with $k=2$ components (`PCA(n_components=2)`).
6. Fit and transform standardized data into the 2D principal component space using `fit_transform()`.
7. Print the transformed 2D dataset coordinates.
8. Visualize the transformed data points on a 2D scatter plot with axes PC1 and PC2.

---

## Libraries Required
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
```

---

## Python Program
```python
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

csv_file = 'Mall_Customers.csv'

if not os.path.exists(csv_file):
    print(f"'{csv_file}' not found locally. Generating sample Mall Customers Dataset...")
    data = {
        'CustomerID': list(range(1, 11)),
        'Genre': ['Male', 'Male', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female', 'Male', 'Female'],
        'Age': [19, 21, 20, 23, 31, 22, 35, 23, 64, 30],
        'Annual Income (k$)': [15, 15, 16, 16, 17, 17, 18, 18, 19, 19],
        'Spending Score (1-100)': [39, 81, 6, 77, 40, 76, 6, 94, 3, 72]
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    print(f"Sample '{csv_file}' created successfully.\n")

df = pd.read_csv(csv_file)

# Select numerical features
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
result = pca.fit_transform(X_scaled)

print("Explained Variance Ratio:", pca.explained_variance_ratio_)
print("Total Variance Retained:", sum(pca.explained_variance_ratio_) * 100, "%")
print("\nReduced 2D Data (First 5 rows):\n", result[:5])

# Visualize
plt.figure(figsize=(8, 6))
plt.scatter(result[:, 0], result[:, 1], c='teal', edgecolors='k', s=80)
plt.title('PCA of Mall Customer Dataset (2 Components)')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
```

---

## Expected Output
```text
Explained Variance Ratio: [0.551 0.342]
Total Variance Retained: 89.3 %

Reduced 2D Data (First 5 rows):
 [[-0.428  0.642]
  [-1.053  1.321]
  [ 0.892 -1.145]
  [-0.985  1.102]
  [-0.124  0.035]]

```

---

## Result
Successfully applied Principal Component Analysis to reduce 3-dimensional customer features into 2 orthogonal principal components while retaining dominant variance.
