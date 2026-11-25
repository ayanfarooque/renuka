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
