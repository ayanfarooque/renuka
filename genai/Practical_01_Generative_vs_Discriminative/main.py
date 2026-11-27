import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA

# 1. Load offline dataset and normalize
X, y = load_digits(return_X_y=True)
X = X / 16.0  # Normalize pixel intensity to [0, 1]

# 2. Split dataset
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)

# 3. Discriminative Model: Logistic Regression (models P(y|x))
disc_model = LogisticRegression(max_iter=2000)
disc_model.fit(Xtr, ytr)
disc_acc = disc_model.score(Xte, yte)
print("Discriminative (LogisticRegression) acc:", round(disc_acc, 4))

# 4. Generative Model: Gaussian Naive Bayes (models P(x|y)P(y))
gen_model = GaussianNB()
gen_model.fit(Xtr, ytr)
gen_acc = gen_model.score(Xte, yte)
print("Generative (GaussianNB) acc:          ", round(gen_acc, 4))

# 5. Visualization: Sample NEW digits and reconstruct test digits
fig, ax = plt.subplots(2, 10, figsize=(12, 3))

# Row 1: Sample NEW digits from learned class-conditional distributions P(x|y=c)
for c in range(10):
    sample = np.random.normal(gen_model.theta_[c], np.sqrt(gen_model.var_[c]))
    ax[0, c].imshow(sample.reshape(8, 8), cmap='gray')
    ax[0, c].axis('off')

# Row 2: Reconstruct test images using PCA latent space
pca = PCA(n_components=20).fit(Xtr)
reconstructed = pca.inverse_transform(pca.transform(Xte[:10]))
for i in range(10):
    ax[1, i].imshow(reconstructed[i].reshape(8, 8), cmap='gray')
    ax[1, i].axis('off')

ax[0, 0].set_title("Generated (GNB)", fontsize=9, loc='left')
ax[1, 0].set_title("Reconstructed (PCA)", fontsize=9, loc='left')
plt.suptitle("Generative vs Discriminative Modeling Demonstration")
plt.tight_layout()
plt.show()\n