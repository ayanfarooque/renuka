import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

# 1. Synthesize bimodal ground-truth real data (500 samples from N(0,1), 500 from N(5, 1.5))
np.random.seed(0)
real = np.r_[np.random.normal(0, 1, 500), np.random.normal(5, 1.5, 500)].reshape(-1, 1)

# 2. Fit a single Gaussian distribution
mu, sd = float(real.mean()), float(real.std())
print("Single Gaussian fit: mean=%.2f std=%.2f" % (mu, sd))

# 3. Fit a Gaussian Mixture Model (GMM with 2 components)
gmm = GaussianMixture(n_components=2, random_state=0).fit(real)
print("GMM means:          ", gmm.means_.ravel().round(2))
print("GMM weights (priors):", gmm.weights_.round(2))

# 4. Generate synthetic samples from the fitted GMM
syn, _ = gmm.sample(1000)

# 5. Density evaluation and plotting
x = np.linspace(-5, 10, 300)
plt.figure(figsize=(10, 5))
plt.hist(real, bins=60, density=True, alpha=0.5, color='blue', label="Real Data")
plt.hist(syn, bins=60, density=True, alpha=0.5, color='orange', label="Synthetic Data (GMM)")
plt.plot(x, norm.pdf(x, mu, sd), 'r--', linewidth=2, label="Single Gaussian PDF")
plt.plot(x, np.exp(gmm.score_samples(x.reshape(-1, 1))), 'k', linewidth=2, label="GMM PDF")

plt.legend()
plt.title("Probabilistic Modeling & Synthetic Data: Single Gaussian vs GMM")
plt.xlabel("Value")
plt.ylabel("Probability Density")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()\n