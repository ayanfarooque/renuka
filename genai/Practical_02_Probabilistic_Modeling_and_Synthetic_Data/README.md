# Practical 2: Probabilistic Modeling and Synthetic Data Generation

**Subject:** Generative AI  
**Level:** Comprehensive Practical Exam Manual  
**Platform:** Python 3 (PyTorch, Scikit-learn, Offline & Self-Contained)

---

## Aim
To model complex multimodal continuous data distributions using single Gaussian vs. Gaussian Mixture Models (GMM) and synthesize novel realistic data samples.

---

## Comprehensive Theory

### 1. Foundations of Probabilistic Modeling
Probabilistic modeling is the statistical cornerstone of Generative AI. Given an unobserved data-generating process that produces observable real-world data points $x \in \mathcal{X}$, probabilistic modeling seeks to estimate a parameterized probability density function $p_\theta(x)$ that closely approximates the true empirical distribution $p_{data}(x)$.

Once a probability density function $p_\theta(x)$ is estimated, it serves two critical functions:
1. **Density Estimation:** Quantifying the likelihood of any observed sample $x$ (useful for anomaly detection and likelihood evaluation).
2. **Generative Sampling:** Drawing new pseudo-random samples $\tilde{x} \sim p_\theta(x)$ that never existed before but match the statistical properties of the original population.

---

### 2. Parametric Single Gaussian vs. Semi-Parametric Mixture Models

#### A. Single Gaussian Distribution (Unimodal)
A standard single Gaussian assumes that data is concentrated around a single central mean $\mu$ with dispersion variance $\sigma^2$:
$$p(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)$$
**Parameter Estimation:** Direct Maximum Likelihood Estimation (MLE):
$$\hat{\mu} = \frac{1}{N} \sum_{i=1}^N x_i, \quad \hat{\sigma}^2 = \frac{1}{N} \sum_{i=1}^N (x_i - \hat{\mu})^2$$
**Limitation:** Fails completely when data is **multimodal** (e.g., mixtures of populations, financial transactions, or multi-cluster measurements). Forcing a single Gaussian over a bimodal distribution places peak probability in the middle valley where zero real data points actually exist!

#### B. Gaussian Mixture Models (GMM - Multimodal)
A Gaussian Mixture Model represents an arbitrary complex distribution as a convex linear combination of $K$ distinct Gaussian components:
$$p(x | \boldsymbol{\theta}) = \sum_{k=1}^K \pi_k \mathcal{N}(x | \mu_k, \sigma_k^2)$$
Where:
- $K$: Number of mixture components.
- $\pi_k$: Mixing coefficient / prior probability of component $k$, satisfying $\sum_{k=1}^K \pi_k = 1$ and $\pi_k \ge 0$.
- $\mu_k, \sigma_k^2$: Mean and variance of the $k$-th Gaussian sub-population.

---

### 3. The Expectation-Maximization (EM) Algorithm
Because the identity of which Gaussian component generated each sample is a hidden (latent) variable $z$, standard analytical MLE cannot be solved directly. GMM parameters are learned using the **Expectation-Maximization (EM)** algorithm:

1. **E-Step (Expectation):** Calculate the posterior probability (responsibility $\gamma_{ik}$) that component $k$ generated sample $x_i$:
   $$\gamma_{ik} = P(z_i = k | x_i) = \frac{\pi_k \mathcal{N}(x_i | \mu_k, \sigma_k^2)}{\sum_{j=1}^K \pi_j \mathcal{N}(x_i | \mu_j, \sigma_j^2)}$$
2. **M-Step (Maximization):** Re-estimate parameters using the weighted responsibilities:
   $$N_k = \sum_{i=1}^N \gamma_{ik}$$
   $$\hat{\pi}_k = \frac{N_k}{N}, \quad \hat{\mu}_k = \frac{1}{N_k} \sum_{i=1}^N \gamma_{ik} x_i, \quad \hat{\sigma}_k^2 = \frac{1}{N_k} \sum_{i=1}^N \gamma_{ik} (x_i - \hat{\mu}_k)^2$$
3. Iterate until the total log-likelihood $\sum_i \log p(x_i)$ converges.

---

### 4. Synthetic Data Generation Process
Sampling from a fitted GMM follows ancestral sampling:
1. Choose a component index $k \in \{1, \dots, K\}$ according to the categorical distribution $\text{Categorical}(\pi_1, \dots, \pi_K)$.
2. Draw sample $\tilde{x}$ from the selected Gaussian component: $\tilde{x} \sim \mathcal{N}(\mu_k, \sigma_k^2)$.

---

### 5. Viva Voce Questions & Answers

**Q1: Why does a single Gaussian distribution fail on bimodal data?**  
*Answer:* A single Gaussian is constrained to be symmetric and unimodal. When fitted to bimodal data, it sets its mean in the intermediate gap between the two modes, assigning highest probability density to a region where no actual data exists.

**Q2: What is the role of the Expectation-Maximization (EM) algorithm in GMM?**  
*Answer:* EM resolves the chicken-and-egg problem of unknown latent cluster memberships. The E-step calculates the responsibility of each Gaussian component for each data point, and the M-step updates the means, variances, and mixture weights to maximize likelihood.

**Q3: How do you sample synthetic data from a trained GMM?**  
*Answer:* First, pick one Gaussian component $k$ at random based on mixture weights $\pi_k$. Second, draw a sample from that chosen Gaussian distribution using its mean $\mu_k$ and standard deviation $\sigma_k$.

**Q4: How do you choose the optimal number of mixture components $K$?**  
*Answer:* By computing model selection criteria such as Akaike Information Criterion (AIC) or Bayesian Information Criterion (BIC) across various values of $K$ and choosing the $K$ that minimizes AIC/BIC.


---

## Step-by-Step Algorithm
1. Set random seed for reproducibility and generate synthetic bimodal real data by combining two distinct Gaussians ($\mathcal{N}(0, 1)$ and $\mathcal{N}(5, 1.5)$).
2. Fit a naive single Gaussian model by computing empirical mean $\mu$ and standard deviation $\sigma$.
3. Fit a 2-component Gaussian Mixture Model (`GaussianMixture(2)`) using the Expectation-Maximization algorithm.
4. Synthesize 1000 novel data points by calling `gmm.sample(1000)`.
5. Compute probability density functions across a continuous evaluation grid for both the single Gaussian and the GMM.
6. Plot superimposed histograms comparing real vs. synthetic data alongside analytical PDF curves.

---

## Libraries Required
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
```

---

## Python Program (Self-Contained & Offline)
```python
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
plt.show()
```

---

## Expected Output
```text
Single Gaussian fit: mean=2.53 std=2.83
GMM means:           [0.05 4.96]
GMM weights (priors): [0.49 0.51]

```

---

## Result
Successfully demonstrated density estimation on multimodal data, proving that GMMs accurately capture distinct data clusters and generate high-fidelity synthetic distributions.\n