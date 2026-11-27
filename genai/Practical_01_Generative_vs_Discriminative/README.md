# Practical 1: Generative vs Discriminative Modeling

**Subject:** Generative AI  
**Level:** Comprehensive Practical Exam Manual  
**Platform:** Python 3 (PyTorch, Scikit-learn, Offline & Self-Contained)

---

## Aim
To implement, compare, and analyze Generative (Gaussian Naive Bayes, PCA) and Discriminative (Logistic Regression) models on handwritten digit data, illustrating classification and synthetic sample generation.

---

## Comprehensive Theory

### 1. Conceptual Distinction: Generative vs. Discriminative Models
In statistical machine learning and artificial intelligence, probabilistic models are divided into two fundamental paradigms based on how they formulate probability distributions over observed data $X$ and target labels $Y$:

1. **Discriminative Models:**
   - **Target:** Directly model the posterior conditional distribution $P(Y | X)$ or learn a direct mapping function $f(X) \to Y$.
   - **Mechanism:** Focus solely on finding the optimal decision boundary that separates classes in the feature space. They ignore how the input features $X$ were originally distributed or generated.
   - **Objective:** Maximize conditional log-likelihood $\sum \log P(y_i | x_i)$.
   - **Representative Algorithms:** Logistic Regression, Support Vector Machines (SVM), Multi-Layer Perceptrons (MLP), Random Forests.
   - **Limitation:** Cannot generate new data samples, synthesize missing attributes, or detect out-of-distribution inputs.

2. **Generative Models:**
   - **Target:** Model the joint probability distribution $P(X, Y) = P(X | Y) P(Y)$, or in unsupervised contexts, the data density $P(X)$.
   - **Mechanism:** Learn the underlying data-generating distribution of each individual class. They capture what features "look like" for a given class.
   - **Inference via Bayes' Rule:** Classification is performed by inverting the joint distribution:
     $$P(Y=c | X) = \frac{P(X | Y=c) P(Y=c)}{\sum_{k} P(X | Y=k) P(Y=k)}$$
   - **Generative Capability:** By sampling from the learned class-conditional density $P(X | Y=c)$, generative models can synthesize novel, unseen data points that mimic the training distribution.
   - **Representative Algorithms:** Gaussian Naive Bayes, Gaussian Mixture Models (GMM), Variational Autoencoders (VAE), Generative Adversarial Networks (GAN), Diffusion Models, Autoregressive Transformers.

---

### 2. Mathematical Formulations

#### A. Discriminative: Multi-Class Logistic Regression (Softmax Regression)
Models class probability directly via the normalized exponential transformation of a linear combination:
$$P(Y=c | X) = \frac{e^{\mathbf{w}_c^T X + b_c}}{\sum_{k=1}^K e^{\mathbf{w}_k^T X + b_k}}$$
Where $\mathbf{w}_c$ is the parameter weight vector for class $c$.

#### B. Generative: Gaussian Naive Bayes (GNB)
Assumes that within each class $c$, each continuous pixel feature $X_j$ follows an independent Gaussian (Normal) distribution parameterized by class mean $\mu_{c, j}$ and class variance $\sigma_{c, j}^2$:
$$P(X_j | Y=c) = \frac{1}{\sqrt{2\pi \sigma_{c, j}^2}} \exp\left( -\frac{(X_j - \mu_{c, j})^2}{2\sigma_{c, j}^2} \right)$$
$$P(X | Y=c) = \prod_{j=1}^d P(X_j | Y=c)$$
**Synthetic Sampling Mechanism:** To generate a novel digit of class $c$, draw independent Gaussian samples for every pixel:
$$\tilde{X}_j \sim \mathcal{N}(\mu_{c, j}, \sigma_{c, j}^2) = \text{g.theta\_}[c, j] + \sqrt{\text{g.var\_}[c, j]} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, 1)$$

#### C. Unsupervised Generative Latent Projection: Principal Component Analysis (PCA)
Projects high-dimensional pixel vectors $X \in \mathbb{R}^{64}$ into a compact $k$-dimensional latent space $Z \in \mathbb{R}^{20}$ using orthogonal principal eigenvectors $W$:
$$Z = (X - \mu) W$$
**Reconstruction / Generation:** Reconstructs the image from its latent representation:
$$\hat{X} = Z W^T + \mu$$
Filters out noise while preserving structural digit geometry.

---

### 3. Comparison Matrix

| Attribute | Discriminative Models | Generative Models |
|---|---|---|
| **What is learned?** | Conditional distribution $P(Y \mid X)$ | Joint distribution $P(X, Y)$ or density $P(X)$ |
| **Primary Goal** | Classify / Discriminate classes | Understand data generation & sample new instances |
| **Generation Capability** | No (cannot synthesize data) | Yes (can draw synthetic samples) |
| **Missing Data Handling** | Poor / Requires preprocessing imputation | Excellent (can marginalize over unobserved variables) |
| **Classification Accuracy** | Typically higher when training data is abundant | Lower asymptotic accuracy due to strong distribution assumptions |
| **Computational Focus** | Optimizes decision boundaries | Optimizes density estimation over the entire feature space |

---

### 4. Viva Voce Questions & Answers

**Q1: What is the fundamental difference between Generative and Discriminative models?**  
*Answer:* Discriminative models learn the conditional probability $P(Y|X)$ to distinguish between classes (finding the boundary). Generative models learn the joint probability $P(X, Y)$ to model how data was generated, enabling both classification via Bayes' rule and the creation of new synthetic samples.

**Q2: Why does Gaussian Naive Bayes allow synthetic sample generation while Logistic Regression does not?**  
*Answer:* Gaussian Naive Bayes explicitly calculates and stores the probability density parameters (mean $\mu$ and variance $\sigma^2$) for every feature given each class ($P(X|Y)$). By sampling random values from $\mathcal{N}(\mu, \sigma^2)$, new synthetic data points can be generated. Logistic Regression only calculates boundary weights $\mathbf{w}$ and possesses no representation of what the inputs look like.

**Q3: How does PCA act as a generative/reconstruction tool?**  
*Answer:* PCA learns an orthogonal latent subspace. By projecting an image into low-dimensional latent coordinates and applying the inverse transformation ($Z W^T + \mu$), it reconstructs the original observation, demonstrating dimensionality reduction and generative decoding.

**Q4: Which type of model typically achieves higher classification accuracy when data is abundant?**  
*Answer:* Discriminative models usually achieve higher asymptotic accuracy because they allocate all model capacity towards optimizing the decision boundary, whereas generative models divide capacity between modeling input distributions and classification.


---

## Step-by-Step Algorithm
1. Load the offline handwritten digits dataset (`load_digits()`) containing $8 \times 8$ grayscale images and scale pixel values to $[0, 1]$ by dividing by $16$.
2. Partition the dataset into 80% training and 20% testing sets using `train_test_split`.
3. Train a discriminative model (`LogisticRegression(max_iter=2000)`) and evaluate its test accuracy.
4. Train a generative model (`GaussianNB()`) and evaluate its test accuracy.
5. Sample novel synthetic digit images for all 10 digits ($0-9$) directly from the learned Gaussian distributions: $X_{new} \sim \mathcal{N}(\mu_c, \sigma_c^2)$.
6. Fit a 20-component PCA model on training data and reconstruct test digit images using `inverse_transform()`.
7. Plot a 2-row comparison grid displaying generated digits in row 1 and PCA-reconstructed digits in row 2.

---

## Libraries Required
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
```

---

## Python Program (Self-Contained & Offline)
```python
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
plt.show()
```

---

## Expected Output
```text
Discriminative (LogisticRegression) acc: 0.9667
Generative (GaussianNB) acc:           0.8250

```

---

## Result
Successfully contrasted discriminative (Logistic Regression) and generative (Gaussian Naive Bayes, PCA) paradigms, proving that discriminative models optimize classification accuracy while generative models synthesize novel data samples.\n