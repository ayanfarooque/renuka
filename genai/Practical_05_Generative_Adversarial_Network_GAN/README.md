# Practical 5: Generative Adversarial Networks (GAN)

**Subject:** Generative AI  
**Level:** Comprehensive Practical Exam Manual  
**Platform:** Python 3 (PyTorch, Scikit-learn, Offline & Self-Contained)

---

## Aim
To implement, train, and evaluate a Generative Adversarial Network (GAN) from scratch in PyTorch to synthesize realistic handwritten digits through adversarial minimax competition.

---

## Comprehensive Theory

### 1. The Generative Adversarial Framework
Formulated by Ian Goodfellow et al. (2014), **Generative Adversarial Networks (GANs)** revolutionized generative modeling by framing generation as an adversarial game between two deep neural sub-networks:

1. **The Generator ($G$):**
   - Takes a random latent noise vector $z \sim p_z(z)$ (typically Gaussian noise $\mathcal{N}(0, I)$) as input.
   - Maps $z$ through neural layers to output a synthetic sample $G(z)$ mimicking the true data distribution $p_{data}$.
   - Acts as a "counterfeiter" attempting to produce indistinguishable synthetic fakes.
2. **The Discriminator ($D$):**
   - Takes an image $x$ (either real from training dataset or fake from $G$) as input.
   - Outputs a scalar probability $D(x) \in [0, 1]$ representing the likelihood that $x$ is genuine rather than generated.
   - Acts as the "detective" attempting to differentiate authentic samples from synthetic counterfeits.

---

### 2. The Minimax Game Formulation
The training objective is formalized as a zero-sum two-player game with value function $V(D, G)$:

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]$$

#### Training Dynamics:
- **Training the Discriminator ($D$):**
  Maximize $V(D, G)$:
  $$\mathcal{L}_D = -\left[ \log D(x_{real}) + \log(1 - D(G(z))) \right]$$
  Implemented using Binary Cross-Entropy with Logits (`nn.BCEWithLogitsLoss()`).
- **Training the Generator ($G$):**
  Minimize $\log(1 - D(G(z)))$. In practice, this saturates early in training when $D$ easily rejects fakes. Hence, Goodfellow proposed the **non-saturating heuristic**: maximize $\log D(G(z))$:
  $$\mathcal{L}_G = -\log D(G(z))$$

#### Theoretical Equilibrium:
At the global Nash Equilibrium, the generator perfectly models the true data distribution $p_g = p_{data}$, and the discriminator is completely baffled:
$$D^*(x) = \frac{1}{2} \quad \forall x$$

---

### 3. Practical Architecture & Training Considerations
- **Latent Dimension $Z$:** 16-dimensional standard Gaussian noise vector.
- **LeakyReLU in Discriminator:** Avoids sparse gradients (dying ReLU problem) by allowing a small negative slope ($0.2$).
- **Tanh in Generator Output:** Forces synthetic pixels strictly into range $[-1, 1]$, matching the scaled real input distribution.
- **Adam Optimizer Settings:** Learning rate $2 \times 10^{-3}$ with momentum $\beta_1 = 0.5$ stabilizes adversarial oscillations.

---

### 4. Common Failure Modes in GANs
1. **Mode Collapse:** The generator finds a single output pattern that fools $D$ and collapses to producing only that digit repeatedly, ignoring diversity.
2. **Vanishing Gradients:** If $D$ learns much faster than $G$, $D(G(z)) \to 0$ and gradients vanish, terminating $G$'s learning.
3. **Non-Convergence:** The adversarial game oscillates wildly without reaching equilibrium.

---

### 5. Viva Voce Questions & Answers

**Q1: What is the roles of the Generator and Discriminator in a GAN?**  
*Answer:* The Generator transforms random noise into synthetic images to deceive the Discriminator. The Discriminator evaluates images and classifies whether they are real training samples or synthetic fakes.

**Q2: What is Mode Collapse in GAN training?**  
*Answer:* Mode collapse occurs when the generator learns to produce only a limited variety of outputs (or a single class/sample) that reliably fools the discriminator, failing to capture the full diversity of the real training distribution.

**Q3: Why is LeakyReLU preferred over ReLU in the Discriminator?**  
*Answer:* Standard ReLU outputs zero for negative inputs, causing dead neurons and zero gradients. LeakyReLU provides a small non-zero slope (e.g., $0.2$), ensuring gradient flow throughout the network even for non-activated regions.

**Q4: What is the theoretical optimal value of the Discriminator at convergence?**  
*Answer:* When the generator reaches optimal distribution matching ($p_g = p_{data}$), the discriminator cannot distinguish between real and fake, outputting probability $0.5$ (or logit $0$) for all inputs.


---

## Step-by-Step Algorithm
1. Load the offline digits dataset, normalize pixels from $[0, 16]$ to $[-1, 1]$, and convert to PyTorch tensors.
2. Define Generator network $G: \mathbb{R}^{16} \to \mathbb{R}^{64}$ with Linear, ReLU, and Tanh activation layers.
3. Define Discriminator network $D: \mathbb{R}^{64} \to \mathbb{R}^{1}$ with Linear and LeakyReLU activations.
4. Initialize Adam optimizers with $\beta_1 = 0.5$ and BCEWithLogitsLoss.
5. In each training step:
   - Sample batch of real images and batch of latent noise vectors.
   - Train $D$ on real images (target 1) and fake images (target 0) with detached generator graphs.
   - Train $G$ by evaluating fake images through $D$ against target 1.
6. Sample 16 latent vectors from $\mathcal{N}(0, I)$ and render a $2 \times 8$ grid of generated digits.

---

## Libraries Required
```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
```

---

## Python Program (Self-Contained & Offline)
```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

torch.manual_seed(0)

# 1. Load data and normalize to [-1, 1]
digits = load_digits().data
X = torch.tensor(digits / 8.0 - 1.0, dtype=torch.float32)

# 2. Network Definitions
Z = 16  # Latent noise dimension

# Generator: Latent noise Z -> 64-pixel image (Tanh output in [-1, 1])
G = nn.Sequential(
    nn.Linear(Z, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.Tanh()
)

# Discriminator: 64-pixel image -> Real/Fake logit
D = nn.Sequential(
    nn.Linear(64, 128),
    nn.LeakyReLU(0.2),
    nn.Linear(128, 1)
)

# 3. Optimizers & Loss Function
oG = torch.optim.Adam(G.parameters(), lr=2e-3, betas=(0.5, 0.999))
oD = torch.optim.Adam(D.parameters(), lr=2e-3, betas=(0.5, 0.999))
bce = nn.BCEWithLogitsLoss()

# Target label tensors (batch size 64)
batch_size = 64
one = torch.ones(batch_size, 1)
zero = torch.zeros(batch_size, 1)

# 4. Adversarial Training Loop
print("Training GAN over 3000 adversarial steps...")
for i in range(3001):
    # Sample real and fake data
    idx = torch.randint(0, len(X), (batch_size,))
    real_imgs = X[idx]
    noise = torch.randn(batch_size, Z)
    fake_imgs = G(noise)

    # Train Discriminator: Maximize log(D(x)) + log(1 - D(G(z)))
    ld = bce(D(real_imgs), one) + bce(D(fake_imgs.detach()), zero)
    oD.zero_grad()
    ld.backward()
    oD.step()

    # Train Generator: Maximize log(D(G(z)))
    lg = bce(D(fake_imgs), one)
    oG.zero_grad()
    lg.backward()
    oG.step()

    if i % 500 == 0:
        print(f"Step {i:4d} | D Loss: {ld.item():.4f} | G Loss: {lg.item():.4f}")

# 5. Generate and Visualize Synthetic Digits
with torch.no_grad():
    sample_noise = torch.randn(16, Z)
    generated_images = G(sample_noise).view(-1, 8, 8)

fig, ax = plt.subplots(2, 8, figsize=(10, 2.8))
for a, im in zip(ax.flat, generated_images):
    a.imshow(im, cmap='gray')
    a.axis('off')
plt.suptitle("Synthesized Digits Generated by Trained GAN")
plt.tight_layout()
plt.show()
```

---

## Expected Output
```text
Training GAN over 3000 adversarial steps...
Step    0 | D Loss: 1.4111 | G Loss: 0.6974
Step  500 | D Loss: 0.7712 | G Loss: 1.2588
Step 1000 | D Loss: 0.8924 | G Loss: 1.1895
Step 1500 | D Loss: 0.8143 | G Loss: 1.3411
Step 2000 | D Loss: 0.8521 | G Loss: 1.1214
Step 2500 | D Loss: 0.8402 | G Loss: 1.1543
Step 3000 | D Loss: 0.7915 | G Loss: 1.2180

```

---

## Result
Successfully implemented and trained a Generative Adversarial Network from scratch, producing novel handwritten digit images via adversarial minimax optimization.\n