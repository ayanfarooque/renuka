# Practical 6: Diffusion Model (DDPM from scratch)

**Subject:** Generative AI  
**Level:** Comprehensive Practical Exam Manual  
**Platform:** Python 3 (PyTorch, Scikit-learn, Offline & Self-Contained)

---

## Aim
To implement, train, and sample from a Denoising Diffusion Probabilistic Model (DDPM) from scratch in PyTorch, executing forward Gaussian noise corruption and iterative reverse denoising.

---

## Comprehensive Theory

### 1. Introduction to Diffusion Models (DDPM)
Denoising Diffusion Probabilistic Models (DDPM), introduced by Sohl-Dickstein et al. (2015) and popularized by Ho, Jain, & Abbeel (2020), represent the state-of-the-art foundation for visual Generative AI (powering Midjourney, Stable Diffusion, and DALL-E 3).

Unlike GANs (which suffer from adversarial instability and mode collapse) or VAEs (which produce blurry samples due to surrogate loss bounds), Diffusion Models offer **stable training objectives** grounded in non-equilibrium thermodynamics and variational inference.

---

### 2. The Forward (Diffusion) Process
The forward process is a fixed, non-learnable Markov chain that incrementally corrupts clean image data $x_0$ by adding Gaussian noise over $T$ discrete timesteps:
$$q(x_t | x_{t-1}) = \mathcal{N}\left( x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t \mathbf{I} \right)$$
Where $\beta_1 < \beta_2 < \dots < \beta_T$ is a predefined variance schedule.

#### The "Nice Property" (Direct Closed-Form Sampling):
Let $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$. Using the reparameterization trick, we can sample $x_t$ at **any arbitrary timestep $t$ directly from $x_0$ without stepping through intermediate iterations**:
$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$
As $t \to T$, $\bar{\alpha}_T \approx 0$, and $x_T$ becomes indistinguishable from pure isotropic Gaussian noise $\mathcal{N}(0, \mathbf{I})$.

---

### 3. The Reverse (Denoising) Process
The generative task consists of learning the time-reversed conditional distribution:
$$p_\theta(x_{t-1} | x_t) = \mathcal{N}\left( x_{t-1}; \boldsymbol{\mu}_\theta(x_t, t), \sigma_t^2 \mathbf{I} \right)$$
Ho et al. demonstrated that rather than predicting the mean $\boldsymbol{\mu}_\theta$ directly, it is mathematically equivalent and numerically superior to train a neural network $\epsilon_\theta(x_t, t)$ to **predict the exact noise vector $\epsilon$ that was added to $x_0$**.

#### Training Loss:
$$L_{\text{simple}}(\theta) = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$
The network is conditioned on both the corrupted sample $x_t$ and normalized timestep $t / T$.

#### Reverse Ancestral Sampling:
Starting from pure noise $x_T \sim \mathcal{N}(0, \mathbf{I})$, iterate backward for $t = T, T-1, \dots, 1$:
$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z, \quad z \sim \mathcal{N}(0, \mathbf{I})$$
When $t=1$, the stochastic term $z=0$, yielding the clean synthesized image $x_0$.

---

### 4. Comparison: Diffusion vs. GANs

| Dimension | Generative Adversarial Networks (GAN) | Diffusion Models (DDPM) |
|---|---|---|
| **Training Dynamics** | Adversarial minimax game; unstable | Supervised regression (MSE loss); stable |
| **Mode Collapse** | Frequent risk | Practically absent |
| **Sampling Speed** | Instantaneous (single forward pass) | Slow (requires $T$ iterative denoising steps) |
| **Mathematical Basis** | Game theory & Jensen-Shannon divergence | Non-equilibrium thermodynamics & Langevin dynamics |

---

### 5. Viva Voce Questions & Answers

**Q1: What happens during the forward process in a diffusion model?**  
*Answer:* The forward process systematically adds small amounts of Gaussian noise to a clean image over $T$ timesteps according to a variance schedule, gradually destroying all structured content until it becomes pure random noise.

**Q2: What is the training target of the neural network in DDPM?**  
*Answer:* The neural network takes the noisy image $x_t$ and timestep $t$, and is trained via Mean Squared Error (MSE) to predict the exact random Gaussian noise vector $\epsilon$ that was added to the clean image.

**Q3: How does sampling work in reverse diffusion?**  
*Answer:* Starting with pure Gaussian noise $x_T$, the model iteratively predicts the noise component at each step $t$, subtracts it according to the reverse transition formula, and adds a small variance perturbation until reaching a clean synthetic image $x_0$.

**Q4: Why do diffusion models avoid the mode collapse problem seen in GANs?**  
*Answer:* Because diffusion models are trained using maximum likelihood estimation (minimizing the variational lower bound) across all data modes with standard MSE loss, rather than relying on adversarial competition.


---

## Step-by-Step Algorithm
1. Define linear variance schedule $\beta_t \in [10^{-4}, 0.05]$ over $T=100$ steps and compute $\alpha_t = 1 - \beta_t$ and cumulative product $\bar{\alpha}_t$.
2. Implement forward diffusion function: $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$.
3. Define neural network `net` taking concatenated vector $[x_t, t/T]$ and predicting noise vector $\epsilon$.
4. Train network for 4000 steps using Adam optimizer and MSE loss between predicted noise and actual added noise.
5. Implement reverse ancestral sampling: start with pure noise $x_T \sim \mathcal{N}(0, I)$ and iteratively denoise from $t=99$ down to $0$.
6. Plot forward degradation sequence in Row 1 and synthesized images in Rows 2-3.

---

## Libraries Required
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
```

---

## Python Program (Self-Contained & Offline)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

torch.manual_seed(0)

# 1. Load digits data and scale to [-1, 1]
digits = load_digits().data
X = torch.tensor(digits / 8.0 - 1.0, dtype=torch.float32)

# 2. Diffusion Noise Schedule (T = 100 timesteps)
T = 100
beta = torch.linspace(1e-4, 0.05, T)
alpha = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0)

# 3. Denoising Neural Network (takes noisy image + normalized timestep t)
net = nn.Sequential(
    nn.Linear(65, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 64)
)

# 4. Forward Diffusion Function (closed-form sampling at timestep t)
def forward_diffusion(x0, t, noise):
    mean_scale = alpha_bar[t].sqrt()[:, None]
    std_scale = (1.0 - alpha_bar[t]).sqrt()[:, None]
    return mean_scale * x0 + std_scale * noise

# 5. Training Loop: Learn to Predict Added Noise
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
print("Training DDPM to predict noise over 4000 steps...")
for i in range(4001):
    # Sample batch
    batch_idx = torch.randint(0, len(X), (128,))
    x0 = X[batch_idx]
    t = torch.randint(0, T, (128,))
    noise = torch.randn_like(x0)
    
    # Forward corrupt
    xt = forward_diffusion(x0, t, noise)
    
    # Predict noise
    t_norm = (t[:, None] / float(T)).float()
    input_tensor = torch.cat([xt, t_norm], dim=1)
    pred_noise = net(input_tensor)
    
    loss = F.mse_loss(pred_noise, noise)
    opt.zero_grad()
    loss.backward()
    opt.step()
    
    if i % 1000 == 0:
        print(f"Step {i:4d} | MSE Loss: {loss.item():.4f}")

# 6. Reverse Diffusion (Sampling / Denoising from pure Gaussian noise)
@torch.no_grad()
def reverse_diffusion_sample(n=16):
    # Start with isotropic Gaussian noise
    x = torch.randn(n, 64)
    for t in reversed(range(T)):
        t_tensor = torch.full((n, 1), t / float(T))
        pred_eps = net(torch.cat([x, t_tensor], dim=1))
        
        # Reverse step formula
        coef1 = 1.0 / alpha[t].sqrt()
        coef2 = beta[t] / (1.0 - alpha_bar[t]).sqrt()
        x = coef1 * (x - coef2 * pred_eps)
        
        # Add Langevin noise perturbation for t > 0
        if t > 0:
            x = x + beta[t].sqrt() * torch.randn_like(x)
    return x

# 7. Visualization
fig, ax = plt.subplots(3, 8, figsize=(10, 4))

# Row 1: Forward process corruption (clean -> noise)
x0_example = X[:1]
timesteps = [0, 15, 30, 45, 60, 75, 90, 99]
for j, t_val in enumerate(timesteps):
    noisy_img = forward_diffusion(x0_example, torch.tensor([t_val]), torch.randn(1, 64))
    ax[0, j].imshow(noisy_img.view(8, 8), cmap='gray')
    ax[0, j].set_title(f"t={t_val}", fontsize=7)

# Rows 2-3: Reverse generated images
samples = reverse_diffusion_sample(16).view(-1, 8, 8)
for j, im in enumerate(samples):
    ax[1 + j // 8, j % 8].imshow(im, cmap='gray')

for a in ax.flat:
    a.axis('off')
ax[0, 0].set_title("Forward t=0", fontsize=7)
plt.suptitle("DDPM: Forward Process (Row 1) & Reverse Generated Digits (Rows 2-3)")
plt.tight_layout()
plt.show()
```

---

## Expected Output
```text
Training DDPM to predict noise over 4000 steps...
Step    0 | MSE Loss: 1.0524
Step 1000 | MSE Loss: 0.1652
Step 2000 | MSE Loss: 0.1221
Step 3000 | MSE Loss: 0.1043
Step 4000 | MSE Loss: 0.0917

```

---

## Result
Successfully implemented forward Gaussian diffusion degradation and reverse generative sampling using a Denoising Diffusion Probabilistic Model (DDPM).\n