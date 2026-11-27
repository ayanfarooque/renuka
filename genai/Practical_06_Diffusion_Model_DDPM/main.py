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
plt.show()\n