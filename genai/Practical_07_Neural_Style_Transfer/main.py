import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(0)

# 1. Procedurally synthesize Content and Style images in code (offline, self-contained)
S = 64
yy, xx = torch.meshgrid(torch.linspace(-1, 1, S), torch.linspace(-1, 1, S), indexing='ij')

# Content: Geometric circle on smooth spatial gradient
content = torch.stack([
    (xx**2 + yy**2 < 0.5).float(),
    yy * 0.5 + 0.5,
    xx * 0.5 + 0.5
])[None]  # Shape: (1, 3, 64, 64)

# Style: High-frequency sinusoidal wave interference pattern
style = torch.stack([
    (torch.sin(xx * 20 + yy * 10) > 0).float(),
    (torch.sin(yy * 15) > 0).float(),
    torch.sin(xx * 10) * 0.5 + 0.5
])[None]  # Shape: (1, 3, 64, 64)

# 2. Fixed Convolutional Feature Extractor
layers = nn.ModuleList([
    nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU()),
    nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.ReLU()),
    nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU())
])
for p in layers.parameters():
    p.requires_grad = False

def extract_features(x):
    outputs = []
    for layer in layers:
        x = layer(x)
        outputs.append(x)
    return outputs

def gram_matrix(feat):
    # feat: (1, C, H, W) -> flatten to (C, H*W)
    v = feat[0].flatten(1)
    return (v @ v.T) / v.numel()

# 3. Target Representations
content_targets = [f.detach() for f in extract_features(content)]
style_targets = [gram_matrix(f).detach() for f in extract_features(style)]

# 4. Target Image Initialization (optimized directly)
img = content.clone().requires_grad_(True)
opt = torch.optim.Adam([img], lr=0.05)

# 5. Optimization Loop
print("Optimizing image pixels for Neural Style Transfer...")
for i in range(301):
    feats = extract_features(img)
    
    # Content loss (from high-level layer 2)
    lc = F.mse_loss(feats[2], content_targets[2])
    
    # Style loss (summed across all layers)
    ls = sum(F.mse_loss(gram_matrix(f), g) for f, g in zip(feats, style_targets))
    
    total_loss = lc + 1e4 * ls
    
    opt.zero_grad()
    total_loss.backward()
    opt.step()
    
    if i % 100 == 0:
        print(f"Step {i:3d} | Content Loss: {lc.item():.4f} | Style Loss: {ls.item():.6f}")

# 6. Visualization
fig, ax = plt.subplots(1, 3, figsize=(9, 3))
titles = ["Content Image", "Style Image", "Stylized Output"]
images = [content, style, img.detach().clamp(0, 1)]

for a, im, t in zip(ax, images, titles):
    a.imshow(im[0].permute(1, 2, 0).numpy())
    a.set_title(t)
    a.axis('off')

plt.suptitle("Neural Style Transfer using Gram Matrix Optimization")
plt.tight_layout()
plt.show()\n