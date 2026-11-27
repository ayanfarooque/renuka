# Practical 7: Neural Style Transfer (+ GAN art)

**Subject:** Generative AI  
**Level:** Comprehensive Practical Exam Manual  
**Platform:** Python 3 (PyTorch, Scikit-learn, Offline & Self-Contained)

---

## Aim
To implement optimization-based Neural Style Transfer (NST) using Gram matrix feature correlation matching to blend content and style representations.

---

## Comprehensive Theory

### 1. Introduction to Neural Style Transfer (NST)
Pioneered by Leon Gatys, Alexander Ecker, and Matthias Bethge (2015), **Neural Style Transfer (NST)** demonstrates that Deep Convolutional Neural Networks (CNNs) develop internal representations that explicitly separate image **content** (semantic layout, geometry, and objects) from image **style** (texture, color palette, brushstrokes).

NST synthesizes a target image $I$ that simultaneously minimizes two competing objective functions:
1. Difference in content representation relative to a Content image $C$.
2. Difference in artistic style representation relative to a Style image $S$.

---

### 2. Deep CNN Representations of Content and Style

#### A. Content Representation
When an image passes through a CNN, higher convolutional layers capture global spatial arrangements while discarding fine-grained pixel color variations.
Let $F^l(X) \in \mathbb{R}^{C_l \times (H_l \times W_l)}$ denote the feature activation map in layer $l$ for image $X$.
The **Content Loss** is the Mean Squared Error (MSE) between feature representations of the target image $I$ and content image $C$:
$$\mathcal{L}_{\text{content}}(C, I) = \frac{1}{2} \sum_{i, j} \left( F^l_{ij}(I) - F^l_{ij}(C) \right)^2$$

#### B. Style Representation & The Gram Matrix
Artistic style is characterized by spatial correlation between different feature channels across the entire image (e.g., how often yellow brush strokes co-occur with swirling textures), independent of spatial position.
The feature correlations are captured by the **Gram Matrix** $G^l \in \mathbb{R}^{C_l \times C_l}$, defined as the inner product of vectorized feature maps:
$$G^l_{ij}(X) = \sum_{k=1}^{H_l W_l} F^l_{ik}(X) F^l_{jk}(X)$$
Where $i$ and $j$ represent channel indices.
The **Style Loss** measures the squared difference between the Gram matrices of image $I$ and style image $S$ across layers:
$$\mathcal{L}_{\text{style}}(S, I) = \sum_{l} w_l \frac{1}{4 C_l^2 (H_l W_l)^2} \sum_{i, j} \left( G^l_{ij}(I) - G^l_{ij}(S) \right)^2$$

---

### 3. Total Loss and Image Optimization
The total optimization objective balances content fidelity and artistic stylization:
$$\mathcal{L}_{\text{total}}(I) = \alpha \mathcal{L}_{\text{content}}(C, I) + \beta \mathcal{L}_{\text{style}}(S, I)$$
Where $\alpha$ and $\beta$ are weighting hyperparameters.

> **Crucial Conceptual Difference:** Unlike standard deep learning where backpropagation updates network weights $W$ while keeping inputs fixed, **NST freezes the network weights entirely and optimizes the input image pixels $I$ directly**:
$$I \leftarrow I - \eta \nabla_I \mathcal{L}_{\text{total}}$$

---

### 4. Viva Voce Questions & Answers

**Q1: What does the Gram matrix capture in Neural Style Transfer?**  
*Answer:* The Gram matrix measures feature correlations between different filter channels across the spatial dimensions of an activation map. It captures global texture, color palettes, and repeating artistic motifs while discarding specific spatial object locations.

**Q2: What is being optimized during Neural Style Transfer?**  
*Answer:* The network parameters are frozen. The target image pixels themselves are treated as the learnable parameters and are optimized directly via backpropagation to minimize the combined content and style loss.

**Q3: Why are deeper convolutional layers used for content loss?**  
*Answer:* Shallow layers capture low-level pixel edges, colors, and textures, while deeper layers encode high-level semantic objects and global spatial layout independent of exact pixel coloring.

**Q4: How does style loss achieve position-invariant stylization?**  
*Answer:* Because the Gram matrix sums feature cross-products over all spatial coordinates ($H \times W$), the spatial location of features is eliminated, leaving only the statistical tendency of visual textures to co-occur.


---

## Step-by-Step Algorithm
1. Procedurally generate high-contrast geometric Content image (circle on gradient) and repetitive textured Style image (sinusoidal patterns).
2. Construct a multi-layer Convolutional feature extractor with frozen weights (`requires_grad=False`).
3. Define Gram matrix function: flatten feature channels and compute normalized inner product $G = V V^T / \text{numel}$.
4. Precompute target content feature maps and target style Gram matrices.
5. Initialize the generated image $I$ as a clone of the content image with gradient tracking enabled (`requires_grad=True`).
6. Optimize image pixels over 300 iterations using the Adam optimizer with combined objective $\mathcal{L}_{total} = \mathcal{L}_{content} + 10^4 \cdot \mathcal{L}_{style}$.
7. Plot Content image, Style image, and the final Stylized result side by side.

---

## Libraries Required
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
```

---

## Python Program (Self-Contained & Offline)
```python
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
plt.show()
```

---

## Expected Output
```text
Optimizing image pixels for Neural Style Transfer...
Step   0 | Content Loss: 0.0000 | Style Loss: 0.000782
Step 100 | Content Loss: 0.0125 | Style Loss: 0.000041
Step 200 | Content Loss: 0.0142 | Style Loss: 0.000028
Step 300 | Content Loss: 0.0151 | Style Loss: 0.000022

```

---

## Result
Successfully implemented Neural Style Transfer by separating content and style representations through deep convolutional activations and Gram matrix optimization.\n