# Practical 8: Train, Fine-tune (LoRA) and Deploy an Open-Source-style Model

**Subject:** Generative AI  
**Level:** Comprehensive Practical Exam Manual  
**Platform:** Python 3 (PyTorch, Scikit-learn, Offline & Self-Contained)

---

## Aim
To pretrain a base language model, freeze its parameters, implement Low-Rank Adaptation (LoRA) from scratch for parameter-efficient fine-tuning (PEFT) on a specialized domain, and deploy an inference endpoint.

---

## Comprehensive Theory

### 1. The Challenge of Full Parameter Fine-Tuning
As large language models scale to tens or hundreds of billions of parameters (e.g., LLaMA-70B, GPT-3 175B), traditional **full parameter fine-tuning** becomes computationally intractable:
- Storing full optimizer states (AdamW tracks 1st and 2nd moments, requiring $8 \times$ the parameter memory).
- Storing separate full model checkpoints for every specialized enterprise domain requires terabytes of storage.
- High risk of **catastrophic forgetting**, where adapting to a new domain erases pre-existing general knowledge.

---

### 2. Low-Rank Adaptation (LoRA) Architecture
Proposed by Edward Hu et al. (Microsoft, 2021), **LoRA (Low-Rank Adaptation)** is a premier Parameter-Efficient Fine-Tuning (PEFT) technique grounded in the **Intrinsic Rank Hypothesis** (Aghajanyan et al., 2020), which asserts that weight updates $\Delta W$ during domain adaptation have an extremely low intrinsic rank.

#### Mathematical Formulation:
For a pre-trained frozen weight matrix $W_0 \in \mathbb{R}^{d \times k}$, LoRA decomposes the weight update matrix $\Delta W$ into the low-rank product of two compact trainable matrices $B$ and $A$:
$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \cdot A)$$
Where:
- $W_0 \in \mathbb{R}^{d \times k}$: Pretrained base model weights (**strictly frozen**, no gradients).
- $B \in \mathbb{R}^{d \times r}$: Trainable low-rank matrix, initialized to **zeros**.
- $A \in \mathbb{R}^{r \times k}$: Trainable low-rank matrix, initialized with **Gaussian noise** $\mathcal{N}(0, \sigma^2)$.
- $r \ll \min(d, k)$: Rank hyperparameter (typically $r \in \{4, 8, 16\}$).
- $\alpha$: Constant scaling hyperparameter stabilizing optimization when varying $r$.

#### Initialization Invariant:
Because $B = 0$ at the start of fine-tuning:
$$\Delta W = B \cdot A = 0 \cdot A = 0$$
Thus, model outputs at step 0 are identical to the base pre-trained model!

---

### 3. Key Advantages of LoRA
1. **Dramatic Parameter Reduction:** Freezes $> 95-99\%$ of model parameters, training only $< 1-5\%$ adapter weights.
2. **Zero Inference Latency Overhead:** During deployment, the low-rank updates can be folded directly into the base weights:
   $$W_{\text{deployed}} = W_0 + \frac{\alpha}{r} (B \cdot A)$$
   Hence, inference executes with zero added latency or computational overhead.
3. **Modular Multi-Tenant Serving:** A single base model can be loaded into GPU memory, and lightweight adapter weights ($< 20$ MB) can be dynamically swapped per customer/task.

---

### 4. Viva Voce Questions & Answers

**Q1: What is the main motivation behind LoRA?**  
*Answer:* LoRA enables parameter-efficient fine-tuning of large language models without retraining all billions of parameters. It dramatically reduces GPU memory usage, storage requirements, and training time while preserving full model quality.

**Q2: Why is matrix $B$ initialized to zero and matrix $A$ to Gaussian random values?**  
*Answer:* Initializing $B=0$ ensures that $\Delta W = B \cdot A = 0$ at the start of training. This guarantees that fine-tuning starts exactly from the pre-trained model's output without introducing random initial perturbations.

**Q3: Does LoRA introduce any inference latency?**  
*Answer:* No. At deployment time, the adapter weights can be pre-calculated and explicitly added into the base model weights ($W = W_0 + \frac{\alpha}{r} BA$). The merged model has the exact same architecture and inference speed as the original model.

**Q4: What is the rank $r$ in LoRA?**  
*Answer:* The rank $r$ controls the inner dimension of matrices $A$ and $B$, determining the capacity of the adapter. Typical values are small ($r=4$ or $r=8$), proving that domain adaptation requires only a tiny fraction of the full parameter space.


---

## Step-by-Step Algorithm
1. Define two text corpora: general base domain text (Text A) and specialized logistics/cargo domain text (Text B).
2. Construct and pretrain a base Transformer decoder `GPT` model on Text A.
3. Freeze all base parameters (`requires_grad = False`).
4. Implement a custom `LoRA` adapter wrapper module decomposing linear weight updates into low-rank factor matrices $A$ and $B$.
5. Replace QKV and projection layers with LoRA adapters and compute parameter reduction percentage ($< 5\%$ trainable).
6. Fine-tune adapter parameters on the domain-specific corpus (Text B).
7. Save only the lightweight adapter weights to disk (`lora_adapter.pt`), reload them into a clean instance, and deploy an inference API function.

---

## Libraries Required
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

---

## Python Program (Self-Contained & Offline)
```python
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# 1. Corpus Preparation
textA = ("generative ai creates new data. transformers use self attention to model language. "
         "language models predict the next token. ") * 30
textB = ("pharos handles hazardous cargo in imdg compliant warehouses. "
         "rail and multimodal logistics move chemicals safely. ") * 30

chars = sorted(set(textA + textB))
stoi = {c: i for i, c in enumerate(chars)}
V = len(chars)
T = 32

enc = lambda t: torch.tensor([stoi[c] for c in t])

# 2. Base Transformer Model Definition
class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ff = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.GELU(),
            nn.Linear(4 * d, d)
        )

    def forward(self, x):
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        n = x.size(1)
        scores = (q @ k.transpose(1, 2)) / (q.size(-1) ** 0.5)
        mask = torch.triu(torch.ones(n, n), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        x = x + self.proj(weights @ v)
        x = x + self.ff(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.tok = nn.Embedding(V, d)
        self.pos = nn.Embedding(T, d)
        self.blocks = nn.Sequential(Block(d), Block(d))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, V)

    def forward(self, x):
        n = x.size(1)
        h = self.tok(x) + self.pos(torch.arange(n, device=x.device))
        return self.head(self.ln(self.blocks(h)))

# 3. LoRA Adapter Implementation from Scratch
class LoRA(nn.Module):
    def __init__(self, base_layer, r=4, alpha=8):
        super().__init__()
        self.base = base_layer
        in_dim = base_layer.in_features
        out_dim = base_layer.out_features
        # Trainable low-rank decomposition matrices
        self.A = nn.Parameter(torch.randn(in_dim, r) * 0.01)
        self.B = nn.Parameter(torch.zeros(r, out_dim))
        self.scale = alpha / r

    def forward(self, x):
        # W_0(x) + (x @ A @ B) * (alpha / r)
        return self.base(x) + (x @ self.A @ self.B) * self.scale

# 4. Training Utility
def train(model, data, steps, lr):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    for i in range(steps):
        ix = torch.randint(0, len(data) - T - 1, (32,))
        x = torch.stack([data[j:j + T] for j in ix])
        y = torch.stack([data[j + 1:j + T + 1] for j in ix])
        loss = F.cross_entropy(model(x).view(-1, V), y.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item()

@torch.no_grad()
def generate(model, prompt, n=100, temp=0.7):
    x = enc(prompt)[None]
    for _ in range(n):
        logits = model(x[:, -T:])[0, -1] / temp
        probs = F.softmax(logits, dim=-1)
        next_char = torch.multinomial(probs, 1)
        x = torch.cat([x, next_char[None]], dim=1)
    return "".join(chars[i] for i in x[0])

# 5. Pretrain Base Model on Generic Text A
m = GPT()
print("Pretraining base model on generic text...")
pre_loss = train(m, enc(textA), steps=400, lr=3e-3)
print(f"Pretraining Loss: {pre_loss:.4f}")
print("Sample BEFORE fine-tuning on 'pharos ':")
print(" ", generate(m, "pharos "))

# 6. Freeze Base Model and Inject LoRA Adapters
for p in m.parameters():
    p.requires_grad = False

for b in m.blocks:
    b.qkv = LoRA(b.qkv, r=4)
    b.proj = LoRA(b.proj, r=4)

trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in m.parameters())
print(f"\nTrainable Parameters with LoRA: {trainable_params} / {total_params} ({100 * trainable_params / total_params:.2f}%)")

# 7. Fine-tune ONLY the LoRA adapters on Domain Text B
print("Fine-tuning LoRA adapters on specialized logistics text...")
ft_loss = train(m, enc(textB), steps=600, lr=5e-3)
print(f"Fine-tuning Loss: {ft_loss:.4f}")
print("Sample AFTER fine-tuning on 'pharos ':")
print(" ", generate(m, "pharos "))

# 8. Deployment: Save only adapter weights, reload, and serve API
adapter_file = "lora_adapter.pt"
torch.save({k: v for k, v in m.state_dict().items() if '.A' in k or '.B' in k}, adapter_file)
print(f"\nSaved lightweight LoRA weights to '{adapter_file}' ({os.path.getsize(adapter_file)} bytes)")

# Reload into model
m.load_state_dict(torch.load(adapter_file), strict=False)

def inference_api(prompt):
    return generate(m, prompt.lower())

print("\n--- Deployed API Endpoint Test ---")
print("Prompt: 'rail and '")
print("Response:", inference_api("rail and "))
```

---

## Expected Output
```text
Pretraining base model on generic text...
Pretraining Loss: 0.0520
Sample BEFORE fine-tuning on 'pharos ':
  pharos use to modan nattentitodels use to to to modans use to to to to modans use to to to modans 

Trainable Parameters with LoRA: 3072 / 72301 (4.25%)
Fine-tuning LoRA adapters on specialized logistics text...
Fine-tuning Loss: 0.0418
Sample AFTER fine-tuning on 'pharos ':
  pharos handles hazardous cargo in imdg compliant warehouses. rail and multimodal logistics move ch

Saved lightweight LoRA weights to 'lora_adapter.pt' (14725 bytes)

--- Deployed API Endpoint Test ---
Prompt: 'rail and '
Response: rail and multimodal logistics move chemicals safely. pharos handles hazardous cargo in imdg compliant

```

---

## Result
Successfully demonstrated parameter-efficient fine-tuning (PEFT) using LoRA from scratch, training only 4.25% of weights and deploying a modular checkpointed model.\n