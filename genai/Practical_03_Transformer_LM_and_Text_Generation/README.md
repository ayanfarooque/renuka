# Practical 3: Transformer Language Model + Text Generation

**Subject:** Generative AI  
**Level:** Comprehensive Practical Exam Manual  
**Platform:** Python 3 (PyTorch, Scikit-learn, Offline & Self-Contained)

---

## Aim
To build, train, and sample from an autoregressive Decoder-only Transformer language model from scratch in PyTorch, implementing explicit self-attention and causal masking.

---

## Comprehensive Theory

### 1. Introduction to the Transformer Architecture
Introduced by Vaswani et al. (2017) in *"Attention Is All You Need"*, the **Transformer** has superseded Recurrent Neural Networks (RNNs) and LSTMs to become the bedrock architecture for modern generative large language models (GPT-4, LLaMA, Claude, Gemini).

Unlike sequential RNNs that process tokens one by one (inhibiting GPU parallelization and suffering from catastrophic forgetting over long context windows), Transformers process entire token sequences concurrently through **Self-Attention mechanisms**.

---

### 2. Anatomy of Self-Attention & Causal Masking

#### A. Scaled Dot-Product Attention
Given an input embedding sequence $X \in \mathbb{R}^{T \times d}$, the model projects each token into three distinct latent representations via learnable linear projections:
- **Queries ($Q = X W_Q$):** What the current token is seeking.
- **Keys ($K = X W_K$):** What information the token holds.
- **Values ($V = X W_V$):** The contextual payload to extract.

The attention weights are computed as normalized pairwise dot-products:
$$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} + M \right) V$$
- **Scaling Factor $\frac{1}{\sqrt{d_k}}$:** Counteracts variance expansion as dimension $d_k$ grows, preventing softmax gradients from vanishing.

#### B. Causal (Autoregressive) Masking
In generative language modeling, text generation is strictly **autoregressive**: token $x_t$ must only depend on preceding tokens $x_{<t}$, never on future tokens $x_{>t}$.
To enforce this causality during parallelized training, an upper-triangular mask $M$ is added before softmax:
$$M_{ij} = \begin{cases} 0 & \text{if } i \ge j \\ -\infty & \text{if } i < j \end{cases}$$
Because $e^{-\infty} = 0$, attention weights to future tokens become identically zero.

---

### 3. Decoder Block Architecture
A canonical GPT-style decoder block consists of:
1. **Pre-Layer Normalization (`nn.LayerNorm`):** Normalizes activations across channel dimensions, promoting training stability.
2. **Causal Multi-Head Self-Attention:** Computes context-aware token interactions.
3. **Residual Skip Connections:** $x \leftarrow x + \text{SubLayer}(x)$, alleviating vanishing gradients in deep networks.
4. **Feed-Forward Network (FFN):** Two-layer MLP with non-linear activation (GELU):
   $$\text{FFN}(x) = \max(0, x W_1 + b_1) W_2 + b_2$$

---

### 4. Text Generation & Sampling Strategies
The output head produces unnormalized logits $z \in \mathbb{R}^V$ over vocabulary $V$. Probabilities are computed via temperature-scaled softmax:
$$P(w_i) = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}$$
- **Temperature $\tau$:**
  - $\tau \to 0$: Greedy sampling (deterministic, repetitive).
  - $\tau = 1.0$: Standard categorical sampling.
  - $\tau > 1.0$: High randomness / creativity (diverse, potential gibberish).

---

### 5. Viva Voce Questions & Answers

**Q1: What is the purpose of the causal mask in a generative Transformer?**  
*Answer:* The causal mask prevents tokens from attending to subsequent tokens in the sequence. It enforces the autoregressive property, ensuring that predicting the next token relies strictly on past context.

**Q2: Why do we divide by $\sqrt{d_k}$ in scaled dot-product attention?**  
*Answer:* For large values of $d_k$, the dot product $Q K^T$ grows large in magnitude, pushing the softmax function into regions with near-zero gradients. Dividing by $\sqrt{d_k}$ stabilizes the variance to $1.0$ and prevents vanishing gradients.

**Q3: How does positional embedding work and why is it necessary?**  
*Answer:* Self-attention is permutation-invariant—it treats tokens as an unordered set without regard to sequence order. Positional embeddings inject sequential order information by adding unique spatial coordinates to each token embedding vector.

**Q4: What is the role of Temperature in text generation?**  
*Answer:* Temperature $\tau$ scales logits before softmax. Lower temperature concentrates probability mass onto the highest-scoring tokens (conservative and focused), while higher temperature flattens the distribution (creative and diverse).


---

## Step-by-Step Algorithm
1. Define training text and construct character vocabulary mapping dictionaries (`stoi`, `chars`).
2. Construct a custom `Block` module implementing LayerNorm, linear QKV projections, scaled dot-product attention, causal masking, residual connections, and a GELU feed-forward sublayer.
3. Construct a `GPT` decoder class combining token embeddings, positional embeddings, stacked Transformer blocks, and an output projection head.
4. Sample random mini-batches of context length $T=32$ and compute Cross-Entropy Loss against shifted target tokens.
5. Train the model using the AdamW optimizer over 400 iterations.
6. Implement temperature-controlled autoregressive sampling (`generate()`) to generate new sentences from a seed prompt.

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
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# 1. Corpus definition and character tokenization
text = ("generative ai creates new data. transformers use self attention to model language. "
        "attention lets every token look at every other token. language models predict the next token. ") * 20
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
V = len(chars)
data = torch.tensor([stoi[c] for c in text])
T = 32  # Context window length

# 2. Transformer Decoder Block
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
        # QKV linear projection & chunking
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        n = x.size(1)
        
        # Scaled dot-product attention
        scores = (q @ k.transpose(1, 2)) / (q.size(-1) ** 0.5)
        
        # Causal mask (prevent future look-ahead)
        mask = torch.triu(torch.ones(n, n), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        
        # Residual connections
        x = x + self.proj(weights @ v)
        x = x + self.ff(self.ln2(x))
        return x

# 3. Full GPT Architecture
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
        positions = torch.arange(n, device=x.device)
        h = self.tok(x) + self.pos(positions)
        return self.head(self.ln(self.blocks(h)))

# 4. Batch generation
def batch(n=32):
    ix = torch.randint(0, len(data) - T - 1, (n,))
    x = torch.stack([data[i:i + T] for i in ix])
    y = torch.stack([data[i + 1:i + T + 1] for i in ix])
    return x, y

# 5. Training Loop
m = GPT()
opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
print("Training Transformer from scratch...")
for i in range(401):
    x, y = batch()
    logits = m(x)
    loss = F.cross_entropy(logits.view(-1, V), y.view(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    if i % 100 == 0:
        print(f"Step {i:3d} | Loss: {loss.item():.4f}")

# 6. Autoregressive Text Generation
@torch.no_grad()
def generate(prompt, n=120, temp=0.8):
    x = torch.tensor([[stoi[c] for c in prompt]])
    for _ in range(n):
        logits = m(x[:, -T:])[0, -1] / temp
        probs = F.softmax(logits, dim=-1)
        next_char = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_char[None]], dim=1)
    return "".join(chars[i] for i in x[0])

print("
Generated Text from prompt 'transformers ':")
print(generate("transformers "))
```

---

## Expected Output
```text
Training Transformer from scratch...
Step   0 | Loss: 3.5137
Step 100 | Loss: 0.8173
Step 200 | Loss: 0.1259
Step 300 | Loss: 0.0435
Step 400 | Loss: 0.0270

Generated Text from prompt 'transformers ':
transformers use self attention to model language. attention lets every token look at every other token. language models pre

```

---

## Result
Successfully constructed, trained, and sampled from a complete Transformer decoder language model with self-attention and causal masking.\n