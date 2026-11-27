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
print(generate("transformers "))\n