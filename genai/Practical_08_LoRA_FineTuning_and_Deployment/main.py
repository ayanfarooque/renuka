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
print("Response:", inference_api("rail and "))\n