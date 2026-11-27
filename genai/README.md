# Generative AI Practicals: Comprehensive Exam Manual (All 8 Practicals)

**Subject:** Generative Artificial Intelligence  
**Environment:** Python 3, PyTorch, Scikit-learn, Matplotlib, SciPy  
**Operational Mode:** 100% Offline, Self-Contained, and Deterministic (No external downloads or API keys required)  

---

## Quick Navigation: Practicals Directory

| No. | Practical Title | Key Concepts & Architecture | Directory Link |
|:---:|---|---|:---:|
| **01** | **Generative vs Discriminative** | $P(Y\mid X)$ vs $P(X, Y)$, Logistic Regression, GaussianNB, PCA latent reconstruction | [Practical_01_Generative_vs_Discriminative](./Practical_01_Generative_vs_Discriminative/README.md) |
| **02** | **Probabilistic Modeling & Synthetic Data** | Single Gaussian vs GMM, Expectation-Maximization (EM), Multimodal sampling | [Practical_02_Probabilistic_Modeling_and_Synthetic_Data](./Practical_02_Probabilistic_Modeling_and_Synthetic_Data/README.md) |
| **03** | **Transformer Language Model + Generation** | Causal Self-Attention, Masking, Decoder blocks, Autoregressive sampling, GPT from scratch | [Practical_03_Transformer_LM_and_Text_Generation](./Practical_03_Transformer_LM_and_Text_Generation/README.md) |
| **04** | **Prompt Engineering + RAG** | Document chunking, TF-IDF vectorization, Cosine retrieval, Grounded hallucination prevention | [Practical_04_Prompt_Engineering_and_RAG](./Practical_04_Prompt_Engineering_and_RAG/README.md) |
| **05** | **Generative Adversarial Network (GAN)** | Minimax game, Generator vs Discriminator, LeakyReLU, BCEWithLogitsLoss, Digit synthesis | [Practical_05_Generative_Adversarial_Network_GAN](./Practical_05_Generative_Adversarial_Network_GAN/README.md) |
| **06** | **Diffusion Model (DDPM from scratch)** | Forward Gaussian corruption, Variance schedule $\beta_t, \bar{\alpha}_t$, Noise prediction, Reverse sampling | [Practical_06_Diffusion_Model_DDPM](./Practical_06_Diffusion_Model_DDPM/README.md) |
| **07** | **Neural Style Transfer (+ GAN Art)** | Convolutional feature extraction, Gram matrix style representation, Direct pixel optimization | [Practical_07_Neural_Style_Transfer](./Practical_07_Neural_Style_Transfer/README.md) |
| **08** | **LoRA Fine-Tuning & Deployment** | PEFT, Frozen weights $W_0$, Low-rank adapters $\Delta W = B \cdot A$, Checkpoint saving & serving | [Practical_08_LoRA_FineTuning_and_Deployment](./Practical_08_LoRA_FineTuning_and_Deployment/README.md) |

---

## 1. Environment & Setup

All practicals operate **offline** with zero internet connection required:
- Embedded dataset `sklearn.datasets.load_digits` replaces MNIST downloads.
- Compact from-scratch PyTorch models replace multi-gigabyte pretrained weights (GPT-2, VGG, Stable Diffusion).
- Deterministic random seeds (`torch.manual_seed(0)`, `np.random.seed(0)`) guarantee reproducible outputs.

### Single Installation Command:
```bash
pip install numpy matplotlib scikit-learn scipy torch
```

---

## 2. High-Yield Viva Voce Quick Reference

| Topic | Core Concept & Mechanism |
|---|---|
| **Generative vs Discriminative** | Generative models learn $P(X, Y)$ to generate new data; discriminative models learn $P(Y\mid X)$ to classify inputs. |
| **Self-Attention** | Tokens dynamically weigh all other tokens via $\text{softmax}(QK^T / \sqrt{d_k}) V$. |
| **Causal Mask** | Lower triangular mask preventing tokens from attending to future tokens in autoregressive generation. |
| **RAG** | Retrieves top-$k$ relevant passages and injects them into the prompt to anchor LLM responses and eliminate hallucinations. |
| **GAN Minimax Game** | Generator $G$ counterfeits data; Discriminator $D$ detects fakes; trained simultaneously via zero-sum minimax loss. |
| **Forward Diffusion** | Adds scheduled Gaussian noise $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$ until reaching pure noise. |
| **Reverse Diffusion** | A neural network predicts the noise $\epsilon_\theta(x_t, t)$, iteratively subtracting it to uncover a clean sample. |
| **Gram Matrix** | Computes channel cross-products $G = V V^T$ to measure spatial texture and style correlations independent of position. |
| **LoRA (PEFT)** | Freezes base weights $W_0$ and adds trainable low-rank adapters $\Delta W = \frac{\alpha}{r} BA$, tuning $< 5\%$ of parameters. |

---

## 3. Practical Exam Tips & Checklist

1. **Self-Contained Execution:** Every practical script is completely runnable standalone: `python main.py`.
2. **Speed on CPU:** Every PyTorch training loop in these practicals is optimized to execute in under 30-60 seconds on standard CPU hardware.
3. **Journal Writing Ready:** Each individual `README.md` includes full-page theoretical explanations, mathematical formulas, algorithms, code, output, and viva questions.\n