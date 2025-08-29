
# From Linear Models to LLMs — a High‑School‑Friendly Path (with Labs)

**Audience:** high‑school students (IB/Grade 11–12).  
**Goal:** build intuition (not heavy proofs) from the simplest model to modern Large Language Models (LLMs), with short labs that reinforce learning.

---

## 0) How to use this guide

- **Prereqs:** comfort with functions, vectors, basic derivatives (slope idea), logs, and Python. PyTorch is used in the labs.
- **Path:** Linear → Nonlinear (neurons) → Deep nets → Attention → Transformers → LLMs.
- **Mindset:** always ask: *what is the input, what is the output, what’s the loss, how do weights change to reduce loss?*

---

## 1) Why linear algebra shows up everywhere

**Data as vectors.** We turn things into numbers:
- An image → a long list of pixel intensities → a vector **x**.
- A word/subword → an ID → an *embedding* vector **e**.
- A sound snippet → amplitudes → a vector.

**Weighted sums.** A dot product **w·x** is just “add up each feature × its importance.”  
Matrices do many dot products at once: **y = W x + b**.

**Geometry view.** Linear models draw a separating hyperplane. If two classes can be split by a straight “cut” in feature space, a linear model works great.

> TL;DR: Linear algebra is the language of stacking and mixing features. It’s compact, fast, and maps cleanly to GPUs.

---

## 2) Start simple: the one‑neuron classifier (a.k.a. logistic regression)

**Score:** $ z = \mathbf{w}\cdot\mathbf{x} + b $ → **Probability:** $ \sigma(z) = \frac{1}{1+e^{-z}} $.  
**Loss:** binary cross‑entropy (small if confident & correct; large if confidently wrong).  
**Learning:** nudge $ \mathbf{w}, b $ to reduce loss (gradient descent; the key term is $p - y$).

> This is your “hello world” of ML. It’s already useful and very interpretable.

If you have the separate “Math for the Bunny Classifier (IB IA)” handout, that’s the detailed version of this section.

---

## 3) Why linear isn’t enough (and what to do about it)

**Problem:** Some patterns (like XOR) are *not* separable by a straight line in any simple feature space.  
**Two common fixes:**

1) **Feature engineering / higher‑degree polynomials.**  
   Create new features like $x_1^2, x_1x_2, x_2^2,\dots$. This can work, but in high dimensions, the number of polynomial terms **explodes** combinatorially and models become wiggly/unstable (overfitting, “Runge phenomenon”).

2) **Learn features with neurons (nonlinearities).**  
   A layer does **linear → nonlinearity** (e.g., ReLU). Stacking layers composes simple pieces to carve complicated shapes.  
   - ReLU: $ \mathrm{ReLU}(t)=\max(0,t) $.  
   - Neural nets with ReLU are **piecewise linear**—like many flat tiles arranged into complex boundaries.  
   - **Universal approximation:** with enough hidden units, a one‑hidden‑layer net can approximate any reasonable function—but **depth** often does it **more efficiently** (fewer parameters) by reusing building blocks.

> Intuition: Polynomials try one giant global curve; deep nets build with LEGO bricks.

---

## 4) What a neuron layer does (minimal math)

Given input vector **x**:
- Linear mix: $ \mathbf{h} = W\mathbf{x} + \mathbf{b} $ (many weighted sums at once).
- Bend it: $ \mathbf{a} = \mathrm{ReLU}(\mathbf{h}) $ (or GELU/Tanh).  
Stack several **(Linear → Nonlinear)** blocks, then finish with a simple classifier head.

**Why nonlinearity matters:** If you only stack linear layers, it collapses to a single linear layer. The “bend” is what creates expressive power.

---

## 5) Training deep nets (the gist)

- **Objective:** choose weights to make loss small on training data *and* on new data.
- **Backpropagation:** chain rule to compute gradients efficiently.
- **Optimizer:** SGD/Adam update weights in the direction that reduces loss.
- **Overfitting:** when training loss ↓ but validation loss ↑.  
  Fixes: more data, augmentation, weight decay (L2), dropout, simpler model, early stopping.
- **Learning rate:** too high → explode; too low → crawl. Use schedules / warmup.

---

## 6) From words to vectors (tokenization & embeddings)

Text isn’t numeric. We:
1) **Tokenize** into subwords (e.g., “play”, “##ing”). This keeps vocab manageable.
2) Map token IDs to **embedding vectors** (rows of a learned matrix).  
   Nearby vectors tend to represent related meanings—because the model *learned* to put them near to reduce prediction error.

> Geometry again: meanings become positions; dot products measure “how related.”

---

## 7) Attention in two steps (the core of Transformers)

**Step 1: Scaled dot‑product attention (one head).**  
Given a sequence of embeddings, make three matrices:
- $Q = X W_Q$ (queries), $K = X W_K$ (keys), $V = X W_V$ (values).  
- Weights between positions: $ \mathrm{AttnWeights} = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) $.  
- Mix values: $ \mathrm{Head} = \mathrm{AttnWeights}\,V $.  

Interpretation: each token *looks* at others and pulls in the most relevant information via dot products.

**Step 2: Multi‑head + MLP + residuals.**  
- Several heads see different relationships; concatenate heads → linear layer.  
- Add a small MLP (feed‑forward) and **residual connections** (skip paths) + LayerNorm.  
- **Positional encoding** tells the model “where” tokens are in the sequence.  
Stack N of these blocks ⇒ a **Transformer**.

---

## 8) What an LLM actually learns

- **Objective:** *next‑token prediction*. Given context, predict the next token.  
- **Loss:** cross‑entropy over the vocabulary (like many‑class logistic regression).  
- **Training data:** huge; many domains and styles.  
- **Sampling:** temperature, top‑k/top‑p; these control creativity vs. accuracy.  
- **Limits & safety:** LLMs imitate patterns; they can hallucinate or reflect biases in data. Guardrails and evaluation matter.

> Big picture: an LLM is a giant stack of attention blocks trained to predict the next symbol really, really well.

---

## 9) Linear vs. polynomial vs. deep nets — quick compare

| Approach | Strengths | Weaknesses | When to use |
|---|---|---|---|
| Linear (logistic/softmax) | Simple, fast, interpretable | Only straight cuts | When features are already good (e.g., tabular with strong signals) |
| Polynomial features / kernels | Can model curves without deep nets | Can blow up in size; tough at large scale | Small/medium problems, classic SVM/RBF settings |
| Deep nets (ReLU/Conv/Attention) | Learn features; parameter‑efficient with depth; scales | Needs data/compute; harder to debug | Images, audio, language; large/complex patterns |

---

## 10) Labs (short, scaffolded)

> All labs use Python + PyTorch. Run in Google Colab or local. Each lab has a **Checkpoint** so you know you’re on track.

### Lab 0 — One‑neuron Bunny Classifier (recap)
- Implement logistic regression on your bunny dataset.
- **Checkpoint:** training & validation loss curves; confusion matrix ≈ reasonable.

### Lab 1 — Make it nonlinear (1‑hidden‑layer MLP)
- Replace the single neuron with: Linear(4096→64) → ReLU → Linear(64→1) → Sigmoid.
- Add weight decay (e.g., 1e‑4). Try different hidden sizes (32, 64, 128).  
**Snippet (skeleton):**
```python
import torch, torch.nn as nn

class MLP(nn.Module):
    def __init__(self, d_in=4096, d_h=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_h),
            nn.ReLU(),
            nn.Linear(d_h, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

# BCEWithLogitsLoss combines sigmoid+loss stably
model = MLP()
loss_fn = nn.BCEWithLogitsLoss()
```
- **Checkpoint:** validation accuracy > Lab 0; plot learning curves.

### Lab 2 — Decision boundaries on toy 2D data (intuition builder)
- Train logistic regression vs. MLP on `sklearn.datasets.make_moons`.  
- **Checkpoint:** visualize boundary; see linear fail and MLP succeed.

### Lab 3 — From n‑grams to a tiny char‑model (Makemore‑style)
- Follow Karpathy’s *Zero‑to‑Hero* char‑model steps: unigram → bigram → MLP.  
- Train on a small names dataset; sample new names.  
- **Checkpoint:** generated names look name‑like.

### Lab 4 — Build one attention head (toy)
- Implement scaled dot‑product attention for a 1D toy sequence (e.g., predict a missing token that repeats earlier).
```python
scores = (Q @ K.transpose(-1,-2)) / (d_k ** 0.5)   # [B, T, T]
weights = scores.softmax(dim=-1)                    # attention over positions
out = weights @ V                                   # [B, T, d_v]
```

### Lab 5 — nanoGPT on tiny data (CPU‑friendly settings)
- Repo: `karpathy/nanoGPT`. Train a **character‑level** model on Tiny Shakespeare *or your own story text*.
- Use a **small config** (e.g., n_layer=2, n_head=2, n_embd=128, block_size=64, batch_size small).  
- Train just enough steps to overfit a few pages (to see learning), then a bit more data to generalize slightly.  
- **Checkpoint:** loss falls from ~4–5 toward ~1–2; samples look text‑like.

### Lab 6 — Sampling & safety
- Play with temperature, top‑k/top‑p; collect examples of factual vs. creative outputs.
- Discuss *hallucination* and how to detect it; add a rule: “verify before trust.”
- **Checkpoint:** a one‑page reflection on when to trust model outputs.

---

## 11) Common misconceptions (fast fixes)

- “Why not just use high‑degree polynomials?” → Parameter blow‑up; poor scaling; deep nets reuse parts (compositionality).  
- “Do LLMs ‘understand’?” → They model patterns in text extremely well; they don’t have human‑style grounded experience by default.  
- “More depth always better?” → Up to a point and with enough data/regularization. Watch validation curves.

---

## 12) Mini‑glossary

- **Activation (ReLU/GELU):** adds nonlinearity so layers don’t collapse into one big linear map.  
- **Embedding:** learned vector for a token.  
- **Attention:** lets a token gather info from others using dot products (similarity).  
- **Transformer block:** attention + MLP + residuals + layer norm.  
- **Cross‑entropy:** loss used for classification/next‑token prediction.  
- **Overfitting:** model memorizes training data patterns that don’t generalize.

---

## 13) Where to learn more (friendly options)

- **Andrej Karpathy — Zero to Hero** (videos + code).  
- **karpathy/nanoGPT** (read the README & training scripts).  
- **3Blue1Brown — Neural Networks** (visual intuition).  
- **Hugging Face — NLP Course** (transformers & practical tips).  
- **fast.ai — Practical Deep Learning** (applied focus).

---

## 14) Optional: a 6‑week plan (≈3–5 hrs/week)

- **W1:** Lab 0 + review vectors/dot product/sigmoid/loss.  
- **W2:** Lab 1 + overfitting/regularization.  
- **W3:** Lab 2 + start Zero‑to‑Hero char model.  
- **W4:** Finish char model; intro to attention (Section 7).  
- **W5:** Lab 4 attention head; read Transformer block code.  
- **W6:** Lab 5 nanoGPT + Lab 6 sampling/safety reflection.

> Output: a folder of working notebooks + a short write‑up per lab (what you tried; curves; what you learned).

---

## 15) Appendix: Why depth helps (intuition only)

Depth = composing simple transforms. Many real‑world rules are **compositional** (edges → shapes → objects; characters → words → phrases → meaning). Deep nets mirror this hierarchy and reuse internal features, often requiring **far fewer** parameters than a flat giant polynomial to reach the same accuracy.
