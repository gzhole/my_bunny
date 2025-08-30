
# From Linear Models to LLMs — a High‑School‑Friendly Path (with Labs)

**Audience:** high‑school students (IB/Grade 11–12).  
**Goal:** build intuition (not heavy proofs) from the simplest model to modern Large Language Models (LLMs), with short labs that reinforce learning.

---

<details>
<summary>## 0) How to use this guide</summary>

- **Prereqs:** comfort with functions, vectors, basic derivatives (slope idea), logs, and Python. PyTorch is used in the labs.
- **Path:** Linear → Nonlinear (neurons) → Deep nets → Attention → Transformers → LLMs.
- **Mindset:** always ask: *what is the input, what is the output, what’s the loss, how do weights change to reduce loss?*
</details>

---

<details>
<summary>## 1) Why linear algebra shows up everywhere</summary>

**Data as vectors.** We turn things into numbers:
- An image → a long list of pixel intensities → a vector **x**.
- A word/subword → an ID → an *embedding* vector **e**.
- A sound snippet → amplitudes → a vector.

**Weighted sums.** A dot product **w·x** is just “add up each feature × its importance.”  
Matrices do many dot products at once: **y = W x + b**.

**Geometry view.** Linear models draw a separating hyperplane. If two classes can be split by a straight "cut" in feature space, a linear model works great.

<details>
<summary>🔍 What's a hyperplane in different dimensions? (Click to expand)</summary>

A hyperplane is a subspace with one dimension less than its surrounding space:

- **1D (1 feature)**: A point (e.g., x = 2.5)  
  → Splits the number line into left/right

- **2D (2 features)**: A line (e.g., y = 2x + 1)  
  → Splits the plane into two half-planes

- **3D (3 features)**: A flat plane (e.g., z = x + y + 5)  
  → Splits 3D space into two half-spaces

- **nD (n features)**: An (n-1) dimensional hyperplane  
  → Still just one "straight cut" in n-dimensional space

**Key insight**: A linear model can only make one straight cut through feature space. For more complex patterns (like spirals or XOR), we'll need nonlinear models.
</details>

<details>
<summary>🔍 How ReLUs can build quadratic functions (Click to expand)</summary>

Let's see how we can use ReLUs to approximate a quadratic function like f(x) = x². The key insight is that while a single ReLU is piecewise linear, we can combine multiple ReLUs to create smooth curves.

### Building x² with ReLUs

We'll use a simple combination of ReLUs to create a quadratic function. The trick is to use the square of ReLUs:

```
f_approx(x) = ReLU(x)² + ReLU(-x)²
```

### How It Works

- For x ≥ 0:
  - ReLU(x) = x
  - ReLU(-x) = 0
  - f_approx(x) = x² + 0 = x²
  
- For x ≤ 0:
  - ReLU(x) = 0
  - ReLU(-x) = -x
  - f_approx(x) = 0 + (-x)² = x²

### Visual Representation

Parabola opens upward; the dot marks the minimum at x = 0.

```
        |                y
        |
        |        \       /
        |         \     /
        |          \   /
--------+-----------•----------- x
                    0
```

### Key Insights

1. The first term `ReLU(x)²` handles the right side of the parabola (x ≥ 0)
2. The second term `ReLU(-x)²` handles the left side (x ≤ 0)
3. At x=0, both terms are zero, creating the minimum point
4. The combination gives the U‑shaped parabola with a minimum at x=0

### Shifted quadratic (x+1)²

We can shift the minimum to x = −1 by shifting the ReLUs:

```
g(x) = ReLU(x + 1)² + ReLU(−x − 1)² = (x + 1)²
```

- For x ≥ −1: ReLU(x+1) = x+1, ReLU(−x−1) = 0 ⇒ g(x) = (x+1)²
- For x ≤ −1: ReLU(x+1) = 0, ReLU(−x−1) = −x−1 ⇒ g(x) = (x+1)²

This matches your earlier polynomial example exactly and shows how a simple shift moves the parabola’s vertex.

### Better Approximations

For more complex curves, we can use more ReLUs with different shifts and scales:

```
f_approx(x) = ∑ w_i * ReLU(x - a_i)² + b_i
```

This is essentially what neural networks do - they learn the weights (w_i), shifts (a_i), and biases (b_i) automatically through training to approximate complex functions.

In practice, deep networks use many such units in parallel to create smooth approximations of complex, high-dimensional functions.
</details>

> TL;DR: Linear algebra is the language of stacking and mixing features. It's compact, fast, and maps cleanly to GPUs.
</details>

---

<details>
<summary>## 2) Start simple: the one‑neuron classifier (a.k.a. logistic regression)</summary>

**Score:** $ z = \mathbf{w}\cdot\mathbf{x} + b $ → **Probability:** $ \sigma(z) = \frac{1}{1+e^{-z}} $.  
**Loss:** binary cross‑entropy (small if confident & correct; large if confidently wrong).  
**Learning:** nudge $ \mathbf{w}, b $ to reduce loss (gradient descent; the key term is $p - y$).

> This is your “hello world” of ML. It’s already useful and very interpretable.
</details>

If you have the separate “Math for the Bunny Classifier (IB IA)” handout, that’s the detailed version of this section.

---

<details>
<summary>## 3) Why linear isn't enough (and what to do about it)</summary>

**Problem:** Some patterns (like XOR) are *not* separable by a straight line in any simple feature space.  
**Two common fixes:**

1) **Feature engineering / higher‑degree polynomials.**  
   Create new features like $x_1^2, x_1x_2, x_2^2,\dots$. This can work, but in high dimensions, the number of polynomial terms **explodes** combinatorially and models become wiggly/unstable (overfitting, "Runge phenomenon").

2) **Learn features with neurons (nonlinearities).**  
   A layer does **linear → nonlinearity** (e.g., ReLU). Stacking layers composes simple pieces to carve complicated shapes.  
   - ReLU: $ \mathrm{ReLU}(t)=\max(0,t) $.  
   - Neural nets with ReLU are **piecewise linear**—like many flat tiles arranged into complex boundaries.  
   - **Universal approximation:** with enough hidden units, a one‑hidden‑layer net can approximate any reasonable function—but **depth** often does it **more efficiently** (fewer parameters) by reusing building blocks.

> Intuition: Polynomials try one giant global curve; deep nets build with LEGO bricks.
</details>

---

<details>
<summary>## 4) What a neuron layer does (minimal math)</summary>

Given input vector **x**:
- Linear mix: $ \mathbf{h} = W\mathbf{x} + \mathbf{b} $ (many weighted sums at once).
- Bend it: $ \mathbf{a} = \mathrm{ReLU}(\mathbf{h}) $ (or GELU/Tanh).  
Stack several **(Linear → Nonlinear)** blocks, then finish with a simple classifier head.

**Why nonlinearity matters:** If you only stack linear layers, it collapses to a single linear layer. The “bend” is what creates expressive power.
</details>

---

<details>
<summary>## 5) Training deep nets (the gist)</summary>

- **Objective:** choose weights to make loss small on training data *and* on new data.
- **Backpropagation:** chain rule to compute gradients efficiently.
- **Optimizer:** SGD/Adam update weights in the direction that reduces loss.
- **Overfitting:** when training loss ↓ but validation loss ↑.  
  Fixes: more data, augmentation, weight decay (L2), dropout, simpler model, early stopping.
- **Learning rate:** too high → explode; too low → crawl. Use schedules / warmup.

<details>
<summary>Why add more neurons (width)?</summary>

* **Expressive power (more “hinges”).** With ReLU, each hidden neuron can add a hinge to the function. More neurons → more pieces → can fit more complex curves/surfaces.
* **Richer learned features.** Each neuron can specialize in a pattern (edge, curve, token interaction). More neurons = a larger “vocabulary” of features.
* **Easier optimization (often).** Over‑parameterized nets (wider than needed) can be easier to train; there are many good solutions for SGD/Adam to find.

When it can hurt:
* **Overfitting risk.** More parameters can memorize small/noisy datasets. Watch validation loss/accuracy.
* **Compute/memory.** Wider layers increase FLOPs and RAM; training slows down.

Rules of thumb for students:
* Start modest (e.g., 32–128 hidden units) → increase until validation stops improving.
* Add regularization as you widen: weight decay, dropout, data augmentation, early stopping.
* Prefer adding a little **depth** before making layers extremely wide; depth reuses features efficiently.
* Plot train vs. validation curves; choose the smallest width that achieves your target validation accuracy.

</details>
</details>

---

<details>
<summary>## 6) From words to vectors (tokenization & embeddings)</summary>

Text isn’t numeric. We:
1) **Tokenize** into subwords (e.g., “play”, “##ing”). This keeps vocab manageable.
2) Map token IDs to **embedding vectors** (rows of a learned matrix).  
   Nearby vectors tend to represent related meanings—because the model *learned* to put them near to reduce prediction error.

> Geometry again: meanings become positions; dot products measure “how related.”
</details>

---

<details>
<summary>## 7) Attention in two steps (the core of Transformers)</summary>

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
</details>

---

<details>
<summary>## 8) What an LLM actually learns</summary>

- **Objective:** *next‑token prediction*. Given context, predict the next token.  
- **Loss:** cross‑entropy over the vocabulary (like many‑class logistic regression).  
- **Training data:** huge; many domains and styles.  
- **Sampling:** temperature, top‑k/top‑p; these control creativity vs. accuracy.  
- **Limits & safety:** LLMs imitate patterns; they can hallucinate or reflect biases in data. Guardrails and evaluation matter.

> Big picture: an LLM is a giant stack of attention blocks trained to predict the next symbol really, really well.
</details>

---

<details>
<summary>## 9) Linear vs. polynomial vs. deep nets — quick compare</summary>

| Approach | Strengths | Weaknesses | When to use |
|---|---|---|---|
| Linear (logistic/softmax) | Simple, fast, interpretable | Only straight cuts | When features are already good (e.g., tabular with strong signals) |
| Polynomial features / kernels | Can model curves without deep nets | Can blow up in size; tough at large scale | Small/medium problems, classic SVM/RBF settings |
| Deep nets (ReLU/Conv/Attention) | Learn features; parameter‑efficient with depth; scales | Needs data/compute; harder to debug | Images, audio, language; large/complex patterns |
</details>

---

<details>
<summary>## 10) Labs (short, scaffolded)</summary>

> All labs use Python + PyTorch. Run in Google Colab or local. Each lab has a **Checkpoint** so you know you're on track.

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

<details>
<summary>### Lab 2 — Decision boundaries on toy 2D data (intuition builder)</summary>

- Train logistic regression vs. MLP on `sklearn.datasets.make_moons`.  
- **Checkpoint:** visualize boundary; see linear fail and MLP succeed.
</details>

<details>
<summary>### Lab 3 — From n‑grams to a tiny char‑model (Makemore‑style)</summary>

- Follow Karpathy’s *Zero‑to‑Hero* char‑model steps: unigram → bigram → MLP.  
- Train on a small names dataset; sample new names.  
- **Checkpoint:** generated names look name‑like.
</details>

<details>
<summary>### Lab 4 — Build one attention head (toy)</summary>

- Implement scaled dot‑product attention for a 1D toy sequence (e.g., predict a missing token that repeats earlier).
```python
scores = (Q @ K.transpose(-1,-2)) / (d_k ** 0.5)   # [B, T, T]
weights = scores.softmax(dim=-1)                    # attention over positions
out = weights @ V                                   # [B, T, d_v]
```
</details>

<details>
<summary>### Lab 5 — nanoGPT on tiny data (CPU‑friendly settings)</summary>

- Repo: `karpathy/nanoGPT`. Train a **character‑level** model on Tiny Shakespeare *or your own story text*.
- Use a **small config** (e.g., n_layer=2, n_head=2, n_embd=128, block_size=64, batch_size small).  
- Train just enough steps to overfit a few pages (to see learning), then a bit more data to generalize slightly.  
- **Checkpoint:** loss falls from ~4–5 toward ~1–2; samples look text‑like.
</details>

<details>
<summary>### Lab 6 — Sampling & safety</summary>

- Play with temperature, top‑k/top‑p; collect examples of factual vs. creative outputs.
- Discuss *hallucination* and how to detect it; add a rule: “verify before trust.”
- **Checkpoint:** a one‑page reflection on when to trust model outputs.
</details>

---

<details>
<summary>## 11) Common misconceptions (fast fixes)</summary>

- “Why not just use high‑degree polynomials?” → Parameter blow‑up; poor scaling; deep nets reuse parts (compositionality).  
- “Do LLMs ‘understand’?” → They model patterns in text extremely well; they don’t have human‑style grounded experience by default.  
- “More depth always better?” → Up to a point and with enough data/regularization. Watch validation curves.
</details>

---

<details>
<summary>## 12) Mini‑glossary</summary>

- **Activation (ReLU/GELU):** adds nonlinearity so layers don’t collapse into one big linear map.  
- **Embedding:** learned vector for a token.  
- **Attention:** lets a token gather info from others using dot products (similarity).  
- **Transformer block:** attention + MLP + residuals + layer norm.  
- **Cross‑entropy:** loss used for classification/next‑token prediction.  
- **Overfitting:** when model memorizes training data patterns that don’t generalize.
</details>

---

<details>
<summary>## 13) Where to learn more (friendly options)</summary>

- **Andrej Karpathy — Zero to Hero** (videos + code).  
- **karpathy/nanoGPT** (read the README & training scripts).  
- **3Blue1Brown — Neural Networks** (visual intuition).  
- **Hugging Face — NLP Course** (transformers & practical tips).  
- **fast.ai — Practical Deep Learning** (applied focus).
</details>

---

<details>
<summary>## 14) Optional: a 6‑week plan (≈3–5 hrs/week)</summary>

- **W1:** Lab 0 + review vectors/dot product/sigmoid/loss.  
- **W2:** Lab 1 + overfitting/regularization.  
- **W3:** Lab 2 + start Zero‑to‑Hero char model.  
- **W4:** Finish char model; intro to attention (Section 7).  
- **W5:** Lab 4 attention head; read Transformer block code.  
- **W6:** Lab 5 nanoGPT + Lab 6 sampling/safety reflection.

> Output: a folder of working notebooks + a short write‑up per lab (what you tried; curves; what you learned).
</details>

---

<details>
<summary>## 15) Appendix: Why depth helps (intuition only)</summary>

Depth = composing simple transforms. Many real‑world rules are **compositional** (edges → shapes → objects; characters → words → phrases → meaning). Deep nets mirror this hierarchy and reuse internal features, often requiring **far fewer** parameters than a flat giant polynomial to reach the same accuracy.
</details>
