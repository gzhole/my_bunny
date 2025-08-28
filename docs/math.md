# Math for the Bunny Classifier (IB Math IA)

This document explains the simple math behind our one‑neuron classifier that decides: “Is this my bunny or not?” The goal is to keep concepts at a Math SL level and show how they work together in the project.

We turn each image into a grayscale 64×64 picture (4,096 numbers between 0 and 1). We flatten it to a vector `x` and send it through a single neuron (logistic regression).

---

## 1) Linear Score: z = w · x + b

- **Why**
  - We need a single number that summarizes the image: high means “looks like my bunny”, low means “does not look like my bunny.” A weighted sum lets the model learn which pixels matter more.
- **What**
  - `x` is the image as a vector of pixel intensities, length 4,096.
  - `w` (same length as `x`) is the weight vector the model learns.
  - `b` is a bias term (a single number).
  - The dot product `w · x` adds up each pixel times its importance, then `+ b` shifts the score.
- **Tiny Example**
  - Suppose x has just 3 pixels: x = [0.2, 0.7, 0.1]
  - Let w = [1.0, −0.5, 0.3], b = 0.2
  - Then z = (1.0)(0.2) + (−0.5)(0.7) + (0.3)(0.1) + 0.2 = 0.2 − 0.35 + 0.03 + 0.2 = 0.08
- **In this project**
  - Real images have 4,096 pixels, so w has 4,096 weights. During training, these values adjust so the score z is higher for “my bunny” images and lower otherwise.

---

## 2) Sigmoid Function: σ(z) = 1 / (1 + e^(−z))

- **Why**
  - The raw score z can be any real number. We want a probability between 0 and 1. The sigmoid smoothly squashes z into that range.
- **What**
  - If z is large and positive, σ(z) is close to 1.
  - If z is around 0, σ(z) ≈ 0.5 (the model is unsure).
  - If z is large and negative, σ(z) is close to 0.
- **Tiny Example**
  - z = −2 → σ(z) ≈ 0.12
  - z = 0 → σ(z) = 0.5
  - z = 2 → σ(z) ≈ 0.88
- **In this project**
  - We interpret p = σ(z) as “probability that the image is my bunny.” At prediction time, we say class = 1 (my bunny) if p ≥ 0.5, else class = 0.

### Optional: Derivative of the Sigmoid (why σ′(z) = σ(z)(1 − σ(z)))

- Start from σ(z) = 1 / (1 + e^(−z)). Let u(z) = 1 + e^(−z), so σ(z) = u(z)^(−1).
- Chain rule: dσ/dz = (−1)·u^(−2) · du/dz.
- Differentiate u: du/dz = d(1 + e^(−z))/dz = −e^(−z).
- Put together: dσ/dz = (−1)·u^(−2)·(−e^(−z)) = e^(−z) / (1 + e^(−z))^2.
- Show it equals σ(1 − σ):
  - σ(z) = 1/(1 + e^(−z)), and 1 − σ(z) = e^(−z)/(1 + e^(−z)).
  - Multiply: σ(z)(1 − σ(z)) = [1/(1 + e^(−z))] · [e^(−z)/(1 + e^(−z))] = e^(−z)/(1 + e^(−z))^2.
  - Therefore, σ′(z) = σ(z)(1 − σ(z)).
- Quick check at z = 0: σ(0) = 0.5 ⇒ σ′(0) = 0.5·(1 − 0.5) = 0.25.

Alternative derivation (quotient rule):
- Write σ(z) = e^z / (1 + e^z).
- Quotient rule: dσ/dz = [(e^z)(1+e^z) − (e^z)(e^z)] / (1 + e^z)^2
  = [e^z + e^{2z} − e^{2z}] / (1 + e^z)^2
  = e^z / (1 + e^z)^2
  = σ(z)(1 − σ(z)).

Useful properties (good for intuition):
- 0 < σ′(z) ≤ 1/4; the maximum slope is 1/4 at z = 0 (where σ = 0.5).
- The curve is steepest near z ≈ 0 and flattens for large |z| (saturation). When saturated, updates are small because σ′(z) is small.
- Using p = σ(z), the derivative is simply σ′(z) = p(1 − p). This is the shortcut used in the gradient derivation.

---

## 3) Binary Cross‑Entropy (Log Loss)

- **Why**
  - We need a number that says how “bad” our prediction is for a training example. A good loss should be small when we are confident and correct, and large when we are confident and wrong.
- **What**
  - Let true label y be 1 for “my bunny” and 0 for “other bunny.” Let p = σ(z).
  - The loss for one example is:
    - L = −[ y · log(p) + (1 − y) · log(1 − p) ]
  - Behavior:
    - If y = 1 and p is close to 1, loss is small.
    - If y = 1 and p is close to 0, loss is large.
    - If y = 0 and p is close to 0, loss is small.
    - If y = 0 and p is close to 1, loss is large.
- **Tiny Example**
  - Correct but unsure: y = 1, p = 0.6 ⇒ L ≈ −log(0.6) ≈ 0.51
  - Confident and correct: y = 1, p = 0.9 ⇒ L ≈ −log(0.9) ≈ 0.105
  - Confident and wrong: y = 1, p = 0.1 ⇒ L ≈ −log(0.1) ≈ 2.303
- **In this project**
  - We average this loss over a batch of images. Training tries to make this average as small as possible on the training set (and also low on the validation set to avoid overfitting).

---

## 4) Gradient Descent (Learning)

- **Why**
  - To improve the model, we must change w and b to reduce the loss. Gradient descent is a method that moves parameters in the direction that most quickly lowers the loss.
- **What**
  - The gradient tells us how the loss would change if we nudge each parameter slightly.
  - Update rule with learning rate η (a small positive number):
    - w ← w − η · (∂L/∂w)
    - b ← b − η · (∂L/∂b)
  - For logistic regression with cross‑entropy, a key quantity is (p − y). Intuitively:
    - If p > y (we predicted too high), gradients push w,b down.
    - If p < y (we predicted too low), gradients push w,b up.
- **Tiny Example (one step)**
  - Start with w = [0, 0], b = 0. Input x = [0.2, 0.5], true label y = 1.
  - Score: z = w·x + b = 0. Probability: p = σ(0) = 0.5.
  - Error signal: (p − y) = −0.5.
  - Gradients: ∂L/∂w = (p − y)·x = −0.5·[0.2, 0.5] = [−0.1, −0.25]; ∂L/∂b = (p − y) = −0.5.
  - Why those gradients (step-by-step):
    1. Binary cross‑entropy (one example) with p = σ(z): L = −[ y·log(p) + (1−y)·log(1−p) ].
    2. Differentiate w.r.t. p: ∂L/∂p = −[ y/p − (1−y)/(1−p) ].
    3. Sigmoid derivative: ∂p/∂z = p(1−p). At z = 0, p = 0.5 ⇒ ∂p/∂z = 0.25.
    4. Chain rule to z: ∂L/∂z = (∂L/∂p)(∂p/∂z) = −[ y/p − (1−y)/(1−p) ]·p(1−p).
       Plug y = 1, p = 0.5: ∂L/∂z = −[ 1/0.5 − 0/0.5 ]·(0.5)(0.5) = −2·0.25 = −0.5 = (p − y).
    5. Since z = w·x + b, we have ∂z/∂w = x and ∂z/∂b = 1.
    6. Chain rule to parameters:
       ∂L/∂w = (∂L/∂z)·x = (−0.5)·[0.2, 0.5] = [−0.1, −0.25];   ∂L/∂b = (∂L/∂z)·1 = −0.5.
  - With learning rate η = 1.0 for illustration:
    - w ← [0, 0] − 1·[−0.1, −0.25] = [0.1, 0.25]
    - b ← 0 − 1·(−0.5) = 0.5
  - New score: z_new = (0.1)(0.2) + (0.25)(0.5) + 0.5 = 0.645
  - New probability: p_new = σ(0.645) ≈ 0.655 (closer to 1, which matches y = 1). Loss goes down.
- **In this project**
  - PyTorch automatically computes gradients for us and applies the updates (we use SGD). We only choose η (learning rate) and how many passes (epochs) to train.

---

## 5) Learning Curves and Evaluation Metrics

- **Why**
  - We need evidence that the model actually learns and generalizes. Learning curves show progress. Metrics on unseen test data show real performance.
- **What**
  - Learning curves: plot training loss and validation loss over epochs. Healthy training usually shows both decreasing; if training loss keeps falling but validation loss rises, that suggests overfitting.
  - Accuracy: percentage of correct predictions on the test set using threshold 0.5.
  - Confusion matrix: counts of predictions vs. truth:
    - True Positive (TP): predict my bunny when it is my bunny.
    - False Positive (FP): predict my bunny when it is not.
    - True Negative (TN): predict not my bunny when it is not.
    - False Negative (FN): predict not my bunny when it is my bunny.
  - Optional: Precision = TP/(TP+FP), Recall = TP/(TP+FN) to analyze types of errors.
- **Tiny Example**
  - Suppose on 20 test images, we get TP=8, FP=2, TN=8, FN=2.
  - Accuracy = (TP+TN)/20 = (8+8)/20 = 0.80.
  - Precision = 8/(8+2) = 0.80; Recall = 8/(8+2) = 0.80.
- **In this project**
  - We will report accuracy and show a confusion matrix. We will also include a learning curve figure to demonstrate training behavior.

---

## 6) How Everything Fits Together (Training Pipeline)

1. Start with random w and b.
2. For a batch of training images:
   - Compute z = w · x + b for each image.
   - Convert z to probabilities p = σ(z).
   - Compute the average binary cross‑entropy loss over the batch.
   - Compute gradients of loss with respect to w and b.
   - Update w and b using gradient descent (optimizer).
3. Repeat for many epochs. Track training and validation loss.
4. After training, evaluate on the test set: compute accuracy and the confusion matrix.

This simple pipeline uses only Math SL concepts: functions, logarithms, derivatives (for the idea of gradients), and optimization (gradient descent). The model remains interpretable because it is just a weighted sum followed by a sigmoid.

---

## 7) Assumptions and Limitations

- A single neuron is a linear model; it works best if a weighted sum of pixels can separate the two classes reasonably well.
- If backgrounds and lighting vary a lot, more data or simple augmentations (flips, small crops) may help.
- Our aim is educational clarity and honest, reproducible results, not beating state‑of‑the‑art methods.

---

## 8) Quick Glossary

- Pixel: a small square in the image; its grayscale value is a number between 0 (black) and 1 (white).
- Dot product: sum of pairwise products of two equal‑length vectors.
- Sigmoid: function that maps any real number to (0,1), used as probability.
- Loss: a number measuring prediction error; smaller is better.
- Gradient: vector of partial derivatives indicating the direction to change parameters to reduce loss.
- Epoch: one full pass through the training dataset.

---

## 9) Optional: Why the gradient is (p − y)

This short derivation is for the curious reader. You don’t need to memorize it; it just explains why gradient descent updates involve the simple term (p − y).

- Start with the binary cross‑entropy for one example, with p = σ(z):
  - L = −[ y · log(p) + (1 − y) · log(1 − p) ]
- Differentiate with respect to p:
  - ∂L/∂p = −[ y/p − (1 − y)/(1 − p) ]
- The sigmoid derivative is: ∂p/∂z = p(1 − p).
- Chain rule: ∂L/∂z = (∂L/∂p)(∂p/∂z)
  - = −[ y/p − (1 − y)/(1 − p) ] · p(1 − p)
  - = −[ y(1 − p) − (1 − y)p ]
  - = −[ y − yp − p + yp ] = −(y − p) = (p − y)
- Since z = w · x + b, by the chain rule:
  - ∂L/∂w = (p − y) · x
  - ∂L/∂b = (p − y)

Intuition: if the model’s probability p is too high compared to the truth y (for example, predicting 0.9 when y=0), the term (p − y) is positive and pushes the parameters down. If p is too low (say 0.1 when y=1), (p − y) is negative and pushes them up.

---

## 10) Further Reading (friendly sources)

- Binary Cross‑Entropy (Log Loss): GeeksforGeeks — https://www.geeksforgeeks.org/deep-learning/binary-cross-entropy-log-loss-for-binary-classification/
- Sigmoid in Logistic Regression: Google ML Crash Course — https://developers.google.com/machine-learning/crash-course/logistic-regression/sigmoid-function
- Gradient Descent (overview): GeeksforGeeks — https://www.geeksforgeeks.org/machine-learning/gradient-descent-algorithm-and-its-variants/
