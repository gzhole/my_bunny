# IB Math IA: My Bunny vs Other Bunnies — Single-Neuron Classifier

## Research Question
Can a single-neuron logistic regression model distinguish my bunny from other bunnies using grayscale images, with accuracy meaningfully above chance, using only Math SL concepts (functions, derivatives, optimization)?

## Motivation
- Apply core Math SL ideas to a real task (classification) with minimal programming complexity.
- Use a transparent model whose math can be fully explained: one neuron (logistic regression).

## Scope and Constraints
- Binary classification: label 1 = my bunny, 0 = other bunnies.
- Keep features simple: grayscale, resized images (e.g., 64×64), flattened to a vector of pixel intensities.
- Minimal library use: Python + PyTorch; no deep architectures; no data scraping code in report.

## Mathematical Ideas (kept simple)
- Sigmoid function: σ(z) = 1 / (1 + e^(−z)). Interprets output as probability of class 1.
- Linear score: z = w·x + b where x = pixel vector, w,b are parameters.
- Loss (cross‑entropy for binary): L = −[y·log(σ(z)) + (1−y)·log(1−σ(z))].
- Gradient descent (learning): update w,b in the opposite direction of the loss gradient to reduce L.
- Evidence shown via learning curve (loss decreasing) and evaluation metrics.

## Data Plan
- My bunny photos: capture multiple poses, backgrounds, lighting. Target ~100–300 images if possible.
- Other bunnies: collect public images (note sources and licenses). Balance classes.
- Split: train 70%, validation 15%, test 15% with no overlap.
- Preprocessing: center-crop (if needed), resize to 64×64, grayscale, normalize to [0,1].

## Modeling Plan (PyTorch)
- Model: single linear layer from 64×64=4096 inputs to 1 output (no hidden layers), followed by sigmoid at inference.
- Training objective: `BCEWithLogitsLoss` (numerically stable cross-entropy); optimizer: SGD.
- Batch training with small learning rate; early stopping via validation loss.

## Evaluation
- Primary metric: accuracy on held-out test set (binary decision threshold 0.5).
- Additional: confusion matrix; precision/recall to show error types.
- Learning curves: training/validation loss vs epochs to demonstrate optimization.

## Success Criteria
- Loss decreases consistently during training (learning observed).
- Test accuracy ≥ 0.80 (adjust if dataset small; justify threshold).
- Confusion matrix shows both classes recognized (not trivial all-one prediction).
- Reproducible run with fixed seed and documented steps.

## Risks and Mitigations
- Class imbalance → enforce balanced sampling or class weighting.
- Overfitting → use validation set, early stopping, simple augmentation (flips, slight crops) if needed.
- Data quality → ensure varied images of my bunny; avoid near-duplicates across splits.

## Ethics and Academic Honesty
- Respect image licenses; cite all sources for other-bunny images.
- Avoid faces/people in background where possible; keep personal data minimal.

## Deliverables
- Code (minimal, well-commented): dataset loader, training, evaluation.
- Tests: dataset shape/label checks; model forward shape; one training step reduces loss.
- Report sections for IA: motivation, method (math), results (tables/plots), discussion, limitations.
- Reproducibility: README with exact steps to run and test.

## Timeline (proposed)
- Week 1: Collect data; finalize goals; scaffold repo.
- Week 2: Implement dataset + model; initial training run; refine preprocessing.
- Week 3: Evaluate; produce plots; write IA math explanation; finalize results.

## Tools
- Python (virtual environment), PyTorch, torchvision, Pillow, NumPy, Matplotlib, PyTest.

## IA Alignment
- Clear research question; mathematical modeling using functions/derivatives; analysis of results; reflection on limitations and next steps.
