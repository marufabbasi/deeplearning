# Deep Learning Training Debug Checklist
Run this **top → bottom**. Stop as soon as something fails.

---

## 1. Single-Batch Overfit Test (NON-NEGOTIABLE)
- [ ] Freeze dataset to **1 batch** (e.g., 32–128 samples)
- [ ] Disable all augmentation & regularization
- [ ] Train for 200–500 steps
- [ ] Training loss → **~0**
- [ ] Training accuracy → **~100%**

If this fails: model wiring, loss, labels, or optimizer is wrong.

---

## 2. Data & Labels
- [ ] Input tensor shape is correct and consistent
- [ ] Inputs normalized / scaled appropriately
- [ ] Labels have correct dtype (e.g., integer indices vs floats)
- [ ] Label shape matches loss expectations
- [ ] Label values are valid (no out-of-range classes)
- [ ] No NaNs / Infs in inputs or labels

---

## 3. Model ↔ Loss Compatibility
- [ ] Final layer outputs **raw logits** when required
- [ ] No extra activation before the loss (unless explicitly needed)
- [ ] Loss function matches task:
  - Classification → Cross-entropy / NLL
  - Binary → BCE with logits
  - Regression → MSE / MAE
- [ ] Output tensor shape matches loss expectation

---

## 4. Training / Evaluation Mode
- [ ] Training uses `train()` mode
- [ ] Validation/testing uses `eval()` mode
- [ ] Dropout active only during training
- [ ] BatchNorm/LayerNorm behaving as intended

---

## 5. Training Loop Correctness
- [ ] Gradients cleared before backward pass
- [ ] `backward()` called exactly once per step
- [ ] Optimizer `step()` called every iteration
- [ ] Loss decreases within first few iterations
- [ ] Model parameters actually change

---

## 6. Optimization Sanity Defaults
- [ ] Start with Adam / AdamW
- [ ] Learning rate in sane range (e.g., `1e-4`–`1e-3`)
- [ ] Batch size reasonable for hardware
- [ ] No scheduler initially
- [ ] No weight decay initially

---

## 7. Validation & Metrics
- [ ] Validation uses `no_grad()` / inference mode
- [ ] Metrics computed on correct dimensions
- [ ] Padding / masked tokens excluded if applicable
- [ ] Validation performance < training performance (expected early)

---

## 8. Numerical Stability
- [ ] Loss is finite every step
- [ ] Gradients are finite
- [ ] Gradient norms not exploding
- [ ] Add gradient clipping if unstable

---

## 9. Scaling Up Safely
- [ ] Overfit tiny dataset (≤1k samples)
- [ ] Add regularization **after** model works
- [ ] Add data augmentation **after** convergence
- [ ] Introduce schedulers & mixed precision last

---

## Final Rule
> **If a model cannot overfit a tiny dataset, it will never generalize.**
