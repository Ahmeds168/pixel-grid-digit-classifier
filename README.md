# Pixel Grid — Digit Classifier

An interactive linear-vs-RBF kernel SVM digit classifier, trained on scikit-learn's `digits` dataset (the classic 8×8 handwritten-digit set).

- Draw a digit on an 8×8 pixel grid (or load a real held-out test sample)
- See live predictions from both a linear-kernel and an RBF-kernel SVM
- Compare each model's held-out test accuracy
- Inference runs entirely client-side in `index.html` — no backend needed

Built from `Classifying_Handwritten_Digits.py`, a lab notebook that trains and compares the two kernels using `StandardScaler`, an 80/20 stratified split, and `GridSearchCV` for tuning.

## Run locally
Just open `index.html` in a browser — it's fully self-contained.
