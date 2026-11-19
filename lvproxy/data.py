"""Offline datasets (bundled with scikit-learn, no download needed)."""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def digits(seed=0):
    d = load_digits()
    X = (d.data / 16.0).astype(np.float64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    Xtr, Xte, ytr, yte = train_test_split(X, d.target, test_size=0.25, random_state=seed,
                                          stratify=d.target)
    return Xtr, ytr, Xte, yte, 10
