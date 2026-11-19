"""Datasets. `digits` is bundled with scikit-learn; `fashion` downloads Fashion-MNIST once
(cached in ~/scikit_learn_data/fashion) and uses a fixed stratified 10k/5k subset."""
from functools import lru_cache

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def _standardize(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    return (Xtr - mu) / sd, (Xte - mu) / sd


def digits(seed=0):
    d = load_digits()
    X = (d.data / 16.0).astype(np.float64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    Xtr, Xte, ytr, yte = train_test_split(X, d.target, test_size=0.25, random_state=seed,
                                          stratify=d.target)
    return Xtr, ytr, Xte, yte, 10


FASHION_URL = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"
FASHION_FILES = ["train-images-idx3-ubyte", "train-labels-idx1-ubyte",
                 "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"]


def _fashion_raw():
    """All 70k Fashion-MNIST images (uint8, flattened) and labels, from the official IDX files
    cached in ~/scikit_learn_data/fashion (downloaded from the Zalando GitHub repo if missing)."""
    import gzip, os, urllib.request
    from sklearn.datasets import get_data_home
    root = os.path.join(get_data_home(), "fashion")
    os.makedirs(root, exist_ok=True)
    arrs = []
    for f in FASHION_FILES:
        path = os.path.join(root, f + ".gz")
        if not os.path.exists(path):
            urllib.request.urlretrieve(FASHION_URL + f + ".gz", path)
        with gzip.open(path, "rb") as fh:
            data = fh.read()
        arrs.append(np.frombuffer(data, np.uint8, offset=16 if "images" in f else 8))
    X = np.concatenate([arrs[0], arrs[2]]).reshape(-1, 784)
    y = np.concatenate([arrs[1], arrs[3]]).astype(int)
    return X, y


@lru_cache(maxsize=None)
def fashion(n_train=10_000, n_test=5_000, seed=0):
    X, y = _fashion_raw()
    X = X.astype(np.float64) / 255.0
    Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=n_train, test_size=n_test,
                                          random_state=seed, stratify=y)
    Xtr, Xte = _standardize(Xtr, Xte)
    return Xtr, ytr, Xte, yte, 10


DATASETS = {"digits": digits, "fashion": fashion}


def load(name):
    return DATASETS[name]()
