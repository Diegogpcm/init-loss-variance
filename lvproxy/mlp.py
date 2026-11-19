"""Minimal NumPy MLP with manual backprop.

Kept dependency-free on purpose (NumPy only) so the first experiments run anywhere.
Architecture:  x -> [W_in] -> h_0 -> L blocks -> [W_out] -> logits
Each block:    z = h W + b ; a = act(z) ; h' = h + a (residual) or a (plain)
Init:          W ~ N(0, gain^2 / fan_in), b = 0
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np

# ---------------------------------------------------------------- activations
_SQ2PI = np.sqrt(2.0 / np.pi)


def _act(name, z):
    if name == "relu":
        return np.maximum(z, 0.0)
    if name == "tanh":
        return np.tanh(z)
    if name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
    if name == "linear":
        return z
    if name == "gelu":
        return 0.5 * z * (1.0 + np.tanh(_SQ2PI * (z + 0.044715 * z ** 3)))
    if name == "sin":
        return np.sin(z)
    raise ValueError(name)


def _dact(name, z):
    if name == "relu":
        return (z > 0).astype(z.dtype)
    if name == "tanh":
        t = np.tanh(z)
        return 1.0 - t * t
    if name == "sigmoid":
        s = 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
        return s * (1.0 - s)
    if name == "linear":
        return np.ones_like(z)
    if name == "gelu":
        u = _SQ2PI * (z + 0.044715 * z ** 3)
        t = np.tanh(u)
        du = _SQ2PI * (1.0 + 3 * 0.044715 * z ** 2)
        return 0.5 * (1.0 + t) + 0.5 * z * (1.0 - t * t) * du
    if name == "sin":
        return np.cos(z)
    raise ValueError(name)


ACTIVATIONS = ["relu", "tanh", "sigmoid", "linear", "gelu", "sin"]


# ---------------------------------------------------------------- architecture
@dataclass(frozen=True)
class Arch:
    depth: int          # number of hidden blocks
    width: int
    act: str
    residual: bool
    gain: float         # init std multiplier (1.0 ~ LeCun, sqrt(2) ~ He)

    def name(self):
        return f"d{self.depth}-w{self.width}-{self.act}-{'res' if self.residual else 'plain'}-g{self.gain:g}"

    def n_params(self, d_in, d_out):
        w = self.width
        return d_in * w + w + self.depth * (w * w + w) + w * d_out + d_out

    def as_dict(self):
        return asdict(self)


def init_params(arch: Arch, d_in: int, d_out: int, rng: np.random.Generator):
    w = arch.width
    shapes = [(d_in, w)] + [(w, w)] * arch.depth + [(w, d_out)]
    params = []
    for i, (fi, fo) in enumerate(shapes):
        # gain applies to hidden blocks; input/output layers use LeCun scale
        g = arch.gain if 0 < i < len(shapes) - 1 else 1.0
        params.append([rng.normal(0, g / np.sqrt(fi), size=(fi, fo)), np.zeros(fo)])
    return params


# ---------------------------------------------------------------- forward / backward
def forward(arch: Arch, params, X, keep=False):
    W, b = params[0]
    h = X @ W + b
    cache = [(X, None, None)]
    pre = []
    for l in range(arch.depth):
        W, b = params[1 + l]
        z = h @ W + b
        a = _act(arch.act, z)
        cache.append((h, z, a))
        pre.append(z)
        h = h + a if arch.residual else a
    W, b = params[-1]
    logits = h @ W + b
    cache.append((h, None, None))
    return (logits, cache, pre) if keep else logits


def log_softmax(logits):
    m = logits.max(1, keepdims=True)
    s = logits - m
    return s - np.log(np.exp(s).sum(1, keepdims=True))


def per_example_ce(logits, y):
    """Cross-entropy for each example (nan/inf safe-ish)."""
    lp = log_softmax(logits)
    return -lp[np.arange(len(y)), y]


def loss_and_grads(arch: Arch, params, X, y):
    logits, cache, _ = forward(arch, params, X, keep=True)
    n = len(y)
    lp = log_softmax(logits)
    loss = -lp[np.arange(n), y].mean()
    g = np.exp(lp)
    g[np.arange(n), y] -= 1.0
    g /= n                                  # dL/dlogits
    grads = [None] * len(params)
    h_last = cache[-1][0]
    W_out = params[-1][0]
    grads[-1] = [h_last.T @ g, g.sum(0)]
    dh = g @ W_out.T
    for l in reversed(range(arch.depth)):
        h_in, z, _ = cache[1 + l]
        dz = dh * _dact(arch.act, z)
        W = params[1 + l][0]
        grads[1 + l] = [h_in.T @ dz, dz.sum(0)]
        dh_in = dz @ W.T
        dh = dh + dh_in if arch.residual else dh_in
    X0 = cache[0][0]
    grads[0] = [X0.T @ dh, dh.sum(0)]
    return loss, grads


# ---------------------------------------------------------------- training
def train(arch: Arch, Xtr, ytr, Xte, yte, n_classes, seed=0, epochs=60, bs=64, lr=1e-3,
          clip=5.0):
    rng = np.random.default_rng(seed)
    params = init_params(arch, Xtr.shape[1], n_classes, rng)
    m = [[np.zeros_like(p) for p in layer] for layer in params]
    v = [[np.zeros_like(p) for p in layer] for layer in params]
    b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
    diverged = False
    with np.errstate(all="ignore"):
        for ep in range(epochs):
            perm = rng.permutation(len(ytr))
            for s in range(0, len(perm), bs):
                idx = perm[s:s + bs]
                loss, grads = loss_and_grads(arch, params, Xtr[idx], ytr[idx])
                if not np.isfinite(loss):
                    diverged = True
                    break
                gn = np.sqrt(sum((g ** 2).sum() for layer in grads for g in layer))
                scale = min(1.0, clip / (gn + 1e-12))
                t += 1
                for i, layer in enumerate(grads):
                    for j, g in enumerate(layer):
                        g = g * scale
                        m[i][j] = b1 * m[i][j] + (1 - b1) * g
                        v[i][j] = b2 * v[i][j] + (1 - b2) * g * g
                        mh = m[i][j] / (1 - b1 ** t)
                        vh = v[i][j] / (1 - b2 ** t)
                        params[i][j] -= lr * mh / (np.sqrt(vh) + eps)
            if diverged:
                break
        out = {}
        for split, X, y in (("train", Xtr, ytr), ("test", Xte, yte)):
            logits = forward(arch, params, X)
            ce = per_example_ce(logits, y)
            out[f"{split}_loss"] = float(np.nanmean(ce)) if np.isfinite(ce).all() else float("nan")
            out[f"{split}_acc"] = float((logits.argmax(1) == y).mean()) if np.isfinite(logits).all() else 1.0 / n_classes
    out["diverged"] = diverged
    return out
