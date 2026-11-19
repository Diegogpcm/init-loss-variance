"""Exp 1: random architecture sweep on digits.

For each sampled architecture: compute init-time proxies (K inits on a fixed probe batch),
then train with S seeds and record train/test accuracy and loss.

Usage: python experiments/run_sweep.py --n-archs 150 --n-train full --out results/sweep_full.jsonl
"""
import argparse, json, os, sys, time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lvproxy import Arch, ACTIVATIONS, train, all_proxies  # noqa: E402
from lvproxy.data import digits  # noqa: E402

DEPTHS = [1, 2, 4, 8, 12]
WIDTHS = [16, 32, 64, 128]
GAINS = [0.5, 1.0, 1.414, 2.0, 3.0]


def sample_archs(n, seed):
    rng = np.random.default_rng(seed)
    seen, archs = set(), []
    while len(archs) < n:
        a = Arch(int(rng.choice(DEPTHS)), int(rng.choice(WIDTHS)), str(rng.choice(ACTIVATIONS)),
                 bool(rng.integers(2)), float(rng.choice(GAINS)))
        if a not in seen:
            seen.add(a)
            archs.append(a)
    return archs


def job(args):
    arch, n_train, n_seeds, epochs, K, probe = args
    Xtr, ytr, Xte, yte, C = digits()
    if n_train != "full":
        Xtr, ytr = Xtr[: int(n_train)], ytr[: int(n_train)]
    t0 = time.time()
    prox = all_proxies(arch, Xtr[:probe], ytr[:probe], C, K=K)
    runs = [train(arch, Xtr, ytr, Xte, yte, C, seed=s, epochs=epochs) for s in range(n_seeds)]
    res = {k: float(np.mean([r[k] for r in runs])) for k in ("train_loss", "train_acc", "test_loss", "test_acc")}
    res["test_acc_std"] = float(np.std([r["test_acc"] for r in runs]))
    res["n_diverged"] = int(sum(r["diverged"] for r in runs))
    return {"arch": arch.as_dict(), "name": arch.name(), "proxies": prox, "trained": res,
            "seconds": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-archs", type=int, default=150)
    ap.add_argument("--n-train", default="full")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--K", type=int, default=32, help="number of inits for proxies")
    ap.add_argument("--probe", type=int, default=256, help="probe batch size")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--arch-seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    archs = sample_archs(a.n_archs, a.arch_seed)
    jobs = [(ar, a.n_train, a.seeds, a.epochs, a.K, a.probe) for ar in archs]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f, Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(job, jobs)):
            f.write(json.dumps(r) + "\n")
            f.flush()
            if i % 10 == 0:
                print(f"[{i+1}/{len(jobs)}] {r['name']} test_acc={r['trained']['test_acc']:.3f}", flush=True)


if __name__ == "__main__":
    main()
