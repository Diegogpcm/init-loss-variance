"""Exp 2: controlled init-gain sweep.

Fix depth/width, vary activation x residual x init gain on a fine grid. Init gain moves
networks between the ordered and chaotic phases, which changes the init-loss distribution
dramatically while the parameter count stays fixed -- a clean test of whether loss
variance tracks trainability/generalization or just logit scale.

Usage: python experiments/run_gain_sweep.py --out results/gain_sweep.jsonl
"""
import argparse, json, os, sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lvproxy import Arch, ACTIVATIONS  # noqa: E402
from run_sweep import job  # noqa: E402

GAINS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--n-train", default="full")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--probe", type=int, default=256)
    ap.add_argument("--dataset", default="digits")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    archs = [Arch(a.depth, a.width, act, res, g) for act in ACTIVATIONS for res in (False, True) for g in GAINS]
    jobs = [(ar, a.n_train, a.seeds, a.epochs, a.K, a.probe, a.dataset) for ar in archs]
    with open(a.out, "w") as f, Pool(a.workers) as pool:
        for r in pool.imap_unordered(job, jobs, chunksize=1):
            f.write(json.dumps(r) + "\n")
            f.flush()
    print("done", len(jobs))
