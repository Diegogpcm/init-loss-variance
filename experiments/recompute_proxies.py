"""Recompute init-time proxies for an existing sweep file (no retraining).

Usage: python experiments/recompute_proxies.py results/sweep_full.jsonl [--n-train 200]
"""
import argparse, json, os, sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lvproxy import Arch, all_proxies  # noqa: E402
from lvproxy.data import digits  # noqa: E402

ARGS = {}


def job(row):
    Xtr, ytr, _, _, C = digits()
    if ARGS["n_train"] != "full":
        Xtr, ytr = Xtr[: int(ARGS["n_train"])], ytr[: int(ARGS["n_train"])]
    p = ARGS["probe"]
    row["proxies"] = all_proxies(Arch(**row["arch"]), Xtr[:p], ytr[:p], C, K=ARGS["K"])
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--n-train", default="full")
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--probe", type=int, default=256)
    a = ap.parse_args()
    ARGS.update(n_train=a.n_train, K=a.K, probe=a.probe)
    rows = [json.loads(l) for l in open(a.file)]
    with Pool(os.cpu_count()) as pool:
        rows = pool.map(job, rows)
    with open(a.file, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"updated {len(rows)} rows")
