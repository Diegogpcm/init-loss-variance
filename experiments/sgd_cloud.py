"""Is SGD's own randomness (batch order, amplified by chaotic dynamics) a large 'free' noise relative to the
data-to-data variation of its output? (theory.md s15)

Same init, plain SGD (no clipping/noise). Two disjoint halves A, B of the training set (n/2 each). K batch-order seeds
on each. Per layer:
  within  s^2  = mean per-coordinate variance of W_T across seeds (pooled over A and B)
  between D^2  = ||mean_A - mean_B||^2 - (2/K) * within total     (bias-corrected)
  travel  R^2  = ||mean W_T - W_0||^2
Isotropic-Gaussian heuristics at the cloud's own scale: KL(cloud_A || cloud_B) ~ D^2/(2 s^2)   vs   R^2/(2 s^2), the
init-centred Gaussian PAC-Bayes KL for a posterior of the same width.  Also: chaos test (init perturbed by 1e-6).
Heuristic only: isotropy and Gaussianity of the cloud are assumptions; the number of seeds limits resolution.
"""
import argparse, json, math, os, sys
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sgd_influence import setup, err  # noqa: E402


def train_one(job):
    a, half, order_seed, init_eps = job
    E = setup(a)
    torch = E["torch"]
    n = E["n"]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(12345))
    part = perm[: n // 2] if half == "A" else perm[n // 2: 2 * (n // 2)]
    th = E["init"](a.seed)
    if init_eps:
        th = th + init_eps * torch.randn(th.shape, generator=torch.Generator().manual_seed(999))
    th0 = th.clone()
    g = torch.Generator().manual_seed(order_seed)
    gradfn = torch.func.grad(lambda th, Xb, yb: E["losses"](th, Xb, yb).mean())
    m = len(part)
    spe = m // a.batch
    for ep in range(a.epochs):
        p = part[torch.randperm(m, generator=g)]
        for s in range(spe):
            idx = p[s * a.batch:(s + 1) * a.batch]
            th = th - a.lr * gradfn(th, E["X"][idx], E["y"][idx])
    return dict(half=half, order_seed=order_seed, init_eps=init_eps, w=th.numpy(), w0=th0.numpy(),
                train01=err(E, th, E["X"][part], E["y"][part]), test01=err(E, th, E["Xte"], E["yte"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--workers", type=int, default=34)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    jobs = [(a, h, 100 + k, 0.0) for h in "AB" for k in range(a.K)] + [(a, "A", 100, 1e-6)]
    with Pool(min(a.workers, len(jobs))) as pool:
        runs = pool.map(train_one, jobs)
    E = setup(a)
    groups = {f"L{l}": None for l in range(a.depth + 1)}
    dims = [E["X"].shape[1]] + [a.width] * a.depth + [int(E["y"].max()) + 1]
    offs, o = {}, 0
    for l in range(len(dims) - 1):
        size = dims[l] * dims[l + 1] + dims[l + 1]
        offs[f"L{l}"] = (o, o + size)
        o += size
    WA = np.stack([r["w"] for r in runs if r["half"] == "A" and not r["init_eps"]]).astype(np.float64)
    WB = np.stack([r["w"] for r in runs if r["half"] == "B" and not r["init_eps"]]).astype(np.float64)
    w0 = runs[0]["w0"].astype(np.float64)
    pert = [r for r in runs if r["init_eps"]][0]["w"].astype(np.float64)
    K = a.K
    rep = dict(cfg={k: v for k, v in vars(a).items()}, runs=[{k: r[k] for k in ("half", "order_seed", "init_eps", "train01", "test01")} for r in runs], layers={})
    tot = dict(D2=0, R2=0, kl_cloud=0, kl_init=0)
    for gname, (lo, hi) in offs.items():
        A, Bm = WA[:, lo:hi], WB[:, lo:hi]
        within_tot = 0.5 * (A.var(0, ddof=1).sum() + Bm.var(0, ddof=1).sum())     # E||W - mean||^2 per cloud
        s2 = within_tot / (hi - lo)
        D2_raw = float(((A.mean(0) - Bm.mean(0)) ** 2).sum())
        D2 = D2_raw - 2 * within_tot / K
        R2 = float(((0.5 * (A.mean(0) + Bm.mean(0)) - w0[lo:hi]) ** 2).sum())
        chaos = float(((pert[lo:hi] - WA[0, lo:hi]) ** 2).sum())
        seedpair = float(((WA[0, lo:hi] - WA[1, lo:hi]) ** 2).sum())
        L = dict(dim=hi - lo, within_total=float(within_tot), s=float(math.sqrt(s2)), D2_raw=D2_raw, D2=float(D2),
                 R2=R2, kl_cloud=float(max(D2, 0) / (2 * s2)), kl_init=float(R2 / (2 * s2)),
                 chaos_init_eps_sq=chaos, seed_pair_sq=seedpair)
        rep["layers"][gname] = L
        for k in tot:
            tot[k] += L[k] if k in L else 0
    rep["total"] = tot
    json.dump(rep, open(a.out, "w"), indent=1)
    print(json.dumps(rep["layers"], indent=1))
    print("total", tot)
    # function space: disagreement of argmax predictions and mean symmetric KL of softmax outputs, within vs between
    torch = E["torch"]
    n = E["n"]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(12345))
    sets = {"test": E["Xte"], "A_train": E["X"][perm[:5000]], "B_train": E["X"][perm[n // 2: n // 2 + 5000]]}
    runsA = [r for r in runs if r["half"] == "A" and not r["init_eps"]]
    runsB = [r for r in runs if r["half"] == "B" and not r["init_eps"]]
    with torch.no_grad():
        P = {name: {id(r): torch.log_softmax(E["logits"](torch.from_numpy(r["w"]), Xs), 1)
                    for r in runsA + runsB} for name, Xs in sets.items()}

    def stats(name, pairs):
        dis, skl = [], []
        for r1, r2 in pairs:
            l1, l2 = P[name][id(r1)], P[name][id(r2)]
            dis.append(float((l1.argmax(1) != l2.argmax(1)).float().mean()))
            skl.append(float(0.5 * ((l1.exp() * (l1 - l2)).sum(1) + (l2.exp() * (l2 - l1)).sum(1)).mean()))
        return dict(disagree=float(np.mean(dis)), sym_kl=float(np.mean(skl)))
    within = [(runsA[i], runsA[j]) for i in range(len(runsA)) for j in range(i + 1, len(runsA))] + \
             [(runsB[i], runsB[j]) for i in range(len(runsB)) for j in range(i + 1, len(runsB))]
    between = [(r1, r2) for r1 in runsA for r2 in runsB]
    rep["function"] = {name: dict(within=stats(name, within), between=stats(name, between)) for name in sets}
    rep["function"]["note"] = "A_train: training points of the A-models (held out for B-models); B_train vice versa"
    json.dump(rep, open(a.out, "w"), indent=1)
    print("function space:", json.dumps(rep["function"], indent=1))
    np.save(a.out.replace(".json", "_weights.npy"), np.stack([r["w"] for r in runs]).astype(np.float32))
    errs = np.array([[r["train01"], r["test01"]] for r in runs if not r["init_eps"]])
    print("train/test err mean over seeds:", errs.mean(0), "std:", errs.std(0))
