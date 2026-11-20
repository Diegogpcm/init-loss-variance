"""Chaos as self-generated noise in function space: supersample (CMI) information under chaotic vs non-chaotic training,
fixed vs random seed, true vs random labels (theory.md s15.3-15.4). Explanatory, not a certificate.

Supersample z~ in Z^{n x 2} (disjoint rows from the Fashion training set); U ~ Unif{0,1}^n picks the training half.
For each (condition, supersample) we train k2 models on z~_U and record the 0-1 losses on all 2n points.
Per pair i: ld-CMI  I(dL_i; U_i | z~) with dL_i = L_i1 - L_i0 in {-1,0,1}  (Wang & Mao 2023, Thm 3.1:
|E gen| <= (1/n) sum_i E sqrt(2 I(dL_i; U_i | z~)) ), and e-CMI I(L_i; U_i | z~) with L_i in {0,1}^2.
Plug-in estimates over the k2 draws, with a permutation null (shuffle U across draws) to estimate the small-sample bias.
Fixed seed: init and batch order identical across draws (deterministic algorithm, randomness only from U).
Random seed: init and batch order drawn afresh for every draw (randomised algorithm, R marginalised).
Chaos index per condition: train twice on the same data with the init perturbed by 1e-6; report the fraction of test
predictions that differ and ||dW|| / ||W - W0||.
"""
import argparse, json, math, os, sys, time
import multiprocessing as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CONDS = {
    "sgd": dict(opt="sgd", lr=0.1, batch=128, epochs=60),
    "gd": dict(opt="gd", lr=None, steps=None),          # set by --gd_lr / --gd_steps after calibration
}


def data(a):
    import torch
    d = np.load(a.data)
    X = (torch.tensor(d["Xtr"], dtype=torch.float32) - 0.5) / 0.5
    y = torch.tensor(d["ytr"]).long()
    Xte = (torch.tensor(d["Xte"], dtype=torch.float32) - 0.5) / 0.5
    yte = torch.tensor(d["yte"]).long()
    return X, y, Xte, yte


def make_net(din, width, depth, K):
    import torch
    dims = [din] + [width] * depth + [K]

    def init(seed):
        g = torch.Generator().manual_seed(seed)
        ps = []
        for l in range(len(dims) - 1):
            std = math.sqrt((2.0 if l < len(dims) - 2 else 1.0) / dims[l])
            ps += [torch.randn(dims[l], dims[l + 1], generator=g) * std, torch.zeros(dims[l + 1])]
        return ps

    def logits(ps, Xb):
        h = Xb
        L = len(ps) // 2
        for l in range(L):
            h = h @ ps[2 * l] + ps[2 * l + 1]
            if l < L - 1:
                h = torch.relu(h)
        return h
    return init, logits


def train(a, cond, Xtr, ytr, init_seed, order_seed, init_eps=0.0):
    import torch
    torch.set_num_threads(a.threads)
    init, logits = make_net(Xtr.shape[1], a.width, a.depth, 10)
    ps = init(init_seed)
    if init_eps:
        g = torch.Generator().manual_seed(999)
        ps = [p + init_eps * torch.randn(p.shape, generator=g) for p in ps]
    w0 = torch.cat([p.flatten() for p in ps]).clone()
    ps = [p.requires_grad_() for p in ps]
    n = len(Xtr)
    g = torch.Generator().manual_seed(order_seed)
    if cond["opt"] == "sgd":
        steps = [(torch.randperm(n, generator=g)) for _ in range(cond["epochs"])]
        batches = [p[i:i + cond["batch"]] for p in steps for i in range(0, n - cond["batch"] + 1, cond["batch"])]
        lr = cond["lr"]
    else:
        batches = [None] * cond["steps"]
        lr = cond["lr"]
    for idx in batches:
        Xb, yb = (Xtr, ytr) if idx is None else (Xtr[idx], ytr[idx])
        loss = torch.nn.functional.cross_entropy(logits(ps, Xb), yb)
        grads = torch.autograd.grad(loss, ps)
        with torch.no_grad():
            for p, gr in zip(ps, grads):
                p -= lr * gr
    ps = [p.detach() for p in ps]
    return ps, logits, w0


def _calib_job(job):
    a, cname, cond, Xtr, ytr, Xte, yte = job
    import torch
    t0 = time.time()
    p1, logits, w0 = train(a, cond, Xtr, ytr, 0, 0)
    p2, _, _ = train(a, cond, Xtr, ytr, 0, 0, init_eps=1e-6)
    w1, w2 = torch.cat([p.flatten() for p in p1]), torch.cat([p.flatten() for p in p2])
    with torch.no_grad():
        pr1, pr2 = logits(p1, Xte).argmax(1), logits(p2, Xte).argmax(1)
        tr = float((logits(p1, Xtr).argmax(1) != ytr).float().mean())
    return dict(cond=cname, cfg=cond, train01=tr, test01=float((pr1 != yte).float().mean()),
                chaos_pred=float((pr1 != pr2).float().mean()), chaos_w=float((w1 - w2).norm() / (w1 - w0).norm()),
                seconds=time.time() - t0)


_CACHE = {}


def supersample(a, j, labels):
    import torch
    if "data" not in _CACHE:
        _CACHE["data"] = data(a)
    X, y, _, _ = _CACHE["data"]
    perm = np.random.default_rng(123).permutation(len(X))
    rows = torch.from_numpy(perm[2 * a.n * j: 2 * a.n * (j + 1)].reshape(a.n, 2))
    Zy = y[rows] if labels == "true" else torch.from_numpy(np.random.default_rng(500 + j).integers(0, 10, (a.n, 2)))
    return X[rows], Zy


def _main_job(job):
    a, cname, cond, seed_mode, labels, j, k = job
    import torch
    Zx, Zy = supersample(a, j, labels)
    n = Zx.shape[0]
    U = torch.from_numpy(np.random.default_rng([j, k, 17]).integers(0, 2, n))
    tr_idx = torch.arange(n)
    Xtr = Zx[tr_idx, U]
    ytr = Zy[tr_idx, U]
    if seed_mode == "fixed":
        init_seed, order_seed = 0, 0
    else:
        init_seed, order_seed = 1000 + 97 * k + j, 2000 + 97 * k + j
    ps, logits, _ = train(a, cond, Xtr, ytr, init_seed, order_seed)
    with torch.no_grad():
        L = torch.stack([(logits(ps, Zx[:, c]).argmax(1) != Zy[:, c]) for c in (0, 1)], 1).to(torch.int8)
    return dict(cond=cname, seed_mode=seed_mode, labels=labels, j=j, k=k, U=U.numpy().astype(np.int8), L=L.numpy())


def mi_plugin(x, u):
    """plug-in MI (nats) between discrete x (values hashed to ints) and binary u, columns = pairs, rows = draws."""
    k, n = x.shape
    out = np.zeros(n)
    vals = np.unique(x)
    for v in vals:
        px = (x == v).mean(0)
        for b in (0, 1):
            pu = (u == b).mean(0)
            pj = ((x == v) & (u == b)).mean(0)
            with np.errstate(divide="ignore", invalid="ignore"):
                out += np.where(pj > 0, pj * np.log(pj / (px * pu)), 0.0)
    return out


def analyse(runs, n_perm=20, seed=0):
    U = np.stack([r["U"] for r in runs])                 # k x n
    L = np.stack([r["L"] for r in runs]).astype(int)     # k x n x 2
    dL = L[:, :, 1] - L[:, :, 0]
    Lpair = L[:, :, 0] * 2 + L[:, :, 1]
    rng = np.random.default_rng(seed)
    res = {}
    for name, x in (("ld", dL), ("e", Lpair)):
        mi = mi_plugin(x, U)
        null = np.mean([mi_plugin(x, U[rng.permutation(len(U))]) for _ in range(n_perm)], 0)
        corr = np.maximum(mi - null, 0)
        res[f"{name}_mi_raw"] = float(mi.mean())
        res[f"{name}_mi_null"] = float(null.mean())
        res[f"{name}_mi_corr"] = float(corr.mean())
        res[f"{name}_bound_raw"] = float(np.sqrt(2 * mi).mean())
        res[f"{name}_bound_corr"] = float(np.sqrt(2 * corr).mean())
    tr = np.where(U == 0, L[:, :, 0], L[:, :, 1]).mean()
    te = np.where(U == 0, L[:, :, 1], L[:, :, 0]).mean()
    res.update(train01=float(tr), ghost01=float(te), gen=float(te - tr), k=len(runs))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", default="main", choices=["calib", "main"])
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--k1", type=int, default=2)
    ap.add_argument("--k2", type=int, default=40)
    ap.add_argument("--gd_lr", type=float, default=0.5)
    ap.add_argument("--gd_steps", type=int, default=2000)
    ap.add_argument("--calib", default="sgd:0.1:128:30,gd:0.05:4000,gd:0.2:2000,gd:0.5:2000,gd:1.0:1000")
    ap.add_argument("--conds", default="sgd,gd")
    ap.add_argument("--modes", default="fixed,random")
    ap.add_argument("--labels", default="true,random")
    ap.add_argument("--workers", type=int, default=34)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    import torch
    X, y, Xte, yte = data(a)
    perm = np.random.default_rng(123).permutation(len(X))
    if a.stage == "calib":
        Xtr, ytr = X[perm[:a.n]], y[perm[:a.n]]
        jobs = []
        for spec in a.calib.split(","):
            f = spec.split(":")
            cond = dict(opt="sgd", lr=float(f[1]), batch=int(f[2]), epochs=int(f[3])) if f[0] == "sgd" else \
                dict(opt="gd", lr=float(f[1]), steps=int(f[2]))
            jobs.append((a, spec, cond, Xtr, ytr, Xte, yte))
        out = []
        with mp.get_context("spawn").Pool(min(a.workers, len(jobs))) as pool:
            for r in pool.imap_unordered(_calib_job, jobs):
                out.append(r)
                print(r, flush=True)
        json.dump(out, open(a.out, "w"), indent=1)
        sys.exit()
    CONDS["gd"].update(lr=a.gd_lr, steps=a.gd_steps)
    jobs = [(a, cname, CONDS[cname], mode, lab, j, k) for j in range(a.k1) for cname in a.conds.split(",")
            for mode in a.modes.split(",") for lab in a.labels.split(",") for k in range(a.k2)]
    print(f"{len(jobs)} training runs", flush=True)
    groups = {}
    t0 = time.time()
    with mp.get_context("spawn").Pool(min(a.workers, len(jobs))) as pool:
        for i, r in enumerate(pool.imap_unordered(_main_job, jobs, chunksize=1)):
            groups.setdefault((r["cond"], r["seed_mode"], r["labels"], r["j"]), []).append(r)
            if i % 50 == 0:
                print(f"  {i}/{len(jobs)} [{time.time() - t0:.0f}s]", flush=True)
    out = []
    for key, rs in sorted(groups.items()):
        res = analyse(rs)
        res.update(cond=key[0], seed_mode=key[1], labels=key[2], j=key[3])
        out.append(res)
        print(res, flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
