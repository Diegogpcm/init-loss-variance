"""Trajectory-based certificates from ONE pass of SGD (online-to-batch / online PAC-Bayes).

Data are processed once, in a fixed random order; theta_{j-1} is the iterate before minibatch j.

(a) Progressive validation = OPBTest of Haddouche & Guedj (2022) with Q_i = P_i = delta(theta_{j(i)-1}).
    X_i = l01(theta_{j(i)-1}, z_i) is computed BEFORE the update that uses z_i. Given the past, E[X_i | F_{i-1}] is the
    population risk of that iterate, so for the predictor "pick an example index i uniformly in a window W and use the
    iterate that scored it" (risk mu_W = average of those conditional means):
        kl(mean_W X || mu_W) <= ln(1/delta_w) / |W|     (Chernoff-kl for [0,1] martingale differences: the conditional
                                                         log-MGF log(1 - mu + mu e^t) is concave in mu)
    Windows: last fraction f of the examples, f in F_GRID, union bound over the grid.

(b) OPBTrain (Haddouche & Guedj 2022, Cor. 3.1) with Gaussian posteriors around the trajectory:
    prior for every example of batch j: P = N(theta_{j-1}, sigma^2 I); posterior Q = N(theta_j, sigma^2 I).
    (1/n) sum_i E_Q[risk] <= (1/n) sum_i E_Q[l(h, z_i)] + (sum_i KL_i + ln(1/delta)) / (lambda n) + lambda / 2,
    sum_i KL_i = b * sum_j |theta_j - theta_{j-1}|^2 / (2 sigma^2) ("step energy").  The empirical term is estimated with
    one posterior sample per batch (block estimator, independent given the trajectory). Union bound over sigma x lambda.

Usage: python experiments/online_bounds.py --data fashion_norm.npz --out results/v2/online_bounds.jsonl
"""
import argparse, itertools, json, math, os, sys, time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pacbayes_norm import init_params, forward, kl_inv, load  # noqa: E402

F_GRID = [1.0, 0.5, 0.25, 0.1]
SIGMA_GRID = [0.003, 0.01, 0.03, 0.1, 0.3]
LAMBDA_GRID = list(np.logspace(-3, 0, 13))


def run(cfg):
    import torch
    torch.set_num_threads(cfg["threads"])
    t0 = time.time()
    Xtr, ytr, _, Xte, yte = load(cfg["data"])
    Xtr, Xte = Xtr.float(), Xte.float()
    norm, depth, width, lr, B = cfg["norm"], cfg["depth"], cfg["width"], cfg["lr"], cfg["batch"]
    n, C = len(Xtr), int(ytr.max()) + 1
    gen = torch.Generator().manual_seed(cfg["seed"])
    m0d, _ = init_params(norm, depth, width, Xtr.shape[1], C, gen)
    keys = list(m0d)
    shapes, sizes = [m0d[k].shape for k in keys], [m0d[k].numel() for k in keys]
    theta = torch.cat([m0d[k].reshape(-1) for k in keys]).float().requires_grad_(True)
    P = theta.numel()

    def unflat(v):
        out, i = {}, 0
        for k, sh, sz in zip(keys, shapes, sizes):
            out[k] = v[i:i + sz].view(sh); i += sz
        return out

    def err(v, X, y, bs=10000):
        with torch.no_grad():
            return sum(int((forward(unflat(v), X[i:i + bs], norm, depth).argmax(1) != y[i:i + bs]).sum())
                       for i in range(0, len(X), bs)) / len(X)

    opt = torch.optim.SGD([theta], lr=lr, momentum=0.9)
    perm = torch.randperm(n, generator=gen)
    J = math.ceil(n / B)
    # iterates whose test error we measure (chosen before training, data-independent)
    probe = sorted(set(int(j) for j in np.linspace(0, J - 1, 40)))
    pre_err = np.zeros(n)                    # prequential 0-1 loss of each example (before its update)
    batch_of = np.zeros(n, dtype=int)
    step_sq = np.zeros(J)                    # |theta_j - theta_{j-1}|^2
    noisy_post = {s: np.zeros(J) for s in SIGMA_GRID}   # one-sample estimate of E_{N(theta_j, s^2)} l01 on batch j
    probe_test = {}
    for j in range(J):
        idx = perm[j * B:(j + 1) * B]
        if j in probe:
            probe_test[j] = err(theta.detach(), Xte, yte)
        with torch.no_grad():
            pre_err[idx.numpy()] = (forward(unflat(theta.detach()), Xtr[idx], norm, depth).argmax(1) != ytr[idx]).float().numpy()
            batch_of[idx.numpy()] = j
        before = theta.detach().clone()
        out = forward(unflat(theta), Xtr[idx], norm, depth)
        loss = torch.nn.functional.cross_entropy(out, ytr[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            step_sq[j] = float((theta.detach() - before).pow(2).sum())
            for s in SIGMA_GRID:
                h = theta.detach() + s * torch.randn(P, generator=gen)
                noisy_post[s][j] = float((forward(unflat(h), Xtr[idx], norm, depth).argmax(1) != ytr[idx]).float().mean())
    final_test = err(theta.detach(), Xte, yte)

    delta = 0.025
    res = dict(norm=norm, depth=depth, width=width, lr=lr, batch=B, seed=cfg["seed"], n=n, n_params=P,
               final_iterate_test=final_test, prog=[], opb=[])
    # (a) progressive validation, union bound over windows
    order = perm.numpy()                     # processing order
    for f in F_GRID:
        W = order[int((1 - f) * n):]
        mean = float(pre_err[W].mean())
        cert = kl_inv(mean, math.log(len(F_GRID) / delta) / len(W))
        jW = sorted(set(batch_of[W]))
        pj = [j for j in probe if j >= jW[0]]
        res["prog"].append(dict(f=f, N=len(W), prequential=mean, cert01=cert,
                                window_test_probe=float(np.mean([probe_test[j] for j in pj])) if pj else None))
    # (b) OPBTrain with Gaussian noise, union bound over sigma x lambda (+ block estimator for the empirical term)
    nb = len(SIGMA_GRID) * len(LAMBDA_GRID)
    for s in SIGMA_GRID:
        energy = B * step_sq.sum() / (2 * s ** 2)
        emp = float(noisy_post[s].mean())
        emp_ub = kl_inv(emp, math.log(2 / 0.01) / J)            # estimation error of the noisy empirical term
        best = min(emp_ub + (energy + math.log(nb / delta)) / (lam * n) + lam / 2 for lam in LAMBDA_GRID)
        res["opb"].append(dict(sigma=s, energy=float(energy), energy_per_n=float(energy / n), noisy_post_err=emp,
                               cert01=float(min(best, 1.0))))
    res["seconds"] = time.time() - t0
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--norms", default="none,ln")
    ap.add_argument("--widths", default="64,128")
    ap.add_argument("--lrs", default="0.01,0.05")
    ap.add_argument("--batches", default="32,128")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    L = lambda x, f: [f(v) for v in x.split(",")]
    cfgs = [dict(data=a.data, depth=2, norm=nm, width=w, lr=lr, batch=B, seed=s, threads=a.threads)
            for nm in L(a.norms, str) for w in L(a.widths, int) for lr in L(a.lrs, float)
            for B in L(a.batches, int) for s in L(a.seeds, int)]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh, Pool(min(a.workers, len(cfgs))) as pool:
        for r in pool.imap_unordered(run, cfgs):
            fh.write(json.dumps(r) + "\n"); fh.flush()
            pv = min(r["prog"], key=lambda x: x["cert01"])
            ob = min(r["opb"], key=lambda x: x["cert01"])
            print(f"{r['norm']} w{r['width']} lr={r['lr']} B={r['batch']} s{r['seed']}: final test {r['final_iterate_test']:.4f} | "
                  f"prog.val. cert {pv['cert01']:.4f} (f={pv['f']}, prequential {pv['prequential']:.4f}, window test "
                  f"{pv['window_test_probe']:.4f}) | OPBTrain cert {ob['cert01']:.4f} (sigma={ob['sigma']}, "
                  f"energy/n {ob['energy_per_n']:.3f}) [{r['seconds']:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
