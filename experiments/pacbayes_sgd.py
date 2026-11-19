"""PAC-Bayes certificates with a posterior shaped by SGD noise.

1. SGD (momentum 0.9, constant lr) from the prior mean m0, optionally with an L2 pull to m0 (wd_init), the point
   estimate analogue of the KL mean term. For bn_stat (BatchNorm with statistics as parameters, see pacbayes_norm.py):
   batch statistics in the first half, then the statistics are frozen into parameters and SGD continues in inference
   mode, so the iterates include them. T iterates are collected over the last epochs.
2. Posterior in prior-whitened coordinates z = (theta - m0) / s (s = sigma0 for weights, 0.5 for BN statistics):
        Q = N(zbar, alpha * V diag(lam) V^T + tau^2 (I - V V^T))
   V, lam = top-K principal directions / variances of the SGD iterates. SGD's noise lives where the data constrain the
   weights; the complement gets an isotropic width tau (relative to the prior), which costs nothing at tau = 1.
        KL(Q || N(0, I)) = 1/2 [alpha sum lam + tau^2 (P-K) + |zbar|^2 - P - sum ln(alpha lam) - (P-K) ln tau^2]
3. sigma0 from a fixed data-free grid with a union bound (delta / |grid|). K, alpha, tau are posterior choices
   (allowed to depend on S): picked on a training subset with few samples; the final certificate estimates the Gibbs
   risk on all of S with the block estimator (fresh sample per block of b examples, N = passes * n / b independent
   terms, correction ln(2/delta')/N). The isotropic posterior (K = 0) is certified separately as a baseline.

Needs PyTorch.  Usage:
  python experiments/pacbayes_sgd.py --data fashion_norm.npz --out results/v2/pacbayes_sgd.jsonl
"""
import argparse, itertools, json, math, os, sys, time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pacbayes_norm import init_params, forward, bn_stats, kl_inv, load, gibbs_block, STAT_PRIOR_SD  # noqa: E402

S0_GRID = [0.01, 0.02, 0.05, 0.1, 0.2]
K_GRID = [0, 10, 40]
ALPHA_GRID = [0.25, 1, 4, 16, 64]
TAU_GRID = [0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0]


def run(cfg):
    import torch
    torch.set_num_threads(cfg["threads"])
    t0 = time.time()
    Xtr, ytr, _, Xte, yte = load(cfg["data"])
    Xtr, Xte = Xtr.float(), Xte.float()
    norm, depth, width, lr, B, wd = cfg["norm"], cfg["depth"], cfg["width"], cfg["lr"], cfg["batch"], cfg["wd_init"]
    n, C = len(Xtr), int(ytr.max()) + 1
    gen = torch.Generator().manual_seed(cfg["seed"])
    m0d, sd = init_params(norm, depth, width, Xtr.shape[1], C, gen)
    keys = list(m0d)
    shapes = [m0d[k].shape for k in keys]
    sizes = [m0d[k].numel() for k in keys]
    m0 = torch.cat([m0d[k].reshape(-1) for k in keys]).float()
    stat_mask = torch.cat([torch.full((m0d[k].numel(),), k in sd) for k in keys])
    P = m0.numel()

    def unflat(v):
        out, i = {}, 0
        for k, sh, sz in zip(keys, shapes, sizes):
            out[k] = v[i:i + sz].view(sh); i += sz
        return out

    # ---------------- 1. SGD with iterate collection
    theta = m0.clone().requires_grad_(True)
    opt = torch.optim.SGD([theta], lr=lr, momentum=0.9)
    E, E_col, T = cfg["epochs"], cfg["collect_epochs"], cfg["T"]
    EA = E // 2 if norm == "bn_stat" else 0
    steps_per_epoch = math.ceil(n / B)
    every = max(1, (E_col * steps_per_epoch) // T)
    snaps, step = [], 0
    for ep in range(E):
        if norm == "bn_stat" and ep == EA:
            with torch.no_grad():
                for k, v in bn_stats(unflat(theta.detach()), Xtr, depth).items():
                    unflat(theta.data)[k].copy_(v)
            opt = torch.optim.SGD([theta], lr=lr, momentum=0.9)
        train_bn = norm == "bn_stat" and ep < EA
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, B):
            idx = perm[i:i + B]
            out = forward(unflat(theta), Xtr[idx], norm, depth, None, train_bn=train_bn)
            loss = torch.nn.functional.cross_entropy(out, ytr[idx])
            if wd > 0:
                loss = loss + 0.5 * wd * (theta - m0).pow(2).sum()
            opt.zero_grad(); loss.backward(); opt.step()
            if ep >= E - E_col:
                step += 1
                if step % every == 0 and len(snaps) < T:
                    snaps.append(theta.detach().clone())
    Th = torch.stack(snaps)                                   # (T, P)
    th_bar = Th.mean(0)

    def err(v, X, y, bs=10000):
        wrong = 0
        with torch.no_grad():
            d = unflat(v)
            for i in range(0, len(X), bs):
                wrong += int((forward(d, X[i:i + bs], norm, depth).argmax(1) != y[i:i + bs]).sum())
        return wrong / len(X)

    res = dict(norm=norm, depth=depth, width=width, lr=lr, batch=B, wd_init=wd, seed=cfg["seed"], n=n, n_params=P,
               T=len(snaps), test_err_last=err(theta.detach(), Xte, yte), test_err_swa=err(th_bar, Xte, yte),
               train_err_swa=err(th_bar, Xtr, ytr), dist_init=float((th_bar - m0).norm()))

    # ---------------- 2./3. posterior family, selection, certification
    delta, delta_mc, M = 0.025, 0.01, cfg["M"]
    conf = math.log(2 * math.sqrt(n) * len(S0_GRID) / delta)     # union bound over the prior grid
    sel_idx = torch.randperm(n, generator=gen)[:cfg["sel_n"]]
    Xs, ys = Xtr[sel_idx], ytr[sel_idx]

    def setup(s0):
        svec = torch.where(stat_mask, torch.tensor(STAT_PRIOR_SD), torch.tensor(s0))
        Z = (Th - m0) / svec
        zbar = Z.mean(0)
        _, S, Vh = torch.linalg.svd(Z - zbar, full_matrices=False)
        lam = (S.double() ** 2) / (len(Z) - 1)
        return svec, zbar, Vh, lam

    def kl_terms(zbar, lam, K, alpha, tau):
        lamK = alpha * lam[:K]
        mean = 0.5 * float(zbar.double().pow(2).sum())
        sub = 0.5 * float((lamK - 1 - torch.log(lamK)).sum()) if K else 0.0
        comp = 0.5 * (P - K) * (tau ** 2 - 1 - math.log(tau ** 2))
        return mean + sub + comp, mean, sub, comp

    def sample(svec, zbar, Vh, lam, K, alpha, tau):
        e2 = torch.randn(P, generator=gen)
        z = zbar + tau * e2
        if K:
            VK = Vh[:K]
            z = z - tau * (VK.T @ (VK @ e2)) + VK.T @ (torch.sqrt(alpha * lam[:K]).float() * torch.randn(K, generator=gen))
        return m0 + svec * z

    cands = []
    for s0 in S0_GRID:
        svec, zbar, Vh, lam = setup(s0)
        for K in K_GRID:
            if K >= len(Th):                                    # centred iterates have rank T - 1
                continue
            for alpha, tau in itertools.product(ALPHA_GRID if K else [1.0], TAU_GRID):
                kl = kl_terms(zbar, lam, K, alpha, tau)
                if (kl[0] + conf) / n > 2.0:                    # hopeless, skip the sampling
                    continue
                e = float(np.mean([err(sample(svec, zbar, Vh, lam, K, alpha, tau), Xs, ys)
                                   for _ in range(cfg["M_sel"])]))
                cands.append(dict(s0=s0, K=K, alpha=alpha, tau=tau, KL=kl[0], est=kl_inv(e, (kl[0] + conf) / n),
                                  sel_err=e))

    def certify(c):
        svec, zbar, Vh, lam = setup(c["s0"])
        kl, mean, sub, comp = kl_terms(zbar, lam, c["K"], c["alpha"], c["tau"])
        smp = lambda: sample(svec, zbar, Vh, lam, c["K"], c["alpha"], c["tau"])

        def err_on(X, y):
            def f(v, idx):
                with torch.no_grad():
                    return float((forward(unflat(v), X[idx], norm, depth).argmax(1) != y[idx]).float().mean())
            return f
        emp, N = gibbs_block(smp, err_on(Xtr, ytr), n, cfg["block"], cfg["passes"], gen)
        te, _ = gibbs_block(smp, err_on(Xte, yte), len(Xte), cfg["block"], max(1, cfg["passes"] // 4), gen)
        emp_ub = kl_inv(emp, math.log(2 / delta_mc) / N)
        return dict(c, KL=kl, KL_mean=mean, KL_sub=sub, KL_comp=comp, train01=emp, train01_ub=emp_ub, N_mc=N,
                    test01=te, cert01=kl_inv(emp_ub, (kl + conf) / n))

    if cands:
        best = min(cands, key=lambda c: c["est"])
        best_iso = min((c for c in cands if c["K"] == 0), key=lambda c: c["est"], default=None)
        res["cert"] = certify(best)
        res["cert_iso"] = certify(best_iso) if best_iso else None
        res["best_est_by_K"] = {str(K): min((c["est"] for c in cands if c["K"] == K), default=None) for K in K_GRID}
        res["best_est_by_s0"] = {str(s): min((c["est"] for c in cands if c["s0"] == s), default=None) for s in S0_GRID}
    res["seconds"] = time.time() - t0
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--collect-epochs", type=int, default=10)
    ap.add_argument("--T", type=int, default=60)
    ap.add_argument("--M", type=int, default=150, help="(unused by the block estimator)")
    ap.add_argument("--block", type=int, default=40, help="block size of the Gibbs-risk estimator (must divide n)")
    ap.add_argument("--passes", type=int, default=20)
    ap.add_argument("--M-sel", type=int, default=4)
    ap.add_argument("--sel-n", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--quick", action="store_true", help="one tiny config, for testing")
    a = ap.parse_args()
    common = dict(data=a.data, depth=2, epochs=a.epochs, collect_epochs=a.collect_epochs, T=a.T, M=a.M,
                  M_sel=a.M_sel, sel_n=a.sel_n, threads=a.threads, seed=0, block=a.block, passes=a.passes)
    cfgs = []
    for norm in ("none", "bn_stat"):
        for width in (64, 128):                                  # main grid: L2-to-init strength
            for wd in (0.0, 1e-3, 1e-2):
                cfgs.append(dict(common, norm=norm, width=width, lr=0.05, batch=128, wd_init=wd))
        for lr in (0.01, 0.05, 0.2):                             # algorithm axis: SGD temperature lr / B
            for B in (32, 128, 512):
                if (lr, B) != (0.05, 128):
                    cfgs.append(dict(common, norm=norm, width=128, lr=lr, batch=B, wd_init=1e-3))
    if a.quick:
        cfgs = [dict(common, norm=nm, width=32, lr=0.05, batch=128, wd_init=1e-3) for nm in ("none", "bn_stat")]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh, Pool(min(a.workers, len(cfgs))) as pool:
        for r in pool.imap_unordered(run, cfgs):
            fh.write(json.dumps(r) + "\n"); fh.flush()
            c, ci = r.get("cert", {}), r.get("cert_iso") or {}
            print(f"{r['norm']:8s} w{r['width']} lr={r['lr']} B={r['batch']} wd={r['wd_init']}: SWA test {r['test_err_swa']:.4f} "
                  f"|d|={r['dist_init']:.1f} | cert {c.get('cert01', float('nan')):.4f} (s0={c.get('s0')} K={c.get('K')} "
                  f"a={c.get('alpha')} tau={c.get('tau')} KL={c.get('KL', float('nan')):.0f}) | iso {ci.get('cert01', float('nan')):.4f} "
                  f"[{r['seconds']:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
