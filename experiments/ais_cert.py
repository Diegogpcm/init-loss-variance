"""Data-tempered annealed importance sampling (AIS) as training, with a disintegrated PAC-Bayes certificate for the
final network whose complexity is the realised path log-ratio (theory.md s17).

Targets gamma_k(w) = P(w) exp(-lam L_k(w)), L_k = sum of cross-entropies over the first k data batches (fixed random
order). w_0 ~ P (data-free Gaussian prior). For k = 1..K: record lam * l_{B_k}(w_{k-1}) (online loss of the new batch
at the current sample), then apply S Metropolis-adjusted HMC moves, each reversible w.r.t. gamma_k.
Extended-space prior P~(w_{0:K}) = P(w_K) * prod_k Trev_k(w_k -> w_{k-1}); by detailed balance
    ln dQ~/dP~ (path) = lam * [ sum_k l_{B_k}(w_{k-1}) - L_K(w_K) ]          (=: C, computable on the realised run)
and since the w_K-marginal of P~ is the data-free P, the disintegrated PAC-Bayes-kl bound (Rivasplata et al. 2020)
gives, w.p. >= 1 - delta over S and the run:  kl(err_S(w_K) || err_D(w_K)) <= (C + ln(2 sqrt(n)/delta)) / n.
Kernel parameters (step size schedule, leapfrog count) may depend on S and k but not on the path itself.
"""
import argparse, json, math, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DELTA = 0.025


def kl_inv(p, c, tol=1e-12):
    if c <= 0:
        return p
    lo, hi = p, 1.0
    kl = lambda a, b: (a * math.log(a / b) if a > 0 else 0) + ((1 - a) * math.log((1 - a) / (1 - b)) if a < 1 else 0)
    while hi - lo > tol:
        m = (lo + hi) / 2
        if kl(p, m) > c:
            hi = m
        else:
            lo = m
    return hi


def main(a):
    import torch
    torch.set_num_threads(a.threads)
    dt = torch.float64
    d = np.load(a.data)
    X = torch.tensor(d["Xtr"], dtype=dt)
    y = torch.tensor(d["ytr"]).long()
    Xte = torch.tensor(d["Xte"], dtype=dt)
    yte = torch.tensor(d["yte"]).long()
    X, Xte = (X - 0.5) / 0.5, (Xte - 0.5) / 0.5          # fixed, data-free pixel scaling
    perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(a.order_seed))[:a.n]
    X, y = X[perm], y[perm]                               # the training sample S, in its (random) presentation order
    n = len(X)
    dims = [X.shape[1]] + [a.width] * a.depth + [10]
    shapes, pvar = [], []
    for l in range(len(dims) - 1):
        shapes += [(dims[l], dims[l + 1]), (dims[l + 1],)]
        pvar += [a.prior_scale / dims[l], a.bias_var]
    sizes = [math.prod(s) for s in shapes]
    pv = torch.cat([torch.full((s,), v, dtype=dt) for s, v in zip(sizes, pvar)])   # prior variances = inverse mass

    def unflat(w):
        out, o = [], 0
        for s, k in zip(shapes, sizes):
            out.append(w[o:o + k].view(s))
            o += k
        return out

    def logits(w, Xb):
        ps = unflat(w)
        h = Xb
        for l in range(len(ps) // 2):
            h = h @ ps[2 * l] + ps[2 * l + 1]
            if l < len(ps) // 2 - 1:
                h = torch.relu(h)
        return h

    def ce_sum(w, Xb, yb):
        return torch.nn.functional.cross_entropy(logits(w, Xb), yb, reduction="sum")

    def U_and_grad(w, k_end):
        w = w.detach().requires_grad_()
        u = 0.5 * (w * w / pv).sum() + a.lam * ce_sum(w, X[:k_end], y[:k_end])
        g, = torch.autograd.grad(u, w)
        return float(u), g

    def err(w, Xb, yb):
        with torch.no_grad():
            return float((logits(w, Xb).argmax(1) != yb).double().mean())

    g = torch.Generator().manual_seed(a.seed)
    w = torch.randn(pv.shape, generator=g, dtype=dt) * pv.sqrt()          # w_0 ~ P
    bounds = [0]                                          # stage boundaries: geometric from --m0, then steps of --m
    while bounds[-1] < n:
        bounds.append(min(n, bounds[-1] + (min(a.m, max(a.m0, bounds[-1])) if a.m0 else a.m)))
    K = len(bounds) - 1
    online_ce = 0.0
    online_err = 0
    log = []
    t0 = time.time()
    for k in range(1, K + 1):
        lo, hi = bounds[k - 1], bounds[k]
        with torch.no_grad():
            ce_b = float(ce_sum(w, X[lo:hi], y[lo:hi]))
            online_ce += ce_b
            online_err += int((logits(w, X[lo:hi]).argmax(1) != y[lo:hi]).sum())
        eps = a.c / math.sqrt(1 + a.lam * hi)               # fixed schedule in k (path-independent)
        u, gr = U_and_grad(w, hi)
        acc = 0
        for s in range(a.S):
            p0 = torch.randn(pv.shape, generator=g, dtype=dt) / pv.sqrt()    # momentum ~ N(0, M), M = diag(1/pv)
            wn, p, gn = w.clone(), p0 - 0.5 * eps * gr, gr
            for j in range(a.L):
                wn = wn + eps * pv * p
                un, gn = U_and_grad(wn, hi)
                if j < a.L - 1:
                    p = p - eps * gn
            p = p - 0.5 * eps * gn
            H0 = u + 0.5 * float((p0 * p0 * pv).sum())
            H1 = un + 0.5 * float((p * p * pv).sum())
            if math.log(float(torch.rand(1, generator=g, dtype=dt))) < H0 - H1:
                w, u, gr = wn, un, gn
                acc += 1
        rec = dict(k=k, n_k=hi, eps=eps, acc=acc / a.S, online_ce_batch=ce_b / (hi - lo), seconds=time.time() - t0)
        if k % a.log_every == 0 or k == K:
            rec.update(train_err=err(w, X[:hi], y[:hi]), test_err=err(w, Xte, yte),
                       online_err_cum=online_err / hi, online_ce_cum=online_ce / hi)
            print(rec, flush=True)
        log.append(rec)
    with torch.no_grad():
        final_ce = float(ce_sum(w, X, y))
    C = a.lam * (online_ce - final_ce)
    tr, te = err(w, X, y), err(w, Xte, yte)
    cert = kl_inv(tr, (C + math.log(2 * math.sqrt(n) / DELTA)) / n)
    res = dict(cfg=vars(a), n=n, n_params=int(pv.numel()), train01=tr, test01=te, online_ce=online_ce / n,
               online_err=online_err / n, final_train_ce=final_ce / n, C=C, C_over_n=C / n, cert=cert,
               prog_val_cert=kl_inv(online_err / n, math.log(1 / DELTA) / n),
               mean_acc=float(np.mean([r["acc"] for r in log])), seconds=time.time() - t0, log=log)
    print({k: v for k, v in res.items() if k not in ("log", "cfg")}, flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--width", type=int, default=100)
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--prior_scale", type=float, default=1.0)
    ap.add_argument("--bias_var", type=float, default=0.1)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--m", type=int, default=1000)
    ap.add_argument("--m0", type=int, default=0, help="if >0: stage sizes m0, m0, 2m0, 4m0, ... capped at m")
    ap.add_argument("--S", type=int, default=4)
    ap.add_argument("--L", type=int, default=30)
    ap.add_argument("--c", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--order_seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--log_every", type=int, default=1)
    main(ap.parse_args())
