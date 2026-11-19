"""Parameter-space vs function-space KL for PAC-Bayes-trained Gaussian posteriors on small MLPs.

Prior     P = N(m0, s0^2 I), m0 ~ N(0, 1/fan_in) drawn once (data-free), as in Perez-Ortiz et al. / Garcia-Perez et al.
Posterior Q = N(mu, diag(sq^2)), trained on  E_Q[bounded CE] + sqrt((KL + ln(2 sqrt(n)/delta)) / 2n).
Certificate on the 0-1 loss via kl-inversion of the PAC-Bayes-kl bound.

Function-space KL is measured on the logits at m training inputs, after projecting out the softmax shift direction
(C-1 dims per input). By data processing,  KL_m <= KL_function(all inputs) <= KL_param,  so KL_m is a LOWER bound on
the function-space KL and KL_param - KL_m is an upper bound on what function space could save.
  linear: shared Jacobian J at mu, P_f = N(0, J S0 J^T), Q_f = N(J(mu - m0), J SQ J^T). Exact push-forward of a
          linear map, so KL_lin <= KL_param holds exactly; isolates what m inputs can "see" of the parameter KL.
          (Linearizing P and Q at their own means violates data processing: the ill-conditioned prior covariance
          amplifies small Jacobian differences.)
  mc:     Gaussian fit to logits sampled from P and Q, no linearization (captures nonlinearity; m <= mc_m).
          Keep m*(C-1) small: the log-det bias of a sample covariance is ~ d^2/(2N) nats.
Note: on sklearn digits (n=1347) the bound-optimal posterior is the prior itself (any useful posterior has KL/n > 0.1),
so use a large n (Fashion-MNIST 60k) or KL modulation (--etas < 1; the certificate always uses the true KL).

Needs PyTorch (CPU is fine).  Usage:
  python experiments/function_space_kl.py --data digits.npz --out results/v2/function_space_kl.jsonl
where digits.npz holds Xtr, ytr, Xte, yte (e.g. dumped from lvproxy.data.digits()).
"""
import argparse, json, math, os
from multiprocessing import Pool

import numpy as np

ACTS = None  # filled in worker (torch imported lazily so --help works without torch)


def build(depth, width, d_in, C):
    shapes = [(d_in, width)] + [(width, width)] * depth + [(width, C)]
    return shapes


def unflatten(theta, shapes):
    out, i = [], 0
    for fi, fo in shapes:
        W = theta[i:i + fi * fo].view(fi, fo); i += fi * fo
        b = theta[i:i + fo]; i += fo
        out.append((W, b))
    return out


def forward(theta, X, shapes, act):
    import torch
    h = X
    layers = unflatten(theta, shapes)
    for j, (W, b) in enumerate(layers):
        h = h @ W + b
        if j < len(layers) - 1:
            h = torch.relu(h) if act == "relu" else torch.tanh(h)
    return h


def helmert(C):
    """C x (C-1) orthonormal basis of the subspace orthogonal to the all-ones vector."""
    H = np.zeros((C, C - 1))
    for k in range(1, C):
        H[:k, k - 1] = 1.0 / math.sqrt(k * (k + 1))
        H[k, k - 1] = -k / math.sqrt(k * (k + 1))
    return H


def kl_inv(p, c):
    lo, hi = p, 1.0
    for _ in range(100):
        q = (lo + hi) / 2
        kl = (p * math.log(max(p, 1e-12) / q) if p > 0 else 0) + (1 - p) * math.log((1 - p) / (1 - q))
        lo, hi = (q, hi) if kl <= c else (lo, q)
    return lo


def gauss_kl(mq, Sq, mp, Sp):
    """KL(N(mq,Sq) || N(mp,Sp)) with a tiny relative jitter; returns (total, mean_term, cov_terms)."""
    d = len(mq)
    jit = 1e-9 * (np.trace(Sp) + np.trace(Sq)) / (2 * d)
    Sp = Sp + jit * np.eye(d); Sq = Sq + jit * np.eye(d)
    Lp = np.linalg.cholesky(Sp)
    Lq = np.linalg.cholesky(Sq)
    A = np.linalg.solve(Lp, Lq)
    dm = np.linalg.solve(Lp, mq - mp)
    tr = (A ** 2).sum()
    logdet = 2 * (np.log(np.diag(Lp)).sum() - np.log(np.diag(Lq)).sum())
    mean_term = 0.5 * (dm ** 2).sum()
    cov_term = 0.5 * (tr - d + logdet)
    return mean_term + cov_term, mean_term, cov_term


def run(cfg):
    import torch
    torch.set_num_threads(2)
    depth, width, act, s0, seed, data, ms, mc_m, epochs = (cfg[k] for k in
        ("depth", "width", "act", "s0", "seed", "data", "ms", "mc_m", "epochs"))
    d = np.load(data)
    Xtr, ytr, Xte, yte = (torch.tensor(d[k]) for k in ("Xtr", "ytr", "Xte", "yte"))
    Xtr, Xte = Xtr.double(), Xte.double()
    n, d_in = Xtr.shape
    C = int(ytr.max()) + 1
    shapes = build(depth, width, d_in, C)
    P = sum(fi * fo + fo for fi, fo in shapes)
    g = torch.Generator().manual_seed(seed)
    m0 = torch.cat([torch.cat([torch.randn(fi * fo, generator=g, dtype=torch.float64) / math.sqrt(fi),
                               torch.zeros(fo, dtype=torch.float64)]) for fi, fo in shapes])
    delta, pmin = 0.025, 1e-4
    conf = math.log(2 * math.sqrt(n) / delta)

    mu = m0.clone().requires_grad_(True)
    rho = torch.full((P,), math.log(math.expm1(s0)), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([mu, rho], lr=cfg["lr"])

    def kl_param(mu, sq):
        return 0.5 * (((sq / s0) ** 2).sum() + ((mu - m0) / s0).pow(2).sum() - P - 2 * torch.log(sq / s0).sum())

    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            sq = torch.nn.functional.softplus(rho)
            theta = mu + sq * torch.randn(P, generator=g, dtype=torch.float64)
            p = torch.softmax(forward(theta, Xtr[idx], shapes, act), 1).gather(1, ytr[idx, None]).squeeze(1)
            bce = (-torch.log(p.clamp_min(pmin)) / math.log(1 / pmin)).mean()
            obj = bce + torch.sqrt((cfg["eta"] * kl_param(mu, sq) + conf) / (2 * n))   # eta: KL modulation
            opt.zero_grad(); obj.backward(); opt.step()

    with torch.no_grad():
        sq = torch.nn.functional.softplus(rho)
        KLp = float(kl_param(mu, sq))
        KLp_mean = float(0.5 * ((mu - m0) / s0).pow(2).sum())
        errs_tr, errs_te = [], []
        for _ in range(200):
            theta = mu + sq * torch.randn(P, generator=g, dtype=torch.float64)
            errs_tr.append(float((forward(theta, Xtr, shapes, act).argmax(1) != ytr).double().mean()))
            errs_te.append(float((forward(theta, Xte, shapes, act).argmax(1) != yte).double().mean()))
        emp, te = float(np.mean(errs_tr)), float(np.mean(errs_te))
        det_te = float((forward(mu, Xte, shapes, act).argmax(1) != yte).double().mean())
    cert = kl_inv(emp, (KLp + conf) / n)

    H = torch.tensor(helmert(C))
    out = dict(depth=depth, width=width, act=act, s0=s0, seed=seed, eta=cfg["eta"], n=n, n_params=P, KL_param=KLp,
               KL_param_mean_term=KLp_mean, mean_sq_over_s0=float((sq / s0).mean()),
               train01_Q=emp, test01_Q=te, test01_mean=det_te, cert01=cert, fn=[])

    mu_d, sq_d = mu.detach(), sq.detach()
    for m in ms:
        if m * (C - 1) >= 0.9 * P or m * (C - 1) * P > 4e7:   # rank limit / Jacobian memory cap
            break
        Xm = Xtr[:m]
        f = lambda th: (forward(th, Xm, shapes, act) @ H).reshape(-1)
        # shared-Jacobian linear model (exact push-forward of a linear map, so data processing holds exactly)
        J = torch.func.jacrev(f)(mu_d)
        Sp = ((J * s0 ** 2) @ J.T).numpy()
        Sq = ((J * sq_d ** 2) @ J.T).numpy()
        kl, km, kc = gauss_kl((J @ (mu_d - m0)).numpy(), Sq, np.zeros(J.shape[0]), Sp)
        rec = dict(m=m, dim=m * (C - 1), KL_lin=kl, KL_lin_mean=km, KL_lin_cov=kc)
        if m <= mc_m:  # Gaussian fit to sampled logits: no linearization
            with torch.no_grad():
                N = 30000
                zs = []
                for mean, sd in ((m0, s0), (mu_d, sq_d)):
                    Z = torch.stack([f(mean + sd * torch.randn(P, generator=g, dtype=torch.float64))
                                     for _ in range(N)]).numpy()
                    zs.append((Z.mean(0), np.cov(Z, rowvar=False)))
                kl, km, kc = gauss_kl(zs[1][0], zs[1][1], zs[0][0], zs[0][1])
                rec.update(KL_mc=kl, KL_mc_mean=km, KL_mc_cov=kc)
        out["fn"].append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--depths", default="1,2")
    ap.add_argument("--widths", default="32,64")
    ap.add_argument("--s0s", default="0.02,0.05")
    ap.add_argument("--etas", default="1")
    ap.add_argument("--mc-m", type=int, default=10)
    a = ap.parse_args()
    L = lambda x, t: [t(v) for v in x.split(",")]
    cfgs = [dict(depth=dp, width=w, act=act, s0=s0, eta=eta, seed=0, data=a.data, epochs=a.epochs, lr=a.lr,
                 ms=[1, 2, 5, 10, 20, 40, 80, 160], mc_m=a.mc_m)
            for dp in L(a.depths, int) for w in L(a.widths, int) for act in ("relu", "tanh")
            for s0 in L(a.s0s, float) for eta in L(a.etas, float)]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh, Pool(a.workers) as pool:
        for r in pool.imap_unordered(run, cfgs):
            fh.write(json.dumps(r) + "\n"); fh.flush()
            last = r["fn"][-1] if r["fn"] else {}
            print(f"d{r['depth']} w{r['width']} {r['act']} s0={r['s0']} eta={r['eta']}: P={r['n_params']} KLp={r['KL_param']:.0f} "
                  f"cert={r['cert01']:.3f} test={r['test01_Q']:.3f} | KL_m(m={last.get('m')})={last.get('KL_lin', float('nan')):.0f}",
                  flush=True)


if __name__ == "__main__":
    main()
