"""PAC-Bayes certificates for MLPs with LayerNorm / BatchNorm, and invariance-aware KL.

Hypothesis classes (all predict each example independently, so the usual i.i.d. PAC-Bayes-kl bound applies):
  none      plain ReLU MLP
  ln        LayerNorm after each hidden linear layer (per-example, no i.i.d. issue)
  bn_ref    BatchNorm whose statistics are computed, for every sampled weight vector, on a fixed reference set R of
            UNLABELED inputs disjoint from the training sample S ("virtual batch norm", Salimans et al. 2016).
            The map weights -> classifier does not depend on S, so a data-free prior over weights is a valid prior.
  bn_param  standard BatchNorm (batch statistics during training); the inference-mode statistics (mean, log-std
            per unit) are treated as extra parameters with a data-free prior N(0, 0.5^2) and a Gaussian posterior
            centred at the statistics of the posterior-mean network on S (naive baseline: stats frozen, sd 0.1).
  bn_stat   statistics as parameters, done properly: phase 1 trains with batch statistics; then the statistics are
            frozen into posterior parameters (trainable mean AND sd) and phase 2 fine-tunes everything in inference
            mode with the PAC-Bayes objective, so the statistics see the weight noise. Inference-mode BN
            (x.W_j - m_j)/s_j is EXACTLY invariant to scaling (W_j, m_j, s_j) by c > 0 (no eps involved).

Invariance-aware KL. If a rescaling of the weights leaves every classifier unchanged, the posterior Q and its
rescaled copy Q_t define the same Gibbs classifier, so the certificate may use min_t KL(Q_t || P):
  none: per-unit ReLU homogeneity (incoming column + bias by e^v, outgoing row by e^-v)
  ln:   per-layer scale of (W, b) feeding the LN, plus per-unit ReLU scaling of (gamma, beta) / outgoing row
  bn:   per-unit scale of the incoming column feeding the BN, plus per-unit ReLU scaling of (gamma, beta) / outgoing row
  bn_stat: per-unit joint scale of (incoming column, mean m_j) and additive shift of log s_j, plus the ReLU scaling
Because LN/BN carry an eps, the invariance is only approximate, so the empirical risk is always evaluated by
sampling from Q_t itself: every reported certificate is rigorous for the classifier it describes.
  mode plain: train with t = 0; report the certificate at t = 0 and after post-hoc LBFGS over t
  mode inv:   t is trained jointly with (mu, sigma); only the KL depends on it

Certificate: PAC-Bayes-kl (Maurer) with delta = 0.025. The Gibbs empirical risk is estimated with the BLOCK estimator
(gibbs_block): S is split into blocks of b examples (b divides n) and every block gets its own fresh posterior sample.
Block error rates are independent [0,1] variables given S whose means average exactly to the Gibbs risk, so the
kl-Chernoff bound (Hoeffding 1963, independent non-identical variables) gives kl(est || Gibbs) <= ln(2/delta')/N
with N = passes * n / b, i.e. tens of thousands of terms for the compute of ~20 full evaluations
(the usual estimator, one sample per full pass, has N = number of samples; delta' = 0.01).

Needs PyTorch.  Usage:
  python experiments/pacbayes_norm.py --data fashion_norm.npz --out results/v2/pacbayes_norm.jsonl
  python experiments/pacbayes_norm.py --data fashion_norm.npz --out results/v2/norm_deterministic.jsonl --deterministic
"""
import argparse, json, math, os, time
from multiprocessing import Pool

import numpy as np

EPS = 1e-5
STAT_PRIOR_SD, STAT_POST_SD = 0.5, 0.1


def kl_inv(p, c):
    if c <= 0:
        return p
    lo, hi = p, 1.0
    for _ in range(80):
        q = (lo + hi) / 2
        kl = (p * math.log(p / q) if p > 0 else 0.0) + (1 - p) * math.log((1 - p) / (1 - q))
        lo, hi = (q, hi) if kl <= c else (lo, q)
    return hi


# ------------------------------------------------------------------ parameters
def init_params(norm, depth, width, d_in, C, gen):
    import torch
    m0, sd = {}, {}
    fi = d_in
    for l in range(depth):
        m0[f"W{l}"] = torch.randn(fi, width, generator=gen, dtype=torch.float64) / math.sqrt(fi)
        if norm in ("none", "ln"):
            m0[f"b{l}"] = torch.zeros(width, dtype=torch.float64)
        if norm != "none":
            m0[f"g{l}"] = torch.ones(width, dtype=torch.float64)
            m0[f"be{l}"] = torch.zeros(width, dtype=torch.float64)
        if norm in ("bn_param", "bn_stat"):
            m0[f"mean{l}"] = torch.zeros(width, dtype=torch.float64)
            m0[f"logs{l}"] = torch.zeros(width, dtype=torch.float64)
            sd[f"mean{l}"] = sd[f"logs{l}"] = STAT_PRIOR_SD
        fi = width
    m0["Wout"] = torch.randn(width, C, generator=gen, dtype=torch.float64) / math.sqrt(width)
    m0["bout"] = torch.zeros(C, dtype=torch.float64)
    return m0, sd


def scale_tensors(norm, depth, width):
    """Shapes of the log-scale variables t for the invariances of each hypothesis class."""
    if norm == "none":
        return {f"v{l}": (width,) for l in range(depth)}
    if norm == "ln":
        return {**{f"s{l}": (1,) for l in range(depth)}, **{f"u{l}": (width,) for l in range(depth)}}
    if norm in ("bn_ref", "bn_stat"):
        return {**{f"v{l}": (width,) for l in range(depth)}, **{f"u{l}": (width,) for l in range(depth)}}
    return {}


def log_scales(norm, depth, t):
    """Per-parameter log-scale pattern (broadcastable) implied by t."""
    pat = {}
    if not t:
        return pat
    for l in range(depth):
        prev = None
        if l > 0:
            prev = t[f"v{l-1}"] if norm == "none" else t[f"u{l-1}"]
        if norm == "none":
            pat[f"W{l}"] = t[f"v{l}"][None, :] - (prev[:, None] if prev is not None else 0)
            pat[f"b{l}"] = t[f"v{l}"]
        elif norm == "ln":
            pat[f"W{l}"] = t[f"s{l}"] - (prev[:, None] if prev is not None else 0)
            pat[f"b{l}"] = t[f"s{l}"]
        else:  # bn_ref, bn_stat
            pat[f"W{l}"] = t[f"v{l}"][None, :] - (prev[:, None] if prev is not None else 0)
            if norm == "bn_stat":
                pat[f"mean{l}"] = t[f"v{l}"]
                pat[f"logs{l}+"] = t[f"v{l}"]          # additive shift of log s
        if norm != "none":
            pat[f"g{l}"] = t[f"u{l}"]
            pat[f"be{l}"] = t[f"u{l}"]
    last = t[f"v{depth-1}"] if norm == "none" else t[f"u{depth-1}"]
    pat["Wout"] = -last[:, None]
    return pat


def effective(mu, sq, pat):
    import torch
    mu_e, sq_e = {}, {}
    for k in mu:
        e = torch.exp(pat[k]) if k in pat else 1.0
        mu_e[k], sq_e[k] = mu[k] * e, sq[k] * e
        if k + "+" in pat:                              # additive (log-scale) parameters
            mu_e[k] = mu_e[k] + pat[k + "+"]
    return mu_e, sq_e


def kl_gauss(mu, sq, m0, s0, sd):
    tot = 0.0
    for k in mu:
        p = sd.get(k, s0)
        tot = tot + 0.5 * ((sq[k] / p) ** 2 + ((mu[k] - m0[k]) / p) ** 2 - 1 - 2 * (sq[k] / p).log()).sum()
    return tot


# ------------------------------------------------------------------ forward
def forward(th, X, norm, depth, R=None, train_bn=False):
    import torch
    h, hr = X, R
    for l in range(depth):
        z = h @ th[f"W{l}"]
        if f"b{l}" in th:
            z = z + th[f"b{l}"]
        if norm == "ln":
            z = (z - z.mean(1, keepdim=True)) / torch.sqrt(z.var(1, unbiased=False, keepdim=True) + EPS)
        elif norm == "bn_ref":
            zr = hr @ th[f"W{l}"]
            m, v = zr.mean(0), zr.var(0, unbiased=False)
            z = (z - m) / torch.sqrt(v + EPS)
            zr = (zr - m) / torch.sqrt(v + EPS)
            hr = torch.relu(zr * th[f"g{l}"] + th[f"be{l}"])
        elif norm in ("bn_param", "bn_stat"):
            if train_bn:
                m, v = z.mean(0), z.var(0, unbiased=False)
                z = (z - m) / torch.sqrt(v + EPS)
            else:
                z = (z - th[f"mean{l}"]) / torch.exp(th[f"logs{l}"])
        if norm != "none":
            z = z * th[f"g{l}"] + th[f"be{l}"]
        h = torch.relu(z)
    return h @ th["Wout"] + th["bout"]


def batched_err(th, X, y, norm, depth, R, bs=8192):
    import torch
    wrong = 0
    for i in range(0, len(X), bs):
        wrong += int((forward(th, X[i:i + bs], norm, depth, R).argmax(1) != y[i:i + bs]).sum())
    return wrong / len(X)


def gibbs_block(sample_fn, err_fn, n, b, passes, gen):
    """Block estimator of the Gibbs error E_theta (1/n) sum_i loss(theta, z_i).
    sample_fn() -> fresh posterior sample; err_fn(theta, idx) -> error rate of theta on examples idx.
    Returns (estimate, N independent terms)."""
    import torch
    assert n % b == 0, f"block size {b} must divide n = {n} so that the blocks partition S"
    errs = []
    for _ in range(passes):
        perm = torch.randperm(n, generator=gen)
        for j in range(n // b):
            errs.append(err_fn(sample_fn(), perm[j * b:(j + 1) * b]))
    return float(np.mean(errs)), len(errs)


def bn_stats(th, X, depth, bs=8192):
    """Per-unit pre-BN mean and std of the given (deterministic) network on X, layer by layer."""
    import torch
    stats, h = {}, X
    for l in range(depth):
        z = h @ th[f"W{l}"]
        m, v = z.mean(0), z.var(0, unbiased=False)
        stats[f"mean{l}"], stats[f"logs{l}"] = m, 0.5 * torch.log(v + EPS)
        h = torch.relu((z - m) / torch.sqrt(v + EPS) * th[f"g{l}"] + th[f"be{l}"])
    return stats


# ------------------------------------------------------------------ runs
def load(path):
    import torch
    d = np.load(path)
    return (torch.tensor(d["Xtr"]).double(), torch.tensor(d["ytr"]), torch.tensor(d["Xref"]).double(),
            torch.tensor(d["Xte"]).double(), torch.tensor(d["yte"]))


def run_det(cfg):
    """Ordinary (non-PAC-Bayes) training, to compare normalizations in practice."""
    import torch
    torch.set_num_threads(cfg["threads"])
    Xtr, ytr, R, Xte, yte = load(cfg["data"])
    norm, depth, width = cfg["norm"], cfg["depth"], cfg["width"]
    gen = torch.Generator().manual_seed(cfg["seed"])
    m0, _ = init_params("bn_param" if norm == "bn" else norm, depth, width, Xtr.shape[1], 10, gen)
    th = {k: v.clone().requires_grad_(not k.startswith(("mean", "logs"))) for k, v in m0.items()}
    hnorm = "bn_param" if norm == "bn" else norm
    opt = torch.optim.Adam([v for v in th.values() if v.requires_grad], lr=1e-3)
    n = len(Xtr)
    for ep in range(cfg["epochs"]):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            ridx = torch.randint(0, len(R), (256,), generator=gen)
            out = forward(th, Xtr[idx], hnorm, depth, R[ridx], train_bn=True)
            loss = torch.nn.functional.cross_entropy(out, ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        if norm == "bn":
            th.update(bn_stats(th, Xtr, depth))
        te = batched_err(th, Xte, yte, hnorm, depth, R)
        tr = batched_err(th, Xtr, ytr, hnorm, depth, R)
    return dict(kind="deterministic", norm=norm, depth=depth, width=width, seed=cfg["seed"], train_err=tr, test_err=te)


def run_pb(cfg):
    import torch
    torch.set_num_threads(cfg["threads"])
    t0 = time.time()
    Xtr, ytr, R, Xte, yte = load(cfg["data"])
    norm, depth, width, s0, mode = cfg["norm"], cfg["depth"], cfg["width"], cfg["s0"], cfg["mode"]
    n, C = len(Xtr), int(ytr.max()) + 1
    gen = torch.Generator().manual_seed(cfg["seed"])
    m0, sd = init_params(norm, depth, width, Xtr.shape[1], C, gen)
    delta, delta_mc, pmin, M = 0.025, 0.01, 1e-4, cfg["M"]
    conf = math.log(2 * math.sqrt(n) / delta)

    stat_trainable = norm == "bn_stat"
    mu = {k: v.clone().requires_grad_(stat_trainable or not k.startswith(("mean", "logs"))) for k, v in m0.items()}
    rho = {k: torch.full_like(v, math.log(math.expm1(sd.get(k, s0) * (STAT_POST_SD / STAT_PRIOR_SD
                                                                       if k in sd else 1.0))),
                              requires_grad=stat_trainable or k not in sd) for k, v in m0.items()}
    t = {k: torch.zeros(s, dtype=torch.float64, requires_grad=(mode == "inv"))
         for k, s in scale_tensors(norm, depth, width).items()}
    params = [v for v in mu.values() if v.requires_grad] + [v for v in rho.values() if v.requires_grad]
    if mode == "inv":
        params += list(t.values())
    opt = torch.optim.Adam(params, lr=cfg["lr"])
    sp = torch.nn.functional.softplus

    def sample(mu_e, sq_e):
        return {k: mu_e[k] + sq_e[k] * torch.randn(mu_e[k].shape, generator=gen, dtype=torch.float64) for k in mu_e}

    phase1 = cfg["epochs"] // 2 if norm == "bn_stat" else cfg["epochs"]
    for ep in range(cfg["epochs"]):
        if norm == "bn_stat" and ep == phase1:     # freeze batch statistics into posterior parameters
            with torch.no_grad():
                sq_now = {k: sp(v) for k, v in rho.items()}
                mu_e0, _ = effective(mu, sq_now, log_scales(norm, depth, t))
                pat = log_scales(norm, depth, t)
                for k, v in bn_stats(mu_e0, Xtr, depth).items():   # store raw values: undo the current rescaling
                    raw = v - pat[k + "+"] if k + "+" in pat else v / torch.exp(pat[k]) if k in pat else v
                    mu[k].copy_(raw)
        train_bn = ep < phase1 if norm == "bn_stat" else True
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            sq = {k: sp(v) for k, v in rho.items()}
            mu_e, sq_e = effective(mu, sq, log_scales(norm, depth, t))
            th = sample(mu_e, sq_e)
            ridx = torch.randint(0, len(R), (256,), generator=gen)
            out = forward(th, Xtr[idx], norm, depth, R[ridx], train_bn=train_bn)
            p = torch.softmax(out, 1).gather(1, ytr[idx, None]).squeeze(1)
            bce = (-torch.log(p.clamp_min(pmin)) / math.log(1 / pmin)).mean()
            kl = kl_gauss(mu_e, sq_e, m0, s0, sd)
            obj = bce + torch.sqrt((kl + conf) / (2 * n))
            opt.zero_grad(); obj.backward(); opt.step()

    with torch.no_grad():
        sq = {k: sp(v) for k, v in rho.items()}
        mu_d = {k: v.detach().clone() for k, v in mu.items()}
        if norm == "bn_param":  # inference statistics of the posterior-mean network on S become posterior means
            mu_e0, _ = effective(mu_d, sq, log_scales(norm, depth, t))
            mu_d.update(bn_stats(mu_e0, Xtr, depth))
        t_trained = {k: v.detach().clone() for k, v in t.items()}

    def evaluate(tt):
        with torch.no_grad():
            mu_e, sq_e = effective(mu_d, sq, log_scales(norm, depth, tt))
            kl = float(kl_gauss(mu_e, sq_e, m0, s0, sd))
            b = cfg["block_ref"] if norm == "bn_ref" else cfg["block"]
            smp = lambda: sample(mu_e, sq_e)
            err_tr = lambda th, idx: float((forward(th, Xtr[idx], norm, depth, R).argmax(1) != ytr[idx]).double().mean())
            err_te = lambda th, idx: float((forward(th, Xte[idx], norm, depth, R).argmax(1) != yte[idx]).double().mean())
            emp, N = gibbs_block(smp, err_tr, n, b, cfg["passes"], gen)
            te, _ = gibbs_block(smp, err_te, len(Xte), b, max(1, cfg["passes"] // 4), gen)
            emp_ub = kl_inv(emp, math.log(2 / delta_mc) / N)
            return dict(KL=kl, KL_per_n=kl / n, train01=emp, train01_ub=emp_ub, N_mc=N, block=b, test01=te,
                        cert01=kl_inv(emp_ub, (kl + conf) / n))

    res = dict(kind="pacbayes", norm=norm, mode=mode, depth=depth, width=width, s0=s0, seed=cfg["seed"], n=n,
               n_params=int(sum(v.numel() for v in m0.values())))
    res["trained"] = evaluate(t_trained)
    if t:  # post-hoc: minimize the KL over the invariance group (mu, sigma fixed)
        tt = {k: v.clone().requires_grad_(True) for k, v in t_trained.items()}
        lb = torch.optim.LBFGS(list(tt.values()), max_iter=200, line_search_fn="strong_wolfe")

        def closure():
            lb.zero_grad()
            mu_e, sq_e = effective(mu_d, sq, log_scales(norm, depth, tt))
            k = kl_gauss(mu_e, sq_e, m0, s0, sd)
            k.backward()
            return k
        lb.step(closure)
        res["posthoc"] = evaluate({k: v.detach() for k, v in tt.items()})
    if norm in ("bn_param", "bn_stat"):
        mu_e, sq_e = effective(mu_d, sq, log_scales(norm, depth, t_trained))   # stats KL of the rescaled posterior
        res["KL_stats"] = float(kl_gauss({k: mu_e[k] for k in sd}, {k: sq_e[k] for k in sd},
                                         {k: m0[k] for k in sd}, s0, sd))
    res["seconds"] = time.time() - t0
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--M", type=int, default=150, help="(unused by the block estimator; kept for compatibility)")
    ap.add_argument("--block", type=int, default=40, help="block size of the Gibbs-risk estimator (must divide n)")
    ap.add_argument("--block-ref", type=int, default=250, help="block size for bn_ref (each sample needs R)")
    ap.add_argument("--passes", type=int, default=20)
    ap.add_argument("--widths", default="64,128")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--s0s", default="0.02,0.05")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--norms", default="none,ln,bn_ref,bn_param")
    ap.add_argument("--modes", default="plain,inv")
    a = ap.parse_args()
    L = lambda x, f: [f(v) for v in x.split(",")]
    common = dict(data=a.data, depth=a.depth, epochs=a.epochs, lr=a.lr, M=a.M, threads=a.threads, block=a.block,
                  block_ref=a.block_ref, passes=a.passes)
    if a.deterministic:
        fn = run_det
        cfgs = [dict(common, norm=nm, width=w, seed=s) for nm in ("none", "ln", "bn", "bn_ref")
                for w in L(a.widths, int) for s in L(a.seeds, int)]
    else:
        fn = run_pb
        cfgs = [dict(common, norm=nm, mode=md, width=w, s0=s0, seed=s)
                for nm in L(a.norms, str) for md in L(a.modes, str)
                if not (nm == "bn_param" and md == "inv")
                for w in L(a.widths, int) for s0 in L(a.s0s, float) for s in L(a.seeds, int)]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh, Pool(a.workers) as pool:
        for r in pool.imap_unordered(fn, cfgs):
            fh.write(json.dumps(r) + "\n"); fh.flush()
            if r["kind"] == "deterministic":
                print(f"[det] {r['norm']:8s} w{r['width']} seed{r['seed']}: test {r['test_err']:.4f}", flush=True)
            else:
                ph = r.get("posthoc", {})
                print(f"[pb] {r['norm']:8s} {r['mode']:5s} w{r['width']} s0={r['s0']}: cert {r['trained']['cert01']:.4f} "
                      f"(KL {r['trained']['KL']:.0f}) posthoc {ph.get('cert01', float('nan')):.4f} "
                      f"(KL {ph.get('KL', float('nan')):.0f}) test {r['trained']['test01']:.4f} [{r['seconds']:.0f}s]",
                      flush=True)


if __name__ == "__main__":
    main()
