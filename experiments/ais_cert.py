"""Data-tempered annealed importance sampling (AIS) / sequential Monte Carlo (SMC) as training, with a disintegrated
PAC-Bayes certificate for the final network whose complexity is the realised path log-ratio (theory.md s17, proofs in
s17.1-17.3, validity tests in s17.7: experiments/ais_validity.py).

Targets gamma_k(w) = P(w) exp(-lam L_k(w)), L_k = sum of cross-entropies over the first n_k training points (fixed
random order). Stage k adds the points (n_{k-1}, n_k].

Single chain (--P 1, AIS). w_0 ~ P (data-free Gaussian prior). For k = 1..K: record lam * l_{B_k}(w_{k-1}) (online
loss of the new block at the current sample), then apply S Metropolis-adjusted HMC moves, each reversible w.r.t.
gamma_k. Extended-space prior P~(w_{0:K}) = P(w_K) * prod_k T_k(w_k -> w_{k-1}) (T_k is its own gamma_k-reversal);
by detailed balance
    ln dQ~/dP~ (path) = lam * [ sum_k l_{B_k}(w_{k-1}) - L_K(w_K) ]          (=: C, computable on the realised run)
P~ depends on S through the kernels, but its w_K-marginal is the data-free P, and the disintegrated PAC-Bayes-kl
argument only integrates functions of w_K under P~ (s17.2). W.p. >= 1 - delta over S and the run:
    kl(err_S(w_K) || err_D(w_K)) <= (C + ln(2 sqrt(n)/delta)) / n.

Particles (--P > 1, SMC with multinomial resampling at every stage, resample-then-move). P particles w_0^j ~ P iid.
Stage k: incremental weights a_k^j = exp(-lam l_{B_k}(w_{k-1}^j)); Zhat_k = mean_j a_k^j; resample ancestors
~ a_k / sum a_k; move each particle with the HMC kernel. Output the particle in slot b = 0 (exchangeable, so it has the
law of a uniformly selected particle). Particle-Gibbs extended target (Andrieu, Doucet & Holenstein 2010; s17.3):
    C = ln dQ~/dP~ = -lam L_K(w_b) - ln Zhat,   Zhat = prod_k Zhat_k
i.e. the online loss of the particle-mixture predictor in place of the single chain's. P = 1 recovers AIS.
The best particle is certified with an extra ln P (union bound), reported as cert_best_union.

Groups (--R > 1, used by the validity tests): R independent runs batched together; resampling stays within groups.

Kernel parameters (step size schedule, leapfrog count, stage sizes) may depend on S and k but not on the path itself:
no adaptation from acceptance rates or particle states.

Architectures: --arch mlp (on pixels or data-free Gabor features) or --arch cnn (on pixels; particles via vmap).
Prior: N(0, prior_scale / fan_in) on weights, N(0, bias_var) on biases.
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


_FEAT_CACHE = {}


def load(a, torch, dt):
    key = (a.data, a.gabor, a.second_order)
    if key not in _FEAT_CACHE:
        d = np.load(a.data)
        X, Xte = torch.tensor(d["Xtr"]), torch.tensor(d["Xte"])
        if a.gabor:                                       # data-free scattering-like features (dp_sgd_bound.py)
            from dp_sgd_bound import gabor_features
            X, Xte = (gabor_features(Z.float(), second_order=a.second_order) for Z in (X, Xte))
        else:
            X, Xte = (X - 0.5) / 0.5, (Xte - 0.5) / 0.5  # fixed, data-free pixel scaling
        _FEAT_CACHE[key] = (X.to(dt), torch.tensor(d["ytr"]).long(), Xte.to(dt), torch.tensor(d["yte"]).long())
    return _FEAT_CACHE[key]


def build_model(a, din, K, torch, dt, dev):
    """returns (prior variances pv (D,), logits(w (B, D), X (N, din)) -> (B, N, K))"""
    shapes, pvar = [], []
    if a.arch == "mlp":
        dims = [din] + [a.width] * a.depth + [K]
        for l in range(len(dims) - 1):
            shapes += [(dims[l], dims[l + 1]), (dims[l + 1],)]
            pvar += [a.prior_scale / dims[l], a.bias_var]
    else:                                                 # conv5-relu-pool2, conv5-relu-pool2, [fc-relu], fc
        ch, side = (1, 28) if din == 784 else (3, 32)
        c1, c2 = a.channels
        shapes += [(c1, ch, 5, 5), (c1,), (c2, c1, 5, 5), (c2,)]
        pvar += [a.prior_scale / (ch * 25), a.bias_var, a.prior_scale / (c1 * 25), a.bias_var]
        flat = c2 * (side // 4) ** 2
        fc = [flat] + ([a.width] if a.depth else []) + [K]
        for l in range(len(fc) - 1):
            shapes += [(fc[l], fc[l + 1]), (fc[l + 1],)]
            pvar += [a.prior_scale / fc[l], a.bias_var]
    sizes = [math.prod(s) for s in shapes]
    pv = torch.cat([torch.full((s,), v, dtype=dt) for s, v in zip(sizes, pvar)]).to(dev)
    from fractions import Fraction
    fan = [s[0] if len(s) == 2 else (s[1] * s[2] * s[3] if len(s) == 4 else None) for s in shapes]
    o, groups = 0, []                                     # (start, end, exact variance): Fraction(prior_scale)/fan_in
    for s, k, f in zip(shapes, sizes, fan):
        v = Fraction(str(a.bias_var)) if len(s) == 1 else Fraction(str(a.prior_scale)) / f
        groups.append((o, o + k, v)); o += k
    assert all(abs(float(v) - float(pv[g0])) < 1e-15 for g0, _, v in groups)
    build_model.groups = groups

    def unflat(w):
        out, o = [], 0
        for s, k in zip(shapes, sizes):
            out.append(w[..., o:o + k].reshape(*w.shape[:-1], *s))
            o += k
        return out

    if a.arch == "mlp":
        def logits(w, Xb):
            ps = unflat(w)
            h = Xb.expand(len(w), *Xb.shape)
            nl = len(ps) // 2
            for l in range(nl):
                h = torch.baddbmm(ps[2 * l + 1][:, None, :], h, ps[2 * l])
                if l < nl - 1:
                    h = torch.relu(h)
            return h
        return pv, logits

    F = torch.nn.functional

    def single(w, Xb):
        ps = unflat(w)
        h = Xb.view(len(Xb), ch, side, side)
        h = F.max_pool2d(torch.relu(F.conv2d(h, ps[0], ps[1], padding=2)), 2)
        h = F.max_pool2d(torch.relu(F.conv2d(h, ps[2], ps[3], padding=2)), 2).flatten(1)
        nl = (len(ps) - 4) // 2
        for l in range(nl):
            h = h @ ps[4 + 2 * l] + ps[5 + 2 * l]
            if l < nl - 1:
                h = torch.relu(h)
        return h
    batched = torch.func.vmap(single, in_dims=(0, None))

    def logits(w, Xb):
        return single(w[0], Xb)[None] if len(w) == 1 else batched(w, Xb)
    return pv, logits


def stage_bounds(n, m, m0):
    """stage boundaries: geometric from m0 (m0, m0, 2 m0, 4 m0, ...) capped at m, then steps of m; data-free"""
    bounds = [0]
    while bounds[-1] < n:
        bounds.append(min(n, bounds[-1] + (min(m, max(m0, bounds[-1])) if m0 else m)))
    return bounds


def sampler(X, y, logits, pv, lam, bounds, S, L, c, n0, R, P, gen, on_stage=None, chunk=10000, cheat=None):
    """Runs R independent groups of P particles (R*P rows). Returns w (R*P, D), lnZ (R,), final_ce (R*P,),
    C (R*P,) = -lam L_K(w) - lnZ[group], online_ce_slot (R*P,) (lineage-free per-slot record) and per-stage logs.
    cheat: None, or a deliberately invalid variant, used only as negative controls of the identity test:
      "no_mh"        always accept the leapfrog proposal (kernel no longer reversible w.r.t. gamma_k);
      "score_after"  charge block k at the post-move sample w_k instead of w_{k-1} (P = 1; optimistic C)."""
    import torch
    dt, dev = pv.dtype, pv.device
    B = R * P

    def ce_sum(w, Xb, yb):
        out = 0
        for s in range(0, len(Xb), chunk):
            lg = logits(w, Xb[s:s + chunk])
            yy = yb[s:s + chunk].expand(len(w), -1)[..., None]
            out = out + (torch.logsumexp(lg, 2) - lg.gather(2, yy)[..., 0]).sum(1)
        return out

    def U_and_grad(w, k_end):
        w = w.detach().requires_grad_()
        u = 0.5 * (w * w / pv).sum(1) + lam * ce_sum(w, X[:k_end], y[:k_end])
        g, = torch.autograd.grad(u.sum(), w)
        return u.detach(), g

    w = torch.randn((B, len(pv)), generator=gen, dtype=dt, device=dev) * pv.sqrt()      # w_0 ~ P iid
    lnZ = torch.zeros(R, dtype=dt, device=dev)
    online_slot = torch.zeros(B, dtype=dt, device=dev)
    logs = []
    for k in range(1, len(bounds)):
        lo, hi = bounds[k - 1], bounds[k]
        with torch.no_grad():
            lb = ce_sum(w, X[lo:hi], y[lo:hi])            # (B,) online block loss, evaluated BEFORE moving on block k
            online_slot += lb
            la = (-lam * lb).view(R, P)
            lnZk = torch.logsumexp(la, 1) - math.log(P)   # (R,)
            lnZ += lnZk
            wn_ = torch.softmax(la, 1)
            ess = float((1 / (wn_ * wn_).sum(1)).mean())
            if P > 1:                                     # multinomial resampling within each group
                anc = torch.multinomial(wn_, P, replacement=True, generator=gen)            # (R, P)
                anc = (anc + P * torch.arange(R, device=dev)[:, None]).flatten()
                w = w[anc]
        eps = c / math.sqrt(1 + lam * max(hi, n0))        # fixed schedule in k (path-independent)
        u, gr = U_and_grad(w, hi)
        acc = torch.zeros(B, dtype=dt, device=dev)
        for s in range(S):
            p0 = torch.randn(w.shape, generator=gen, dtype=dt, device=dev) / pv.sqrt()  # momentum ~ N(0, diag(1/pv))
            wn, p = w.clone(), p0 - 0.5 * eps * gr
            for j in range(L):
                wn = wn + eps * pv * p
                un, gn = U_and_grad(wn, hi)
                if j < L - 1:
                    p = p - eps * gn
            p = p - 0.5 * eps * gn
            H0 = u + 0.5 * (p0 * p0 * pv).sum(1)
            H1 = un + 0.5 * (p * p * pv).sum(1)
            ok = torch.log(torch.rand(B, generator=gen, dtype=dt, device=dev)) < H0 - H1
            ok = ok & torch.isfinite(H1)
            if cheat == "no_mh":
                ok = torch.isfinite(H1)
            w = torch.where(ok[:, None], wn, w)
            u = torch.where(ok, un, u)
            gr = torch.where(ok[:, None], gn, gr)
            acc += ok.to(dt)
        if cheat == "score_after":
            with torch.no_grad():
                lb2 = ce_sum(w, X[lo:hi], y[lo:hi])
                lnZ += (-lam * lb2) - (-lam * lb)
        rec = dict(k=k, n_k=hi, eps=eps, acc=float(acc.mean()) / S, ess=ess,
                   online_ce_mix_batch=float(-lnZk.mean()) / lam / (hi - lo), online_ce_batch=float(lb.mean()) / (hi - lo))
        logs.append(rec)
        if on_stage is not None:
            on_stage(k, rec, w, lo, hi)
    with torch.no_grad():
        final_ce = ce_sum(w, X, y)
    C = -lam * final_ce - lnZ.repeat_interleave(P)
    return dict(w=w, lnZ=lnZ, final_ce=final_ce, C=C, online_slot=online_slot, logs=logs)


def main(a):
    import torch
    torch.set_num_threads(a.threads)
    dt = torch.float64
    dev = torch.device(a.device)
    X, y, Xte, yte = load(a, torch, dt)
    perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(a.order_seed))[:a.n]
    X, y = X[perm].to(dev), y[perm].to(dev)               # the training sample S, in its (random) presentation order
    Xte, yte = Xte.to(dev), yte.to(dev)
    if a.random_labels:                                   # control: labels independent of inputs, train and test
        gl = torch.Generator().manual_seed(10_000 + a.order_seed)
        y = torch.randint(0, 10, (len(y),), generator=gl).to(dev)
        yte = torch.randint(0, 10, (len(yte),), generator=gl).to(dev)
    n, P = len(X), a.P
    pv, logits = build_model(a, X.shape[1], 10, torch, dt, dev)

    def errs(w, Xb, yb, chunk=10000):                     # per-particle 0-1 error
        with torch.no_grad():
            e = sum((logits(w, Xb[s:s + chunk]).argmax(2) != yb[s:s + chunk]).sum(1).double()
                    for s in range(0, len(Xb), chunk))
        return (e / len(Xb)).cpu().numpy()

    bounds = stage_bounds(n, a.m, a.m0)
    K = len(bounds) - 1
    if a.exact:
        return main_exact(a, X, y, Xte, yte, logits, errs, bounds, torch)
    t0 = time.time()

    def on_stage(k, rec, w, lo, hi):
        rec["seconds"] = time.time() - t0
        if k % a.log_every == 0 or k == K:
            rec.update(train_err=float(errs(w[:1], X[:hi], y[:hi])[0]), test_err=float(errs(w[:1], Xte, yte)[0]))
            print(rec, flush=True)

    g = torch.Generator(device=dev).manual_seed(a.seed)
    out = sampler(X, y, logits, pv, a.lam, bounds, a.S, a.L, a.c, a.n0, 1, P, g, on_stage=on_stage)
    w, lnZ = out["w"], float(out["lnZ"][0])
    final_ce = out["final_ce"].cpu().numpy()
    Cs = out["C"].cpu().numpy()
    trs, tes = errs(w, X, y), errs(w, Xte, yte)
    base = math.log(2 * math.sqrt(n) / DELTA)
    certs = np.array([kl_inv(float(t), (float(c) + base) / n) for t, c in zip(trs, Cs)])
    certs_u = np.array([kl_inv(float(t), (float(c) + base + math.log(P)) / n) for t, c in zip(trs, Cs)])
    with torch.no_grad():
        pm = torch.cat([torch.softmax(logits(w, Xte[s:s + 2000]), 2).mean(0) for s in range(0, len(Xte), 2000)])
    log = out["logs"]
    res = dict(cfg=vars(a), n=n, P=P, K=K, n_params=int(pv.numel()),
               train01=float(trs[0]), test01=float(tes[0]), C=float(Cs[0]), C_over_n=float(Cs[0]) / n,
               cert=float(certs[0]), online_ce_mix=-lnZ / a.lam / n,
               online_ce_slot0=float(out["online_slot"][0]) / n, final_train_ce=float(final_ce[0]) / n,
               cert_mean=float(certs.mean()), cert_best_union=float(certs_u.min()),
               train01_all=trs.tolist(), test01_all=tes.tolist(), test01_bma=float((pm.argmax(1) != yte).double().mean()),
               mean_acc=float(np.mean([r["acc"] for r in log])), seconds=time.time() - t0, log=log)
    print({k: v for k, v in res.items() if k not in ("log", "cfg", "train01_all", "test01_all")}, flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"))
    return res


def main_exact(a, X, y, Xte, yte, logits, errs, bounds, torch):
    """floating-point-free lattice HMC (experiments/exact_hmc.py, theory.md s17.5); P = 1, CPU"""
    from exact_hmc import ExactAIS
    assert a.P == 1 and a.device == "cpu"
    n, K = len(X), len(bounds) - 1
    t0 = time.time()

    def on_stage(k, rec, i):
        rec["seconds"] = time.time() - t0
        if k % a.log_every == 0 or k == K:
            w = torch.from_numpy(i * 2.0 ** -24)[None]
            rec.update(train_err=float(errs(w, X[:bounds[k]], y[:bounds[k]])[0]), test_err=float(errs(w, Xte, yte)[0]))
            print(rec, flush=True)
    lam = a.lam
    r = ExactAIS(torch, X, y, logits, build_model.groups, lam, bounds, a.S, a.L, a.c, a.n0, seed=a.seed,
                 audit=True).run(on_stage=on_stage, roundtrip_every=a.roundtrip_every)
    w = torch.from_numpy(r["w"])[None]
    tr, te = float(errs(w, X, y)[0]), float(errs(w, Xte, yte)[0])
    Cf = float(r["C"]) + 1e-9 * max(1.0, abs(float(r["C"])))   # round the exact rational C upward
    cert = kl_inv(tr, (Cf + math.log(2 * math.sqrt(n) / DELTA)) / n)
    res = dict(cfg=vars(a), exact=True, n=n, K=K, n_params=int(r["w"].size), train01=tr, test01=te,
               C=Cf, C_exact=f"{r['C'].numerator}/{r['C'].denominator}" if a.keep_rational else None, C_over_n=Cf / n,
               cert=cert, online_ce=sum(r["online"]) / n, final_train_ce=sum(r["final"]) / n,
               mean_acc=float(np.mean(r["acc"])), stats=r["stats"], seconds=time.time() - t0, acc=r["acc"])
    print({k: v for k, v in res.items() if k not in ("cfg", "acc", "C_exact")}, flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"))
    return res


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out")
    ap.add_argument("--arch", default="mlp", choices=["mlp", "cnn"])
    ap.add_argument("--channels", type=lambda s: [int(v) for v in s.split(",")], default=[16, 32], help="cnn: c1,c2")
    ap.add_argument("--gabor", action="store_true", help="use data-free Gabor/scattering features instead of pixels")
    ap.add_argument("--second_order", action="store_true", help="with --gabor: add second-order scattering maps")
    ap.add_argument("--random_labels", action="store_true", help="control: replace train and test labels by noise")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--width", type=int, default=100)
    ap.add_argument("--depth", type=int, default=1, help="mlp: hidden layers; cnn: 1 = one hidden fc layer")
    ap.add_argument("--prior_scale", type=float, default=1.0)
    ap.add_argument("--bias_var", type=float, default=0.1)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--m", type=int, default=1000)
    ap.add_argument("--m0", type=int, default=0, help="if >0: stage sizes m0, m0, 2m0, 4m0, ... capped at m")
    ap.add_argument("--P", type=int, default=1, help="number of SMC particles (1 = plain AIS)")
    ap.add_argument("--S", type=int, default=4)
    ap.add_argument("--L", type=int, default=30)
    ap.add_argument("--c", type=float, default=0.05)
    ap.add_argument("--n0", type=int, default=0, help="step size eps = c / sqrt(1 + lam max(n_k, n0))")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--order_seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--log_every", type=int, default=1)
    ap.add_argument("--exact", action="store_true", help="floating-point-free lattice HMC (exact_hmc.py)")
    ap.add_argument("--roundtrip_every", type=int, default=0, help="exact: verify the involution every k-th move")
    ap.add_argument("--keep_rational", action="store_true", help="exact: store C as an exact rational")
    return ap


if __name__ == "__main__":
    main(parser().parse_args())
