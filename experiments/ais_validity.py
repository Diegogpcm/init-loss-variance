"""Validity tests for the prequential AIS / SMC certificate (theory.md s17.7).

Stage "identity": the certificate rests on one identity. With C = ln dQ~/dP~ on the realised run and P~ having the
data-free prior P as its w_K-marginal,
    E_{run ~ Q~} [ exp(-C) g(w_K) ] = E_{w ~ P} [ g(w) ]        for every g >= 0 (and integrable g),
in particular E[exp(-C)] = 1. We check it by brute force on a tiny network (same code path as the real runs:
experiments/ais_cert.py:sampler), batching R independent runs, for AIS (P = 1) and SMC (P = 4), against direct prior
sampling, for g in {1, w_K[0], w_K[0]^2, L_K(w_K)/n, 1[err_S(w_K) = 0]}. The last two are where the run's law and the
prior differ most. Negative control: the same sampler with the Metropolis test switched off (an invalid,
non-reversible kernel) must fail it, and so must charging each block after the moves on it ("score_after").
Heavy-tailed weights bias the plain sample mean DOWN, so the dangerous direction (an optimistic C) shows as
weighted > prior; validity only needs the one-sided E[exp(-C) f(w_K)] <= E_P f for f >= 0.

Stage "real": coverage and random-label controls on MNIST with Gabor features, calling ais_cert.main.
    cov:  n = 1000, true labels, lam in {0.25, 1, 4}, 20 seeds each (different subsets and orders): count runs where
          the certificate is below the test error (true error estimated on the 10k test set).
    rand: same with random labels (true error 0.9 for every classifier); and one n = 60000, lam = 0.25 run.
"""
import argparse, json, math, os, sys, time
import multiprocessing as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def identity(a):
    import torch
    from ais_cert import build_model, sampler
    torch.set_num_threads(a.threads)
    dt = torch.float64
    g0 = torch.Generator().manual_seed(0)
    n, K = 8, 3
    X = torch.randn(n, 2, generator=g0, dtype=dt) * 1.5
    y = (X @ torch.tensor([[1.0, -0.5, -0.5], [0.0, 1.0, -1.0]], dtype=dt) + 0.3 * torch.randn(n, K, generator=g0, dtype=dt)).argmax(1)
    cfg = argparse.Namespace(arch="mlp", width=4, depth=1, prior_scale=1.0, bias_var=0.1)
    pv, logits = build_model(cfg, 2, K, torch, dt, torch.device("cpu"))
    bounds, lam, S, L, c = [0, 2, 4, 8], a.lam, 2, 5, a.c

    def stats(w, final_ce):
        with torch.no_grad():
            err0 = (logits(w, X).argmax(2) != y).double().mean(1) == 0
        return np.stack([np.ones(len(w)), w[:, 0].numpy(), (w[:, 0] ** 2).numpy(), final_ce.numpy() / n,
                         err0.double().numpy()], 1)
    names = ["1", "w[0]", "w[0]^2", "L_K/n", "1[err_S=0]"]

    # ground truth: direct prior sampling
    gp = torch.Generator().manual_seed(1)
    wp = torch.randn(a.R_prior, len(pv), generator=gp, dtype=dt) * pv.sqrt()
    with torch.no_grad():
        lg = logits(wp, X)
        ce = (torch.logsumexp(lg, 2) - lg.gather(2, y.expand(len(wp), -1)[..., None])[..., 0]).sum(1)
    Gp = stats(wp, ce)
    truth = Gp.mean(0)
    truth_se = Gp.std(0) / math.sqrt(len(Gp))
    out = dict(n=n, lam=lam, bounds=bounds, S=S, L=L, c=c, n_params=int(pv.numel()), prior=dict(zip(names, truth.tolist())))
    cases = [("ais", 1, None, a.R), ("smc_P4", 4, None, a.R // 4), ("ais_no_mh (negative control)", 1, "no_mh", a.R),
             ("ais_score_after (negative control)", 1, "score_after", a.R)]
    for name, P, cheat, R in cases:
        gen = torch.Generator().manual_seed(2)
        t0 = time.time()
        res = sampler(X, y, logits, pv, lam, bounds, S, L, c, 0, R, P, gen, cheat=cheat)
        idx = torch.arange(R) * P                         # slot 0 of every group
        C = res["C"][idx]
        Wt = torch.exp(-C).numpy()
        G = stats(res["w"][idx], res["final_ce"][idx])
        unweighted = G.mean(0)
        est = (Wt[:, None] * G).mean(0)
        se = (Wt[:, None] * G).std(0) / math.sqrt(R)
        z = (est - truth) / np.sqrt(se ** 2 + truth_se ** 2)
        ess = Wt.sum() ** 2 / (Wt ** 2).sum()
        rec = dict(R=R, P=P, seconds=time.time() - t0, ess=float(ess), mean_C=float(C.mean()),
                   acc=float(np.mean([l["acc"] for l in res["logs"]])),
                   rows={nm: dict(prior=float(t), unweighted_run=float(u), weighted=float(e), se=float(s), z=float(zz))
                         for nm, t, u, e, s, zz in zip(names, truth, unweighted, est, se, z)})
        out[name] = rec
        print(name, json.dumps(rec, indent=1), flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


def _real_job(args):
    from ais_cert import parser, main
    r = main(parser().parse_args(args))
    keep = ("n", "train01", "test01", "C", "C_over_n", "cert", "online_ce_mix", "final_train_ce", "mean_acc", "seconds")
    return dict({k: r[k] for k in keep}, lam=r["cfg"]["lam"], random_labels=r["cfg"]["random_labels"],
                seed=r["cfg"]["seed"], order_seed=r["cfg"]["order_seed"])


def real(a):
    base = ["--data", a.data, "--gabor", "--prior_scale", "2", "--m", "1000", "--m0", "10", "--S", "10", "--L", "50",
            "--c", "1.5", "--n0", "300", "--log_every", "1000", "--threads", str(a.threads)]
    jobs = []
    if a.big:
        jobs.append(base + ["--n", "60000", "--lam", "0.25", "--random_labels"])
    for lab in ([], ["--random_labels"]):
        for lam in a.lams.split(","):
            for s in range(a.seeds):
                jobs.append(base + ["--n", str(a.n), "--lam", lam, "--seed", str(s), "--order_seed", str(s)] + lab)
    out = open(a.out, "w")
    with mp.get_context("spawn").Pool(min(a.workers, len(jobs))) as pool:
        for r in pool.imap_unordered(_real_job, jobs):
            out.write(json.dumps(r) + "\n")
            out.flush()
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)


# ---------- exact (lattice) sampler tests, theory.md s17.5 ----------
def exact_unit(a):
    """discrete-Gaussian samplers against exact pmfs (chi-square), and exact Bernoulli(exp(-g)) frequencies"""
    import random
    from fractions import Fraction
    from scipy import stats as st
    from exact_hmc import dg_sample_vec, dg_sample_exact, bern_exp
    out = {}
    for N in (3, 50, 2 ** 24):
        x = dg_sample_vec(N, a.unit_n, np.random.default_rng(N))
        sd = math.sqrt(N)
        lo, hi = -int(4 * sd), int(4 * sd)
        edges = np.unique(np.linspace(lo, hi + 1, min(60, hi - lo + 2)).astype(np.int64))
        ks = np.arange(lo - 50 * int(sd) - 10, hi + 50 * int(sd) + 10) if N < 10 ** 4 else None
        if ks is not None:                                 # exact pmf on a wide range
            p = np.exp(-ks.astype(float) ** 2 / (2 * N)); p /= p.sum()
            cnt = np.array([(x == k).sum() for k in ks])
            m = p * len(x) > 5
            chi = (((cnt[m] - p[m] * len(x)) ** 2) / (p[m] * len(x))).sum()
            pv = 1 - st.chi2.cdf(chi, m.sum() - 1)
        else:                                              # binned against the (numerically exact) Gaussian limit
            cdf = st.norm.cdf((edges - 0.5) / sd)
            pb = np.diff(cdf)
            cnt = np.histogram(x, bins=edges - 0.5)[0]
            chi = (((cnt - pb * len(x)) ** 2) / (pb * len(x))).sum()
            pv = 1 - st.chi2.cdf(chi, len(pb) - 1)
        out[f"dg_vec_N{N}"] = dict(n=len(x), mean=float(x.mean()), var_over_N=float(x.var() / N), chi2_p=float(pv))
    rng = random.Random(0)
    s2 = Fraction(9, 4)
    x = np.array([dg_sample_exact(s2, rng) for _ in range(a.unit_n // 20)])
    ks = np.arange(-30, 31)
    p = np.exp(-ks.astype(float) ** 2 / (2 * 2.25)); p /= p.sum()
    cnt = np.array([(x == k).sum() for k in ks]); m = p * len(x) > 5
    chi = (((cnt[m] - p[m] * len(x)) ** 2) / (p[m] * len(x))).sum()
    out["dg_exact_s2_9/4"] = dict(n=len(x), mean=float(x.mean()), var=float(x.var()), chi2_p=float(1 - st.chi2.cdf(chi, m.sum() - 1)))
    for g in (Fraction(1, 3), Fraction(5, 2)):
        f = np.mean([bern_exp(g, rng) for _ in range(a.unit_n // 20)])
        out[f"bern_exp_{g}"] = dict(freq=float(f), target=math.exp(-float(g)), se=math.sqrt(math.exp(-float(g)) * (1 - math.exp(-float(g))) / (a.unit_n // 20)))
    print(json.dumps(out, indent=1), flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


def _exact_chain(job):
    import torch
    from fractions import Fraction
    from ais_cert import build_model
    from exact_hmc import ExactAIS
    torch.set_num_threads(1)
    seeds, lam, c = job
    dt = torch.float64
    g0 = torch.Generator().manual_seed(0)
    n, K = 8, 3
    X = torch.randn(n, 2, generator=g0, dtype=dt) * 1.5
    y = (X @ torch.tensor([[1.0, -0.5, -0.5], [0.0, 1.0, -1.0]], dtype=dt) + 0.3 * torch.randn(n, K, generator=g0, dtype=dt)).argmax(1)
    cfg = argparse.Namespace(arch="mlp", width=4, depth=1, prior_scale=1.0, bias_var=0.1)
    pv, logits = build_model(cfg, 2, K, torch, dt, torch.device("cpu"))
    groups, o = [], 0                                      # (start, end, exact prior variance) per tensor
    for size, v in ((8, Fraction(1, 2)), (4, Fraction(1, 10)), (12, Fraction(1, 4)), (3, Fraction(1, 10))):
        groups.append((o, o + size, v)); o += size
    assert o == len(pv) and all(abs(float(v) - float(pv[a])) < 1e-15 for a, b, v in groups)
    rows = []
    for s in seeds:
        r = ExactAIS(torch, X, y, logits, groups, lam, [0, 2, 4, 8], 2, 5, c, 0, seed=s, audit=True).run(roundtrip_every=2 if s % 50 == 0 else 0)
        w = torch.from_numpy(r["w"])[None]
        with torch.no_grad():
            err0 = bool(((logits(w, X)[0].argmax(1) != y).double().mean() == 0))
        rows.append((float(r["C"]), float(r["w"][0]), sum(r["final"]) / n, err0, r["stats"]["audit_fail"],
                     r["stats"]["roundtrips"], r["stats"]["roundtrip_fail"], float(np.mean(r["acc"]))))
    return rows


def identity_exact(a):
    jobs = [(range(s, s + a.R // a.workers // 10), a.lam, a.c) for s in range(0, a.R, a.R // a.workers // 10)]
    rows = []
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for r in pool.imap_unordered(_exact_chain, jobs):
            rows += r
    A = np.array(rows, dtype=float)
    Wt = np.exp(-A[:, 0])
    G = np.stack([np.ones(len(A)), A[:, 1], A[:, 1] ** 2, A[:, 2], A[:, 3]], 1)
    prior = json.load(open(a.prior_from))["prior"]
    names = ["1", "w[0]", "w[0]^2", "L_K/n", "1[err_S=0]"]
    est, se = (Wt[:, None] * G).mean(0), (Wt[:, None] * G).std(0) / math.sqrt(len(A))
    res = dict(R=len(A), seconds=time.time() - t0, ess=float(Wt.sum() ** 2 / (Wt ** 2).sum()), mean_C=float(A[:, 0].mean()),
               acc=float(A[:, 7].mean()), audit_fail=int(A[:, 4].sum()), roundtrips=int(A[:, 5].sum()),
               roundtrip_fail=int(A[:, 6].sum()),
               rows={nm: dict(prior=prior[nm], weighted=float(e), se=float(s), z=float((e - prior[nm]) / s))
                     for nm, e, s in zip(names, est, se)})
    print(json.dumps(res, indent=1), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["identity", "real", "exact_unit", "identity_exact"], required=True)
    ap.add_argument("--unit_n", type=int, default=400000)
    ap.add_argument("--prior_from", help="identity_exact: json of a float identity run (for the prior column)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--R", type=int, default=200000)
    ap.add_argument("--R_prior", type=int, default=2000000)
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--c", type=float, default=1.2, help="identity: HMC step scale (large, so the no-MH control is visibly biased)")
    ap.add_argument("--data", default="/tmp/lv_mnist_raw.npz")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--lams", default="0.25,1,4")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--big", action="store_true")
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    dict(identity=identity, real=real, exact_unit=exact_unit, identity_exact=identity_exact)[a.stage](a)
