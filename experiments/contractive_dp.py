"""Contractive noisy SGD on data-free features with a time-uniform (privacy-amplification-by-iteration) information
bound: Altschuler & Talwar 2022 Thm 5.1 constants (experiments/pabi_account.py) -> L-CL-Z Thm 8 -> PAC-Bayes-kl
certificate for the final iterate (theory.md s16).

Model: linear softmax on x~ = [a F/||F||, a] (F = per-example standardised Gabor features; a = 1/2, so ||x~||^2 = 1/2).
Per-example CE gradient wrt W is (p - e_y) x~^T: norm <= sqrt(2) ||x~|| = 1 =: L on all of R^d (no clipping needed);
Hessian <= (1/2) ||x~||^2 = 1/4 =: M.  Update, fixed-size uniform batches (size b, without replacement):
    W <- Proj_{||W|| <= rho} [ (1 - eta lam) W - eta (g_b(W) + Z) ],   Z ~ N(0, nu^2 I)
c = 1 - eta lam is the contraction of phi_t when eta (2 lam + M) <= 2.  W_0 = 0 (data-free).
Only the final iterate is certified (the bound is on the law of W_T, not on functions of the trajectory).
"""
import argparse, json, math, os, sys, time
import multiprocessing as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
M_SMOOTH, L_LIP = 0.25, 1.0


def feats(raw, out):
    import torch
    from dp_sgd_bound import gabor_features
    d = np.load(raw)
    res = {}
    for split in ("tr", "te"):
        F = gabor_features(torch.tensor(d["X" + split]).float())
        F = 0.5 * F / F.norm(dim=1, keepdim=True)
        res["X" + split] = torch.cat([F, torch.full((len(F), 1), 0.5)], 1).numpy().astype(np.float32)
        res["y" + split] = d["y" + split]
    np.savez(out, **res)


def rho_of(lam, eta, nu, d):
    """data-free radius: ||w*_lam|| <= sqrt(2 ln 10 / lam) (since lam/2 ||w*||^2 <= F(0) = ln 10), plus 3x the
    stationary noise norm sqrt(d eta nu^2 / (2 lam - eta lam^2)) of the linear recursion w <- c w + eta Z."""
    return math.sqrt(2 * math.log(10) / lam) + 3 * math.sqrt(d * eta * nu * nu / (2 * lam - eta * lam * lam))


def run(job):
    a, cfg = job
    import torch
    torch.set_num_threads(a.threads)
    from pabi_account import pabi_eps, S, certificate
    from tail_bounds import lz8_kappa
    d = np.load(a.feats)
    Xtr, ytr, Xte, yte = (torch.from_numpy(d[k]) for k in ("Xtr", "ytr", "Xte", "yte"))
    ytr, yte = ytr.long(), yte.long()
    n, p = Xtr.shape
    K = 10
    b, nu, lam, eta, T = cfg["b"], cfg["nu"], cfg["lam"], cfg["eta"], cfg["T"]
    assert eta * (2 * lam + M_SMOOTH) <= 2
    rho = rho_of(lam, eta, nu, p * K)
    g = torch.Generator().manual_seed(cfg["seed"])
    W = torch.zeros(p, K)
    Y = torch.nn.functional.one_hot(ytr, K).float()
    proj = 0
    t0 = time.time()
    for t in range(T):
        idx = torch.randperm(n, generator=g)[:b]
        Xb = Xtr[idx]
        G = Xb.T @ (torch.softmax(Xb @ W, 1) - Y[idx]) / b
        W = (1 - eta * lam) * W - eta * (G + nu * torch.randn(p, K, generator=g))
        nw = W.norm()
        if nw > rho:
            W *= rho / nw
            proj += 1
    with torch.no_grad():
        tr = float(((Xtr @ W).argmax(1) != ytr).float().mean())
        te = float(((Xte @ W).argmax(1) != yte).float().mean())
    D = 2 * rho
    q = b / n
    cert_p, k_p, info_p = certificate(tr, n, lambda al: pabi_eps(al, q, nu, b, T, eta, lam, D))
    cert_c, k_c, _ = certificate(tr, n, lambda al: T * S(q, round(nu * b / (2 * L_LIP), 9), al))
    det = pabi_eps(info_p["alpha"], q, nu, b, T, eta, lam, D, detail=True)[1]
    return dict(cfg, n=n, rho=rho, wnorm=float(W.norm()), proj_frac=proj / T, train01=tr, test01=te,
                kappa_pabi=k_p, cert_pabi=cert_p, kappa_comp=k_c, cert_comp=cert_c, alpha=info_p["alpha"],
                R=det["R"], split=det["f"], seconds=time.time() - t0)


def L(s, f):
    return [f(x) for x in s.split(",")]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw")
    ap.add_argument("--feats", required=True)
    ap.add_argument("--out")
    ap.add_argument("--stage", default="grid", choices=["feats", "grid"])
    ap.add_argument("--b", default="1024,4096")
    ap.add_argument("--nu", default="0.005,0.01,0.02")
    ap.add_argument("--lam", default="0.01,0.03,0.1")
    ap.add_argument("--eta", default="2,4,7")
    ap.add_argument("--T", default="300,1000,3000")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--workers", type=int, default=34)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    if a.stage == "feats":
        feats(a.raw, a.feats)
        sys.exit()
    cfgs = [dict(b=b, nu=nu, lam=lam, eta=eta, T=T, seed=s) for b in L(a.b, int) for nu in L(a.nu, float)
            for lam in L(a.lam, float) for eta in L(a.eta, float) for T in L(a.T, int) for s in L(a.seeds, int)
            if eta * (2 * lam + M_SMOOTH) <= 2]
    print(f"{len(cfgs)} runs", flush=True)
    out = open(a.out, "w")
    with mp.get_context("spawn").Pool(min(a.workers, len(cfgs))) as pool:
        for r in pool.imap_unordered(run, [(a, c) for c in cfgs]):
            out.write(json.dumps(r) + "\n")
            out.flush()
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
