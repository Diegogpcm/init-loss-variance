"""Certificate for a single multi-epoch network trained with noisy, clipped SGD (DP-SGD), via mutual information.

Algorithm A (Abadi et al. 2016): init W_0 (data-free); for t = 1..T: Poisson-sample each example with rate q = B/n,
G_t = sum_{i in batch} clip_C(grad l(W_{t-1}, z_i)) + N(0, sigma^2 C^2 I); W_t = SGD-momentum step with G_t / B.

Bound (derived in docs/theory.md section 14):
  1. Prior P* = marginal law of W_T over S ~ D^n and the algorithm's randomness (data-free, unknown).
     Posterior Q_S = law of W_T given S. Information density i(W, S) = ln dQ_S/dP*(W); E[i] = I(S; W_T).
  2. Disintegrated PAC-Bayes-kl (Maurer + change of measure, single draw W ~ Q_S, i.e. the one trained network):
        P[ kl(L_S(W) || L(W)) > (i(W,S) + ln(2 sqrt(n)/d2)) / n ] <= d2.
  3. E[i^+] <= I + 1/e (since E[e^{-i}] <= 1 and -x <= e^{-x-1}), Markov: P[i > (I + 1/e)/d1] <= d1.
  4. I(S; W_T) <= I(S; G_1..G_T) <= sum_i sum_t E KL(P_{G_t | S, past} || P_{G_t | S without z_i, past})
                                 <= n T KL_step(q, sigma)
     (chain rule over examples for a product distribution, conditional MI <= KL to the leave-one-out mechanism,
     adaptive chain rule over steps; KL_step = KL((1-q) N(0, s^2) + q N(1, s^2) || N(0, s^2)), the worst case over
     clipped gradients by data processing).
  Certificate (probability >= 1 - d1 - d2): kl(train err || risk) <= ((I_ub + 1/e)/d1 + ln(2 sqrt(n)/d2)) / n.
  The Markov step is the price: the KL is charged 1/d1 times.

Usage: python experiments/dp_sgd_bound.py --data /tmp/lv_fashion_norm.npz --out results/v2/dp_sgd.jsonl
"""
import argparse, json, math, os, sys, time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pacbayes_norm import init_params, forward, kl_inv, load  # noqa: E402

D1, D2 = 0.024, 0.001          # Markov / PAC-Bayes split of delta = 0.025 (the KL term is linear in 1/d1)


def kl_step(q, sigma):
    """KL((1-q) N(0, s^2) + q N(1, s^2) || N(0, s^2)): Poisson-subsampled Gaussian, unit sensitivity, remove-one."""
    x = np.linspace(-12 * sigma - 1, 12 * sigma + 2, 400001)
    dx = x[1] - x[0]
    phi = lambda m: np.exp(-(x - m) ** 2 / (2 * sigma ** 2)) / math.sqrt(2 * math.pi * sigma ** 2)
    p = (1 - q) * phi(0) + q * phi(1)
    lr = np.log1p(q * np.expm1((2 * x - 1) / (2 * sigma ** 2)))
    return float(np.sum(p * lr) * dx)


def renyi2_step(q, sigma):
    return math.log1p(q * q * math.expm1(1 / sigma ** 2))


def gabor_features(X):
    """Data-free first-order scattering-like features (cf. Tramer & Boneh 2021): |X * gabor| for 2 scales x 8
    orientations, 4x4 average pooling, plus a pooled low-pass image; then per-example standardization."""
    import torch
    imgs = X.reshape(-1, 1, 28, 28)
    r = torch.arange(-5, 6, dtype=torch.float32)
    yy, xx = torch.meshgrid(r, r, indexing="ij")
    ks = []
    for sig in (1.5, 3.0):
        for o in range(8):
            th = math.pi * o / 8
            u = xx * math.cos(th) + yy * math.sin(th)
            env = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sig ** 2))
            for ph in (torch.cos, torch.sin):
                k = env * ph(2 * math.pi * u / (3 * sig))
                ks.append(k - k.mean() * (ph is torch.cos))      # zero-mean real part
    K = torch.stack(ks)[:, None]
    feats = []
    for i in range(0, len(imgs), 5000):
        z = torch.nn.functional.conv2d(imgs[i:i + 5000], K, padding=5)
        mod = torch.sqrt(z[:, 0::2] ** 2 + z[:, 1::2] ** 2 + 1e-8)
        f = torch.cat([torch.nn.functional.avg_pool2d(mod, 4), torch.nn.functional.avg_pool2d(imgs[i:i + 5000], 4)], 1)
        feats.append(f.flatten(1))
    F = torch.cat(feats)
    return (F - F.mean(1, keepdim=True)) / F.std(1, keepdim=True)


def run(cfg):
    import torch
    from torch.func import grad, vmap
    torch.set_num_threads(cfg["threads"])
    t0 = time.time()
    Xtr, ytr, _, Xte, yte = load(cfg["data"])
    Xtr, Xte = Xtr.float(), Xte.float()
    if cfg["features"] == "gabor":            # cfg["data"] must then hold raw [0,1] pixels (lv_fashion_raw.npz)
        Xtr, Xte = gabor_features(Xtr), gabor_features(Xte)
    norm, depth, width, B, nu, epochs, lr, C = (cfg[k] for k in ("norm", "depth", "width", "batch", "nu", "epochs",
                                                                    "lr", "clip"))
    n, K = len(Xtr), int(ytr.max()) + 1
    q, sigma = B / n, nu * B / C               # nu = noise std on the mean gradient, in units of C
    T = int(round(epochs / q))
    gen = torch.Generator().manual_seed(cfg["seed"])
    if depth == 0:                             # linear softmax (zero init: data-free)
        th = {"Wout": torch.zeros(Xtr.shape[1], K), "bout": torch.zeros(K)}
    else:
        th = {k: v.float() for k, v in init_params(norm, depth, width, Xtr.shape[1], K, gen)[0].items()}

    def loss1(th, x, y):
        return torch.nn.functional.cross_entropy(forward(th, x[None], norm, depth), y[None])
    pgrad = vmap(grad(loss1), in_dims=(None, 0, 0))

    def err(th, X, y, bs=10000):
        with torch.no_grad():
            return sum(int((forward(th, X[i:i + bs], norm, depth).argmax(1) != y[i:i + bs]).sum())
                       for i in range(0, len(X), bs)) / len(X)

    mom = {k: torch.zeros_like(v) for k, v in th.items()}
    avg, n_avg = {k: torch.zeros_like(v) for k, v in th.items()}, 0   # tail average of iterates (post-processing)
    clipped_frac = []
    for t in range(T):
        lr_t = lr * (0.5 * (1 + math.cos(math.pi * t / T)) if cfg["sched"] == "cos" else 1.0)
        idx = torch.nonzero(torch.rand(n, generator=gen) < q).flatten()
        G = {k: torch.zeros_like(v) for k, v in th.items()}
        for i in range(0, len(idx), cfg["chunk"]):
            j = idx[i:i + cfg["chunk"]]
            g = pgrad(th, Xtr[j], ytr[j])
            nrm = torch.sqrt(sum(v.reshape(len(j), -1).pow(2).sum(1) for v in g.values()))
            f = torch.clamp(C / (nrm + 1e-12), max=1.0)
            clipped_frac.append(float((nrm > C).float().mean()))
            for k in G:
                G[k] += torch.einsum("b,b...->...", f, g[k])
        for k in th:
            G[k] += sigma * C * torch.randn(G[k].shape, generator=gen)
            mom[k] = 0.9 * mom[k] + G[k] / B
            th[k] = th[k] - lr_t * mom[k]
        if t >= T // 2:
            n_avg += 1
            for k in th:
                avg[k] += (th[k] - avg[k]) / n_avg
        if not all(torch.isfinite(v).all() for v in th.values()):
            return dict(cfg=cfg, diverged=True, step=t)
    train01, test01 = err(th, Xtr, ytr), err(th, Xte, yte)
    train_avg, test_avg = err(avg, Xtr, ytr), err(avg, Xte, yte)

    kls, r2 = kl_step(q, sigma), renyi2_step(q, sigma)
    I_ub = n * T * kls
    conf = ((I_ub + 1 / math.e) / D1 + math.log(2 * math.sqrt(n) / D2)) / n
    # both the final and the averaged iterate are certified, each at (D1/2, D2/2), so the min of the two is valid
    conf2 = ((I_ub + 1 / math.e) / (D1 / 2) + math.log(2 * math.sqrt(n) / (D2 / 2))) / n
    exp_conf = (I_ub + math.log(2 * math.sqrt(n) / 0.025)) / n      # NOT a certificate: E[KL] plugged in, no Markov
    res = dict(features=cfg["features"], n_params=sum(v.numel() for v in th.values()), norm=norm, depth=depth, width=width, batch=B, nu=nu, sigma=sigma, epochs=epochs, lr=lr, clip=C,
               seed=cfg["seed"], n=n, q=q, T=T, train01=train01, test01=test01,
               kl_step=kls, renyi2_step=r2, I_ub=I_ub, I_per_n=I_ub / n, conf=conf,
               cert01=kl_inv(train01, conf), exp_cert01=kl_inv(train01, exp_conf), sched=cfg["sched"],
               train_avg=train_avg, test_avg=test_avg,
               cert01_best2=min(kl_inv(train01, conf2), kl_inv(train_avg, conf2)),
               clipped_frac=float(np.mean(clipped_frac)), seconds=time.time() - t0)
    return res


def safe_run(cfg):
    try:
        return run(cfg)
    except Exception as e:  # noqa: BLE001
        return dict(cfg=cfg, error=repr(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--norms", default="ln,none")
    ap.add_argument("--widths", default="128")
    ap.add_argument("--depths", default="2")
    ap.add_argument("--features", default="pixels", choices=["pixels", "gabor"])
    ap.add_argument("--batches", default="1024,4096")
    ap.add_argument("--nus", default="0.005,0.01,0.02")
    ap.add_argument("--epochs", default="10,30")
    ap.add_argument("--lrs", default="0.1")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--sched", default="cos", choices=["cos", "const"])
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    L = lambda x, f: [f(v) for v in x.split(",")]
    cfgs = [dict(data=a.data, features=a.features, depth=dp, norm=nm, width=w, batch=B, nu=nu, epochs=e, lr=lr, clip=a.clip, sched=a.sched, seed=s,
                 chunk=a.chunk, threads=a.threads)
            for dp in L(a.depths, int) for nm in L(a.norms, str) for w in L(a.widths, int) for B in L(a.batches, int) for nu in L(a.nus, float)
            for e in L(a.epochs, int) for lr in L(a.lrs, float) for s in L(a.seeds, int)]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh, Pool(min(a.workers, len(cfgs))) as pool:
        for r in pool.imap_unordered(safe_run, cfgs):
            fh.write(json.dumps(r) + "\n"); fh.flush()
            if "cert01" not in r:
                print("FAILED/DIVERGED", r, flush=True); continue
            print(f"{r['features']} d{r['depth']} {r['norm']} w{r['width']} P={r['n_params']} B={r['batch']} nu={r['nu']} (sigma={r['sigma']:.1f}) ep={r['epochs']} "
                  f"lr={r['lr']} {r['sched']} s{r['seed']}: train {r['train01']:.4f} test {r['test01']:.4f} | avg train "
                  f"{r['train_avg']:.4f} test {r['test_avg']:.4f} best-of-2 cert {r['cert01_best2']:.4f} | I<= {r['I_ub']:.1f} "
                  f"nats | cert {r['cert01']:.4f} (no-Markov {r['exp_cert01']:.4f}) | clipped {r['clipped_frac']:.2f} "
                  f"[{r['seconds']:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
