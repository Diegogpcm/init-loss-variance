"""Diagnostic for certifying a deterministic SGD-trained network via leave-one-out influence (theory.md s15).

Question: is the per-example influence of the data on the final weights, propagated through the training dynamics,
far smaller than the distance travelled from init? If so, an oracle-prior / information bound for the perturbed output
W~ = W_T + N(0, diag sigma^2) is far tighter than the usual init-centred Gaussian PAC-Bayes KL.

  I(S; W~) <= sum_i E || W_T(S) - W_T(S \\ z_i) ||^2_{diag(1/sigma^2)} / 2            (chain rule + Neu et al. Lemma 4)
  W_T(S) - W_T(S \\ z_i) ~= - sum_{t: i in B_t} (eta/B) J_{T<-t+1} grad l(W_t; z_i),   J_{T<-t+1} = prod_{s>t} (I - eta H_s)

Stages (all on one plain-SGD run of a standard ReLU MLP, no clipping, no noise):
  train    : SGD, store the full trajectory (memmap) and the minibatch indices
  adjoint  : Hutchinson probes v restricted to one layer: backprop v through training (u <- u - eta H_t u), accumulate
             a_i += (eta/B) u_{t+1} . grad l(W_t; z_i);   E_v sum_i a_i^2 = sum_i ||Delta W_i (layer)||^2
  check    : exact leave-one-out retraining (same batches, same divisor B) vs forward-tangent linearisation, m points
  spectrum : per-example gradient covariance spectrum at checkpoints; d(sigma) = sum_k min(1, eta^2 lambda_k / (B sigma^2))
  report   : flatness E_xi err(W_T + xi) on train/test, KL_init vs I_infl per sigma scale, plug-in bound values
Nothing here is a certificate: I_infl is an *expected* quantity estimated on one run, and sigma is chosen post hoc.
"""
import argparse, json, math, os, sys, time
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def setup(a):
    import torch
    torch.set_num_threads(a.threads)
    d = np.load(a.data)
    X = torch.tensor(d["Xtr"], dtype=torch.float32)
    y = torch.tensor(d["ytr"]).long()
    Xte = torch.tensor(d["Xte"], dtype=torch.float32)
    yte = torch.tensor(d["yte"]).long()
    X, Xte = (X - 0.5) / 0.5, (Xte - 0.5) / 0.5            # fixed, data-free scaling of raw [0,1] pixels
    dims = [X.shape[1]] + [a.width] * a.depth + [int(y.max()) + 1]
    shapes = []
    for l in range(len(dims) - 1):
        shapes += [(f"W{l}", (dims[l], dims[l + 1])), (f"b{l}", (dims[l + 1],))]
    sizes = [int(np.prod(s)) for _, s in shapes]
    offs = np.concatenate([[0], np.cumsum(sizes)])
    groups = {f"L{l}": (int(offs[2 * l]), int(offs[2 * l + 2])) for l in range(len(dims) - 1)}   # W and b of a layer

    def unflat(th):
        return {k: th[offs[j]:offs[j + 1]].view(s) for j, (k, s) in enumerate(shapes)}

    def logits(th, Xb):
        p = unflat(th)
        h = Xb
        L = len(dims) - 1
        for l in range(L):
            h = h @ p[f"W{l}"] + p[f"b{l}"]
            if l < L - 1:
                h = torch.relu(h)
        return h

    def losses(th, Xb, yb):
        return torch.nn.functional.cross_entropy(logits(th, Xb), yb, reduction="none")

    def init(seed):
        g = torch.Generator().manual_seed(seed)
        parts = []
        for k, s in shapes:
            if k.startswith("W"):
                std = math.sqrt((2.0 if int(k[1:]) < len(dims) - 2 else 1.0) / s[0])
                parts.append(torch.randn(s, generator=g).flatten() * std)
            else:
                parts.append(torch.zeros(s).flatten())
        return torch.cat(parts)

    return dict(torch=torch, X=X, y=y, Xte=Xte, yte=yte, P=int(offs[-1]), groups=groups, logits=logits, losses=losses,
                init=init, n=len(X))


def err(E, th, X, y, bs=10000):
    with E["torch"].no_grad():
        return sum(int((E["logits"](th, X[i:i + bs]).argmax(1) != y[i:i + bs]).sum()) for i in range(0, len(X), bs)) / len(X)


def stage_train(a):
    E = setup(a)
    torch = E["torch"]
    n, B = E["n"], a.batch
    steps_per_epoch = n // B
    T = steps_per_epoch * a.epochs
    g = torch.Generator().manual_seed(a.seed + 1000)
    batches = np.concatenate([torch.randperm(n, generator=g)[:steps_per_epoch * B].view(steps_per_epoch, B).numpy()
                              for _ in range(a.epochs)])
    traj = np.lib.format.open_memmap(f"{a.dir}/traj.npy", mode="w+", dtype=np.float32, shape=(T + 1, E["P"]))
    th = E["init"](a.seed)
    traj[0] = th.numpy()
    gradfn = torch.func.grad(lambda th, Xb, yb: E["losses"](th, Xb, yb).mean())
    t0 = time.time()
    for t in range(T):
        idx = torch.from_numpy(batches[t])
        th = th - a.lr * gradfn(th, E["X"][idx], E["y"][idx])
        traj[t + 1] = th.numpy()
    traj.flush()
    np.save(f"{a.dir}/batches.npy", batches)
    meta = dict(vars(a), T=T, P=E["P"], n=n, groups=E["groups"], train01=err(E, th, E["X"], E["y"]),
                test01=err(E, th, E["Xte"], E["yte"]), train_seconds=time.time() - t0)
    json.dump(meta, open(f"{a.dir}/meta.json", "w"), indent=1)
    print("train done:", {k: meta[k] for k in ("T", "P", "train01", "test01", "train_seconds")}, flush=True)


def _adjoint_job(job):
    a, gname, k = job
    E = setup(a)
    torch = E["torch"]
    meta = json.load(open(f"{a.dir}/meta.json"))
    traj = np.load(f"{a.dir}/traj.npy", mmap_mode="r")
    batches = np.load(f"{a.dir}/batches.npy")
    lo, hi = meta["groups"][gname]
    g = torch.Generator().manual_seed(10_000 + 97 * k + sorted(meta["groups"]).index(gname))
    u = torch.zeros(meta["P"])
    u[lo:hi] = torch.randn(hi - lo, generator=g)
    acc = torch.zeros(E["n"], dtype=torch.float64)
    B, eta = a.batch, a.lr
    meanloss = lambda th, Xb, yb: E["losses"](th, Xb, yb).mean()
    gradfn = torch.func.grad(meanloss)
    unorm = []
    for t in range(meta["T"] - 1, -1, -1):
        th = torch.from_numpy(np.array(traj[t]))
        idx = torch.from_numpy(batches[t])
        Xb, yb = E["X"][idx], E["y"][idx]
        _, d = torch.func.jvp(lambda p: E["losses"](p, Xb, yb), (th,), (u,))      # u_{t+1} . grad l(W_t; z_j), j in B_t
        acc[idx] += (eta / B) * d.double()
        _, Hu = torch.func.jvp(lambda p: gradfn(p, Xb, yb), (th,), (u,))
        u = u - eta * Hu
        if t % 500 == 0:
            unorm.append((t, float(u.norm())))
    return dict(group=gname, probe=k, S=float((acc ** 2).sum()), a=acc.float().numpy().tolist() if a.save_a else None,
                u_norm=unorm, v_dim=hi - lo)


def stage_adjoint(a):
    meta = json.load(open(f"{a.dir}/meta.json"))
    jobs = [(a, gname, k) for gname in meta["groups"] for k in range(a.probes)]
    out = []
    with Pool(min(a.workers, len(jobs))) as pool:
        for r in pool.imap_unordered(_adjoint_job, jobs):
            out.append(r)
            print(f"adjoint {r['group']} probe {r['probe']}: sum_i a_i^2 = {r['S']:.4e}  |u| trace {r['u_norm'][:3]}...", flush=True)
    json.dump(out, open(f"{a.dir}/adjoint.json", "w"))


def _check_job(job):
    a, i = job
    E = setup(a)
    torch = E["torch"]
    meta = json.load(open(f"{a.dir}/meta.json"))
    traj = np.load(f"{a.dir}/traj.npy", mmap_mode="r")
    batches = np.load(f"{a.dir}/batches.npy")
    B, eta = a.batch, a.lr
    gsum = torch.func.grad(lambda th, Xb, yb: E["losses"](th, Xb, yb).sum())
    gw = torch.func.grad(lambda th, Xb, yb, w: (w * E["losses"](th, Xb, yb)).sum())
    gradfn = torch.func.grad(lambda th, Xb, yb: E["losses"](th, Xb, yb).mean())
    th_loo = torch.from_numpy(np.array(traj[0])).double()     # float64 replay of the original run, example i weighted 1-eps
    th_ref = th_loo.clone()                                     # float64 replay of the original run (eps = 0)
    eps = a.check_eps
    delta = torch.zeros(meta["P"])                              # linearised W(S) - W(S \ z_i)
    for t in range(meta["T"]):
        idx = batches[t]
        th = torch.from_numpy(np.array(traj[t]))
        Xb, yb = E["X"][torch.from_numpy(idx)], E["y"][torch.from_numpy(idx)]
        _, Hd = torch.func.jvp(lambda p: gradfn(p, Xb, yb), (th,), (delta,))
        delta = delta - eta * Hd
        w = torch.ones(len(idx), dtype=torch.float64)
        if (idx == i).any():
            delta = delta - (eta / B) * gsum(th, E["X"][i:i + 1], E["y"][i:i + 1])
            w[torch.from_numpy(idx == i)] = 1 - eps
        Xd, yd = Xb.double(), yb
        th_ref = th_ref - (eta / B) * gw(th_ref, Xd, yd, torch.ones_like(w))
        th_loo = th_loo - (eta / B) * gw(th_loo, Xd, yd, w)       # same batches, same divisor B
    exact = ((th_ref - th_loo) / eps).float()                   # -> linearised delta as eps -> 0
    res = dict(i=int(i), eps=eps, n_used=int(sum((batches == i).any(1))),
               replay_drift=float((th_ref.float() - torch.from_numpy(np.array(traj[-1]))).norm()))
    for gname, (lo, hi) in meta["groups"].items():
        res[gname] = dict(exact=float(exact[lo:hi].pow(2).sum()), linear=float(delta[lo:hi].pow(2).sum()),
                          cos=float(torch.nn.functional.cosine_similarity(exact[lo:hi], delta[lo:hi], dim=0)))
    return res


def stage_check(a):
    meta = json.load(open(f"{a.dir}/meta.json"))
    rng = np.random.default_rng(a.seed + 7)
    pts = rng.choice(meta["n"], a.check, replace=False)
    out = []
    with Pool(min(a.workers, len(pts))) as pool:
        for r in pool.imap_unordered(_check_job, [(a, int(i)) for i in pts]):
            out.append(r)
            print("check", r, flush=True)
    json.dump(out, open(f"{a.dir}/check.json", "w"))


def stage_spectrum(a):
    E = setup(a)
    torch = E["torch"]
    meta = json.load(open(f"{a.dir}/meta.json"))
    traj = np.load(f"{a.dir}/traj.npy", mmap_mode="r")
    m = min(1000, meta["n"])
    idx = torch.from_numpy(np.random.default_rng(a.seed + 11).choice(meta["n"], m, replace=False))
    pg = torch.func.vmap(torch.func.grad(lambda th, x, y: E["losses"](th, x[None], y[None])[0]), in_dims=(None, 0, 0))
    out = []
    thT = torch.from_numpy(np.array(traj[-1]))
    for t in np.linspace(0, meta["T"], 5).astype(int):
        th = torch.from_numpy(np.array(traj[t]))
        G = torch.cat([pg(th, E["X"][idx[j:j + 200]], E["y"][idx[j:j + 200]]) for j in range(0, m, 200)]).double()
        G = G - G.mean(0, keepdim=True)
        rec = dict(t=int(t))
        for c in a.cs:   # whiten by the per-layer sigma_g = c * rms(W_T[g]); lambda_k of the whitened covariance
            scale = torch.ones(meta["P"], dtype=torch.float64)
            for gname, (lo, hi) in meta["groups"].items():
                scale[lo:hi] = 1.0 / (c * float(thT[lo:hi].pow(2).mean().sqrt()))
            ev = torch.linalg.eigvalsh((G * scale) @ (G * scale).T / m).clamp(min=0).flip(0)
            rec[f"d_c{c}"] = float(torch.clamp(a.lr ** 2 * ev / a.batch, max=1.0).sum())
        ev = torch.linalg.eigvalsh(G @ G.T / m).clamp(min=0).flip(0)
        rec.update(trace=float(ev.sum()), eff_rank=float(ev.sum() ** 2 / (ev ** 2).sum()), top5=[float(e / ev.sum()) for e in ev[:5]])
        out.append(rec)
        print("spectrum", rec, flush=True)
    json.dump(out, open(f"{a.dir}/spectrum.json", "w"))


def stage_report(a):
    from tail_bounds import kl_inv
    E = setup(a)
    torch = E["torch"]
    meta = json.load(open(f"{a.dir}/meta.json"))
    traj = np.load(f"{a.dir}/traj.npy", mmap_mode="r")
    adj = json.load(open(f"{a.dir}/adjoint.json"))
    th0, thT = torch.from_numpy(np.array(traj[0])), torch.from_numpy(np.array(traj[-1]))
    n = meta["n"]
    S = {g: float(np.mean([r["S"] for r in adj if r["group"] == g])) for g in meta["groups"]}
    S_se = {g: float(np.std([r["S"] for r in adj if r["group"] == g]) / math.sqrt(a.probes)) for g in meta["groups"]}
    dist = {g: float((thT[lo:hi] - th0[lo:hi]).pow(2).sum()) for g, (lo, hi) in meta["groups"].items()}
    rms = {g: float(thT[lo:hi].pow(2).mean().sqrt()) for g, (lo, hi) in meta["groups"].items()}
    gsub = torch.Generator().manual_seed(5)
    tr_idx = torch.randperm(n, generator=gsub)[:10000]
    pb = math.log(2 * math.sqrt(n) / 0.025)
    rows = []
    for c in a.cs:
        sig = {g: c * rms[g] for g in meta["groups"]}
        kl_init = sum(dist[g] / (2 * sig[g] ** 2) for g in sig)
        i_infl = sum(S[g] / (2 * sig[g] ** 2) for g in sig)
        tr, te = [], []
        for s in range(a.flat_samples):
            xi = torch.zeros(meta["P"])
            for g, (lo, hi) in meta["groups"].items():
                xi[lo:hi] = sig[g] * torch.randn(hi - lo, generator=gsub)
            tr.append(err(E, thT + xi, E["X"][tr_idx], E["y"][tr_idx]))
            te.append(err(E, thT + xi, E["Xte"], E["yte"]))
        gtr, gte = float(np.mean(tr)), float(np.mean(te))
        rows.append(dict(c=c, KL_init=kl_init, I_infl=i_infl, ratio=kl_init / max(i_infl, 1e-30), gibbs_train=gtr,
                         gibbs_test=gte, flat_gap_train=gtr - meta["train01"], flat_gap_test=gte - meta["test01"],
                         plugin_init=kl_inv(gtr, (kl_init + pb) / n), plugin_infl=kl_inv(gtr, (i_infl + pb) / n)))
    rep = dict(meta={k: meta[k] for k in ("T", "P", "n", "train01", "test01", "lr", "batch", "epochs", "width", "depth")},
               S=S, S_se=S_se, dist=dist, rms=rms, rows=rows)
    if os.path.exists(f"{a.dir}/check.json"):
        chk = json.load(open(f"{a.dir}/check.json"))
        rep["check"] = chk
        if chk and chk[0]["eps"] == 1.0:   # unbiased MC estimate of sum_i ||Delta W_i||^2 from exact LOO retrains
            rep["S_exact"] = {g: n * float(np.mean([c[g]["exact"] for c in chk])) for g in meta["groups"]}
            rep["S_exact_se"] = {g: n * float(np.std([c[g]["exact"] for c in chk])) / math.sqrt(len(chk)) for g in meta["groups"]}
            rep["S_lin_sample"] = {g: n * float(np.mean([c[g]["linear"] for c in chk])) for g in meta["groups"]}
            for r in rows:
                sig = {g: r["c"] * rms[g] for g in meta["groups"]}
                r["I_exact"] = sum(rep["S_exact"][g] / (2 * sig[g] ** 2) for g in sig)
                r["plugin_exact"] = kl_inv(r["gibbs_train"], (r["I_exact"] + pb) / n)
    if os.path.exists(f"{a.dir}/spectrum.json"):
        rep["spectrum"] = json.load(open(f"{a.dir}/spectrum.json"))
    json.dump(rep, open(a.out, "w"), indent=1)
    print(json.dumps({k: rep[k] for k in ("meta", "S", "S_se", "dist", "rms", "S_exact", "S_exact_se", "S_lin_sample") if k in rep}, indent=1))
    print("| c | KL_init | I_lin | I_exact | Gibbs train | Gibbs test | flat gap train/test | plug-in init | plug-in lin | plug-in exact |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['c']} | {r['KL_init']:.3g} | {r['I_infl']:.3g} | {r.get('I_exact', float('nan')):.3g} | {r['gibbs_train']:.4f} | {r['gibbs_test']:.4f} | "
              f"{r['flat_gap_train']:+.4f}/{r['flat_gap_test']:+.4f} | {r['plugin_init']:.4f} | {r['plugin_infl']:.4f} | {r.get('plugin_exact', float('nan')):.4f} |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--stages", default="train,adjoint,check,spectrum,report")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probes", type=int, default=8)
    ap.add_argument("--check", type=int, default=6)
    ap.add_argument("--flat_samples", type=int, default=8)
    ap.add_argument("--cs", type=lambda s: [float(x) for x in s.split(",")], default=[0.01, 0.03, 0.1, 0.3, 1.0])
    ap.add_argument("--save_a", action="store_true")
    ap.add_argument("--check_eps", type=float, default=1.0, help="down-weight (not remove) the checked point by eps")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.dir, exist_ok=True)
    a.out = a.out or f"{a.dir}/report.json"
    for st in a.stages.split(","):
        t0 = time.time()
        globals()[f"stage_{st}"](a)
        print(f"[{st} {time.time() - t0:.0f}s]", flush=True)
