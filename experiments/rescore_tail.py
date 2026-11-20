"""Rescore DP-SGD runs with the L-CL-Z Thm 8 tail bound (experiments/tail_bounds.py) instead of Markov.

kappa depends only on (n, q, sigma, T); beta = 0.024 for the tail, d2 = 0.001 for disintegrated PAC-Bayes (total 0.025,
as before). Both iterates (final, averaged) are functions of the same trajectory, so ONE tail event covers both and
best-of-2 costs only d2 -> d2/2 in the PAC-Bayes term.
Usage: python experiments/rescore_tail.py results/v2/dp_*.jsonl  [--top 25]
"""
import json, math, sys, functools
sys.path.insert(0, "experiments")
from tail_bounds import lz8_kappa, eps_rep_conv, kl_inv

BETA, D2 = 0.024, 0.001
ALPHAS = [k / 2 for k in range(3, 20001, 2)]   # 1.5, 2.5, ... (step 1)


@functools.lru_cache(None)
def kappa(n, q, sigma, T):
    k, info = lz8_kappa(n, lambda a: eps_rep_conv(q, sigma, T, a), BETA, ALPHAS)
    return k, info["alpha"], info["beta1"]


@functools.lru_cache(None)
def kappa_filtered(n, q, sigma, T0, alpha):
    """individual-filter runs: replace-one RDP is that of an unfiltered T0-step run, at the single order alpha."""
    k, info = lz8_kappa(n, lambda a: eps_rep_conv(q, sigma, T0, a), BETA, [alpha])
    return k, alpha, info["beta1"]


def rescore(r):
    if r.get("filter_alpha"):
        k, a, b1 = kappa_filtered(r["n"], round(r["q"], 12), round(r["sigma"], 9), r["filter_T0"], r["filter_alpha"])
    else:
        k, a, b1 = kappa(r["n"], round(r["q"], 12), round(r["sigma"], 9), r["T"])
    n = r["n"]
    c1 = (k + math.log(2 * math.sqrt(n) / D2)) / n
    c2 = (k + math.log(2 * math.sqrt(n) / (D2 / 2))) / n
    r.update(kappa_lz8=k, lz8_alpha=a, lz8_beta1=b1, cert01_lz8=kl_inv(r["train01"], c1),
             cert01_lz8_best2=min(kl_inv(r["train01"], c2), kl_inv(r["train_avg"], c2)) if "train_avg" in r else None)
    return r


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 25
    rows = []
    for f in args:
        if f.endswith(".jsonl"):
            for line in open(f):
                r = json.loads(line)
                if "cert01" in r:
                    r["file"] = f.split("/")[-1]
                    rows.append(rescore(r))
    rows.sort(key=lambda r: r["cert01_lz8"])
    print("| file | feat | d | w | B | nu | ep | lr | seed | train | test | I | kappa_M | kappa_LZ8 | cert Markov | cert LZ8 | best2 LZ8 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows[:top]:
        kM = (r["I_ub"] + 1 / math.e) / BETA
        print(f"| {r['file']} | {r.get('features','pix')} | {r['depth']} | {r['width']} | {r['batch']} | {r['nu']} | {r['epochs']} | {r['lr']} | {r['seed']} | "
              f"{r['train01']:.4f} | {r['test01']:.4f} | {r['I_ub']:.1f} | {kM:.0f} | {r['kappa_lz8']:.0f} | {r['cert01']:.4f} | {r['cert01_lz8']:.4f} | "
              f"{(r['cert01_lz8_best2'] or float('nan')):.4f} |")
