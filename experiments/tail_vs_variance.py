"""Is the lower risk tail (best-of-K untrained acc) determined by the variance? Usage: python experiments/tail_vs_variance.py"""
import json, sys, numpy as np
from scipy.stats import spearmanr, rankdata, beta as betad
sys.path.insert(0, "experiments")
from analyze import partial_spearman
for f in ["grid_full", "grid_n200", "gain_sweep", "fashion_full", "fashion_n1000"]:
    rows = [json.loads(l) for l in open(f"results/v2/{f}.jsonl")]
    g = lambda k: np.array([r["proxies"][k] for r in rows])
    acc, sd, best, y = g("untrained_acc"), g("acc_std_init"), g("acc_best"), np.array([r["trained"]["test_acc"] for r in rows])
    K = rows[0]["K"]
    # tail predicted from (mean, var) alone: expected max of K draws from a Beta with same mean/var
    m, v = acc, np.maximum(sd**2, 1e-10)
    k = m*(1-m)/v - 1
    a, b = np.maximum(m*k, 1e-3), np.maximum((1-m)*k, 1e-3)
    pred = betad.ppf(1 - 1/(K+1), a, b)
    z = (best - acc) / np.maximum(sd, 1e-9)       # standardized max: shape beyond variance
    print(f"{f:14s} mean acc: median {np.median(acc):.4f}, |acc-0.1|>3SE in {np.mean(np.abs(acc-0.1) > 3*sd/np.sqrt(K)+1e-9):.0%} | "
          f"rho(best,std)={spearmanr(best, sd)[0]:+.2f} rho(best,beta-pred)={spearmanr(best, pred)[0]:+.2f} | "
          f"z=(best-mean)/sd: median {np.median(z):.2f} IQR [{np.percentile(z,25):.2f},{np.percentile(z,75):.2f}] | "
          f"test-acc partial rho: best|std {partial_spearman(best, y, sd):+.2f}, std|best {partial_spearman(sd, y, best):+.2f}, z {spearmanr(z, y)[0]:+.2f}")
