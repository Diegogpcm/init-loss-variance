"""Summarize results/v2/pacbayes_norm.jsonl and norm_deterministic.jsonl.
Usage: python experiments/summarize_norm.py"""
import json, os
from collections import defaultdict

import numpy as np

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "v2")

det = [json.loads(l) for l in open(os.path.join(RES, "norm_deterministic.jsonl"))]
g = defaultdict(list)
for r in det:
    g[(r["norm"], r["width"])].append(r["test_err"])
print("### Ordinary training (Adam, 20 epochs, 3 seeds): test error\n")
print("| norm | width 64 | width 128 |\n|---|---|---|")
for nm in ("none", "ln", "bn", "bn_ref"):
    cells = [f"{np.mean(g[(nm, w)]):.4f} ± {np.std(g[(nm, w)]):.4f}" for w in (64, 128)]
    print(f"| {nm} | " + " | ".join(cells) + " |")

pb = [json.loads(l) for f in ("pacbayes_norm.jsonl", "pacbayes_bnstat.jsonl") if os.path.exists(os.path.join(RES, f))
      for l in open(os.path.join(RES, f))]
ORDER = ["none", "ln", "bn_ref", "bn_param", "bn_stat"]
pb.sort(key=lambda r: (r["s0"], r["width"], ORDER.index(r["norm"]), r["mode"]))
print("\n### PAC-Bayes (Fashion-MNIST, n = 55k): 0-1 certificate\n")
print("| σ₀ | width | norm | mode | test err (Gibbs) | KL | cert | KL after post-hoc rescaling | cert after |")
print("|---|---|---|---|---|---|---|---|---|")
for r in pb:
    tr, ph = r["trained"], r.get("posthoc")
    # stats KL of mode=inv runs in the first bn_stat batch was computed on unrescaled parameters (fixed since)
    extra = f" (stats {r['KL_stats']:.0f})" if "KL_stats" in r and r["mode"] == "plain" else ""
    print(f"| {r['s0']} | {r['width']} | {r['norm']} | {r['mode']} | {tr['test01']:.4f} | {tr['KL']:.0f}{extra} | "
          f"{tr['cert01']:.4f} | {ph['KL']:.0f} | {ph['cert01']:.4f} |" if ph else
          f"| {r['s0']} | {r['width']} | {r['norm']} | {r['mode']} | {tr['test01']:.4f} | {tr['KL']:.0f}{extra} | "
          f"{tr['cert01']:.4f} | — | — |")
