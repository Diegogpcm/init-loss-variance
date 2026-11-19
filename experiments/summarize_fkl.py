"""Summarize results/v2/function_space_kl.jsonl. Usage: python experiments/summarize_fkl.py [file]"""
import json, sys

path = sys.argv[1] if len(sys.argv) > 1 else "results/v2/function_space_kl.jsonl"
rows = sorted((json.loads(l) for l in open(path)),
              key=lambda r: (r["s0"], r["act"], r["depth"], r["width"]))
print("| arch | σ₀ | #params | test err (Q) | cert (0-1) | KL_param (mean term) | KL_m linear: m=10 / 40 / max m | "
      "KL_m Monte Carlo, m=10 | MC m=10 / KL_param |")
print("|---|---|---|---|---|---|---|---|---|")
for r in rows:
    fn = {f["m"]: f for f in r["fn"]}
    last = r["fn"][-1]
    lin = " / ".join(f"{fn[m]['KL_lin']:.0f}" if m in fn else "—" for m in (10, 40)) + f" / {last['KL_lin']:.0f} (m={last['m']})"
    mc = fn.get(10, {}).get("KL_mc", float("nan"))
    print(f"| d{r['depth']}-w{r['width']}-{r['act']} | {r['s0']} | {r['n_params']:,} | {r['test01_Q']:.3f} | "
          f"{r['cert01']:.3f} | {r['KL_param']:.0f} ({r['KL_param_mean_term'] / r['KL_param']:.0%}) | {lin} | "
          f"{mc:.0f} | {mc / r['KL_param']:.0%} |")
