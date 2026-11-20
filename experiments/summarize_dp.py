"""Summarize DP-SGD / mutual-information certificates (experiments/dp_sgd_bound.py).
Usage: python experiments/summarize_dp.py results/v2/dp_sgd_r1.jsonl [more files] [--top 15]"""
import json, sys

args = sys.argv[1:]
top = 15
if "--top" in args:
    i = args.index("--top"); top = int(args[i + 1]); args = args[:i] + args[i + 2:]
rows = []
for f in args:
    for l in open(f):
        r = json.loads(l)
        if "cert01" in r:
            r["file"] = f.split("/")[-1].replace(".jsonl", "")
            rows.append(r)
rows.sort(key=lambda r: r["cert01"])
print("| file | features | depth | norm | width | params | B | ν | T | lr | sched | train | test | I ≤ (nats) | **cert** | "
      "no-Markov (not a cert.) |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows[:top]:
    print(f"| {r['file']} | {r.get('features', 'pixels')} | {r['depth']} | {r['norm']} | {r['width'] if r['depth'] else '—'} | "
          f"{r.get('n_params', '')} | {r['batch']} | {r['nu']} | {r['T']} | {r['lr']} | {r.get('sched', 'const')} | "
          f"{r['train01']:.4f} | {r['test01']:.4f} | {r['I_ub']:.1f} | **{r['cert01']:.4f}** | {r['exp_cert01']:.4f} |")
