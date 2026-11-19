"""Mean ± sd over seeds for SGD-posterior runs (pacbayes_sgd.py --grid custom) and variational runs (pacbayes_norm.py).
Usage: python experiments/summarize_seeds.py sgd results/v2/sgd_r1_wdsweep.jsonl [more files]
       python experiments/summarize_seeds.py norm results/v2/pacbayes_norm.jsonl results/v2/pacbayes_norm_seeds.jsonl"""
import json, sys
from collections import defaultdict

import numpy as np


def ms(v):
    v = [x for x in v if x is not None]
    if not v:
        return "—"
    return f"{np.mean(v):.3f} ± {np.std(v):.3f}" if len(v) > 1 else f"{v[0]:.3f}"


kind, files = sys.argv[1], sys.argv[2:]
import os
rows = [json.loads(l) for f in files if os.path.exists(f) for l in open(f)]
bad = [r for r in rows if r.get("diverged") or r.get("error")]
if bad:
    print(f"({len(bad)} diverged/failed runs excluded: " + ", ".join(
        f"{r['norm']} w{r['width']} lr{r.get('lr')} B{r.get('batch')} wd{r.get('wd_init')} s{r['seed']}" for r in bad) + ")\n")
rows = [r for r in rows if not (r.get("diverged") or r.get("error"))]
g = defaultdict(list)
if kind == "sgd":
    for r in rows:
        g[(r["norm"], r["width"], r["lr"], r["batch"], r["wd_init"], r.get("objective", "mcallester"))].append(r)
    print("| norm | width | lr | batch | L2-to-init | objective | seeds | ‖θ̄−m₀‖ | SWA test | cert SGD posterior | cert isotropic | "
          "**cert hybrid** | hybrid Gibbs test | hybrid KL |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k in sorted(g):
        rs = g[k]
        get = lambda f: [f(r) for r in rs]
        c = lambda key, fld: get(lambda r: (r.get(key) or {}).get(fld))
        print(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {k[4]} | {k[5]} | {len(rs)} | {ms(get(lambda r: r['dist_init']))} | "
              f"{ms(get(lambda r: r['test_err_swa']))} | {ms(c('cert', 'cert01'))} | {ms(c('cert_iso', 'cert01'))} | "
              f"{ms(c('cert_ft', 'cert01'))} | {ms(c('cert_ft', 'test01'))} | {ms(c('cert_ft', 'KL'))} |")
else:
    for r in rows:
        if r.get("kind") != "pacbayes":
            continue
        g[(r["norm"], r["mode"], r["width"], r["s0"], r.get("objective", "mcallester"))].append(r)
    print("| norm | mode | width | σ₀ | objective | seeds | cert | Gibbs test | KL |\n|---|---|---|---|---|---|---|---|---|")
    for k in sorted(g):
        rs = g[k]
        print(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {k[4]} | {len(rs)} | {ms([r['trained']['cert01'] for r in rs])} | "
              f"{ms([r['trained']['test01'] for r in rs])} | {ms([r['trained']['KL'] for r in rs])} |")
