import json, numpy as np
from collections import defaultdict
rows=[json.loads(l) for l in open("results/v2/online_bounds.jsonl")]
g=defaultdict(list)
for r in rows: g[(r["norm"],r["width"],r["lr"],r["batch"])].append(r)
ms=lambda v: f"{np.mean(v):.3f} ± {np.std(v):.3f}"
print("| norm | width | lr | batch | final iterate test | prog. val. cert (best window) | window | prequential | probe test in window | OPBTrain best cert | its σ | energy/n at that σ |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for k in sorted(g):
    rs=g[k]
    pv=[min(r["prog"],key=lambda x:x["cert01"]) for r in rs]
    ob=[min(r["opb"],key=lambda x:x["cert01"]) for r in rs]
    print(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {ms([r['final_iterate_test'] for r in rs])} | {ms([p['cert01'] for p in pv])} | "
          f"{sorted(set(p['f'] for p in pv))} | {ms([p['prequential'] for p in pv])} | {ms([p['window_test_probe'] for p in pv])} | "
          f"{ms([o['cert01'] for o in ob])} | {sorted(set(o['sigma'] for o in ob))} | {ms([o['energy_per_n'] for o in ob])} |")
best=min(rows,key=lambda r:min(p["cert01"] for p in r["prog"]))
print("\nOPBTrain per sigma (best config by prog.val.):", [(o["sigma"],round(o["energy_per_n"],3),round(o["noisy_post_err"],3),round(o["cert01"],3)) for o in best["opb"]])
