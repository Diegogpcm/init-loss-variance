"""Summarize results/v2/pacbayes_sgd.jsonl. Usage: python experiments/summarize_sgd.py [file]"""
import json, sys

# later files override earlier ones for the same configuration (e.g. the bn_stat rerun with its own stats width)
paths = sys.argv[1:] or ["results/v2/pacbayes_sgd.jsonl", "results/v2/pacbayes_sgd_bnfix.jsonl"]
cfgs = {}
for path in paths:
    try:
        for l in open(path):
            r = json.loads(l)
            cfgs[(r["norm"], r["width"], r["lr"], r["batch"], r["wd_init"])] = r
    except FileNotFoundError:
        pass
rows = list(cfgs.values())


def f(x, p=4):
    return "—" if x is None else f"{x:.{p}f}"


def line(r):
    c, ci = r.get("cert") or {}, r.get("cert_iso") or {}
    sub = (f"K={c['K']}, α={c['alpha']}, τ={c['tau']}, σ₀={c['s0']}" + (f", τ_s={c['tau_s']}" if r["norm"] == "bn_stat"
           and "tau_s" in c else "") if c else "—")
    kl = (f"{c['KL']:.0f} (mean {c['KL_mean']:.0f}, sub {c['KL_sub']:.0f}, compl. {c['KL_comp']:.0f})" if c else "—")
    return (f"| {r['norm']} | {r['width']} | {r['lr']} | {r['batch']} | {r['wd_init']} | {f(r['test_err_swa'])} | "
            f"{r['dist_init']:.1f} | {f(c.get('test01'))} | {f(c.get('train01'))} | {f(c.get('cert01'))} | "
            f"{f(ci.get('cert01'))} | {sub} | {kl} |")


hdr = ("| norm | width | lr | batch | L2-to-init | SWA test err | ‖θ̄−m₀‖ | Gibbs test err | Gibbs train err | "
       "**cert** | cert (isotropic, K=0) | chosen posterior | KL (nats) |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|")
main = [r for r in rows if r["lr"] == 0.05 and r["batch"] == 128]
algo = [r for r in rows if r["width"] == 128 and r["wd_init"] == 1e-3]
print("### Main grid (lr 0.05, batch 128)\n")
print(hdr)
for r in sorted(main, key=lambda r: (r["norm"], r["width"], r["wd_init"])):
    print(line(r))
print("\n### SGD temperature: lr × batch (width 128, L2-to-init 1e-3)\n")
print(hdr)
for r in sorted(algo, key=lambda r: (r["norm"], r["lr"], r["batch"])):
    print(line(r))
