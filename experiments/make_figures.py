"""Figures for the README. Usage: python experiments/make_figures.py"""
import json, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import table, load  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
RES = os.path.join(ROOT, "results")
os.makedirs(FIG, exist_ok=True)

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
ACTS = ["relu", "tanh", "sigmoid", "linear", "gelu", "sin"]
MARK = dict(zip(ACTS, ["o", "s", "^", "D", "v", "P"]))
COL = dict(zip(ACTS, CAT))

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "legend.frameon": False,
})

SHOW = [("var_init", "Var$_{init}$ (raw)"), ("var_data", "Var$_{data}$ (raw)"),
        ("var_init_norm", "Var$_{init}$ (temp.-normalized)"), ("var_pointwise_norm", "Var$_{pointwise}$ (normalized)"),
        ("free_energy_1_norm", "Free energy, λ=1 (normalized)"), ("mean_loss", "Mean init loss"),
        ("cvu", "CVU (Gracheva 2021)"), ("zico", "ZiCo (simplified)"), ("grad_norm", "Grad norm"),
        ("naswot", "NASWOT"), ("log_n_params", "log #params")]


def fig_correlations():
    fig, ax = plt.subplots(figsize=(7, 4.6))
    series = [("sweep_full.jsonl", "full data (n=1347)", CAT[0]), ("sweep_n200.jsonl", "low data (n=200)", CAT[1])]
    h = 0.38
    ys = np.arange(len(SHOW))[::-1]
    for j, (f, lab, c) in enumerate(series):
        out, _ = table(os.path.join(RES, f))
        d = {o["proxy"]: o for o in out}
        r = np.array([d[k]["rho_all"] for k, _ in SHOW])
        lo = np.array([d[k]["ci_lo"] for k, _ in SHOW])
        hi = np.array([d[k]["ci_hi"] for k, _ in SHOW])
        yy = ys + (h / 2 if j == 0 else -h / 2)
        ax.barh(yy, r, height=h - 0.04, color=c, label=lab, zorder=2)
        ax.errorbar(r, yy, xerr=[r - lo, hi - r], fmt="none", ecolor=INK2, elinewidth=0.8, capsize=0, zorder=3)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(ys, [l for _, l in SHOW])
    ax.set_xlabel("Spearman ρ with trained test accuracy (95% bootstrap CI)")
    ax.set_title("Init-time proxies vs. trained accuracy, 150 random MLPs on digits", loc="left", fontsize=10)
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "proxy_correlations.png"), dpi=160)


def fig_scatter():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    rows, P, T, acts = load(os.path.join(RES, "sweep_n200.jsonl"))
    for ax, (k, lab, logx) in zip(axes, [("var_init", "Var$_{init}$ of raw init loss", True),
                                         ("var_init_norm", "Var$_{init}$ after temperature normalization", True)]):
        for a in ACTS:
            m = acts == a
            ax.scatter(P[k][m], T["test_acc"][m], s=22, marker=MARK[a], color=COL[a], label=a,
                       edgecolor=SURF, linewidth=0.6, zorder=2)
        if logx:
            ax.set_xscale("log")
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel(lab)
    axes[0].set_ylabel("test accuracy (mean of 3 seeds)")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, title="activation",
                   title_fontsize=8)
    fig.suptitle("Low-data regime (n=200): raw variance mostly tracks logit scale; normalized variance tracks accuracy",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "scatter_n200.png"), dpi=160)


def fig_gain():
    path = os.path.join(RES, "gain_sweep.jsonl")
    if not os.path.exists(path):
        return
    rows = [json.loads(l) for l in open(path)]
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.4), sharex=True)
    for col, res in enumerate([False, True]):
        for a in ACTS:
            rr = sorted([r for r in rows if r["arch"]["act"] == a and r["arch"]["residual"] == res],
                        key=lambda r: r["arch"]["gain"])
            if not rr:
                continue
            g = [r["arch"]["gain"] for r in rr]
            kw = dict(color=COL[a], marker=MARK[a], ms=4, lw=1.6, label=a)
            axes[0, col].plot(g, [r["trained"]["test_acc"] for r in rr], **kw)
            axes[1, col].plot(g, [r["proxies"]["var_init_norm"] for r in rr], **kw)
        axes[0, col].set_title(("residual" if res else "plain") + " MLP, depth 8, width 64", loc="left", fontsize=9)
        axes[1, col].set_xlabel("init gain (σ_w · √fan_in)")
        axes[1, col].set_yscale("log")
    axes[0, 0].set_ylabel("test accuracy")
    axes[1, 0].set_ylabel("Var$_{init}$ (normalized)")
    hs, ls = axes[0, 0].get_legend_handles_labels()
    fig.legend(hs, ls, loc="lower center", ncol=6, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(FIG, "gain_sweep.png"), dpi=160)


if __name__ == "__main__":
    fig_correlations()
    fig_scatter()
    fig_gain()
    print("figures written to", FIG)
