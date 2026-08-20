#!/usr/bin/env python3
"""Generate camera-quality paper figures from the experiment artifacts.

Design follows the dataviz skill's transferable rules for static print figures:
- categorical colors in a FIXED order from the Okabe-Ito colorblind-safe palette;
- one axis per chart (never dual-axis); thin marks; recessive grid; legend + a few
  direct labels; text in ink (not series color).

Writes PDF (for LaTeX) + PNG (preview) to paper/figs/. Needs matplotlib + the JSON
artifacts in experiments/. Usage: python3 scripts/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
OUT = ROOT / "paper" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

# Okabe-Ito CVD-safe categorical palette (fixed order — assign by entity, never cycle).
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "vermillion": "#D55E00",
      "skyblue": "#56B4E9", "yellow": "#F0E442", "purple": "#CC79A7", "black": "#000000"}
INK, MUTED = "#222222", "#666666"

mpl.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "figure.dpi": 150, "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "grid.color": "#DDDDDD", "grid.linewidth": 0.6, "legend.frameon": False,
    "axes.axisbelow": True,   # gridlines behind bars/marks, never showing through
})


def _load(name):
    p = EXP / name
    return json.loads(p.read_text()) if p.exists() else None


def _save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}")
    plt.close(fig)
    print(f"wrote paper/figs/{stem}.pdf|.png")


def fig_budget_frontier():
    d = _load("m9_budget_frontier.json")
    if not d:
        return
    fr = d["frontier"]
    xs = [p["tokens_used"] / 1e6 for p in fr]
    ys = [p["dollar_cost"] for p in fr]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.plot(xs, ys, "-o", color=OI["blue"], lw=2, ms=5, label="STOA (budget-conditioned)")
    st = d["static_baselines"]
    for key, name, col in [("all_plaintext", "all-plaintext", OI["vermillion"]),
                           ("all_vector", "all-vector", OI["orange"]),
                           ("all_latent", "all-latent", OI["green"])]:
        s = st[key]
        ax.plot(s["tokens_used"] / 1e6, s["dollar_cost"], "s", color=col, ms=6)
        ax.annotate(name, (s["tokens_used"] / 1e6, s["dollar_cost"]), textcoords="offset points",
                    xytext=(6, 3), fontsize=7, color=INK)
    ax.set_xlabel("context tokens used (millions)")
    ax.set_ylabel("consolidation cost (USD)")
    ax.set_title("Budget-conditioned placement frontier (RQ3)")
    ax.grid(True, axis="both")
    ax.legend(loc="upper right", fontsize=7)
    _save(fig, "fig_budget_frontier")


def fig_locomo(name="eval_locomo.json", stem="fig_locomo"):
    d = _load(name)
    if not d:
        return
    def series(rows):
        return [r["budget_tokens"] for r in rows], [r["accuracy"] * 100 for r in rows]
    sx, sy = series(d["stoa"])
    rx, ry = series(d["random"])
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.plot(sx, sy, "-o", color=OI["blue"], lw=2, ms=5, label="STOA (evidence-demand)")
    ax.plot(rx, ry, "--s", color=OI["orange"], lw=2, ms=5, label="random selection")
    ax.set_xlabel("context-token budget")
    ax.set_ylabel("QA accuracy (%)")
    metric = d.get("config", {}).get("metric", "substring")
    ax.set_title(f"LoCoMo accuracy vs budget ({metric})")
    ax.set_ylim(-3, 103)
    ax.grid(True, axis="y")
    ax.legend(loc="lower right", fontsize=7)
    _save(fig, stem)


def fig_tier_ablation():
    d = _load("m3_ablation.json")
    if not d:
        return
    tier = d["tier_axis_online"]
    fracs = sorted(tier.keys(), key=float)
    policies = [("belady", "Belady (oracle)", OI["blue"]), ("h2o", "H2O/LFU", OI["orange"]),
                ("lru", "LRU", OI["green"]), ("static", "static", OI["vermillion"])]
    import numpy as np
    x = np.arange(len(fracs))
    w = 0.2
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for i, (key, name, col) in enumerate(policies):
        vals = [tier[f][key]["hit_rate"] * 100 for f in fracs]
        ax.bar(x + (i - 1.5) * w, vals, w, color=col, label=name, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{float(f):.0%}" for f in fracs])
    ax.set_xlabel("hot-tier cache size (fraction of items)")
    ax.set_ylabel("hit rate (%)")
    ax.set_title("Tier-axis eviction ablation (RQ1)")
    ax.grid(True, axis="y")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    _save(fig, "fig_tier_ablation")


def fig_locomo_final(name="eval_locomo_final.json", stem="fig_locomo_final"):
    """LoCoMo accuracy vs budget FRACTION, averaged over samples (schema: lists of floats)."""
    d = _load(name)
    if not d or "budget_fractions" not in d:
        return
    xs = [f * 100 for f in d["budget_fractions"]]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.plot(xs, [a * 100 for a in d["stoa"]], "-o", color=OI["blue"], lw=2, ms=5,
            label="STOA (evidence-demand)")
    ax.plot(xs, [a * 100 for a in d["random"]], "--s", color=OI["orange"], lw=2, ms=5,
            label="random selection")
    cfg = d.get("config", {})
    ax.set_xlabel("context-token budget (% of full)")
    ax.set_ylabel("QA accuracy (%)")
    ax.set_title(f"LoCoMo ({cfg.get('samples', '?')} dialogues, {cfg.get('metric', 'LLM-judge')})")
    ax.set_ylim(-3, 103)
    ax.grid(True, axis="y")
    ax.legend(loc="upper left", fontsize=7)
    _save(fig, stem)


def fig_locomo_baselines():
    """Leak-free LoCoMo: STOA vs random/centrality (query-agnostic) vs retrieval (query-aware)."""
    d = _load("eval_locomo_leakfree.json")
    if not d:
        return
    xs = [f * 100 for f in d["budget_fractions"]]
    r = d["results"]
    series = [("retrieval", "per-query retrieval (query-aware)", OI["vermillion"], "-^"),
              ("stoa", "STOA demand (placement)", OI["blue"], "-o"),
              ("random", "random", OI["orange"], "--s"),
              ("centrality", "embedding centrality", OI["green"], "--d")]
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    for key, label, col, style in series:
        if key in r:
            ax.plot(xs, [a * 100 for a in r[key]], style, color=col, lw=2, ms=5, label=label)
    ax.set_xlabel("context-token budget (% of full)")
    ax.set_ylabel("QA accuracy (%)")
    ax.set_title("LoCoMo, leak-free protocol")
    ax.set_ylim(-3, 60)
    ax.grid(True, axis="y")
    ax.legend(loc="upper left", fontsize=6.2)
    _save(fig, "fig_locomo_baselines")


def fig_learnability():
    """The paper's headline: what each feature set buys, on each source."""
    d = _load("kvct_placement_summary.json")
    if not d:
        return
    import numpy as np
    runs = d["runs"]
    feats = ["block", "block+session", "block+session+think"]
    labels = ["block", "+session", "+think time"]
    x = np.arange(len(feats)); w = 0.35
    fig, ax = plt.subplots(figsize=(3.7, 2.7))
    for i, run in enumerate(runs):
        by = {r["features"]: r for r in run["results"]}
        # report the better of the two belief constructions; the spread is the error bar
        lo = [min(by[f]["cls"], by[f]["regr"]) for f in feats]
        hi = [max(by[f]["cls"], by[f]["regr"]) for f in feats]
        mid = [(a + b) / 2 for a, b in zip(lo, hi)]
        err = [[m - a for m, a in zip(mid, lo)], [b - m for m, b in zip(mid, hi)]]
        ax.bar(x + (i - 0.5) * w, mid, w, yerr=err, capsize=3,
               color=[OI["blue"], OI["skyblue"]][i], edgecolor="white", linewidth=0.5,
               label=f'{run["convs"]} conversations', error_kw={"lw": 1, "ecolor": MUTED})
    ax.axhline(0, color=MUTED, lw=1)
    ax.annotate("Mooncake: $\\approx$0% for every feature set", (-0.45, -108), fontsize=6.3,
                color=OI["vermillion"])
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("placement headroom\ncaptured (%)")
    ax.set_title("Request timing is what makes it learnable")
    ax.set_ylim(-140, 110)
    ax.grid(True, axis="y"); ax.legend(loc="lower right", fontsize=6.5)
    _save(fig, "fig_learnability")


def fig_reactive():
    """Reactive tiering on production traces: hit rate vs Belady, by workload.

    Uses the FULL traces; an 800-request subsample gave different values and reversed the
    trend in tier size, so the subsample artifact is deliberately not the figure's source.
    """
    d = _load("reactive_real_full_lrb.json") or _load("reactive_real_full.json")
    if not d:
        return
    import numpy as np
    tr = d["traces"]
    names = list(tr)
    fracs = [r["cache_frac"] for r in tr[names[0]]["rows"]]
    x = np.arange(len(fracs)); w = 0.35
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    cols = {"toolagent": OI["blue"], "conversation": OI["orange"]}
    # The bar is an oracle OVER policies, so its label must say how many it picks from --
    # a run with LRB enabled picks from five, not four, and mislabelling it would overstate
    # what a single deployable policy achieves.
    n_pol = 5 if all("lrb_over_belady" in r for n in names for r in tr[n]["rows"]) else 4
    for i, n in enumerate(names):
        vals = [r["best_reactive_over_belady"] * 100 for r in tr[n]["rows"]]
        ax.bar(x + (i - 0.5) * w, vals, w, color=cols.get(n, OI["green"]),
               label=f"{n} (best-of-{n_pol} envelope)", edgecolor="white", linewidth=0.5)
        # ARC is one deployable policy, drawn as a mark on top of the envelope bar so the
        # reader can see how much of the per-operating-point oracle a single policy gets.
        arc = [r.get("arc_over_belady", 0) * 100 for r in tr[n]["rows"]]
        ax.plot(x + (i - 0.5) * w, arc, "D", color=INK, ms=3.5, ls="none",
                label="ARC (adaptive)" if i == 0 else None)
        if all("lrb_over_belady" in r for r in tr[n]["rows"]):
            lrb = [r["lrb_over_belady"] * 100 for r in tr[n]["rows"]]
            ax.plot(x + (i - 0.5) * w, lrb, "x", color=OI["vermillion"], ms=4.5, mew=1.2,
                    ls="none", label="LRB-style (learned)" if i == 0 else None)
    ax.axhline(100, color=MUTED, lw=1, ls=":")
    ax.annotate("Belady (offline optimum)", (len(fracs) - 0.6, 101), fontsize=6, color=MUTED)
    ax.set_xticks(x); ax.set_xticklabels([f"{f:.0%}" for f in fracs])
    ax.set_xlabel("hot-tier size (fraction of blocks)")
    ax.set_ylabel("reactive hit rate\n(% of Belady)")
    ax.set_ylim(0, 115)
    ax.set_title("Reaction improves with tier size")
    ax.set_ylim(0, 140)          # headroom for the legend, which otherwise covers the 2% bars
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(True, axis="y")
    ax.legend(loc="upper left", fontsize=6.5, ncol=1, handletextpad=0.5, labelspacing=0.3)
    _save(fig, "fig_reactive")


def fig_capacity_ladder():
    """The decoupling, across model class: capacity moves AUC and not placement.

    One axis, one message -- captured headroom against ranking quality, with the true-future
    belief drawn as the ceiling. Rungs are labeled directly rather than by a color legend,
    because the reader needs to know WHICH model each point is, not which series.
    """
    d = _load("capacity_ladder.json")
    if not d:
        return
    rows = d["results"]
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    marks = {"conversation": ("o", OI["orange"]), "toolagent": ("s", OI["blue"])}
    for trace, (mk, col) in marks.items():
        pts = [r for r in rows if r["eval"] == trace and r["rung"] != "oracle"]
        ax.plot([r["auc_cross_trace"] for r in pts], [r["captured_pct"] for r in pts],
                mk, color=col, ms=5, ls="none", label=trace)
        # Label only the two floors individually. The three learned rungs land on top of one
        # another -- that coincidence IS the finding, so they get one bracket, not three labels
        # fighting for the same few pixels.
        for r in pts:
            if r["rung"] in ("constant", "count"):
                ax.annotate(r["rung"], (r["auc_cross_trace"], r["captured_pct"]),
                            textcoords="offset points", xytext=(5, -8), fontsize=6, color=INK)

    # All six learned points (3 model classes x 2 traces) fall in a band ~2 points wide.
    # Drawing the band and labelling it once states the finding as geometry; six separate
    # labels would collide and say less.
    learned = [r["captured_pct"] for r in rows if r["rung"] in ("linear", "gbdt", "mlp")]
    if learned:
        ax.axhspan(min(learned) - 0.4, max(learned) + 0.4, color=MUTED, alpha=0.13, zorder=0)
        ax.annotate("every learned model, both traces:\nlinear, GBDT and MLP alike",
                    (0.47, max(learned) + 3), fontsize=6, color=INK, va="bottom")
    ax.plot([1.0], [100.0], "*", color=OI["green"], ms=11)
    ax.annotate("true future counts\n(the ceiling)", (1.0, 100.0), textcoords="offset points",
                xytext=(-8, -14), fontsize=6, color=INK, ha="right")
    ax.axhline(0, color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel("ranking quality (cross-trace AUC)")
    ax.set_ylabel("headroom captured (%)")
    ax.set_title("Capacity buys ranking, not placement")
    ax.set_xlim(0.45, 1.08)
    ax.set_ylim(-35, 118)
    ax.grid(True, axis="y")
    ax.legend(loc="upper left", fontsize=6.5)
    _save(fig, "fig_capacity_ladder")


def fig_sequential():
    d = _load("seq_placement.json")
    if not d:
        return
    r = d["results"]
    rows = [("random-feasible", "random_feasible", OI["skyblue"]),
            ("static BC", "static_bc", OI["vermillion"]),
            ("sequential-RL", "sequential_rl", OI["blue"]),
            ("greedy oracle", "greedy_feasible_oracle", OI["green"])]
    names = [n for n, _, _ in rows]
    costs = [r[k]["mean_cost"] for _, k, _ in rows]
    cols = [c for _, _, c in rows]
    feas = [r[k]["feasible"] for _, k, _ in rows]
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    bars = ax.bar(names, costs, color=cols, edgecolor="white", linewidth=0.5)
    for b, f in zip(bars, feas):
        ax.annotate(f"{f}/3 feas", (b.get_x() + b.get_width() / 2, b.get_height()),
                    textcoords="offset points", xytext=(0, 2), ha="center", fontsize=6.5, color=INK)
    ax.set_ylabel("mean placement cost")
    ax.set_title("Sequential occupancy-aware placement (§5.7)")
    ax.grid(True, axis="y")
    ax.tick_params(axis="x", labelrotation=20)
    _save(fig, "fig_sequential")


def fig_architecture():
    """Schematic block diagram of the STOA control plane (Fig 2). Not data-driven."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(4.6, 4.9))
    ax.set_xlim(0, 10); ax.set_ylim(0, 11); ax.axis("off")

    def box(x, y, w, h, text, fc="#F2F2F2", ec=MUTED, tc=INK, fs=8, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.12",
                                    fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color=tc, weight="bold" if bold else "normal")

    def arrow(x1, y1, x2, y2, color=MUTED, style="-|>"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=12,
                                     lw=1.3, color=color, shrinkA=2, shrinkB=2))

    # inputs
    box(0.4, 9.9, 5.6, 0.9, "memory items  +  systems telemetry", fc="#FFFFFF")
    box(6.6, 9.9, 3.0, 0.9, "budget knob (L, C, B)", fc="#FFF3E0", ec=OI["orange"], tc=INK)
    # perception
    box(1.2, 8.3, 7.6, 0.95, "Perception:  GNN over memory graph (co-access / entity edges)",
        fc="#E8F0FB", ec=OI["blue"])
    # policy
    box(0.6, 6.0, 6.2, 1.7, "", fc="#E8F0FB", ec=OI["blue"])
    ax.text(3.7, 7.35, "Factored policy  $\\pi_{rep}\\cdot\\pi_{tier}\\cdot\\pi_{time}$", ha="center",
            fontsize=8.5, color=INK, weight="bold")
    box(0.9, 6.15, 1.7, 0.85, "$\\pi_{rep}$", fc="#FFFFFF", fs=8)
    box(2.75, 6.15, 1.7, 0.85, "$\\pi_{tier}$", fc="#FFFFFF", fs=8)
    box(4.6, 6.15, 1.9, 0.85, "$\\pi_{time}$", fc="#FFFFFF", fs=8)
    ax.text(3.7, 5.72, "masked softmax over 50 legal actions", ha="center", fontsize=6.5, color=MUTED)
    box(7.0, 6.0, 2.4, 1.7, "Critic\n(value head)\nunder (L,C,B)", fc="#E8F5EE", ec=OI["green"], fs=7.5)
    # execution
    box(0.6, 3.7, 4.1, 1.4, "Online execution\nMemCube migrate/fuse\nLMCache put/get",
        fc="#FFFFFF", fs=7.5)
    box(5.1, 3.7, 4.3, 1.4, "Offline (sleep-time)\nconsolidation queue:\nplaintext$\\to$latent/param",
        fc="#FFFFFF", fs=7.5)
    # substrates
    box(0.6, 1.6, 8.8, 1.2, "Substrates:  MemOS store  |  LMCache tiers (GPU/CPU/disk/remote)  |  M+ latent",
        fc="#F2F2F2", fs=7.0)
    # reward feedback
    box(0.6, 0.2, 8.8, 0.85, "reward  r = U (task utility)  $-\\ \\lambda_L$lat $-\\ \\lambda_C$cost $-\\ \\lambda_B$tok",
        fc="#FDECEA", ec=OI["vermillion"], fs=7.8)

    # flow arrows
    arrow(3.2, 9.9, 3.2, 9.25)
    arrow(5.0, 8.3, 3.7, 7.75)
    arrow(8.1, 9.9, 8.2, 7.7)          # budget -> critic
    arrow(2.2, 6.0, 2.2, 5.15)         # policy -> online
    arrow(6.2, 6.0, 6.6, 5.15)         # policy -> offline
    arrow(2.6, 3.7, 3.0, 2.8)
    arrow(7.0, 3.7, 6.6, 2.8)
    # feedback (reward up to critic/policy) on the right
    arrow(9.4, 0.62, 9.7, 0.62, color=OI["vermillion"], style="-")
    ax.add_patch(FancyArrowPatch((9.7, 0.62), (9.7, 6.85), arrowstyle="-", lw=1.3, color=OI["vermillion"]))
    arrow(9.7, 6.85, 9.4, 6.85, color=OI["vermillion"])
    ax.text(9.85, 3.7, "credit assignment", rotation=90, va="center", fontsize=6.8, color=OI["vermillion"])

    _save(fig, "fig_architecture")


def main():
    fig_architecture()
    fig_budget_frontier()
    fig_tier_ablation()
    fig_locomo("eval_locomo.json", "fig_locomo")
    if (EXP / "eval_locomo_judge.json").exists():
        fig_locomo("eval_locomo_judge.json", "fig_locomo_judge")
    fig_locomo_final()
    fig_locomo_baselines()
    fig_learnability()
    fig_reactive()
    fig_capacity_ladder()
    fig_sequential()


if __name__ == "__main__":
    main()
