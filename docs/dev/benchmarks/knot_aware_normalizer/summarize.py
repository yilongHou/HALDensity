"""Tables and figures for the knot-aware normalizer benchmark. Usage: python summarize.py RESULTS_DIR

Writes RESULTS_DIR/summary.md and RESULTS_DIR/*.png from cv.jsonl, fits.jsonl and bounds.json (bench.py).
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ORDER = ["A", "B", "C100", "C250", "C500", "C1000", "D"]
LABEL = {"A": "A midpoint, all knots", "B": "B knot-aware, all knots", "D": "D knot-aware dense (reference)",
         **{f"C{K}": f"C knot-aware, K = {K}" for K in (100, 250, 500, 1000)}}


def load(out: Path):
    fits = pd.DataFrame([json.loads(line) for line in open(out / "fits.jsonl")])
    cv = [json.loads(line) for line in open(out / "cv.jsonl")]
    bounds = json.loads((out / "bounds.json").read_text())
    fits["exceeded"] = fits["status"].astype(str).str.startswith("exceeded")
    fits["error"] = fits["status"].astype(str).str.startswith(("error", "no output"))
    fits["M_factor"] = [round(m / bounds[f"{k}/{t}"]) for m, k, t in zip(fits["M"], fits["k"], fits["truth"])]
    fits["config"] = pd.Categorical(fits["config"], ORDER, ordered=True)
    return fits, cv, bounds


def md(df: pd.DataFrame, floatfmt: str = "{:.3g}") -> str:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, rule, *rows])


def cost_tables(fits: pd.DataFrame) -> str:
    out = []
    for k in (0, 1):
        sub = fits[fits.k == k]
        for col, name, scale in (("wall_s", "wall time per fit (s)", 1.0), ("cpu_s", "CPU time per fit (s)", 1.0),
                                 ("peak_rss_gb", "peak RSS per fit (GB)", 1.0)):
            t = sub.pivot_table(index="config", columns="n", values=col, aggfunc="median", observed=True)
            mx = sub.pivot_table(index="config", columns="n", values=col, aggfunc="max", observed=True)
            cells = t.map(lambda v: f"{v:.3g}") + " (" + mx.map(lambda v: f"{v:.3g}") + ")"
            cells = cells.reset_index()
            out.append(f"**k = {k}: {name}**, median (max) over truths, bounds and seeds\n\n" + md(cells))
        exc = sub[sub.exceeded].groupby(["config", "n"], observed=True).size()
        if len(exc):
            out.append(f"Exceeded the 64 GB / 1 h cap (k = {k}): " + ", ".join(f"{c} n={n}: {v}" for (c, n), v in exc.items()))
    return "\n\n".join(out)


def thresholds(fits: pd.DataFrame) -> str:
    rows = []
    for (k, config), g in fits.groupby(["k", "config"], observed=True):
        over_t = g[(g.wall_s > 600) | g.exceeded].n.min()
        over_m = g[(g.peak_rss_gb > 16) | g.exceeded].n.min()
        rows.append(dict(k=k, config=config, first_n_over_10_min=over_t, first_n_over_16_GB=over_m,
                         largest_n_run=int(g.n.max())))
    return md(pd.DataFrame(rows))


def accuracy_tables(fits: pd.DataFrame) -> str:
    ok = fits[~fits.exceeded & ~fits.error].copy()
    ok["spike"] = ok["spike_flag"].astype(float)
    out = []
    agg = dict(L2=("l2", "median"), KL=("kl", "median"), spike_share=("spike", "mean"),
               max_gap=("normalizer_gap", lambda v: float(np.max(np.abs(v)))),
               max_cell_share=("max_cell_mass_share", "median"), mass_over_truth=("max_cell_mass_over_truth", "max"),
               opposite_pairs=("opposite_sign_neighbours", "mean"), selected=("n_selected", "median"),
               certified=("certified", "mean"))
    for (k, truth), g in ok.groupby(["k", "truth"]):
        for n in (1000, 5000, 20000):
            t = g[g.n == n].groupby(["M_factor", "config"], observed=True).agg(**agg).reset_index()
            if len(t):
                out.append(f"**k = {k}, truth {truth}, n = {n}** (3 seeds per row; L2 and KL medians; spike_share = "
                           f"share of fits with |normalizer gap| > 0.01 nats; max_gap = largest |gap|; mass_over_truth "
                           f"= largest ratio of the fitted to the true mass in the fit's heaviest cell)\n\n" + md(t))
    return "\n\n".join(out)


def l2_vs_n(fits: pd.DataFrame) -> str:
    ok = fits[~fits.exceeded & ~fits.error & (fits.M_factor == 1)]
    out = []
    for (k, truth), g in ok.groupby(["k", "truth"]):
        t = g.pivot_table(index="config", columns="n", values="l2", aggfunc="median", observed=True).reset_index()
        out.append(f"**L2 at the CV bound, k = {k}, truth {truth}** (median of 3 seeds)\n\n" + md(t))
    return "\n\n".join(out)


def solver_table(fits: pd.DataFrame) -> str:
    ok = fits[~fits.exceeded & ~fits.error].copy()
    ok["mosek"] = ok["mosek_status"].fillna("not tried").astype(str).str.slice(0, 24)
    t = ok.groupby(["k", "config"], observed=True).agg(
        fits=("n", "size"), certified=("certified", "mean"), mosek_optimal=("mosek", lambda s: float(np.mean(s == "optimal"))),
        mosek_near_optimal=("mosek", lambda s: float(np.mean(s == "optimal_inaccurate"))),
        final_by_mosek=("solver", lambda s: float(np.mean(s == "MOSEK"))), mean_attempts=("n_attempts", "mean"),
        dense_fallback=("formulation", lambda s: int(np.sum(s == "dense")))).reset_index()
    return md(t)


def cv_shape(cv: list) -> tuple[str, pd.DataFrame]:
    rows = []
    for r in cv:
        if "curve" not in r:
            continue
        for p in r["curve"]:
            rows.append(dict(k=r["k"], truth=r["truth"], config=r["config"], n=r["n"], seed=r["seed"], M=p["M"],
                             score=p["mean"], failed=p["failed_folds"]))
    d = pd.DataFrame(rows)
    mean = d.groupby(["k", "truth", "config", "n", "M"]).score.mean().reset_index()
    out = []
    for (k, truth, config, n), g in mean.groupby(["k", "truth", "config", "n"]):
        g = g.sort_values("M")
        i = int(np.nanargmax(g.score.values))
        peak, last = g.score.values[i], g.score.values[-1]
        out.append(dict(k=k, truth=truth, config=config, n=n, M_best=g.M.values[i], best=peak,
                        drop_to_largest_M=peak - last, drop_at_10x=peak - np.interp(np.log(g.M.values[i] * 10), np.log(g.M.values), g.score.values),
                        failed_folds=int(d[(d.k == k) & (d.truth == truth) & (d.config == config) & (d.n == n)].failed.sum())))
    t = pd.DataFrame(out)
    t["config"] = pd.Categorical(t["config"], ORDER, ordered=True)
    return md(t.sort_values(["k", "truth", "n", "config"])), mean


def figures(fits: pd.DataFrame, cv_mean: pd.DataFrame, out: Path) -> None:
    ok = fits[~fits.error]
    for col, ylabel, name in (("wall_s", "wall time per fit (s)", "time"), ("peak_rss_gb", "peak RSS per fit (GB)", "memory")):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
        for ax, k in zip(axes, (0, 1)):
            for config in ORDER:
                g = ok[(ok.k == k) & (ok.config == config)].groupby("n")[col].median()
                if len(g):
                    ax.plot(g.index, g.values, marker="o", label=LABEL[config])
            ax.set(xscale="log", yscale="log", xlabel="n", title=f"k = {k}", ylabel=ylabel if k == 0 else None)
            ax.axhline(600 if name == "time" else 16, color="grey", ls=":", lw=1)
            ax.grid(alpha=0.3)
        axes[1].legend(fontsize=7)
        fig.suptitle(f"{ylabel}: median over truths, bounds (1x, 10x, 100x) and seeds; dotted: "
                     + ("10 min" if name == "time" else "16 GB"), fontsize=9)
        fig.tight_layout()
        fig.savefig(out / f"{name}_vs_n.png", dpi=130)
        plt.close(fig)
    combos = [(0, "jump"), (0, "smooth"), (1, "kink"), (1, "smooth")]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for col, (k, truth) in enumerate(combos):
        for row, n in enumerate((1000, 2500)):
            ax = axes[row, col]
            for config in ORDER:
                g = cv_mean[(cv_mean.k == k) & (cv_mean.truth == truth) & (cv_mean.n == n) & (cv_mean.config == config)]
                if len(g):
                    ax.plot(g.M, g.score, marker=".", label=LABEL[config])
            ax.set(xscale="log", title=f"k = {k}, {truth}, n = {n}", xlabel="M")
            ax.grid(alpha=0.3)
            if col == 0:
                ax.set_ylabel("5-fold CV log-likelihood (mean of 3 seeds)")
    axes[0, 3].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "cv_curves.png", dpi=130)
    plt.close(fig)


def main() -> None:
    out = Path(sys.argv[1])
    fits, cv, bounds = load(out)
    cv_text, cv_mean = cv_shape(cv)
    figures(fits, cv_mean, out)
    text = [
        "# Knot-aware normalizer benchmark: tables", "",
        f"Bounds M (best 3-seed mean CV score of config B at n = 1,000): {json.dumps(bounds)}", "",
        f"Fits: {len(fits)}; exceeded a cap: {int(fits.exceeded.sum())}; errors: {int(fits.error.sum())}.", "",
        "## Time and memory", "", cost_tables(fits), "", "Where each configuration first exceeds 10 min or 16 GB per fit:", "",
        thresholds(fits), "", "## Solver ladder (final fits, solver_tolerance=\"tight\")", "", solver_table(fits), "",
        "## Accuracy at the CV bound vs n", "", l2_vs_n(fits), "",
        "## Accuracy and spike diagnostics vs K and M", "", accuracy_tables(fits), "",
        "## CV curves (exact held-out score; drop = nats below the peak)", "", cv_text, "",
    ]
    (out / "summary.md").write_text("\n".join(text))
    print(f"wrote {out / 'summary.md'}")


if __name__ == "__main__":
    main()
