"""Benchmark: knot-aware objective normalizer and knot cap (time, memory, accuracy at large n).

Spec: Part 2 of the 2026-10-23 knot-aware normalizer / knot cap task. Not part of the package.

Configurations (CVXPYEstimator, basis order k in {0, 1}):
  A      midpoint objective normalizer, all knots (the default);
  B      knot-aware objective normalizer, all knots (structured form);
  C<K>   knot-aware, max_knots = K in {100, 250, 500, 1000} (structured form);
  D      knot-aware, all knots, the dense n x m form (reference, n <= 1,000 only: ~3 min per fit at 2,500).
Truths on [0, 1]: "jump" (piecewise constant; k = 0), "kink" (piecewise linear; k = 1), "smooth" (Beta(2, 5);
both k). Fixed-bound fits use the final-fit default (solver_tolerance="tight"); CV folds use "default", as
OptunaHyperparameterTuner does.

Usage (each fit / CV curve runs in its own process):
  python bench.py run cv   OUT_DIR --workers W --mosek-threads T
  python bench.py run fits OUT_DIR --workers W --mosek-threads T     (needs OUT_DIR/cv.jsonl for the bounds)
  python bench.py report OUT_DIR
  python bench.py fit  K TRUTH CONFIG N M SEED      (one fit; prints one JSON line)
  python bench.py cv   K TRUTH CONFIG N SEED        (one CV curve; prints one JSON line)
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

NS = (500, 1000, 2500, 5000, 10000, 20000)
SEEDS = (0, 1, 2)
KS = (100, 250, 500, 1000)
CONFIGS = ("A", "B") + tuple(f"C{K}" for K in KS) + ("D",)
DENSE_MAX_N = 1000
TRUTHS = {0: ("jump", "smooth"), 1: ("kink", "smooth")}
CV_NS = (1000, 2500)
CV_FOLDS = 5
M_GRID = {0: np.geomspace(0.5, 2000.0, 17), 1: np.geomspace(2.0, 20000.0, 17)}
M_FACTORS = (1, 10, 100)
MEMORY_CAP_GB = 64.0
TIME_CAP_S = 3600.0
SPIKE_GAP_NATS = 0.01
OPPOSITE_FRACTION = 0.1          # "large": both coefficients >= 10% of the largest knot coefficient
# Final-fit ladder for the fixed-bound fits: the tight ladder without its SCS step (option (b), Yilong
# 2026-10-24: SCS takes 20-60+ min per fit at n = 20,000); uncertified fits are recorded with their violations.
FIT_SOLVERS = ["MOSEK", "CLARABEL", "ECOS"]

# --------------------------------------------------------------------------------------------- truths
JUMP = (np.array([0.0, 0.3, 0.6, 1.0]), np.array([0.6, 1.8, 0.7]))          # breaks, levels (integral 1)
KINK = (np.array([0.0, 0.3, 0.7, 1.0]), np.array([0.5, 1.6, 1.2, 0.4]))     # knots, values (rescaled to 1)
FINE = np.linspace(0.0, 1.0, 200_001)


def truth_pdf(name: str, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if name == "jump":
        breaks, levels = JUMP
        return levels[np.clip(np.searchsorted(breaks, x, side="right") - 1, 0, levels.size - 1)]
    if name == "kink":
        knots, values = KINK
        area = np.sum(np.diff(knots) * (values[:-1] + values[1:]) / 2)
        return np.interp(x, knots, values) / area
    if name == "smooth":
        from scipy.stats import beta
        return beta.pdf(x, 2.0, 5.0)
    raise ValueError(name)


def sample(name: str, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if name == "smooth":
        return rng.beta(2.0, 5.0, size=n)
    pdf = truth_pdf(name, FINE)
    cdf = np.concatenate(([0.0], np.cumsum((pdf[1:] + pdf[:-1]) / 2 * np.diff(FINE))))
    return np.interp(rng.uniform(size=n), cdf / cdf[-1], FINE)


# --------------------------------------------------------------------------------------------- children
def _limit_threads(mosek_threads: int) -> None:
    """MOSEK uses every core by default: cap it (and BLAS) so parallel workers do not compete."""
    import cvxpy as cp
    original = cp.Problem.solve

    def solve(self, *args, **kwargs):
        if kwargs.get("solver") == "MOSEK":
            params = dict(kwargs.get("mosek_params") or {})
            params["MSK_IPAR_NUM_THREADS"] = mosek_threads
            kwargs["mosek_params"] = params
        return original(self, *args, **kwargs)

    cp.Problem.solve = solve


def _estimator(k: int, config: str, M: float, solver_tolerance: str):
    from haldensity.full_uni import CVXPYEstimator
    extra = dict(_solver_waterfall=FIT_SOLVERS) if solver_tolerance == "tight" else {}
    fit = CVXPYEstimator(norm_constraint=M, basis_order=k, log_frequency=-1,
                         _objective_normalizer="midpoint" if config == "A" else "knot_aware",
                         max_knots=int(config[1:]) if config.startswith("C") else None,
                         __solver_tolerance=solver_tolerance, **extra)
    if config == "D":
        fit._knot_aware_formulation = "dense"
    return fit


def _peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1e9 if sys.platform == "darwin" else 1e6)


def _log_f(points: np.ndarray, theta: np.ndarray, knots: np.ndarray, k: int) -> np.ndarray:
    """Unnormalized log density for k = 0, 1 by cumulative sums over the active knots (exact, O(N log m))."""
    first = 1 if k == 0 else 2
    coef = theta[first:]
    nonzero = coef != 0
    a, c = knots[nonzero], coef[nonzero]
    idx = np.searchsorted(a, points, side="right")                  # active knots <= x
    s1 = np.concatenate(([0.0], np.cumsum(c)))
    if k == 0:
        return theta[0] + s1[idx]
    s2 = np.concatenate(([0.0], np.cumsum(c * a)))
    return theta[0] + theta[1] * points + points * s1[idx] - s2[idx]


def _log_cell_integrals(edges: np.ndarray, theta: np.ndarray, knots: np.ndarray, k: int) -> np.ndarray:
    """log of the integral of exp(f) over each cell between consecutive edges (active knots), exactly."""
    widths = np.diff(edges)
    left = _log_f(edges[:-1], theta, knots, k)
    if k == 0:
        return left + np.log(widths)
    d = _log_f(edges[1:], theta, knots, k) - left                    # f is linear on the cell
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        factor = np.where(np.abs(d) < 1e-8, d / 2,
                          np.where(d > 0, d + np.log(-np.expm1(-np.abs(d))) - np.log(np.abs(d)),
                                   np.log(-np.expm1(-np.abs(d))) - np.log(np.abs(d))))
    return left + np.log(widths) + factor


def _diagnostics(fit, k: int, truth: str) -> dict:
    from scipy.integrate import trapezoid
    from scipy.special import logsumexp
    theta, knots = np.asarray(fit.theta_hat, dtype=float), np.asarray(fit._grid_points_hal, dtype=float)
    first = 1 if k == 0 else k + 1
    coef = theta[first:]
    active = knots[coef != 0]
    edges = np.unique(np.concatenate(([0.0], active[(active > 0) & (active < 1)], [1.0])))
    log_cells = _log_cell_integrals(edges, theta, knots, k)
    log_z = float(logsumexp(log_cells))                              # exact for k = 0, 1
    density = np.exp(_log_f(FINE, theta, knots, k) - log_z)
    f0 = truth_pdf(truth, FINE)
    out = dict(l2=float(np.sqrt(trapezoid((density - f0) ** 2, FINE))))
    with np.errstate(divide="ignore", invalid="ignore"):
        integrand = np.where(f0 > 0, f0 * (np.log(f0) - np.log(np.maximum(density, 1e-300))), 0.0)
    out["kl"] = float(trapezoid(integrand, FINE))
    out["normalizer_gap"] = float(log_z - fit.objective_log_normalizer(theta))
    out["spike_flag"] = bool(abs(out["normalizer_gap"]) > SPIKE_GAP_NATS)
    # largest share of the mass in one cell between consecutive active knots, and the truth's mass there
    share = np.exp(log_cells - log_z)
    j = int(np.argmax(share))
    cdf0 = np.concatenate(([0.0], np.cumsum((f0[1:] + f0[:-1]) / 2 * np.diff(FINE))))
    truth_mass = float(np.diff(np.interp(edges[j:j + 2], FINE, cdf0))[0])
    out["max_cell_mass_share"] = float(share[j])
    out["max_cell_width"] = float(edges[j + 1] - edges[j])
    out["max_cell_mass_over_truth"] = float(share[j] / max(truth_mass, 1e-12))
    big = np.abs(coef) >= OPPOSITE_FRACTION * np.max(np.abs(coef), initial=0.0)
    pairs = big[:-1] & big[1:] & (np.sign(coef[:-1]) * np.sign(coef[1:]) < 0)
    out["opposite_sign_neighbours"] = int(pairs.sum())
    out["n_selected"] = int(np.count_nonzero(coef))
    out["max_density"] = float(density.max())
    return out


def _violations(fit, x: np.ndarray, k: int, config: str) -> dict:
    """The KKT failures of the final fit's raw solution: kept coefficients with |gradient| < 0.99 lambda
    (should be zero) and coefficients with |gradient| > 1.01 lambda (missing), with their sizes."""
    import pandas as pd
    from haldensity.core.univariate.support import KKT_SLACK, MAX_RATIO, FullDataHALProblem
    problem = FullDataHALProblem(pd.DataFrame({"W1": x}), k, 200, "midpoint" if config == "A" else "knot_aware",
                                 int(config[1:]) if config.startswith("C") else None)
    theta = np.asarray(fit.optimized_theta_raw, dtype=float)
    first = 1 if k == 0 else k + 1
    g, t = problem.gradient(theta)[first:], theta[first:]
    lam = fit.kkt_["lambda_hat"]
    kept = np.abs(t) > fit.tol
    ratio = np.abs(g) / lam if lam > 0 else np.zeros_like(g)
    low, high = kept & (ratio < KKT_SLACK), ratio > MAX_RATIO
    scale = float(np.abs(theta[1:]).sum())
    return dict(viol_low=int(low.sum()), viol_low_l1=float(np.abs(t[low]).sum()),
                viol_low_l1_share=float(np.abs(t[low]).sum() / scale) if scale else 0.0,
                viol_low_max=float(np.abs(t[low]).max(initial=0.0)), viol_high=int(high.sum()),
                kkt_max_ratio=fit.kkt_["max_ratio"], kkt_lambda=lam, kkt_bound=fit.kkt_["bound"],
                kkt_n_kept=fit.kkt_["n_kept"], l1_over_M=scale / float(fit.norm_constraint))


def child_fit(k: int, truth: str, config: str, n: int, M: float, seed: int) -> dict:
    import pandas as pd
    x = sample(truth, n, seed)
    out = dict(task="fit", k=k, truth=truth, config=config, n=n, M=M, seed=seed)
    fit = _estimator(k, config, M, "tight")
    wall, cpu = time.perf_counter(), time.process_time()
    try:
        fit.fit(pd.DataFrame({"W1": x}))
        ok = True
    except Exception as exc:
        ok = False
        out["status"] = f"error: {type(exc).__name__}: {exc}"[:300]
    out["wall_s"], out["cpu_s"] = time.perf_counter() - wall, time.process_time() - cpu
    out["peak_rss_gb"] = _peak_rss_gb()
    m = int(len(fit._grid_points_hal)) if getattr(fit, "_grid_points_hal", None) is not None else None
    out["n_knots"] = m
    if m is not None:
        p = (1 if k == 0 else k + 1) + m
        dense = config in ("A", "D") or fit.knot_aware_formulation_used_ == "dense"
        cells = 199 if config == "A" else m + 1
        out["largest_matrix_gb"] = (n * p * 4 / 1e9) if dense else 0.0            # float32 n x p data basis
        out["normalizer_matrix_gb"] = (cells * p * 4 / 1e9) if (dense and not (k == 0 and config != "A")) else 0.0
    if ok:
        out.update(status=fit.problem_status_, solver=fit.solver_used_, solver_option=fit.solver_option_,
                   certified=fit.certified_, kkt_bound=fit.kkt_["bound"], n_attempts=len(fit.solve_attempts_),
                   mosek_status=next((a.get("status", a["outcome"][:60]) for a in fit.solve_attempts_
                                      if a["solver"] == "MOSEK"), None),
                   formulation=fit.knot_aware_formulation_used_,
                   attempts=[{key: a.get(key) for key in ("solver", "option", "formulation", "status", "outcome",
                                                          "certified", "seconds")} for a in fit.solve_attempts_])
        out.update(_diagnostics(fit, k, truth))
        out.update(_violations(fit, x, k, config))
    return out


def child_cv(k: int, truth: str, config: str, n: int, seed: int) -> dict:
    import pandas as pd
    from sklearn.model_selection import KFold
    x = sample(truth, n, seed)
    folds = list(KFold(CV_FOLDS, shuffle=True, random_state=seed).split(x))
    wall, cpu = time.perf_counter(), time.process_time()
    curve = []
    for M in M_GRID[k]:
        scores, errors = [], 0
        for train, valid in folds:
            fit = _estimator(k, config, float(M), "default")
            try:
                fit.fit(pd.DataFrame({"W1": x[train]}))
                score = float(np.mean(fit.get_log_density_at_points(x[valid])))
                if np.isfinite(score):
                    scores.append(score)
                else:
                    errors += 1
            except Exception:
                errors += 1
        curve.append(dict(M=float(M), mean=float(np.mean(scores)) if scores else None,
                          sd=float(np.std(scores, ddof=1)) if len(scores) > 1 else None, failed_folds=errors))
    return dict(task="cv", k=k, truth=truth, config=config, n=n, seed=seed, curve=curve,
                wall_s=time.perf_counter() - wall, cpu_s=time.process_time() - cpu, peak_rss_gb=_peak_rss_gb())


# --------------------------------------------------------------------------------------------- parent
def plan_cv() -> list[tuple]:
    tasks = []
    for k, truths in TRUTHS.items():
        for truth in truths:
            for n in CV_NS:
                for config in CONFIGS:
                    if config == "D" or (config.startswith("C") and int(config[1:]) >= n * (CV_FOLDS - 1) // CV_FOLDS):
                        continue
                    for seed in SEEDS:
                        tasks.append(("cv", k, truth, config, n, seed))
    return tasks


def chosen_bounds(out_dir: Path) -> dict:
    """M per (k, truth): the grid bound with the best 3-seed mean CV score of config B at n = 1,000."""
    rows = [json.loads(line) for line in open(out_dir / "cv.jsonl")]
    bounds = {}
    for k, truths in TRUTHS.items():
        for truth in truths:
            curves = [r["curve"] for r in rows if r.get("curve") and r["k"] == k and r["truth"] == truth
                      and r["config"] == "B" and r["n"] == 1000]
            means = np.nanmean([[c["mean"] if c["mean"] is not None else np.nan for c in cur] for cur in curves], axis=0)
            bounds[f"{k}/{truth}"] = float(M_GRID[k][int(np.nanargmax(means))])
    return bounds


def plan_fits(out_dir: Path) -> list[tuple]:
    bounds = chosen_bounds(out_dir)
    (out_dir / "bounds.json").write_text(json.dumps(bounds, indent=1))
    tasks = []
    for n in sorted(NS, reverse=True):         # large n first, so the longest fits do not form the tail
        for k, truths in TRUTHS.items():
            for truth in truths:
                for config in CONFIGS:
                    if config == "D" and n > DENSE_MAX_N:
                        continue
                    if config.startswith("C") and int(config[1:]) >= n:
                        continue               # K >= n: the same knots as B
                    for factor in M_FACTORS:
                        for seed in SEEDS:
                            tasks.append(("fit", k, truth, config, n, bounds[f"{k}/{truth}"] * factor, seed))
    return tasks


def _key(task: tuple) -> str:
    return json.dumps([str(v) for v in task])


def run_one(task: tuple, mosek_threads: int) -> dict:
    """Run one task in a child process; stop it at the memory or time cap and record "exceeded"."""
    import psutil
    # CLARABEL's Rayon pool and BLAS otherwise use every core: on a 256-core machine one dense k = 1 fit at
    # n = 5,000 took 17,000 CPU-s in 300 s wall without RAYON_NUM_THREADS=1, 177 CPU-s in 176 s with it
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", RAYON_NUM_THREADS="1",
               BENCH_MOSEK_THREADS=str(mosek_threads))
    cmd = [sys.executable, __file__, *[str(v) for v in task]]
    started = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    watched, peak, reason = psutil.Process(proc.pid), 0.0, None
    while proc.poll() is None:
        try:
            rss = watched.memory_info().rss / 1e9
        except psutil.Error:
            rss = 0.0
        peak = max(peak, rss)
        if rss > MEMORY_CAP_GB:
            reason = f"exceeded memory cap {MEMORY_CAP_GB:g} GB"
        elif time.perf_counter() - started > TIME_CAP_S * (17 * CV_FOLDS if task[0] == "cv" else 1):
            reason = f"exceeded time cap {TIME_CAP_S:g} s"
        if reason:
            proc.kill()
            break
        time.sleep(0.5)
    stdout, stderr = proc.communicate()
    head = dict(zip(("task", "k", "truth", "config", "n") + (("M", "seed") if task[0] == "fit" else ("seed",)), task))
    if reason:
        return dict(head, status=reason, wall_s=time.perf_counter() - started, peak_rss_gb=peak)
    lines = [line for line in stdout.splitlines() if line.startswith("{")]
    if not lines:
        return dict(head, status=f"no output (exit {proc.returncode}): {stderr[-300:]}", peak_rss_gb=peak)
    result = json.loads(lines[-1])
    result["watched_peak_rss_gb"] = peak
    return result


def run(kind: str, out_dir: Path, workers: int, mosek_threads: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ("cv.jsonl" if kind == "cv" else "fits.jsonl")
    tasks = plan_cv() if kind == "cv" else plan_fits(out_dir)
    done = set()
    if path.exists():
        for line in open(path):
            r = json.loads(line)
            done.add(r.get("_key"))
    todo = [t for t in tasks if _key(t) not in done]
    print(f"{kind}: {len(tasks)} tasks, {len(todo)} to run, {workers} workers", flush=True)
    with open(path, "a") as fh, ThreadPoolExecutor(workers) as pool:
        for i, (task, result) in enumerate(zip(todo, pool.map(lambda t: run_one(t, mosek_threads), todo))):
            result["_key"] = _key(task)
            fh.write(json.dumps(result) + "\n")
            fh.flush()
            if i % 25 == 0:
                print(f"{i + 1}/{len(todo)} {result.get('status')} {task}", flush=True)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in ("fit", "cv"):
        threads = int(os.environ.get("BENCH_MOSEK_THREADS", "1"))
        _limit_threads(threads)
        import warnings
        warnings.filterwarnings("ignore")
        a = sys.argv[2:]
        if sys.argv[1] == "fit":
            result = child_fit(int(a[0]), a[1], a[2], int(a[3]), float(a[4]), int(a[5]))
        else:
            result = child_cv(int(a[0]), a[1], a[2], int(a[3]), int(a[4]))
        print(json.dumps(result), flush=True)
        return
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("kind", choices=("cv", "fits"))
    r.add_argument("out_dir", type=Path)
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--mosek-threads", type=int, default=2)
    args = parser.parse_args()
    if args.cmd == "run":
        run(args.kind, args.out_dir, args.workers, args.mosek_threads)


if __name__ == "__main__":
    main()
