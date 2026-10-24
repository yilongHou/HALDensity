"""Evaluate the proposed final-fit repair rule on the 192 survey problems (seed 0). Usage: rule_eval.py OUT WORKERS

Run from the benchmark's run directory: it expects the repository at ./code and the knot-aware benchmark's
bounds at ./results/bounds.json (../knot_aware_normalizer/results/bounds.json).

Per problem: MOSEK 1e-12 + KKT check. If uncertified: repair "zero" and "polish" on MOSEK's solution; the
rest of the current ladder (CLARABEL 1e-12/1e-10, ECOS 1e-12/1e-10); if still uncertified, SCS 1e-9 with a
600 s time limit (rule (ii)). Each candidate is compared with the reference (first certified ladder or SCS
solution): density (sup relative and L1 difference) and delta-method SE (max relative difference).
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[v] = "1"
import importlib.util, json, sys, time, warnings
from multiprocessing import get_context

GRID = None
SCS_CAP_S = 600.0
SE_MAX_SUPPORT = 2000


def task(args):
    k, truth, config, n, M, Mf = args
    warnings.filterwarnings("ignore")
    import numpy as np, pandas as pd
    from scipy.special import logsumexp
    spec = importlib.util.spec_from_file_location("bench", "code/docs/dev/benchmarks/knot_aware_normalizer/bench.py")
    b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
    b._limit_threads(2)
    from haldensity.core.univariate.support import FullDataHALProblem, kkt_report, repair_support, _solve_quietly, PRUNE
    from haldensity.core.solvers.cvxpy import TIGHT_LADDER
    from haldensity.core.univariate.normalizer import log_unnormalized_density, log_normalizer
    from haldensity.core.variance import estimate_covariance_beta, density_confidence_interval
    x = b.sample(truth, n, 0)
    data = pd.DataFrame({"W1": x})
    problem = FullDataHALProblem(data, k, 200, "midpoint" if config == "A" else "knot_aware",
                                 int(config[1:]) if config.startswith("C") else None)
    first = 1 if k == 0 else k + 1
    out = dict(k=k, truth=truth, config=config, n=n, M=M, M_factor=Mf)
    t = time.perf_counter(); built = problem.build(M); out["build_s"] = time.perf_counter() - t

    def run(solver, label, opts, cap=None):
        t = time.perf_counter()
        try:
            o = dict(opts)
            if cap:
                o["time_limit_secs"] = cap
            _solve_quietly(built.problem, solver, o)
            th = built.theta()
            th = None if th is None else np.asarray(th, float).copy()
            rep = kkt_report(problem, th, norm_constraint=M) if th is not None and np.all(np.isfinite(th)) else None
            return dict(solver=solver, option=label, status=str(built.problem.status), s=time.perf_counter() - t,
                        certified=None if rep is None else rep["certified"], theta=th, kkt=rep)
        except Exception as e:
            return dict(solver=solver, option=label, status="error " + type(e).__name__, s=time.perf_counter() - t,
                        certified=None, theta=None, kkt=None)

    mosek = run(*TIGHT_LADDER[0])
    out.update(mosek_status=mosek["status"], mosek_s=mosek["s"], mosek_certified=mosek["certified"],
               mosek_violating=None if mosek["kkt"] is None else mosek["kkt"]["n_kept_violating"],
               mosek_max_ratio=None if mosek["kkt"] is None else mosek["kkt"]["max_ratio"],
               mosek_kept=None if mosek["kkt"] is None else mosek["kkt"]["n_kept"])
    if mosek["certified"] or mosek["theta"] is None:
        return out
    candidates = {"mosek": mosek["theta"]}
    for name, polish in (("zero", False), ("polish", True)):
        r = repair_support(problem, mosek["theta"], M, polish=polish)
        out.update({f"{name}_{key}": r[key] for key in ("applied", "certified", "removed", "removed_l1",
                                                         "removed_share", "added", "rounds", "reason", "seconds")})
        if r["applied"]:
            candidates[name] = r["theta_raw"]
    ladder = [run(*step) for step in TIGHT_LADDER[1:5]]
    out["ladder"] = [{key: s[key] for key in ("solver", "option", "status", "s", "certified")} for s in ladder]
    out["ladder_s"] = sum(s["s"] for s in ladder)
    reference = next((s for s in ladder if s["certified"]), None)
    usable = [s for s in ladder if s["theta"] is not None]
    if usable:
        candidates["ladder_best"] = min(usable, key=lambda s: (not s["certified"], s["kkt"]["n_kept_violating"]))["theta"]
    if reference is None:
        scs = run(*TIGHT_LADDER[5], cap=SCS_CAP_S)
        out.update(scs_status=scs["status"], scs_s=scs["s"], scs_certified=scs["certified"],
                   scs_violating=None if scs["kkt"] is None else scs["kkt"]["n_kept_violating"])
        if scs["theta"] is not None:
            candidates["scs600"] = scs["theta"]
        if scs["certified"]:
            reference = scs
    out["reference"] = None if reference is None else f"{reference['solver']} {reference['option']}"
    if reference is None:
        return out

    # compare every candidate with the reference: density and delta-method SE
    grid = np.linspace(0.0025, 0.9975, 400)
    points = np.linspace(0.02, 0.98, 49)

    def pruned(th):
        th = th.copy(); th[first:] = np.where(np.abs(th[first:]) > PRUNE, th[first:], 0.0); return th

    def density(th):
        lz = log_normalizer(th, problem.knots, k).log_z
        return np.exp(log_unnormalized_density(grid, th, problem.knots, k) - lz)

    def se(th):
        if np.count_nonzero(th[first:]) > SE_MAX_SUPPORT:
            return None
        res = dict(theta_hat=th, data_points=problem.knots, grid_points=np.linspace(0, 1, 200), norm_constraint=M)
        cov = estimate_covariance_beta(data, res, k, support_refit=None)
        return density_confidence_interval(points, res, cov, k, support_refit=None)["se"].to_numpy()

    ref = pruned(reference["theta"])
    d_ref, se_ref = density(ref), se(ref)
    for name, th in candidates.items():
        th = pruned(th)
        d = density(th)
        out[f"{name}_density_sup_rel"] = float(np.max(np.abs(d - d_ref) / d_ref))
        out[f"{name}_density_l1"] = float(np.mean(np.abs(d - d_ref)))      # grid mean ~ integral on [0, 1]
        s = se(th) if se_ref is not None else None
        out[f"{name}_se_max_rel"] = None if s is None else float(np.max(np.abs(s - se_ref) / se_ref))
        out[f"{name}_support"] = int(np.count_nonzero(th[first:]))
    out["reference_support"] = int(np.count_nonzero(ref[first:]))
    return out


if __name__ == "__main__":
    bounds = json.load(open("results/bounds.json"))
    tasks = [(k, t, c, n, bounds[f"{k}/{t}"] * f, f) for n in (20000, 10000, 5000, 2500)
             for k, t in ((0, "jump"), (0, "smooth"), (1, "kink"), (1, "smooth"))
             for c in ("A", "B", "C250", "C1000") for f in (1, 10, 100)]
    with open(sys.argv[1], "w") as fh, get_context("spawn").Pool(int(sys.argv[2]), maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(task, tasks):
            fh.write(json.dumps(r, default=str) + "\n"); fh.flush()
