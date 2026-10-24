# Knot-aware objective normalizer and knot cap: time, memory and accuracy at large n

Run 2026-10-24 on blackbear (commit `aeba275`; Part 2 of the 2026-10-23 spec, option (b) of 2026-10-24).

**Decision on the knot cap (Yilong, 2026-10-24):** default `max_knots="auto"`, K = 1,000 for k ≥ 1 and
K = max(1,000, ⌈n^(2/3)⌉) for k = 0 (the k = 0 term acts only above n ≈ 31,600; see
`haldensity.core.univariate.objective.default_max_knots`). The default objective per order is still
pending.

**Recommendation.** Make the knot-aware objective normalizer the default, with a knot cap of
K = 1,000 candidate knots (`objective_normalizer="knot_aware"`, `max_knots=1000`; the cap only acts
when n > 1,000).

- **Spikes.** Knot-aware never spiked in 1,044 fits, even at 100× the CV bound. The midpoint
  objective spiked in half its k = 0 fits at the CV bound and in all of them at 10× M.
- **Accuracy at the CV bound.** K = 1,000 is within 13% of all knots in L2 for a density with a
  jump (k = 0) and equal or better elsewhere. The midpoint objective is up to 2.5× worse in L2 at
  n = 20,000.
- **Cost.** At n = 20,000 a capped fit takes 0.25–0.3 s and 0.22 GB, the same at every n tested;
  all knots takes 9–19 s and 1.8–2.5 GB; the midpoint objective takes 54–57 s and 17 GB.
- **Certification.** Every capped fit was certified with the ladder stopping before SCS. Without
  a cap, 5.6% of fits were not.

## Set-up

- **Configurations** (`CVXPYEstimator`, k = 0 and 1):
  - A: midpoint objective, all knots (the current default).
  - B: knot-aware objective, all knots, structured (sparse) form.
  - C: knot-aware with `max_knots` K ∈ {100, 250, 500, 1000}, run only for K < n.
  - D: knot-aware in the old dense form, n ≤ 1,000 only (reference).
- **Truths** on [0, 1]:
  - jump (k = 0): levels 0.6, 1.8, 0.7 with breaks at 0.3 and 0.6;
  - kink (k = 1): piecewise linear through (0, 0.5), (0.3, 1.6), (0.7, 1.2), (1, 0.4), normalized;
  - smooth (both k): Beta(2, 5).
- n ∈ {500, 1,000, 2,500, 5,000, 10,000, 20,000}; 3 seeds.
- **Bounds M:** the best grid M of config B's 3-seed mean 5-fold CV score at n = 1,000 (k = 0:
  jump 2.37, smooth 6.68; k = 1: kink 11.2, smooth 63.2), then 10× and 100× that M.
- **Final fits:** the package's tight final fit (MOSEK 1e-12, prune 1e-4, KKT check) with the
  ladder stopping before SCS (MOSEK → CLARABEL 1e-12/1e-10 → ECOS 1e-12/1e-10). Uncertified fits
  are recorded with their violations.
- **CV curves:** default solver tolerance, as in the tuner.
- **Measurement:** each fit runs in its own process (peak RSS from `getrusage`), capped at 64 GB
  and 1 h. Threads: MOSEK 2; BLAS, OpenMP and CLARABEL's Rayon pool 1.
- **Cost:** 1,260 fits in 2.7 CPU-h plus 132 CV curves in 1.4 CPU-h; 20 workers, about 20 min wall.
  No fit exceeded a cap; the longest took 404 s (A, k = 1, n = 20,000).
- **Files:** `bench.py` runs it; `summarize.py results` writes `results/summary.md` (all tables)
  and the figures.

## Time and memory per fit (`results/time_vs_n.png`, `results/memory_vs_n.png`)

Median over truths, bounds and seeds, at n = 20,000 (max in brackets):

| | A midpoint | B all knots | C K = 100 | C K = 250 | C K = 500 | C K = 1,000 |
| --- | --- | --- | --- | --- | --- | --- |
| k = 0 wall (s) | 54 (296) | 19 (44) | 0.07 | 0.10 | 0.15 | 0.25 (0.46) |
| k = 1 wall (s) | 57 (404) | 9 (68) | 0.09 | 0.12 | 0.18 | 0.31 (0.48) |
| k = 0 peak RSS (GB) | 16.6 | 1.8 | 0.19 | 0.19 | 0.19 | 0.22 |
| k = 1 peak RSS (GB) | 17.4 | 2.5 | 0.19 | 0.19 | 0.20 | 0.23 |

- **Capped knots (C):** cost is flat in n.
- **All knots (B):** grows about linearly in time and memory.
- **Midpoint (A):** passes 16 GB at n = 20,000; its basis is the full n × n matrix.
- **Dense knot-aware reference (D):** 23–31 s already at n = 1,000.
- No configuration exceeded 10 minutes per fit.

## Certification (tight final fit, ladder without SCS)

| | A | B | C (all K) | D |
| --- | --- | --- | --- | --- |
| certified | 88.4% | 94.4% | 100% | 100% |

- **A:** 25 fits uncertified, 23 of them at k = 1 with n ≥ 2,500. In the median such fit, the
  kept coefficients failing the check hold 42–63% of ‖θ‖₁ (up to 78%), because the 199-cell
  midpoint objective does not identify θ at large n.
- **B:** the 12 uncertified fits miss only on coefficients that should be zero, holding at most
  0.3% of ‖θ‖₁ (k = 0: n = 1,000–20,000; k = 1: n = 2,500–10,000).

The rule proposal for these cases is in `../final_fit_repair/`.

## Accuracy at the CV bound

L2 relative to B, median of 3 seeds (the full tables are in `results/summary.md`):

| truth (k) | n | A | C 100 | C 250 | C 500 | C 1,000 |
| --- | --- | --- | --- | --- | --- | --- |
| jump (0) | 2,500 | 1.11 | 1.37 | 1.13 | 1.04 | 1.07 |
| jump (0) | 20,000 | 1.49 | 1.88 | 1.43 | 1.14 | 1.07 |
| smooth (0) | 20,000 | 2.47 | 1.16 | 1.00 | 0.99 | 1.00 |
| kink (1) | 20,000 | 1.47 | 0.75 | 0.76 | 0.76 | 0.75 |
| smooth (1) | 20,000 | 1.31 | 0.99 | 1.00 | 1.00 | 1.00 |

- **Small K and jumps:** a small K costs accuracy where the truth has a jump. K = 100 is 1.4–1.9×
  worse from n = 2,500 on; K = 1,000 is within 5–13%.
- **Smooth and kinked truths:** K ≥ 250 is within 8% of all knots, and K ≥ 500 within 3%.
- **Kink at n ≥ 10,000:** capped fits are 8–25% better than all knots.

## Spikes vs M (|objective log Z − accurate log Z| > 0.01 nats)

| share of fits flagged | 1× M | 10× M | 100× M |
| --- | --- | --- | --- |
| A, k = 0 | 0.50 | 1.00 | 1.00 |
| A, k = 1 | 0 | 0 | 0.50 |
| B, C, D (all K) | 0 | 0 | 0 |

- **Midpoint spikes:** at 10× and 100× M, the midpoint objective at k = 0 puts up to 140× the
  truth's mass into one cell, with L2 around 14.
- **Knot-aware normalizer gap:** the largest gap in any knot-aware fit is 0.0025 nats (k = 1,
  where the knot-aware normalizer is a knot-inserted midpoint rule).
- **Neighbouring knots with large opposite-sign coefficients:** none at the CV bound in any
  configuration. They appear only at 10× and 100× M for knot-aware fits: the near-collinearity
  of an over-large bound, without the spikes.
- **A cap regularizes a badly wrong M:** at 100× M (k = 0, n = 20,000), L2 is 0.10 for K = 100,
  0.14 for K = 250, 0.18 for K = 500, 0.25 for K = 1,000 and 0.23 for all knots (jump truth).

## CV curves (`results/cv_curves.png`)

Exact-score 5-fold CV log-likelihood over 17 bounds, n = 1,000 and 2,500, 3 seeds.

- **A at k = 0:** collapses past its peak, 4–26 nats below it at 10× the best M and 760–950 nats
  at the largest M.
- **Knot-aware at k = 0:** declines past its peak (it never rises by more than 4e-4), 0.016–0.056
  nats at 10× (1.5–5 SE of the CV mean). With a cap it then levels off: once M exceeds what K knots can use, a larger M does not
  change the fit.
- **k = 1:** A and the knot-aware configurations give the same curve until the largest bounds,
  where A drops 0.8–1.4 nats.
- **Effect of K on the bound:** none; every configuration's best bound is the same grid point.

## Caveats

- The recommended K = 1,000 rests on n ≤ 20,000. For a jump density the L2 gap to all knots did
  not grow beyond 13% up to that size. Whether K should grow with n beyond 20,000 (for example
  K = max(1,000, n / 20)) was not tested.
- The bounds were chosen at n = 1,000 and held fixed; accuracy at large n is at that bound, not at
  a CV choice per n.
- One seed set per cell (3 seeds): the tables show medians, not a precise comparison of close
  configurations.
