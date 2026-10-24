# Final fit at large n: a proposed rule for uncertified solutions

Proposal for Yilong's decision (2026-10-24). It is core; nothing here is a package default yet.

**Proposed rule:**
- Keep the tight ladder, MOSEK 1e-12 → CLARABEL → ECOS → SCS.
- Cap the SCS step at 600 s (SCS's `time_limit_secs`).
- If nothing certifies, keep the best attempt, flagged and warned, as now.
- Do **not** add rule (i), "set the small violators to zero and re-check": it certified 0 of the 18
  solutions it applied to.
- The active-set polish (i′) is in the package (`repair_support(polish=True)`,
  `certified_solve(repair="polish")`, off by default), but I do not propose it as a default step.
  It certified 11 of 20, every one of which CLARABEL or ECOS also certified, and it took longer.

With the recommended knot-aware objective and K = 1,000 (`../knot_aware_normalizer/`), the ladder
never needed SCS in either study: every capped fit certified by ECOS at the latest.

## Problems and what was run

The 192 problems are the cost survey's cells:
- n ∈ {2,500, 5,000, 10,000, 20,000};
- k = 0 (jump, smooth) and k = 1 (kink, smooth);
- configurations A (midpoint, all knots), B (knot-aware, all knots), C250 and C1000 (knot-aware,
  K = 250 and 1,000);
- bounds 1×, 10× and 100× the CV bound; seed 0.

`rule_eval.py` ran on blackbear at commit `aeba275`, 12 workers, about 20 min. Per problem:
1. MOSEK 1e-12 and the optimality (KKT) check.
2. If MOSEK's solution is not certified:
   - rule (i), "zero": set to zero the kept coefficients whose gradient is below 0.99 λ (the
     conditions say they are zero) if they hold at most 1% of ‖θ[1:]‖₁, then re-check;
   - rule (i′), "polish": the same coefficients removed, the problem re-solved exactly on the
     remaining support (a cells × |support| problem, because the data term is linear in θ), and
     the full gradient checked. Coordinates the check says are missing are added: active set,
     starting from at most 200 coefficients, up to 100 added per round, at most 10 rounds;
   - the rest of the current ladder: CLARABEL 1e-12/1e-10, ECOS 1e-12/1e-10;
   - if still uncertified, rule (ii): SCS 1e-9 with `time_limit_secs` = 600.
3. Every candidate is compared with the reference, which is the first certified CLARABEL, ECOS or
   SCS solution.
   - Density: sup relative and mean absolute difference on 400 points.
   - Delta-method SE: largest relative difference at 49 points, each solution on its own pruned
     basis, for supports up to 2,000.

`rule_summary.py results/rule_eval.jsonl.gz` prints every table.

## Results

- **MOSEK at 1e-12** reports "optimal_inaccurate" (stopped short of 1e-12, its near-optimal stop
  made visible) in 191 of 192 problems, yet its solution is certified in 153 (80%).
- **Capped knots:** C250 and C1000 are certified at MOSEK in 92 of 96 problems, and the other 4 by
  CLARABEL or ECOS.

The 39 problems MOSEK did not certify:

| step | applied to | certified | time per problem |
| --- | --- | --- | --- |
| (i) zero, ≤ 1% of ‖θ‖₁ | 18 (not applied: 19 bound not binding, 2 above 1%) | **0** | < 2 s |
| (i′) polish, active set | 20 | 11 (all also certified by CLARABEL or ECOS) | B, n = 20,000: median 72 s, max 172 s |
| CLARABEL → ECOS | 39 | 30 | B, n = 20,000: median 17 s, max 63 s (until certified) |
| (ii) SCS, 600 s cap | 9 that were still uncertified | 7 (3 of them at the cap) | 5–639 s |

- **Still uncertified after (ii):** 2 problems, both midpoint objective (A), k = 1, smooth truth,
  n = 20,000.
- **What rule (i) removes:** a median of 20 coefficients (max 139), holding 0.003% of ‖θ‖₁ in the
  median (max 0.24%). The remaining coefficients are then not exactly optimal, and the 1% ratio
  test fails.
- **What rule (i′) removes:** a median of 265 coefficients (max 2,733), 16% of ‖θ‖₁ in the median,
  and it adds up to 293. It fails at 100× M, where the support has 1,000+ knots.

**How far an uncertified solution is from the certified one** (the "best attempt, flagged" outcome):

| candidate vs reference | problems | density, sup relative: median (max) | SE, max relative: median (max) |
| --- | --- | --- | --- |
| MOSEK's uncertified solution | 37 | 0.23% (32%) | 87% (490%), 23 with SE |
| — midpoint objective (A) | 12 | 1.7% (16%) | 334% (490%) |
| — knot-aware, all knots (B) | 21 | 0.15% (32%) | 86% (460%) |
| — knot-aware, capped (C) | 4 | 0.2% (0.3%) | 42–184% |
| polish, certified | 11 | 0.03% (0.7%) | 1.5% (58%) |

An uncertified solution is close to the certified one as a density, but its SE basis is not: the
SE differs by 87% in the median. Certification matters for the SE, not for the point estimate.

The 58% SE difference for a certified polish result (B, k = 1, n = 20,000) is between two certified
supports of 224 and 18 knots. With many near-tied knots, the certified support is not unique.

## Decisions for Yilong

1. **Cap the SCS step at 600 s** (`time_limit_secs`) and keep the best attempt flagged. This bounds
   the final fit at large n (without the cap, SCS ran past 17 minutes at n = 20,000). It is reached
   only by the midpoint objective at k = 1 and, rarely, by knot-aware fits without a cap.
2. **Rule (i), zero and re-check:** not recommended (0 of 18).
3. **The active-set polish:** keep it off by default (as implemented) or remove it before the
   release. It never certified a problem the ladder could not.
4. **SE from an uncertified fit:** today it is used with a warning. Since its SE differs by 87% in
   the median, an option is to refuse the SE (or fall back to the original basis with a stronger
   warning) when the final fit is not certified.
