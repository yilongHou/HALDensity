"""Digest rule_eval.jsonl. Usage: rule_summary.py results/rule_eval.jsonl.gz"""
import json
import sys

import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
import gzip
opener = gzip.open if sys.argv[1].endswith(".gz") else open
d = pd.DataFrame([json.loads(line) for line in opener(sys.argv[1], "rt")])
d["mosek_certified"] = d["mosek_certified"].fillna(False).astype(bool)
u = d[~d.mosek_certified].copy()
for c in ("zero_certified", "polish_certified", "scs_certified", "zero_applied", "polish_applied"):
    if c in u:
        u[c] = u[c].fillna(False).astype(bool)
u["ladder_certified"] = u["ladder"].map(lambda L: any(s.get("certified") for s in L) if isinstance(L, list) else False)
u["ladder_cert_s"] = u["ladder"].map(lambda L: sum(s["s"] for s in L[: next((i + 1 for i, s in enumerate(L) if s.get("certified")), len(L))]) if isinstance(L, list) else np.nan)

print(f"{len(d)} problems; MOSEK 1e-12 certified {int(d.mosek_certified.sum())}; uncertified {len(u)}\n")
print("Certified, by configuration and n (counts over all problems in the cell):")
g = d.groupby(["config", "n"])
tab = pd.DataFrame({"problems": g.size(), "mosek": g.mosek_certified.sum()})
for name, col in (("+zero", "zero_certified"), ("+polish", "polish_certified"), ("+ladder (no SCS)", "ladder_certified"),
                  ("SCS 600 s", "scs_certified")):
    tab[name] = u.groupby(["config", "n"])[col].sum() if col in u else 0
tab = tab.fillna(0).astype(int)
tab["mosek or polish"] = tab["mosek"] + tab["+polish"]
tab["mosek or zero"] = tab["mosek"] + tab["+zero"]
print(tab.to_string(), "\n")
print("By k and configuration:")
gk = d.groupby(["k", "config"])
tk = pd.DataFrame({"problems": gk.size(), "mosek": gk.mosek_certified.sum()})
for name, col in (("+zero", "zero_certified"), ("+polish", "polish_certified"), ("+ladder", "ladder_certified"), ("SCS600", "scs_certified")):
    tk[name] = u.groupby(["k", "config"])[col].sum() if col in u else 0
print(tk.fillna(0).astype(int).to_string(), "\n")

print("Time per uncertified problem (s): median / max")
times = pd.DataFrame({
    "mosek": u.mosek_s, "zero": u.zero_seconds, "polish": u.polish_seconds,
    "ladder_until_cert": u.ladder_cert_s, "ladder_all": u.ladder_s, "scs600": u.get("scs_s")})
print(times.groupby([u.config, u.n]).agg(["median", "max"]).round(2).to_string(), "\n")

print("Removed by the rule (uncertified problems where it applied): count and share of ||theta[1:]||_1")
for name in ("zero", "polish"):
    a = u[u[f"{name}_applied"]]
    print(name, "applied", len(a), "of", len(u), "| certified", int(a[f"{name}_certified"].sum()),
          "| removed median", a[f"{name}_removed"].median(), "max", a[f"{name}_removed"].max(),
          "| share median", f"{a[f'{name}_removed_share'].median():.3g}", "max", f"{a[f'{name}_removed_share'].max():.3g}",
          "| added max", a[f"{name}_added"].max())
    if name == "zero":
        na = u[~u.zero_applied]
        print("   not applied:", na.zero_reason.str.slice(0, 40).value_counts().to_dict())
print()

print("Change vs the reference (first certified CLARABEL/ECOS, else SCS 600 s) where one exists:")
rows = []
for name in ("mosek", "zero", "polish", "ladder_best", "scs600"):
    col = f"{name}_density_sup_rel"
    if col not in u:
        continue
    m = u[u[col].notna() & u.reference.notna()]
    if name in ("zero", "polish"):
        m = m[m[f"{name}_certified"]]
    if not len(m):
        continue
    rows.append(dict(candidate=name + (" (certified only)" if name in ("zero", "polish") else ""), problems=len(m),
                     density_sup_rel_median=m[col].median(), density_sup_rel_max=m[col].max(),
                     density_l1_max=m[f"{name}_density_l1"].max(),
                     se_max_rel_median=m[f"{name}_se_max_rel"].median(), se_max_rel_max=m[f"{name}_se_max_rel"].max(),
                     se_computed=int(m[f"{name}_se_max_rel"].notna().sum())))
print(pd.DataFrame(rows).to_string(float_format=lambda v: f"{v:.3g}"), "\n")
print("References:", u.reference.fillna("none").value_counts().to_dict())
cols = ["k", "truth", "config", "n", "M_factor", "mosek_violating", "mosek_kept", "zero_certified", "zero_removed_share",
        "polish_certified", "polish_rounds", "polish_removed", "polish_added", "ladder_certified", "scs_certified", "scs_s", "reference"]
print(u[[c for c in cols if c in u]].sort_values(["config", "n", "k"]).to_string())
