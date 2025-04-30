#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy.stats import poisson

# ---------------------------- 0. Parameters ----------------------------
INPUT_FILE  = "gene_dnds_trinucleotide_context_summary.csv"
FULL_OUTPUT = "gene_dnds_trinucleotide_context_summary_with_Bonferroni.csv"
SIG_OUTPUT  = "significant_genes_Bonferroni_0.01.csv"
ALPHA       = 0.01           # family-wise error rate threshold

# ---------------------------- 1. Load data -----------------------------
df = pd.read_csv(INPUT_FILE)

# ---------------------- 2. Two-sided Poisson p-value -------------------
def poisson_p_two_sided(obs, exp):
    """Exact two-sided Poisson p-value."""
    if pd.isna(obs) or pd.isna(exp) or exp <= 0:
        return np.nan
    if obs >= exp:
        p = poisson.sf(obs - 1, exp) * 2.0     # right tail *2
    else:
        p = poisson.cdf(obs, exp) * 2.0        # left tail *2
    return min(p, 1.0)

df["p_value"] = [
    poisson_p_two_sided(o, e)
    for o, e in zip(df["observed_dNdS"], df["expected_dNdS"])
]

# ---------------------- 3. Bonferroni adjustment -----------------------
pvals = pd.to_numeric(df["p_value"], errors="coerce")
valid_mask = pvals.notna()
m = valid_mask.sum()          # number of valid tests

if m == 0:
    raise ValueError("No valid p-values to adjust.")

try:
    from statsmodels.stats.multitest import multipletests
    reject, p_bonf, _, _ = multipletests(
        pvals[valid_mask], alpha=ALPHA, method="bonferroni"
    )
    df.loc[valid_mask, "p_bonf"]     = p_bonf
    df.loc[valid_mask, "signif_bonf"] = reject
except ModuleNotFoundError:
    # Manual Bonferroni: p_adj = p_raw * m
    p_bonf = np.minimum(pvals[valid_mask] * m, 1.0)
    df.loc[valid_mask, "p_bonf"]     = p_bonf
    df.loc[valid_mask, "signif_bonf"] = p_bonf < ALPHA

# ---------------------- 4. Save full result table ----------------------
df.to_csv(FULL_OUTPUT, index=False)
print(f"Full results written to {FULL_OUTPUT}")

# ---------------------- 5. Select & rank significant -------------------
sig_df = (
    df[df["p_bonf"] < ALPHA]
      .sort_values("p_bonf")
      .reset_index(drop=True)
)

sig_df.to_csv(SIG_OUTPUT, index=False)
print(f"{len(sig_df)} significant genes (Bonferroni, α={ALPHA}) "
      f"written to {SIG_OUTPUT}")

# Optional preview
print(sig_df.head())
