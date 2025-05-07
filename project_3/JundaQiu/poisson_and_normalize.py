#!/usr/bin/env python3
"""
poisson_and_normalize.py
------------------------
One Poisson resample + three normalizations (GMM, 100×, log₂).

Notebook use:
    from poisson_and_normalize import load_matrix, one_iteration
    M = load_matrix("mutation_matrix.csv")
    out = one_iteration(M, seed=0)          # dict with keys poisson/gmm/100x/log2

CLI use inside Jupyter or shell:
    %run poisson_and_normalize.py --matrix mutation_matrix.csv --seed 7 --save
"""

import math
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

def _infer_sep(path: Path) -> str:
    with path.open("r") as f:
        head = f.readline()
    return "," if head.count(",") >= head.count("\t") else "\t"


def load_matrix(filepath: str | Path) -> pd.DataFrame:
    path = Path(filepath)
    sep = _infer_sep(path)
    M = pd.read_csv(path, sep=sep, index_col=0)

    # transpose if samples are accidentally in rows
    if M.shape[0] > M.shape[1]:
        M = M.T
    return M

def _scale_columns(df: pd.DataFrame, targets: dict) -> pd.DataFrame:
    scaled = df.copy().astype(float)
    for col, tgt in targets.items():
        s = df[col].sum()
        if s > tgt > 0:
            scaled[col] *= tgt / s
    return scaled


def _gmm_cutoff(totals: np.ndarray, random_state: int) -> float:
    gmm = GaussianMixture(n_components=2, random_state=random_state).fit(totals)
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_).flatten()
    idx = np.argmin(means)           # lower‑mean component
    return means[idx] + 2 * stds[idx]


def one_iteration(M_raw: pd.DataFrame,
                  seed: int = 0,
                  gmm_random_state: int | None = None) -> Dict[str, pd.DataFrame]:
    if gmm_random_state is None:
        gmm_random_state = seed

    # Poisson resample
    rng = np.random.default_rng(seed)
    M_poi = pd.DataFrame(rng.poisson(M_raw.values),
                         index=M_raw.index,
                         columns=M_raw.columns)

    # GMM normalization
    totals = M_poi.sum(axis=0).astype(float).values.reshape(-1, 1)
    if totals.shape[0] >= 2:
        cutoff = _gmm_cutoff(totals, gmm_random_state)
        targets = {c: min(t, cutoff) for c, t in M_poi.sum(axis=0).items()}
        M_gmm = _scale_columns(M_poi, targets)
    else:
        M_gmm = M_poi.copy()

    # 100× normalization
    upper = M_poi.shape[0] * 100
    targets_100 = {c: min(t, upper) for c, t in M_poi.sum(axis=0).items()}
    M_100 = _scale_columns(M_poi, targets_100)

    # log₂ normalization
    M_log2 = M_poi.copy().astype(float)
    for col, tot in M_poi.sum(axis=0).items():
        if tot > 0:
            M_log2[col] *= math.log2(tot) / tot

    return {"poisson": M_poi, "gmm": M_gmm, "100x": M_100, "log2": M_log2}


if __name__ == "__main__":
    import argparse, sys

    p = argparse.ArgumentParser(description="Run one Poisson+normalize iteration")
    p.add_argument("--matrix", default="mutation_matrix.csv",
                   help="Path to mutation matrix (CSV/TSV)")
    p.add_argument("--seed", type=int, default=0, help="Poisson RNG seed")
    p.add_argument("--save", action="store_true",
                   help="If set, write three CSVs with seed suffix")
    args, unknown = p.parse_known_args()   # ←  **ignore extra '-f ...'**

    if unknown:
        print("⚠  Ignoring extra args:", unknown, file=sys.stderr)

    mat = load_matrix(args.matrix)
    res = one_iteration(mat, seed=args.seed)

    if args.save:
        stem = Path(args.matrix).stem
        res["gmm"].to_csv(f"{stem}_gmm_seed{args.seed}.csv")
        res["100x"].to_csv(f"{stem}_100x_seed{args.seed}.csv")
        res["log2"].to_csv(f"{stem}_log2_seed{args.seed}.csv")
        print("Files written.")
    else:
        print("Iteration finished; DataFrames in variable 'res'.")
