import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

def infer_sep(path: Path) -> str:
    """Guess delimiter by inspecting first line."""
    with path.open("r") as f:
        head = f.readline()
    return "," if head.count(",") >= head.count("\t") else "\t"


def scale_columns(df: pd.DataFrame, targets: dict[str, float]) -> pd.DataFrame:
    """Scale each column so its sum becomes targets[col] if greater."""
    scaled = df.copy().astype(float)
    for col, target in targets.items():
        col_sum = df[col].sum()
        if col_sum > 0 and col_sum > target:
            scaled[col] *= target / col_sum
    return scaled


def main(filepath: str | Path = "mutation_matrix.csv") -> None:
    path = Path(filepath)
    if not path.exists():
        sys.exit(f"Input file not found: {path}")

    # --- load ---
    sep = infer_sep(path)
    M = pd.read_csv(path, sep=sep, index_col=0)

    # Ensure channels are rows, samples are columns
    if M.shape[0] > M.shape[1]:  # likely samples in rows; transpose
        M = M.T

    if M.shape[1] < 1:
        sys.exit("Matrix has no sample columns after reading.")
    print(f"Loaded matrix shape (channels × samples): {M.shape}")

    # --- 1) GMM normalization ---
    if M.shape[1] >= 2:
        totals = M.sum(axis=0).values.reshape(-1, 1).astype(float)
        gmm = GaussianMixture(n_components=2, random_state=0).fit(totals)

        means = gmm.means_.flatten()
        stds = np.sqrt(gmm.covariances_).flatten()
        baseline = np.argmin(means)  # lower‑mean component
        cutoff = means[baseline] + 2 * stds[baseline]

        targets_gmm = {col: min(total, cutoff) for col, total in M.sum(axis=0).items()}
        M_gmm = scale_columns(M, targets_gmm)
    else:
        print("Only one sample available – skipping GMM scaling.")
        M_gmm = M.copy()

    # --- 2) 100× normalization ---
    upper = M.shape[0] * 100  # channels × 100
    targets_100x = {col: min(total, upper) for col, total in M.sum(axis=0).items()}
    M_100x = scale_columns(M, targets_100x)

    # --- 3) log₂ normalization ---
    M_log2 = M.copy().astype(float)
    for col, total in M.sum(axis=0).items():
        if total > 0:
            factor = math.log2(total) / total
            M_log2[col] *= factor

    # --- save ---
    M_gmm.to_csv("mutation_matrix_GMM_norm.csv")
    M_100x.to_csv("mutation_matrix_100x_norm.csv")
    M_log2.to_csv("mutation_matrix_log2_norm.csv")
    print("Normalization complete. Files saved.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "mutation_matrix.csv")
