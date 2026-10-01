"""Algorithmic Pareto-frontier identification (no manual/subjective model selection)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def find_pareto_frontier(df: pd.DataFrame, maximize: list[str], minimize: list[str]) -> pd.Series:
    """Return a boolean Series marking which rows of `df` are Pareto-optimal.

    A row is non-dominated if no other row is at least as good on every
    objective in `maximize`/`minimize` and strictly better on at least one.
    Rows with NaN in any objective column are excluded (cannot be compared).
    """
    cols = maximize + minimize
    valid = df.dropna(subset=cols)
    is_pareto = pd.Series(False, index=df.index)

    values = valid[cols].to_numpy(dtype=float)
    n_maximize = len(maximize)
    # Flip minimize columns so "larger is better" uniformly.
    signed = values.copy()
    signed[:, n_maximize:] *= -1

    for i in range(len(valid)):
        dominated = False
        for j in range(len(valid)):
            if i == j:
                continue
            at_least_as_good = np.all(signed[j] >= signed[i])
            strictly_better = np.any(signed[j] > signed[i])
            if at_least_as_good and strictly_better:
                dominated = True
                break
        if not dominated:
            is_pareto.loc[valid.index[i]] = True

    return is_pareto


def plot_pareto_frontier(
    df: pd.DataFrame, x_col: str, y_col: str, pareto_mask: pd.Series, out_path: str, x_label: str, y_label: str
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    dominated = df[~pareto_mask]
    frontier = df[pareto_mask].sort_values(x_col)

    ax.scatter(dominated[x_col], dominated[y_col], color="gray", alpha=0.5, label="Dominated")
    ax.scatter(frontier[x_col], frontier[y_col], color="crimson", s=90, zorder=3, label="Pareto-optimal")
    ax.plot(frontier[x_col], frontier[y_col], color="crimson", linestyle="--", alpha=0.6)

    for _, row in df.iterrows():
        label = f"{row.get('precision', '')}-{row.get('method', '')}"
        ax.annotate(label, (row[x_col], row[y_col]), fontsize=7, xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"Pareto Frontier: {y_label} vs {x_label}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
