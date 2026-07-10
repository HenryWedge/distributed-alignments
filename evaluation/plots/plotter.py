from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HATCHES = ["", "//", "\\\\", "xx", "..", "++", "oo", "**", "--"]
COLORS = plt.cm.tab10(np.linspace(0, 1, 10))


def generate_plots(
    csv_path: str,
    metrics: List[Tuple[str, str, str]],
    x_col: str = "warmup",
    group_col: str = "dataset",
    title: str = "",
):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        print(f"No data in {csv_path}")
        return

    unique_x = _sorted_unique(rows, x_col)
    groups = _sorted_unique(rows, group_col)

    base = os.path.splitext(os.path.basename(csv_path))[0]
    out_dir = os.path.join(os.path.dirname(csv_path), base)
    os.makedirs(out_dir, exist_ok=True)

    for idx, (col, label, _) in enumerate(metrics):
        _save_single_plot(
            rows=rows,
            x_col=x_col,
            group_col=group_col,
            metric_col=col,
            metric_label=label,
            unique_x=unique_x,
            groups=groups,
            title=title,
            out_path=os.path.join(out_dir, f"{base}_{col}.svg"),
        )


def _sorted_unique(rows: List[Dict], col: str) -> List[str]:
    seen: Dict[str, int] = {}
    for r in rows:
        v = r[col]
        if v not in seen:
            seen[v] = len(seen)
    return sorted(seen, key=lambda k: seen[k])


def _save_single_plot(
    rows, x_col, group_col, metric_col, metric_label,
    unique_x, groups, title, out_path,
):
    x = np.arange(len(unique_x))
    n_groups = len(groups)
    bar_width = 0.8 / n_groups

    fig, ax = plt.subplots(figsize=(max(8, len(unique_x) * 1.2), 5))

    for gi, grp in enumerate(groups):
        vals = []
        for xv in unique_x:
            match = [r for r in rows if r[x_col] == xv and r[group_col] == grp]
            vals.append(float(match[-1][metric_col]) if match else 0.0)
        offset = (gi - (n_groups - 1) / 2) * bar_width
        color = COLORS[gi % len(COLORS)]
        hatch = HATCHES[gi % len(HATCHES)]
        bars = ax.bar(
            x + offset, vals, bar_width, label=grp,
            color=color, hatch=hatch, edgecolor="black", linewidth=0.6,
        )
        for bar, v in zip(bars, vals):
            pass

    ax.set_title(metric_label, fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([_shorten(v) for v in unique_x], fontsize=8, rotation=30, ha="right")
    ax.set_ylabel(metric_label, fontsize=10)
    ax.legend(fontsize=8, loc="best")

    if title:
        ax.set_title(f"{title} — {metric_label}", fontsize=11, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {out_path}")


def _shorten(v: str) -> str:
    if len(v) > 8:
        return v[:4] + ".."
    return v
