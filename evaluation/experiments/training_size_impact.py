from evaluation.base import Evaluator, write_csv, print_results_table
from evaluation.config import DATASETS, RESULTS_DIR, DEFAULT_FIXED, STANDARD_METRICS as METRICS, DEFAULT_N_TRAINING, \
    DEFAULT_N_TEST
from utility.event_log_splitter import EventLogSplitter
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HATCHES = ["", "//", "\\\\", "xx", "..", "++", "oo", "**", "--"]
COLORS = plt.cm.tab10(np.linspace(0, 1, 10))

CSV_PATH = os.path.join(RESULTS_DIR, "training_size_impact.csv")

TRAINING_SIZES = [10, 50, 100]
PREDECESSOR_THRESHOLD_VALUES = [1.0, 0.7, 0.5]
CD_VALUES = [None, 5, 1]

FIXED = {k: v for k, v in DEFAULT_FIXED.items()
         if k not in ("n_training", "predecessor_threshold", "context_depth")}


def _unique_sorted(rows, col):
    seen = {}
    for r in rows:
        v = r[col]
        if v not in seen:
            seen[v] = len(seen)
    return sorted(seen, key=lambda k: seen[k])


def _shorten(v):
    s = str(v)
    if len(s) > 10:
        return s[:7] + ".."
    return s


def _plot_subgrid(rows, x_col, group_col, title, out_dir):
    groups = _unique_sorted(rows, group_col)
    unique_x = _unique_sorted(rows, x_col)
    base = os.path.splitext(os.path.basename(out_dir))[0]
    os.makedirs(out_dir, exist_ok=True)

    for col, label, _ in METRICS:
        x = np.arange(len(unique_x))
        n_groups = len(groups)
        bar_width = 0.8 / n_groups

        fig, ax = plt.subplots(figsize=(max(8, len(unique_x) * 1.2), 5))

        for gi, grp in enumerate(groups):
            vals = []
            for xv in unique_x:
                match = [r for r in rows if r[x_col] == xv and r[group_col] == grp]
                vals.append(float(match[-1][col]) if match else 0.0)
            offset = (gi - (n_groups - 1) / 2) * bar_width
            color = COLORS[gi % len(COLORS)]
            hatch = HATCHES[gi % len(HATCHES)]
            ax.bar(
                x + offset, vals, bar_width, label=grp,
                color=color, hatch=hatch, edgecolor="black", linewidth=0.6,
            )

        ax.set_title(f"{title} — {label}", fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([_shorten(v) for v in unique_x], fontsize=8, rotation=30, ha="right")
        ax.set_ylabel(label, fontsize=10)
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()

        svg_path = os.path.join(out_dir, f"{base}_{col}.svg")
        fig.savefig(svg_path, format="svg", bbox_inches="tight")
        plt.close(fig)
        print(f"  Plot saved to {svg_path}")


def main():
    results = []

    for ds_name, ds_cfg in DATASETS.items():
        print(f"\nDataset: {ds_name}")
        splitter = EventLogSplitter(ds_cfg["path"], location_key=ds_cfg["location_key"])

        for nt in TRAINING_SIZES:
            for l in PREDECESSOR_THRESHOLD_VALUES:
                for cd in CD_VALUES:
                    label = f"{ds_name} n={nt} l={l} cd={cd}"
                    ev = Evaluator(
                        dataset_path=ds_cfg["path"],
                        location_key=ds_cfg["location_key"],
                        splitter=splitter,
                        n_training=nt,
                        predecessor_threshold=l,
                        context_depth=cd,
                        max_depth=None,
                        n_test=DEFAULT_N_TEST,
                    )
                    row = ev.run(label=label)
                    row["dataset"] = ds_name
                    row["n_training"] = nt
                    row["predecessor_threshold"] = l
                    row["context_depth"] = str(cd) if cd is not None else "None"
                    results.append(row)

                    print(f"  \u2713 loss={row['avg_step_loss']:.4f}  "
                          f"exact={row['exact_pct']:.1f}%  "
                          f"entrypoints={row['total_entrypoints']}  "
                          f"remote={row['total_remote']}  "
                          f"queries={row['total_queried']}  "
                          f"{row['elapsed_s']}s")

    write_csv(results, CSV_PATH)

    # Print tables grouped by training size
    for nt in TRAINING_SIZES:
        subset = [r for r in results if r["n_training"] == nt]
        print(f"\n=== Training size: {nt} ===")
        table_metrics = [(c, l) for c, l, _ in METRICS]
        print_results_table(subset, "predecessor_threshold", table_metrics, group_col="dataset")
        print_results_table(subset, "context_depth", table_metrics, group_col="dataset")

    # Predecessor-threshold sweep at each training size (cd=None)
    for nt in TRAINING_SIZES:
        subset = [r for r in results
                  if r["n_training"] == nt and r["context_depth"] == "None"]
        if not subset:
            continue
        _plot_subgrid(
            subset, x_col="predecessor_threshold", group_col="dataset",
            title=f"Predecessor Threshold Impact at Training Size n={nt} (context_depth=None)",
            out_dir=os.path.join(RESULTS_DIR, f"training_size_{nt}_predecessor_threshold"),
        )

    # Context-depth sweep at each training size (predecessor_threshold=1.0)
    for nt in TRAINING_SIZES:
        subset = [r for r in results
                  if r["n_training"] == nt and r["predecessor_threshold"] == 1.0]
        if not subset:
            continue
        _plot_subgrid(
            subset, x_col="context_depth", group_col="dataset",
            title=f"Context-Depth Impact at Training Size n={nt} (top_p=1.0)",
            out_dir=os.path.join(RESULTS_DIR, f"training_size_{nt}_cd"),
        )


if __name__ == "__main__":
    main()
