from __future__ import annotations

import csv
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from conformance.builder import build_network, load_training_data, load_validation_trace
from conformance.instrumentation_wrappers import instrument_network
from evaluation.evaluator import write_csv
from evaluation.config import DATASETS, RESULTS_DIR
from utility.event_log_splitter import EventLogSplitter

CSV_PATH = os.path.join(RESULTS_DIR, "pruning_impact.csv")

N_TRAINING = 50
N_TEST = 30


def run_dataset(
    ds_name: str,
    splitter: EventLogSplitter,
    pruning: bool,
    label: str,
    split_idx: int,
    test_indices: List[int],
) -> Dict[str, Any]:
    training_sequences, mapping = load_training_data(splitter, split_idx, c=False)

    dec_exec = build_network(
        training_sequences,
        mapping,
        max_depth=None,
        pruning=pruning
    )
    inst_net = instrument_network(dec_exec.network)
    dec_exec.network = inst_net

    t0 = time.time()

    for idx in tqdm(test_indices, desc=label, unit="tr", leave=False):
        trace = load_validation_trace(splitter, idx, c=False)
        if len(trace) < 2:
            continue
        case_id = f"case_{idx}"
        for ei in range(len(trace)):
            trace[ei].case_id = case_id
            dec_exec.record_event(trace[ei])

    from conformance.instrumentation_wrappers import InstrumentedParticipant as IP
    for pid, p in list(inst_net.participants.items()):
        if not isinstance(p, IP):
            wrapped = IP(p)
            inst_net.participants[pid] = wrapped

    final_states = sum(
        len(p.participant.alignment_cache) for p in inst_net.participants.values()
    )
    total_route = inst_net.route_calls

    elapsed = time.time() - t0

    return {
        "dataset": ds_name,
        "pruning": str(pruning),
        "elapsed_s": round(elapsed, 3),
        "final_states": final_states,
        "total_route": total_route,
    }


def generate_pruning_plots(csv_path: str):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        print(f"No data in {csv_path}")
        return

    datasets_in_order: List[str] = []
    seen_datasets: set = set()
    for r in rows:
        ds = r["dataset"]
        if ds not in seen_datasets:
            seen_datasets.add(ds)
            datasets_in_order.append(ds)

    metrics: List[Tuple[str, str, str]] = [
        ("elapsed_s", "Elapsed Time [s]", "lower is better"),
        ("final_states", "Alignment States", "lower is better"),
        ("total_route", "Route Calls", "lower is better"),
    ]

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    n_datasets = len(datasets_in_order)
    x = np.arange(n_datasets)
    bar_width = 0.35

    for ax, (col, label, _) in zip(axes, metrics):
        vals_on: List[float] = []
        vals_off: List[float] = []
        for ds in datasets_in_order:
            on_row = [r for r in rows if r["dataset"] == ds and r["pruning"] == "True"]
            off_row = [r for r in rows if r["dataset"] == ds and r["pruning"] == "False"]
            vals_on.append(float(on_row[-1][col]) if on_row else 0.0)
            vals_off.append(float(off_row[-1][col]) if off_row else 0.0)

        bars_on = ax.bar(x - bar_width / 2, vals_on, bar_width,
                         label="Pruning On", color="#2ecc71", edgecolor="white", linewidth=0.8)
        bars_off = ax.bar(x + bar_width / 2, vals_off, bar_width,
                          label="Pruning Off", color="#e74c3c", edgecolor="white", linewidth=0.8)

        for bar, v in zip(bars_on, vals_on):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.1f}", ha="center", va="bottom", fontsize=7, rotation=45)
        for bar, v in zip(bars_off, vals_off):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.1f}", ha="center", va="bottom", fontsize=7, rotation=45)

        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        short_names = [ds[:10] + ".." if len(ds) > 10 else ds for ds in datasets_in_order]
        ax.set_xticklabels(short_names, fontsize=9)
        ax.legend(fontsize=10)

    fig.suptitle("Pruning Impact on Performance", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    png_path = os.path.splitext(csv_path)[0] + ".png"
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {png_path}")


def main():
    results: List[Dict[str, Any]] = []

    for ds_name, ds_cfg in DATASETS.items():
        print(f"\nDataset: {ds_name}")
        splitter = EventLogSplitter(ds_cfg["path"], location_key=ds_cfg["location_key"])
        total_cases = len(splitter.case_ids)
        split_idx = N_TRAINING
        test_indices = [idx for idx in range(split_idx, min(split_idx + N_TEST, total_cases))]

        for pruning in [True, False]:
            label = f"{ds_name} pruning={'on' if pruning else 'off'}"
            row = run_dataset(ds_name, splitter, pruning, label, split_idx, test_indices)
            row["dataset"] = ds_name
            results.append(row)
            print(f"  pruning={pruning}: "
                  f"time={row['elapsed_s']}s  "
                  f"states={row['final_states']}  "
                  f"route={row['total_route']}")

    write_csv(results, CSV_PATH)
    generate_pruning_plots(CSV_PATH)


if __name__ == "__main__":
    main()
