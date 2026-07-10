from evaluation.base import Evaluator, write_csv, print_results_table
from evaluation.config import DATASETS, RESULTS_DIR, DEFAULT_FIXED, STANDARD_METRICS as METRICS, DEFAULT_N_TRAINING, \
    DEFAULT_N_TEST
from evaluation.plots.plotter import generate_plots
from utility.event_log_splitter import EventLogSplitter
import os

CSV_PATH = os.path.join(RESULTS_DIR, "context_depth_impact.csv")

PARAM = "context_depth"
VALUES = [None, 10, 8, 6, 4, 2, 1]
FIXED = {k: v for k, v in DEFAULT_FIXED.items() if k != PARAM}

TITLE = "Impact of Context Depth on Accuracy and Model Size"


def main():
    results = []
    for ds_name, ds_cfg in DATASETS.items():
        print(f"\nDataset: {ds_name}")
        splitter = EventLogSplitter(ds_cfg["path"], location_key=ds_cfg["location_key"])
        for val in VALUES:
            label = str(val) if val is not None else "None"
            ev = Evaluator(
                dataset_path=ds_cfg["path"],
                location_key=ds_cfg["location_key"],
                splitter=splitter,
                context_depth=val,
                max_depth=None,
                predecessor_threshold=1.0,
                n_training=DEFAULT_N_TRAINING,
                n_test=DEFAULT_N_TEST,
            )
            row = ev.run(label=f"{ds_name} cd={label}")
            row["dataset"] = ds_name
            row[PARAM] = label
            results.append(row)
            print(f"  \u2713 loss={row['avg_step_loss']:.4f}  "
                  f"exact={row['exact_pct']:.1f}%  "
                  f"entrypoints={row['total_entrypoints']}  "
                  f"states={row['final_states']}  "
                  f"remote={row['total_remote']}  "
                  f"compute={row['total_compute']}  "
                  f"{row['elapsed_s']}s")

    write_csv(results, CSV_PATH)
    table_metrics = [(c, l) for c, l, _ in METRICS]
    print_results_table(results, PARAM, table_metrics)
    generate_plots(CSV_PATH, METRICS, x_col=PARAM, group_col="dataset", title=TITLE)


if __name__ == "__main__":
    main()
