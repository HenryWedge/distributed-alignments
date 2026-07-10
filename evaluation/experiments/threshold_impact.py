from evaluation.base import Evaluator, write_csv, print_results_table
from evaluation.config import DATASETS, RESULTS_DIR, DEFAULT_FIXED, STANDARD_METRICS as METRICS, DEFAULT_N_TRAINING, \
    DEFAULT_N_TEST
from evaluation.plots.plotter import generate_plots
from utility.event_log_splitter import EventLogSplitter
import os

CSV_PATH = os.path.join(RESULTS_DIR, "threshold_impact.csv")

PARAM = "predecessor_threshold"
#VALUES = [1.0, 0.999, 0.995, 0.99]
VALUES = [1.0, 0.99, 0.95, 0.9, 0.85, 0.8]
FIXED = {k: v for k, v in DEFAULT_FIXED.items() if k != PARAM}

TITLE = "Impact of Predecessor Threshold on Accuracy and Performance"

def main():
    results = []
    for ds_name, ds_cfg in DATASETS.items():
        print(f"\nDataset: {ds_name}")
        splitter = EventLogSplitter(ds_cfg["path"], location_key=ds_cfg["location_key"])
        for val in VALUES:
            label = str(val)
            ev = Evaluator(
                dataset_path=ds_cfg["path"],
                location_key=ds_cfg["location_key"],
                splitter=splitter,
                predecessor_threshold=val,
                max_depth=None,
                context_depth=None,
                n_training=DEFAULT_N_TRAINING,
                n_test=DEFAULT_N_TEST,
            )
            row = ev.run(label=f"{ds_name} l={label}")
            row["dataset"] = ds_name
            row[PARAM] = val
            results.append(row)
            print(f"  l={val:.2f}  "
                  f"loss={row['avg_step_loss']:.4f}  "
                  f"exact={row['exact_pct']:.1f}%  "
                  f"route={row['total_route']}  "
                  f"queries={row['total_queried']}  "
                  f"compute={row['total_compute']}  "
                  f"{row['elapsed_s']}s")

    write_csv(results, CSV_PATH)
    table_metrics = [(c, l) for c, l, _ in METRICS]
    print_results_table(results, PARAM, table_metrics)
    generate_plots(CSV_PATH, METRICS, x_col=PARAM, group_col="dataset", title=TITLE)


if __name__ == "__main__":
    main()
