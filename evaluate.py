#!/usr/bin/env python3
"""
Evaluate strategy combinations on the Sepsis log.

For each experiment a **fresh** optimal executor (FullScan+InfiniteCache) and
a **fresh** heuristic executor are built from the same training data.  They
process the test events **in lockstep** so both see identical cross-case cache
states.  The per-alignment cost difference is the accuracy loss.

Results are written to ``evaluation_results.csv``.
"""

import csv
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from conformance import (
    build_network,
    load_training_data,
    load_validation_trace,
    FullScanStrategy,
    InfiniteCacheStrategy,
    ActivityFilteredStrategy,
    TopKStrategy,
    DepthLimitedCacheStrategy,
)
from conformance.wrappers import instrument_network
from algo.utility.event_log_splitter import EventLogSplitter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_PATH = "../gt/test/datasets/Sepsis.xes"
LOCATION_KEY = "org:group"
N_TRAINING = 100          # traces used to build the model
N_TEST = 100              # traces used for evaluation (first N_TEST after training)

EXPERIMENTS = [
    ("FullScan + InfiniteCache",    FullScanStrategy(), InfiniteCacheStrategy()),
    ("ActivityFiltered(skip=inf)",  ActivityFilteredStrategy(10**9), InfiniteCacheStrategy()),
    ("ActivityFiltered(skip=100)",  ActivityFilteredStrategy(100), InfiniteCacheStrategy()),
    ("ActivityFiltered(skip=10)",   ActivityFilteredStrategy(10), InfiniteCacheStrategy()),
    ("ActivityFiltered(skip=5)",    ActivityFilteredStrategy(5), InfiniteCacheStrategy()),
    ("ActivityFiltered(skip=3)",    ActivityFilteredStrategy(3), InfiniteCacheStrategy()),
    ("ActivityFiltered(skip=2)",    ActivityFilteredStrategy(2), InfiniteCacheStrategy()),
    ("ActivityFiltered(skip=1)",    ActivityFilteredStrategy(1), InfiniteCacheStrategy()),
    ("TopK(AF, 10)",                TopKStrategy(ActivityFilteredStrategy(), k=10), InfiniteCacheStrategy()),
    ("TopK(AF, 1)",                 TopKStrategy(ActivityFilteredStrategy(), k=1), InfiniteCacheStrategy()),
    ("DepthLimited(d=3)",           FullScanStrategy(),
     DepthLimitedCacheStrategy(InfiniteCacheStrategy(), max_depth=3)),
    ("DepthLimited(d=5)",           FullScanStrategy(),
     DepthLimitedCacheStrategy(InfiniteCacheStrategy(), max_depth=5)),
    ("DepthLimited(d=10)",          FullScanStrategy(),
     DepthLimitedCacheStrategy(InfiniteCacheStrategy(), max_depth=10)),
    ("DepthLimited(d=20)",          FullScanStrategy(),
     DepthLimitedCacheStrategy(InfiniteCacheStrategy(), max_depth=20)),
]

CSV_PATH = os.path.join(os.path.dirname(__file__), "evaluation_results.csv")

# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(name, query_strategy, cache_strategy,
                   training_sequences, mapping, splitter, test_indices):
    """Build fresh optimal + heuristic executors, run in lockstep."""
    t0 = time.time()

    # --- fresh optimal executor ---
    opt_exec = build_network(training_sequences, mapping)
    # --- fresh heuristic executor ---
    heur_exec = build_network(
        training_sequences, mapping,
        query_strategy=query_strategy,
        cache_strategy=cache_strategy,
    )
    inst_net = instrument_network(heur_exec.network)
    heur_exec.network = inst_net

    total_step_loss = 0
    total_events = 0
    max_step_loss = 0
    exact_count = 0
    traces_processed = 0
    total_route = 0
    total_remote = 0

    for idx in test_indices:
        trace = load_validation_trace(splitter, idx, c=False)
        if len(trace) < 2:
            continue

        traces_processed += 1
        case_id = f"case_{idx}"
        inst_net.reset()

        for event in trace:
            event.case_id = case_id

            opt_align = opt_exec.record_event(event)
            heur_align = heur_exec.record_event(event)

            loss = heur_align.cost - opt_align.cost
            total_step_loss += loss
            total_events += 1

            if loss > max_step_loss:
                max_step_loss = loss
            if loss == 0:
                exact_count += 1

        total_route += inst_net.route_calls
        total_remote += inst_net.remote_calls

    final_states = sum(
        len(p._p.decision_cache) for p in inst_net.participants.values()
    )
    total_compute = sum(
        p.compute_calls for p in inst_net.participants.values()
    )

    elapsed = time.time() - t0

    return {
        "strategy": name,
        "traces": traces_processed,
        "events": total_events,
        "sum_step_loss": total_step_loss,
        "avg_step_loss": total_step_loss / total_events if total_events else 0.0,
        "max_step_loss": max_step_loss,
        "exact_count": exact_count,
        "exact_pct": 100.0 * exact_count / total_events if total_events else 0.0,
        "total_route": total_route,
        "total_remote": total_remote,
        "final_states": final_states,
        "total_compute": total_compute,
        "elapsed_s": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    splitter = EventLogSplitter(DATA_PATH, location_key=LOCATION_KEY)
    total_cases = len(splitter.case_ids)
    test_indices = list(range(N_TRAINING, min(total_cases, N_TRAINING + N_TEST)))

    print(f"Total cases in dataset: {total_cases}")
    print(f"Training traces:        {N_TRAINING}")
    print(f"Test traces:            {len(test_indices)}")
    print()

    training_sequences, mapping = load_training_data(splitter, N_TRAINING, c=False)
    print(f"Training: {len(training_sequences)} traces, "
          f"{sum(len(s) for s in training_sequences)} events")
    print()

    # ---- Run experiments -------------------------------------------------
    results = []

    for name, qs, cs in EXPERIMENTS:
        sys.stdout.write(f"  [{name:32s}] ... ")
        sys.stdout.flush()
        r = run_experiment(name, qs, cs,
                           training_sequences, mapping,
                           splitter, test_indices)
        results.append(r)
        print(f"events={r['events']:5d}  loss={r['avg_step_loss']:7.4f}  "
              f"exact={r['exact_pct']:5.1f}%  route={r['total_route']:6d}  "
              f"states={r['final_states']:6d}  {r['elapsed_s']:5.1f}s")

    # ---- Write CSV -------------------------------------------------------
    fieldnames = [
        "strategy", "traces", "events",
        "sum_step_loss", "avg_step_loss", "max_step_loss",
        "exact_count", "exact_pct",
        "total_route", "total_remote",
        "final_states", "total_compute",
        "elapsed_s",
    ]
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults written to {CSV_PATH}")

    # ---- Summary table ---------------------------------------------------
    sep = "-" * 108
    print(f"\n{sep}")
    print(f"  {'Strategy':<32s}  {'events':>6s}  {'avg_loss':>8s}  "
          f"{'max_loss':>8s}  {'exact%':>6s}  {'route':>7s}  "
          f"{'remote':>6s}  {'states':>7s}  {'compute':>8s}")
    print(sep)
    for r in results:
        print(f"  {r['strategy']:<32s}  {r['events']:>6d}  "
              f"{r['avg_step_loss']:>8.4f}  {r['max_step_loss']:>8d}  "
              f"{r['exact_pct']:>5.1f}%  {r['total_route']:>7d}  "
              f"{r['total_remote']:>6d}  {r['final_states']:>7d}  "
              f"{r['total_compute']:>8d}")
    print(sep)


if __name__ == "__main__":
    main()
