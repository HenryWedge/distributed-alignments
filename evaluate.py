import csv
import sys
import os
import time

from conformance.builder import load_validation_trace, load_training_data, build_network
from conformance.instrumentation_wrappers import instrument_network
from utility.event_log_splitter import EventLogSplitter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA_PATH = "datasets/Sepsis.xes"
LOCATION_KEY = "org:group"
N_TRAINING = 1000
N_TEST = 50

WARMUP_VALUES = [0, 5, 10, 12, 14, 16, 18, 20, 25, 30, 40, 50, 100, None]

CSV_PATH = os.path.join(os.path.dirname(__file__), "evaluation_results.csv")


def run_experiment(
    warmup, training_sequences, mapping,
    training_sequences_c, mapping_c,
    splitter, test_indices
):
    t0 = time.time()

    # Build a centralized baseline (no warmup) for each test
    # to avoid cross-trace state leakage, build fresh on each experiment
    central_exec = build_network(training_sequences_c, mapping_c, warmup=None)

    # Build decentralized executor with the given warmup
    dec_exec = build_network(training_sequences, mapping, warmup=warmup)
    inst_net = instrument_network(dec_exec.network)
    dec_exec.network = inst_net

    warmup_label = str(warmup) if warmup is not None else "None"

    total_step_loss = 0
    total_events = 0
    max_step_loss = 0
    exact_count = 0
    traces_processed = 0

    for idx in test_indices:
        trace = load_validation_trace(splitter, idx, c=False)
        trace_c = load_validation_trace(splitter, idx, c=True)
        if len(trace) < 2:
            continue

        traces_processed += 1
        case_id = f"case_{idx}"

        for ei in range(len(trace)):
            trace[ei].case_id = case_id
            trace_c[ei].case_id = case_id

            dec_align = dec_exec.record_event(trace[ei])
            central_align = central_exec.record_event(trace_c[ei])

            loss = dec_align.cost - central_align.cost
            total_step_loss += loss
            total_events += 1

            if loss > max_step_loss:
                max_step_loss = loss
            if loss == 0:
                exact_count += 1

    final_states = sum(
        len(p.participant.alignment_cache) for p in inst_net.participants.values()
    )
    total_compute = sum(
        p.compute_calls for p in inst_net.participants.values()
    )
    total_queried = sum(
        p.participant.query_strategy.total_queries for p in inst_net.participants.values()
    )
    # route/remote are already accumulated across all traces (no reset)
    total_route = inst_net.route_calls
    total_remote = inst_net.remote_calls

    elapsed = time.time() - t0

    return {
        "warmup": warmup_label,
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
        "total_queried": total_queried,
        "elapsed_s": round(elapsed, 1),
    }


def main():
    splitter = EventLogSplitter(DATA_PATH, location_key=LOCATION_KEY)
    total_cases = len(splitter.case_ids)
    test_indices = list(range(N_TRAINING, min(total_cases, N_TRAINING + N_TEST)))

    print(f"Total cases in dataset: {total_cases}")
    print(f"Training traces:        {N_TRAINING}")
    print(f"Test traces:            {len(test_indices)}")
    print()

    training_sequences, mapping = load_training_data(splitter, N_TRAINING, c=False)
    training_sequences_c, mapping_c = load_training_data(splitter, N_TRAINING, c=True)
    print(f"Decentral training: {len(training_sequences)} traces, "
          f"{sum(len(s) for s in training_sequences)} events, "
          f"{len(set(mapping.values()))} participants")
    print(f"Central training:   {len(training_sequences_c)} traces, "
          f"{sum(len(s) for s in training_sequences_c)} events, "
          f"{len(set(mapping_c.values()))} participants")
    print()

    results = []

    for warmup in WARMUP_VALUES:
        label = str(warmup) if warmup is not None else "None"
        sys.stdout.write(f"  [warmup={label:>5s}] ... ")
        sys.stdout.flush()
        r = run_experiment(
            warmup, training_sequences, mapping,
            training_sequences_c, mapping_c,
            splitter, test_indices
        )
        results.append(r)
        print(f"events={r['events']:5d}  loss={r['avg_step_loss']:7.4f}  "
              f"exact={r['exact_pct']:5.1f}%  route={r['total_route']:7d}  "
              f"compute={r['total_compute']:6d}  q/ev={r['total_queried']//r['events']:>3d}  "
              f"states={r['final_states']:6d}  {r['elapsed_s']:5.1f}s")

    fieldnames = [
        "warmup", "traces", "events",
        "sum_step_loss", "avg_step_loss", "max_step_loss",
        "exact_count", "exact_pct",
        "total_route", "total_remote",
        "final_states", "total_compute", "total_queried",
        "elapsed_s",
    ]
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults written to {CSV_PATH}")

    sep = "-" * 120
    print(f"\n{sep}")
    print(f"  {'warmup':>7s}  {'events':>6s}  {'exact%':>6s}  "
          f"{'route':>7s}  {'remote':>6s}  {'compute':>8s}  "
          f"{'q/ev':>5s}  {'states':>7s}  {'time':>6s}")
    print(sep)
    for r in results:
        print(f"  {r['warmup']:>7s}  {r['events']:>6d}  "
              f"{r['exact_pct']:>5.1f}%  {r['total_route']:>7d}  "
              f"{r['total_remote']:>6d}  {r['total_compute']:>8d}  "
              f"{r['total_queried']//r['events']:>5d}  "
              f"{r['final_states']:>7d}  {r['elapsed_s']:>6.1f}")
    print(sep)


if __name__ == "__main__":
    main()
