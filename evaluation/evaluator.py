from __future__ import annotations

import csv
import os
import sys
import time
from typing import Any, Dict, List, Optional, Type, Tuple

from tqdm import tqdm

from conformance.builder import build_network, load_training_data, load_validation_trace
from conformance.instrumentation_wrappers import instrument_network
from utility.event_log_splitter import EventLogSplitter

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(CODE_DIR, ".."))


class Evaluator:
    def __init__(
        self,
        dataset_path: str,
        location_key: str,
        splitter: Optional[EventLogSplitter] = None,
        use_batch: bool = False,
        max_depth: Optional[int] = None,
        n_training: Optional[int] = None,
        n_test: Optional[int] = None,
        context_depth: Optional[int] = None,
        predecessor_threshold: float = 1.0,
        verbose: bool = True,
    ):
        self.dataset_path = dataset_path
        self.location_key = location_key
        self.splitter = splitter
        self.use_batch = use_batch
        self.max_depth = max_depth
        self.n_training = n_training
        self.n_test = n_test
        self.context_depth = context_depth
        self.predecessor_threshold = predecessor_threshold
        self.verbose = verbose

    def run(self, label: str = "") -> Dict[str, Any]:
        splitter = (
            self.splitter
            if self.splitter is not None
            else EventLogSplitter(self.dataset_path, location_key=self.location_key)
        )

        total_cases = len(splitter.case_ids)

        if self.n_training is not None:
            split_idx = self.n_training
        else:
            split_idx = int(total_cases * 0.8)

        training_sequences, mapping = load_training_data(splitter, split_idx, c=False)
        training_sequences_c, mapping_c = load_training_data(splitter, split_idx, c=True)

        central_exec = build_network(training_sequences_c, mapping_c)
        dec_exec = build_network(
            training_sequences,
            mapping,
            max_depth=self.max_depth,
            context_depth=self.context_depth,
            predecessor_threshold=self.predecessor_threshold,
        )
        instrumented_network = instrument_network(dec_exec.network)
        dec_exec.network = instrumented_network

        t0 = time.time()

        total_step_loss = 0
        total_events = 0
        max_step_loss = 0
        exact_count = 0
        traces_processed = 0

        test_indices = [idx for idx in range(split_idx, total_cases)]

        if self.n_test is not None:
            test_indices = test_indices[:self.n_test]
        iter_wrapper = tqdm(
            test_indices,
            desc=label,
            unit="tr",
            disable=not self.verbose
        )

        for idx in iter_wrapper:
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

            if self.verbose:
                iter_wrapper.set_postfix(loss=total_step_loss)

        # Wrap any runtime-created participants that instrument_network missed
        from conformance.instrumentation_wrappers import InstrumentedParticipant
        for pid, p in list(instrumented_network.participants.items()):
            if not isinstance(p, InstrumentedParticipant):
                wrapped = InstrumentedParticipant(p)
                instrumented_network.participants[pid] = wrapped

        final_states = sum(
            len(p.participant.alignment_cache) for p in instrumented_network.participants.values()
        )
        total_compute = sum(
            p.compute_calls for p in instrumented_network.participants.values()
        )
        total_cost_evals = sum(
            p.calculate_cost_calls for p in instrumented_network.participants.values()
        )
        total_queried = 0
        total_route = instrumented_network.route_calls
        total_remote = instrumented_network.remote_calls

        elapsed = time.time() - t0

        num_participants = len(instrumented_network.participants)
        total_entrypoints = sum(
            len(p.participant.entrypoints) for p in instrumented_network.participants.values()
        )

        return {
            "dataset": os.path.splitext(os.path.basename(self.dataset_path))[0],
            "use_batch": str(self.use_batch),
            "max_depth": str(self.max_depth),
            "context_depth": str(self.context_depth),
            "n_training": split_idx,
            "n_test": len(test_indices),
            "traces": traces_processed,
            "events": total_events,
            "sum_step_loss": total_step_loss,
            "avg_step_loss": round(total_step_loss / total_events, 4) if total_events else 0.0,
            "max_step_loss": max_step_loss,
            "exact_count": exact_count,
            "exact_pct": round(100.0 * exact_count / total_events, 2) if total_events else 0.0,
            "total_route": total_route,
            "total_remote": total_remote,
            "final_states": final_states,
            "total_compute": total_compute,
            "total_cost_evals": total_cost_evals,
            "total_queried": total_queried,
            "num_participants": num_participants,
            "total_entrypoints": total_entrypoints,
            "elapsed_s": round(elapsed, 3),
        }


def write_csv(results: List[Dict[str, Any]], path: str):
    if not results:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(results[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Results written to {path}")


def _fmt(val: Any, width: int, decimals: int = 0) -> str:
    try:
        f = float(val)
        if decimals:
            return f"{f:>{width}.{decimals}f}"
        return f"{int(f):>{width}d}"
    except (ValueError, TypeError):
        s = str(val)
        return f"{s:>{width}s}"


def print_results_table(
    results: List[Dict[str, Any]],
    param_col: str,
    metric_cols: List[Tuple[str, str]],
    group_col: str = "dataset",
):
    datasets = list({r[group_col] for r in results})
    sep = "-" * 100
    print(f"\n{sep}")

    for ds in datasets:
        ds_rows = [r for r in results if r[group_col] == ds]
        header = f"  {param_col:>12s}  " + "  ".join(f"{label:>10s}" for _, label in metric_cols)
        print(f"\n  Dataset: {ds}")
        print(header)
        print(sep)
        for row in ds_rows:
            cells = [_fmt(row[param_col], 12)]
            for col, _ in metric_cols:
                val = row.get(col, "")
                cells.append(_fmt(val, 10, decimals=2 if "pct" in col or "loss" in col else 0))
            print("  " + "  ".join(cells))
        print(sep)
