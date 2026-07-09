from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .network import Network
from .participant import Participant
from .models import Entrypoint, Event
from .executor import Executor
from utility.event_log_splitter import EventLogSplitter


def _build_predecessor_ranking(
    entrypoints: List[Entrypoint],
    participant_id: str,
    all_participant_ids: set,
) -> Tuple[List[Tuple[str, float]], Dict[str, float]]:
    counts: Dict[str, int] = {}
    for ep in entrypoints:
        if ep.previous_entrypoint is not None:
            pid = ep.previous_entrypoint.participant_id
            counts[pid] = counts.get(pid, 0) + ep.frequency

    total_self = sum(ep.frequency for ep in entrypoints)
    counts[participant_id] = counts.get(participant_id, 0) + total_self

    for pid in all_participant_ids:
        if pid not in counts:
            counts[pid] = 0

    total = sum(counts.values())
    if total == 0:
        ranking = [(pid, 1.0) for pid in counts]
        pred_map = {pid: 0.0 for pid in counts}
        return ranking, pred_map

    sorted_pids = sorted(counts.items(), key=lambda x: -x[1])
    cumulative = 0.0
    ranking: List[Tuple[str, float]] = []
    pred_map: Dict[str, float] = {}
    for pid, count in sorted_pids:
        cum_before = cumulative
        cumulative += count / total
        if cumulative > 1.0:
            cumulative = 1.0
        ranking.append((pid, cumulative))
        pred_map[pid] = cum_before
    return ranking, pred_map


def build_network(
    sequences: List[List[Event]],
    participant_mapping: Dict[str, str],
    max_depth: Optional[int] = None,
    participant_cls: type = Participant,
    context_depth: int | None = None,
    pruning: bool = True,
    predecessor_threshold: float = 1.0,
):
    network = Network()
    participant_entrypoints: Dict[str, List[Entrypoint]] = {}

    for seq in sequences:
        prev = None
        for event in seq:
            participant_id = participant_mapping[event.activity]
            candidate = Entrypoint(prev, event.activity, participant_id, context_depth=context_depth)

            existing = [ep for ep in participant_entrypoints.get(participant_id, []) if ep.hash == candidate.hash]
            if existing:
                existing[0].frequency += 1
                curr = existing[0]
            else:
                participant_entrypoints.setdefault(participant_id, []).append(candidate)
                curr = candidate
            prev = curr

    all_participant_ids = set(participant_entrypoints.keys())

    for pid, entrypoints in participant_entrypoints.items():
        ranking, pred_map = _build_predecessor_ranking(entrypoints, pid, all_participant_ids)
        participant = participant_cls(
            pid, network, entrypoints, ranking, pred_map,
            max_depth=max_depth,
            pruning=pruning,
            predecessor_threshold=predecessor_threshold,
        )
        network.register_participant(participant)

    executor = Executor(network, participant_mapping)
    return executor


def load_training_data(splitter: EventLogSplitter, n: int, c: bool):
    training = splitter.get_training_data(n)

    sequences: List[List[Event]] = []
    mapping: Dict[str, str] = {}

    for case_id in training.traces:
        trace: List[Event] = []
        occurrences: Dict[str, int] = {}
        for event in training.traces[case_id]:
            if event.activity not in occurrences:
                occurrences[event.activity] = 1
            else:
                occurrences[event.activity] += 1
            #activity_str = f"{event.activity}-{event.location}{occurrences[event.activity]}"
            #activity_str = f"{event.activity}-{event.location}"
            activity_str = f"{event.activity}"
            loc = "c" if c else event.location
            trace.append(Event(activity_str, case_id, loc, event.time))
            mapping[activity_str] = loc
        sequences.append(trace)

    return sequences, mapping


def load_validation_trace(splitter: EventLogSplitter, record_index: int, c: bool):
    test_data = splitter.get_test_data(record_index)

    for case_id in test_data.traces:
        trace: List[Event] = []
        occurrences: Dict[str, int] = {}
        for event in test_data.traces[case_id]:
            if event.activity not in occurrences:
                occurrences[event.activity] = 1
            else:
                occurrences[event.activity] += 1
            #activity_str = f"{event.activity}-{event.location}{occurrences[event.activity]}"
            #activity_str = f"{event.activity}-{event.location}"
            activity_str = f"{event.activity}"
            loc = "c" if c else event.location
            trace.append(Event(activity_str, case_id, loc, event.time))
        return trace

    return []
