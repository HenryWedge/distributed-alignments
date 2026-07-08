from __future__ import annotations

from typing import Dict, List, Optional

from .network import Network
from .participant import Participant
from .models import Entrypoint, Event
from .executor import Executor
from utility.event_log_splitter import EventLogSplitter
from .strategy.query.abstract_query_strategy import QueryStrategy

def build_network(
    sequences: List[List[Event]],
    participant_mapping: Dict[str, str],
    query_strategy: Optional[QueryStrategy] = None,
    warmup: Optional[int] = 100,
    max_depth: Optional[int] = None,
    participant_cls: type = Participant,
    top_p: float = 1.0,
    context_depth: int | None = None,
):
    network = Network()
    for seq in sequences:
        prev = None
        for event in seq:
            participant_id = participant_mapping[event.activity]
            if participant_id not in network.participants:
                network.register_participant(
                    participant_cls(
                        participant_id, network,
                        query_strategy,
                        max_depth=max_depth,
                    )
                )
            participant = network.participants[participant_id]
            candidate = Entrypoint(prev, event.activity, participant_id, context_depth=context_depth)
            existing = [ep for ep in participant.entrypoints if ep.hash == candidate.hash]
            if existing:
                existing[0].frequency += 1
                curr = existing[0]
            else:
                participant.add_entrypoint(candidate)
                curr = candidate
            prev = curr

    # All participants must have a strategy that knows ALL participants.
    # Since participants are created incrementally, each one's __init__
    # only captures a partial snapshot. Re-initialize strategies here.
    from .strategy.query.rank_predecessors_query_strategy import RankPredecessorsQueryStrategy
    all_participants = list(network.participants.values())
    participant_frequencies: Dict[str, int] = {}
    for p in all_participants:
        participant_frequencies[p.id] = sum(ep.frequency for ep in p.entrypoints)
    for participant in all_participants:
        participant.query_strategy = RankPredecessorsQueryStrategy(
            all_participants, None, warmup, top_p=top_p,
            participant_frequencies=participant_frequencies,
        )

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
            activity_str = f"{event.activity}"
            loc = "c" if c else event.location
            trace.append(Event(activity_str, case_id, loc, event.time))
        return trace

    return []
