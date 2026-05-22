from __future__ import annotations

import copy
from typing import Dict, List, Optional

from .network import Network
from .participant import Participant
from .models import Entrypoint, Event
from .executor import Executor
from .strategies import QueryStrategy, CacheStrategy
from algo.utility.event_log_splitter import EventLogSplitter


def build_network(
    sequences: List[List[Event]],
    participant_mapping: Dict[str, str],
    query_strategy: Optional[QueryStrategy] = None,
    cache_strategy: Optional[CacheStrategy] = None,
):
    network = Network()
    for seq in sequences:
        prev = None
        for event in seq:
            participant_id = participant_mapping[event.activity]
            if participant_id not in network.participants:
                network.register_participant(
                    Participant(
                        participant_id, network,
                        query_strategy,
                        copy.deepcopy(cache_strategy) if cache_strategy else None,
                    )
                )
            curr = Entrypoint(prev, event.activity, participant_id)
            network.participants[participant_id].add_entrypoint(curr)
            prev = curr

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
            activity_str = f"{event.activity}-{event.location}{occurrences[event.activity]}"
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
            activity_str = f"{event.activity}-{event.location}{occurrences[event.activity]}"
            loc = "c" if c else event.location
            trace.append(Event(activity_str, case_id, loc, event.time))
        return trace

    return []
