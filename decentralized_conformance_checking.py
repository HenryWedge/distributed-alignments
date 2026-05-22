#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import hashlib
from typing import Dict, List, Tuple, Set, Optional

from algo.utility.event_log_splitter import EventLogSplitter


START_HASH = "START"
LOG_COST = 1
MODEL_COST = 1


def compute_hash(prev_hash: str, activity: str) -> str:
    return hashlib.sha256((prev_hash + activity).encode()).hexdigest()


class Transition:
    def __init__(self, prev_hash: str, activity: str):
        self.prev_hash = prev_hash
        self.activity = activity
        self.curr_hash = compute_hash(prev_hash, activity)


class Alignment:
    def __init__(self, moves: Optional[List[Tuple[Optional[str], Optional[str]]]] = None):
        self._moves: List[Tuple[Optional[str], Optional[str]]] = moves if moves is not None else []

    @property
    def moves(self) -> List[Tuple[Optional[str], Optional[str]]]:
        return self._moves

    @property
    def cost(self) -> int:
        c = 0
        for m, l in self._moves:
            if m is None:
                c += LOG_COST
            elif l is None:
                c += MODEL_COST
        return c

    def add_sync_move(self, activity: str) -> 'Alignment':
        return Alignment(self._moves + [(activity, activity)])

    def add_log_move(self, activity: str) -> 'Alignment':
        return Alignment(self._moves + [(None, activity)])

    def add_model_move(self, activity: str) -> 'Alignment':
        return Alignment(self._moves + [(activity, None)])

    @staticmethod
    def from_log_moves(events: List[str], log_idx: int) -> 'Alignment':
        return Alignment([(None, e) for e in events[:log_idx + 1]])


class Participant:
    def __init__(self, participant_id: str):
        self.id = participant_id
        self.transitions: List[Transition] = []
        self._memo: Dict[Tuple[str, int], Alignment] = {}

    def add_transition(self, prev_hash: str, activity: str) -> str:
        t = Transition(prev_hash, activity)
        self.transitions.append(t)
        return t.curr_hash

    def get_transitions_by_curr_hash(self, curr_hash: str) -> List[Transition]:
        return [t for t in self.transitions if t.curr_hash == curr_hash]

    def calculate_alignment(self, target_hash: str, log_idx: int,
                            events: List[str], network: 'Network') -> Alignment:
        if target_hash == START_HASH:
            return Alignment.from_log_moves(events, log_idx)

        key = (target_hash, log_idx)
        if key in self._memo:
            return self._memo[key]

        local_transitions = self.get_transitions_by_curr_hash(target_hash)

        best_align: Optional[Alignment] = None
        best_cost = float('inf')

        for t in local_transitions:
            align = network.route_min_cost(t.prev_hash, log_idx, events)
            candidate = align.add_model_move(t.activity)
            if candidate.cost < best_cost:
                best_cost = candidate.cost
                best_align = candidate

            if log_idx >= 0 and events[log_idx] == t.activity:
                align = network.route_min_cost(t.prev_hash, log_idx - 1, events)
                candidate = align.add_sync_move(t.activity)
                if candidate.cost < best_cost:
                    best_cost = candidate.cost
                    best_align = candidate

        if log_idx >= 0:
            align = network.route_min_cost(target_hash, log_idx - 1, events)
            candidate = align.add_log_move(events[log_idx])
            if candidate.cost < best_cost:
                best_cost = candidate.cost
                best_align = candidate

        if best_cost == float('inf'):
            best_align = Alignment.from_log_moves(events, log_idx)

        self._memo[key] = best_align
        return best_align

    def compute_prefix_alignment(self, events: List[str],
                                 network: 'Network') -> Alignment:
        all_nodes = {START_HASH}
        for t in network.all_transitions():
            all_nodes.add(t.curr_hash)

        log_idx = len(events) - 1

        best_align: Optional[Alignment] = None
        best_cost = float('inf')

        for node in all_nodes:
            align = network.route_min_cost(node, log_idx, events)
            if align.cost < best_cost:
                best_cost = align.cost
                best_align = align

        return best_align


class Network:
    def __init__(self):
        self.event_stream: Dict[str, List[str]] = {}
        self.participants: Dict[str, Participant] = {}
        self.hash_owner: Dict[str, str] = {}

    def all_transitions(self) -> List[Transition]:
        result = []
        for p in self.participants.values():
            result.extend(p.transitions)
        return result

    def register_participant(self, participant: Participant):
        self.participants[participant.id] = participant
        for t in participant.transitions:
            self.hash_owner[t.curr_hash] = participant.id

    def store_event(self, case_id: str, activity: str):
        if case_id not in self.event_stream:
            self.event_stream[case_id] = []
        self.event_stream[case_id].append(activity)

    def get_events(self, case_id: str) -> List[str]:
        return self.event_stream.get(case_id, [])

    def route_min_cost(self, target_hash: str, log_idx: int,
                        events: List[str]) -> Alignment:
        if target_hash == START_HASH:
            return Alignment.from_log_moves(events, log_idx)

        owner = self.hash_owner.get(target_hash)
        if owner is None:
            return Alignment.from_log_moves(events, log_idx)

        return self.participants[owner].calculate_alignment(target_hash, log_idx, events, self)

    def record_event(self, case_id: str, activity: str, participant_id: str) -> Alignment:
        self.store_event(case_id, activity)
        events = self.get_events(case_id)

        participant = self.participants.get(participant_id)
        if participant is None:
            align = self._compute_prefix_fallback(events)
        else:
            align = participant.compute_prefix_alignment(events, self)

        print(f"[{participant_id}] Event: {activity} | Running cost: {align.cost}")
        return align

    def _compute_prefix_fallback(self, events: List[str]) -> Alignment:
        all_nodes = {START_HASH}
        for t in self.all_transitions():
            all_nodes.add(t.curr_hash)
        log_idx = len(events) - 1

        best_align: Optional[Alignment] = None
        best_cost = float('inf')
        for node in all_nodes:
            align = self.route_min_cost(node, log_idx, events)
            if align.cost < best_cost:
                best_cost = align.cost
                best_align = align
        return best_align

def load_training_data(splitter: EventLogSplitter, n: int, c: bool):
    training = splitter.get_training_data(n)
    sequences: List[List[str]] = []
    mapping: Dict[str, str] = {}

    for case_id in training.traces:
        trace = []
        occurrences: Dict[str, int] = {}
        for event in training.traces[case_id]:
            if event.activity not in occurrences:
                occurrences[event.activity] = 1
            else:
                occurrences[event.activity] += 1
            activity_str = f"{event.activity}-{event.location}{occurrences[event.activity]}"
            loc = "c" if c else event.location
            trace.append(activity_str)
            mapping[activity_str] = loc
        sequences.append(trace)

    return sequences, mapping


def load_validation_trace(splitter: EventLogSplitter, record_index: int, c: bool):
    test_data = splitter.get_test_data(record_index)
    for case_id in test_data.traces:
        trace = []
        occurrences: Dict[str, int] = {}
        for event in test_data.traces[case_id]:
            if event.activity not in occurrences:
                occurrences[event.activity] = 1
            else:
                occurrences[event.activity] += 1
            activity_str = f"{event.activity}-{event.location}{occurrences[event.activity]}"
            trace.append(activity_str)
        return trace
    return []


def build_network(sequences: List[List[str]],
                  participant_mapping: Dict[str, str]) -> Network:
    network = Network()

    for seq in sequences:
        prev = START_HASH
        for activity in seq:
            pid = participant_mapping[activity]
            if pid not in network.participants:
                network.register_participant(Participant(pid))
            curr = network.participants[pid].add_transition(prev, activity)
            if curr not in network.hash_owner:
                network.hash_owner[curr] = pid
            prev = curr

    return network
