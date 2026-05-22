from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from .participant import Participant


class Network:
    def __init__(self):
        self.participants: Dict[str, Participant] = {}
        self.activity_index: Dict[str, List[Participant]] = {}
        self.participant_event_counts: Dict[str, Dict[str, int]] = {}
        self.route_calls = 0
        self.remote_calls = 0

    def reset_stats(self):
        self.route_calls = 0
        self.remote_calls = 0

    def get_participant(self, participant_id: str, caller_id: str) -> Participant | None:
        self.route_calls += 1
        if caller_id != participant_id:
            self.remote_calls += 1
        return self.participants.get(participant_id, None)

    def get_total_states(self) -> int:
        return sum(len(p.decision_cache) for p in self.participants.values())

    def register_participant(self, participant: Participant):
        self.participants[participant.id] = participant
        participant.network = self

    def record_local_event(self, participant_id: str, case_id: str):
        self.participant_event_counts.setdefault(case_id, {})
        self.participant_event_counts[case_id][participant_id] = \
            self.participant_event_counts[case_id].get(participant_id, 0) + 1
