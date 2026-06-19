from __future__ import annotations

from typing import Dict

class Network:
    def __init__(self):
        self.participants: 'Dict[str, Participant]' = {}
        self.route_calls = 0
        self.remote_calls = 0

    def reset_stats(self):
        self.route_calls = 0
        self.remote_calls = 0

    def get_participant(self, participant_id: str, caller_id: str) -> 'Participant | None':
        self.route_calls += 1
        if caller_id != participant_id:
            self.remote_calls += 1
        return self.participants.get(participant_id, None)

    def get_total_states(self) -> int:
        return sum(len(participant.alignment_cache) for participant in self.participants.values())

    def register_participant(self, participant: 'Participant'):
        self.participants[participant.id] = participant
        participant.network = self

