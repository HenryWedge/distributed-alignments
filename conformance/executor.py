from __future__ import annotations

from typing import Dict

from utility.event import Event
from .network import Network
from .models import Alignment
from .participant import Participant

class Executor:
    def __init__(self, network: Network, participant_mapping: Dict[str, str]):
        self.network = network
        self.participant_mapping = participant_mapping
        self.case_event_count: Dict[str, int] = {}

    def record_event(self, event: Event) -> Alignment:
        count = self.case_event_count.get(event.case_id, 0) + 1
        self.case_event_count[event.case_id] = count
        log_index = count - 1
        participant_id = self.participant_mapping.get(event.activity)
        participant: Participant = self.network.get_participant(participant_id, None)
        if participant is None:
            new_id = f"_unknown_{event.activity}"
            participant = Participant(new_id, self.network, max_depth=None, predecessor_threshold=1.0)
            self.network.register_participant(participant)
            self.participant_mapping[event.activity] = new_id
        return participant.process_event(event, log_index)
