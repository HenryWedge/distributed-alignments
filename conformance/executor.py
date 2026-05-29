from __future__ import annotations

from typing import Dict

from utility.event import Event
from . import Participant
from .network import Network
from .models import Alignment


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
            return Alignment([(None, "?") for _ in range(log_index + 1)])
        return participant.process_event(event, log_index)
