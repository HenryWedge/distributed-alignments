from typing import Dict, List
from utility.event import Event

class EventLog:

    def __init__(self, traces: Dict[str, List[Event]] = None):
        if traces:
            self.traces: Dict[str, List[Event]] = traces
        else:
            self.traces: Dict[str, List[Event]] = {}

    def add_event(self, event: Event):
        if not event.case_id in self.traces:
            self.traces[event.case_id] = []
        self.traces[event.case_id].append(event)

    def has_event_in_case(self, case_id, activity) -> bool:
        if case_id not in self.traces:
            return False
        return True in [event.activity == activity for event in self.traces[case_id]]

    def filter_case_ids(self, case_ids: List[str]):
        new_event_log = EventLog()
        for case_id in self.traces:
            if case_id in case_ids:
                for event in self.traces[case_id]:
                    new_event_log.add_event(event)
        return new_event_log