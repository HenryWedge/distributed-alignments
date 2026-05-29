from pm4py.objects.log.obj import EventLog as Pm4PyEventLog, Trace
from pm4py.objects.log.obj import Event as Pm4PyEvent
from pm4py.util import constants
import pandas as pd

from utility.event import Event

from utility.event_log import EventLog

class Converter:
    def to_event_log(self, event_log: Pm4PyEventLog, location_key=None) -> EventLog:
        el = EventLog()
        for trace in event_log:
            occurrence = 0
            for event in trace:
                if location_key not in event:
                    print(f"Location Key Missing {event["concept:name"]}")
                    continue
                if el.has_event_in_case(trace._attributes["concept:name"], f"{event["concept:name"]}-{event[location_key]}{occurrence}"):
                    occurrence = occurrence + 1
                el.add_event(self.to_event_location_aware(event, trace._attributes["concept:name"], location_key, occurrence))
        return el

    def to_event_location_aware(self, pm4py_event: Pm4PyEvent, case_id, location_key=None, occurrence="") -> Event:
        location = pm4py_event[location_key] if location_key and location_key in pm4py_event else ""
        return Event(
            time=pm4py_event["time:timestamp"],
            activity=f"{pm4py_event["concept:name"]}",
            case_id=case_id,
            location=location
        )

    def to_event(self, pm4py_event: Pm4PyEvent, case_id, location_key=None, occurrence="") -> Event:
        location = pm4py_event[location_key] if location_key and location_key in pm4py_event else ""
        return Event(
            time=pm4py_event["time:timestamp"],
            activity=f"{pm4py_event["concept:name"]}",
            case_id=case_id,
            location=location
        )

    def from_event_log(self, el: EventLog) -> Pm4PyEventLog:
        event_log = Pm4PyEventLog()
        traces = {}

        for case_id, events in el.traces.items():
            if case_id not in traces:
                new_trace = Trace()
                new_trace.attributes[constants.CASE_CONCEPT_NAME] = case_id
                traces[case_id] = new_trace
            for event in events:
                traces[case_id].append(self.from_event(event))

        for trace in traces:
            event_log.append(traces[trace])

        return Pm4PyEventLog(event_log)

    def from_event(self, event: Event) -> Pm4PyEvent:
        pm4py_event = Pm4PyEvent()
        pm4py_event["concept:name"] = event.activity
        pm4py_event["time:timestamp"] = pd.Timestamp(event.time)
        # pm4pyEvent["resource"] = event.node
        pm4py_event["case:concept:name"] = event.case_id
        return pm4py_event
