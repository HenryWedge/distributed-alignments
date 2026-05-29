import random
import pm4py
from utility.converter import Converter


class EventLogSplitter:

    def __init__(self, file_path: str, training_split: int = 0.8, location_key: str = "org:group") -> None:
        self.converter = Converter()
        self.training_split = training_split
        self.log = self._read_xes_log(file_path, location_key)
        self.case_ids = self._get_case_ids()
        self.split_index = self._calculate_split_index()

    def _read_xes_log(self, file_path, location_key="org:group"):
        return self.converter.to_event_log(
            pm4py.read_xes(file_path, return_legacy_log_object=True),
            location_key=location_key
        )

    def _get_case_ids(self):
        return list(self.log.traces.keys())

    def _calculate_split_index(self):
        random.seed(1)
        random.shuffle(self.case_ids)

        split_index = int(len(self.case_ids) * 0.8)
        return split_index

    def get_training_data(self, number_of_records: int):
        return self.log.filter_case_ids(self.case_ids[:number_of_records])

    def get_test_data(self, record_index: int = 1):
        return self.log.filter_case_ids(self.case_ids[record_index])

    def get_case(self, case_id: str):
        return self.log.filter_case_ids([case_id])