from abc import ABC, abstractmethod
from typing import List

from conformance import Network


class QueryStrategy(ABC):

    @abstractmethod
    def rank_participants(
        self,
        activity: str,
        log_index: int,
        case_id: str,
        network: Network,
    ) -> 'List[Participant]':
        pass

    @abstractmethod
    def should_skip_participant(self, participant: 'Participant', best_so_far: int) -> bool:
        pass

    @abstractmethod
    def should_stop_early(self, best_so_far: int, remaining: 'List[Participant]') -> bool:
        pass

    def notify_computed(self, participant_id: str, case_id: str, cost: int) -> None:
        pass
