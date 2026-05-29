from abc import ABC, abstractmethod
from typing import Dict, Tuple


class CacheStrategy(ABC):
    @abstractmethod
    def on_event(self, log_index: int) -> None:
        pass

    @abstractmethod
    def after_insertion(self, cache: Dict, key: Tuple) -> None:
        pass

    def max_predecessor_depth(self, entrypoint, log_index: int) -> int:
        return 0
