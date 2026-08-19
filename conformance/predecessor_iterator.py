from __future__ import annotations

from typing import Dict, List, Tuple


class PredecessorIterator:
    def __init__(
        self,
        participant_id: str,
        ranking: List[Tuple[str, float]],
        pred_map: Dict[str, float],
        threshold: float = 1.0,
    ):
        self._id = participant_id
        self._ranking = ranking
        self._pred_map = pred_map
        self._threshold = threshold
        self._idx = 0
        self._prev_cum = 0.0
        self._skip_self()

    def reset(self):
        self._idx = 0
        self._prev_cum = 0.0
        self._skip_self()

    def _skip_self(self):
        while self._idx < len(self._ranking) and self._ranking[self._idx][0] == self._id:
            self._prev_cum = self._ranking[self._idx][1]
            self._idx += 1

    def has_next(self) -> bool:
        return self._idx < len(self._ranking) and self._prev_cum < self._threshold

    def next(self) -> str:
        if not self.has_next():
            return None
        pid, cum = self._ranking[self._idx]
        self._prev_cum = cum
        self._idx += 1
        self._skip_self()
        return pid

    def is_blocked(self, participant_id: str) -> bool:
        return self._pred_map.get(participant_id, float('inf')) > self._threshold
