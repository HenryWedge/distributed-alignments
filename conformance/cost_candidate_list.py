from __future__ import annotations

from typing import Callable, List, NamedTuple, Optional

from .constants import LOG_COST, MODEL_COST, MOVE_LOG, MOVE_MODEL, MOVE_SYNC
from .models import DecisionPath, Entrypoint


class MoveCandidate(NamedTuple):
    move_type: int
    predecessor_entrypoint: Optional[Entrypoint]
    predecessor_log_index: int
    add_cost: int

class CostCandidateList:
    def __init__(
        self,
        entrypoint: Entrypoint,
        log_index: int,
        activity: Optional[str],
        pruning: bool,
    ):
        self._pruning = pruning
        self._candidates: List[MoveCandidate] = []
        if activity is not None and activity == entrypoint.activity:
            self._candidates.append(
                MoveCandidate(MOVE_SYNC, entrypoint.previous_entrypoint, log_index - 1, 0)
            )
        self._candidates.append(
            MoveCandidate(MOVE_MODEL, entrypoint.previous_entrypoint, log_index, MODEL_COST)
        )
        if log_index >= 0:
            self._candidates.append(
                MoveCandidate(MOVE_LOG, entrypoint, log_index - 1, LOG_COST)
            )

    def best(
        self,
        request_cost: Callable[[Optional[Entrypoint], int], int],
    ) -> DecisionPath:
        first = self._candidates[0]
        predecessor_cost = request_cost(first.predecessor_entrypoint, first.predecessor_log_index)
        best_cost = predecessor_cost + first.add_cost
        best = DecisionPath(best_cost, first.move_type, first.predecessor_entrypoint, first.predecessor_log_index)

        for candidate in self._candidates[1:]:
            if self._pruning and candidate.add_cost >= best_cost:
                continue
            predecessor_cost = request_cost(candidate.predecessor_entrypoint, candidate.predecessor_log_index)
            total = predecessor_cost + candidate.add_cost
            if total < best_cost:
                best_cost = total
                best = DecisionPath(
                    total,
                    candidate.move_type,
                    candidate.predecessor_entrypoint,
                    candidate.predecessor_log_index
                )
                if self._pruning and best_cost == 0:
                    break

        return best
