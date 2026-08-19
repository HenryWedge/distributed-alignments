from __future__ import annotations

from typing import Dict, List, NamedTuple, Tuple

from .constants import LOG_COST, MODEL_COST, MOVE_LOG, MOVE_MODEL, MOVE_SYNC
from .models import DecisionPath, Entrypoint

MODEL_MOVE = "model"
SYNC_MOVE = "sync"
LOG_MOVE = "log"


class IndexedEntrypoint(NamedTuple):
    entrypoint: Entrypoint
    log_index: int
    case_id: str


class AlignmentBuilder:

    def build(
        self,
        entrypoint: Entrypoint,
        log_index: int,
        predecessor_costs: Dict[str, int],
        case_id: str = "",
    ) -> int:
        key = IndexedEntrypoint(entrypoint, log_index, case_id)
        if key in self._cache:
            return self._cache[key].cost

        paths: List[DecisionPath] = []
        if MODEL_MOVE in predecessor_costs:
            paths.append(
                DecisionPath(
                    predecessor_costs[MODEL_MOVE] + MODEL_COST,
                    MOVE_MODEL,
                    entrypoint.previous_entrypoint,
                    log_index,
                )
            )
        if SYNC_MOVE in predecessor_costs:
            paths.append(
                DecisionPath(
                    predecessor_costs[SYNC_MOVE],
                    MOVE_SYNC,
                    entrypoint.previous_entrypoint,
                    log_index - 1,
                )
            )
        if LOG_MOVE in predecessor_costs:
            paths.append(
                DecisionPath(
                    predecessor_costs[LOG_MOVE] + LOG_COST,
                    MOVE_LOG,
                    entrypoint,
                    log_index - 1,
                )
            )

        best = min(paths)
        return best
