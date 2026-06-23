from __future__ import annotations

from typing import Dict, List, Tuple, Optional, NamedTuple

from .constants import LOG_COST, MODEL_COST, MOVE_LOG, MOVE_MODEL, MOVE_SYNC
from .models import Entrypoint, DecisionPath, Alignment, AlignmentResult, Event
from .network import Network
from .strategy.query.rank_predecessors_query_strategy import RankPredecessorsQueryStrategy

MODEL_MOVE = 'model'
SYNC_MOVE = 'sync'
LOG_MOVE = 'log'


class IndexedEntrypoint(NamedTuple):
    entrypoint: Entrypoint
    log_index: int


class Participant:
    def __init__(
            self,
            participant_id: str,
            network: Network,
            query_strategy: 'Optional[QueryStrategy]' = None,
            max_depth: Optional[int] = None,
    ):
        self.id = participant_id
        self.entrypoints: List[Entrypoint] = []
        self.alignment_cache: Dict[Tuple, DecisionPath] = {}
        self.network = network
        self.local_events: Dict[Tuple[str, int], str] = {}
        # TOD hrei: insert
        # self.query_strategy = query_strategy
        self.query_strategy = RankPredecessorsQueryStrategy(list(self.network.participants.values()), None, warmup=0)
        self.max_depth = max_depth

    def _get_activity_at(self, case_id: str, log_index: int) -> Optional[str]:
        return self.local_events.get((case_id, log_index))

    def process_event(self, event: Event, log_index: int) -> Alignment:
        self.local_events[(event.case_id, log_index)] = event.activity

        if not self.entrypoints:
            if log_index == 0:
                return Alignment([(None, event.activity)])

            best_result = AlignmentResult.max()
            for p in self.network.participants.values():
                if p is self:
                    continue
                result = p.compute_best_cost(log_index - 1, event.case_id)
                if result < best_result:
                    best_result = result

            if best_result.is_valid():
                alignment = best_result.participant.reconstruct_alignment(
                    best_result.entrypoint, log_index - 1, event.case_id
                )
            else:
                alignment = Alignment()

            return alignment.add_log_move(event.activity)

        best_alignment_result = AlignmentResult.max()

        self.query_strategy.reset()
        while self.query_strategy.has_next():
            participant = self.query_strategy.get_next_predecessor()
            result = participant.compute_best_cost(log_index, event.case_id).with_participant(participant)
            if result < best_alignment_result:
                best_alignment_result = result

        if best_alignment_result.participant is not None:
            self.query_strategy.update_predecessor(best_alignment_result.participant)

        if best_alignment_result.is_valid:
            return best_alignment_result.participant.reconstruct_alignment(
                best_alignment_result.entrypoint, log_index, event.case_id
            )

        return Alignment([(None, "?") for _ in range(log_index + 1)])

    def request_predecessor_cost(
            self,
            entrypoint: Entrypoint | None,
            log_index: int,
            case_id: str,
            depth: int = 0,
    ) -> int:
        if entrypoint is None or self.is_max_depth_exceeded(depth):
            return LOG_COST * (log_index + 1)

        return self.network.get_participant(entrypoint.participant_id, self.id).calculate_cost(
            entrypoint, log_index, case_id, depth
        )

    def _build_alignment(
            self,
            entrypoint: Entrypoint,
            log_index: int,
            predecessor_costs: Dict[str, int],
    ) -> int:
        key = IndexedEntrypoint(entrypoint, log_index)
        if key in self.alignment_cache:
            return self.alignment_cache[key].cost

        paths = []
        if MODEL_MOVE in predecessor_costs:
            paths.append(
                DecisionPath(
                    predecessor_costs[MODEL_MOVE] + MODEL_COST,
                    MOVE_MODEL,
                    entrypoint.previous_entrypoint,
                    log_index
                )
            )
        if SYNC_MOVE in predecessor_costs:
            paths.append(
                DecisionPath(
                    predecessor_costs[SYNC_MOVE],
                    MOVE_SYNC,
                    entrypoint.previous_entrypoint,
                    log_index - 1
                )
            )
        if LOG_MOVE in predecessor_costs:
            paths.append(
                DecisionPath(
                    predecessor_costs[LOG_MOVE] + LOG_COST,
                    MOVE_LOG,
                    entrypoint,
                    log_index - 1
                )
            )

        best = min(paths)
        self.alignment_cache[key] = best
        return best.cost

    def is_max_depth_exceeded(self, depth):
        return self.max_depth is not None and depth > self.max_depth

    def calculate_cost(
            self,
            entrypoint: Entrypoint | None,
            log_index: int,
            case_id: str,
            depth: int = 0,
    ) -> int:
        if entrypoint is None or self.is_max_depth_exceeded(depth):
            return LOG_COST * (log_index + 1)

        if IndexedEntrypoint(entrypoint, log_index) in self.alignment_cache:
            return self.alignment_cache[IndexedEntrypoint(entrypoint, log_index)].cost

        activity = self._get_activity_at(case_id, log_index)

        predecessor_cost = {}
        predecessor_cost[MODEL_MOVE] = self.request_predecessor_cost(
            entrypoint.previous_entrypoint, log_index, case_id, depth + 1)
        if activity is not None and activity == entrypoint.activity:
            predecessor_cost[SYNC_MOVE] = self.request_predecessor_cost(
                entrypoint.previous_entrypoint, log_index - 1, case_id, depth + 1)
        if log_index >= 0:
            predecessor_cost[LOG_MOVE] = self.request_predecessor_cost(
                entrypoint, log_index - 1, case_id, depth + 1)

        return self._build_alignment(entrypoint, log_index, predecessor_cost)

    def calculate_costs(
        self,
        entries: List[Tuple[Entrypoint | None, int]],
        case_id: str,
        depth: int = 0
    ) -> List[int]:
        return [self.calculate_cost(ep, li, case_id, depth) for ep, li in entries]

    def request_predecessor_alignment(
            self,
            entrypoint: Entrypoint | None,
            log_index: int,
            case_id: str,
            depth: int = 0,
    ) -> Alignment:
        if entrypoint is None or self.is_max_depth_exceeded(depth):
            return Alignment([(None, "?") for _ in range(log_index + 1)])

        return self.network.get_participant(entrypoint.participant_id, self.id).reconstruct_alignment(
            entrypoint, log_index, case_id, depth
        )

    def reconstruct_alignment(
            self,
            entrypoint: Entrypoint | None,
            log_index: int,
            case_id: str,
            depth: int = 0,
    ) -> Alignment:
        if entrypoint is None or self.is_max_depth_exceeded(depth):
            return Alignment([(None, "?") for _ in range(log_index + 1)])

        key = IndexedEntrypoint(entrypoint, log_index)
        if key not in self.alignment_cache:
            self.calculate_cost(entrypoint, log_index, case_id, depth)

        decision = self.alignment_cache[key]
        if decision.alignment is not None:
            return decision.alignment

        predecessor_alignment = self.request_predecessor_alignment(
            decision.predecessor_entrypoint, decision.predecessor_log_index, case_id, depth + 1
        )

        if decision.move_type == MOVE_LOG:
            activity = self._get_activity_at(case_id, log_index)
            align = predecessor_alignment.add_log_move(activity if activity is not None else "?")
        elif decision.move_type == MOVE_MODEL:
            align = predecessor_alignment.add_model_move(entrypoint.activity)
        else:
            align = predecessor_alignment.add_sync_move(entrypoint.activity)

        decision.alignment = align
        return align

    def compute_best_cost(self, log_index: int, case_id: str, depth: int = 0) -> AlignmentResult:
        if not self.entrypoints:
            return AlignmentResult.max()
        entries = [IndexedEntrypoint(ep, log_index) for ep in self.entrypoints]
        costs = self.calculate_costs(entries, case_id, depth)
        best_idx = min(range(len(costs)), key=lambda i: costs[i])
        return AlignmentResult(costs[best_idx], self.entrypoints[best_idx])

    def add_entrypoint(self, entrypoint: Entrypoint):
        self.entrypoints.append(entrypoint)

    def __lt__(self, other):
        return True
