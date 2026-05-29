from __future__ import annotations

import sys
from typing import Dict, List, Tuple, Optional

from .constants import LOG_COST, MODEL_COST, MOVE_LOG, MOVE_MODEL, MOVE_SYNC
from .models import Entrypoint, DecisionPath, Alignment, Event
from .network import Network
from .strategies import InfiniteCacheStrategy
from .strategy.query.full_query_strategy import FullScanStrategy


def _compute_chain_depth(entrypoint: Optional[Entrypoint]) -> int:
    depth = 0
    while entrypoint is not None:
        depth += 1
        entrypoint = entrypoint.previous_entrypoint
    return depth


class Participant:
    def __init__(
        self,
        participant_id: str,
        network: Network,
        query_strategy: 'Optional[QueryStrategy]' = None,
        cache_strategy: 'Optional[CacheStrategy]' = None,
    ):
        self.id = participant_id
        self.entrypoints: List[Entrypoint] = []
        self.decision_cache: Dict[Tuple, DecisionPath] = {}
        self.network = network
        self.local_events: Dict[Tuple[str, int], str] = {}
        self.query_strategy = query_strategy if query_strategy is not None else FullScanStrategy()
        self.cache_strategy = cache_strategy if cache_strategy is not None else InfiniteCacheStrategy()

    def _get_activity_at(self, case_id: str, log_index: int) -> Optional[str]:
        return self.local_events.get((case_id, log_index))

    def process_event(self, event: Event, log_index: int) -> Alignment:
        self.local_events[(event.case_id, log_index)] = event.activity
        self.network.record_local_event(self.id, event.case_id)
        self.cache_strategy.on_event(log_index)

        best_cost = sys.maxsize
        best_participant = None
        best_entrypoint = None

        ranked = self.query_strategy.rank_participants(
            event.activity, log_index, event.case_id, self.network
        )

        for index, participant in enumerate(ranked):
            if self.query_strategy.should_skip_participant(participant, best_cost):
                continue

            cost, entrypoint = participant.compute_best_cost(log_index, event.case_id)
            self.query_strategy.notify_computed(participant.id, event.case_id, cost)

            if cost < best_cost:
                best_cost = cost
                best_participant = participant
                best_entrypoint = entrypoint

            if self.query_strategy.should_stop_early(best_cost, ranked[index + 1:]):
                break

        if best_participant and best_entrypoint is not None:
            return best_participant.reconstruct_alignment(
                best_entrypoint, log_index, event.case_id
            )
        return Alignment([(None, "?") for _ in range(log_index + 1)])


    def add_entrypoint(self, entrypoint: Entrypoint):
        self.entrypoints.append(entrypoint)

    def request_predecessor_cost(
        self,
        entrypoint: Entrypoint | None,
        log_index: int,
        case_id: str,
    ) -> int:
        if entrypoint is None:
            return LOG_COST * (log_index + 1)

        return self.network.get_participant(entrypoint.participant_id, self.id).calculate_cost(
            entrypoint, log_index, case_id
        )

    def calculate_cost(
        self,
        entrypoint: Entrypoint | None,
        log_index: int,
        case_id: str,
    ) -> int:
        if entrypoint is None:
            return LOG_COST * (log_index + 1)

        max_depth = self.cache_strategy.max_predecessor_depth(entrypoint, log_index)
        if max_depth > 0:
            depth = _compute_chain_depth(entrypoint)
            # REVIEW: Think about this. May be error prone
            if depth > max_depth:
                return LOG_COST * (log_index + 1) + MODEL_COST * (depth - max_depth)

        activity = self._get_activity_at(case_id, log_index)
        key = (entrypoint, log_index)
        if key in self.decision_cache:
            return self.decision_cache[key].cost

        possible_decision_paths = []

        pre_cost = self.request_predecessor_cost(entrypoint.previous_entrypoint, log_index, case_id)
        possible_decision_paths.append(
            DecisionPath(pre_cost + MODEL_COST, MOVE_MODEL, entrypoint.previous_entrypoint, log_index)
        )

        if activity is not None and activity == entrypoint.activity:
            pre_cost = self.request_predecessor_cost(entrypoint.previous_entrypoint, log_index - 1, case_id)
            possible_decision_paths.append(
                DecisionPath(pre_cost, MOVE_SYNC, entrypoint.previous_entrypoint, log_index - 1)
            )

        if log_index >= 0:
            pre_cost = self.request_predecessor_cost(entrypoint, log_index - 1, case_id)
            possible_decision_paths.append(
                DecisionPath(pre_cost + LOG_COST, MOVE_LOG, entrypoint, log_index - 1)
            )

        best_decision = min(possible_decision_paths)
        self.decision_cache[key] = best_decision
        self.cache_strategy.after_insertion(self.decision_cache, key)
        return best_decision.cost

    def request_predecessor_alignment(
        self,
        entrypoint: Entrypoint | None,
        log_index: int,
        case_id: str,
    ) -> Alignment:
        if entrypoint is None:
            return Alignment([(None, "?") for _ in range(log_index + 1)])

        return self.network.get_participant(entrypoint.participant_id, self.id).reconstruct_alignment(
            entrypoint, log_index, case_id
        )

    def reconstruct_alignment(
        self,
        entrypoint: Entrypoint | None,
        log_index: int,
        case_id: str,
    ) -> Alignment:
        if entrypoint is None:
            return Alignment([(None, "?") for _ in range(log_index + 1)])

        key = (entrypoint, log_index)
        if key not in self.decision_cache:
            self.calculate_cost(entrypoint, log_index, case_id)

        decision = self.decision_cache[key]
        if decision.alignment is not None:
            return decision.alignment

        predecessor_alignment = self.request_predecessor_alignment(
            decision.predecessor_entrypoint, decision.predecessor_log_index, case_id
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

    def compute_best_cost(self, log_index: int, case_id: str) -> Tuple[int, object]:
        best_cost = sys.maxsize
        best_entrypoint = None
        for entrypoint in self.entrypoints:
            cost = self.calculate_cost(entrypoint, log_index, case_id)
            if cost < best_cost:
                best_cost = cost
                best_entrypoint = entrypoint
        return best_cost, best_entrypoint
