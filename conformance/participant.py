from __future__ import annotations

import sys
from typing import Dict, List, Tuple, Optional

from .constants import LOG_COST, MODEL_COST, MOVE_LOG, MOVE_MODEL, MOVE_SYNC
from .models import Entrypoint, DecisionPath, Alignment, AlignmentResult, Event
from .network import Network
from .predecessor_iterator import PredecessorIterator
from .alignment_builder import AlignmentBuilder, IndexedEntrypoint, MODEL_MOVE, SYNC_MOVE, LOG_MOVE
from .cost_candidate_list import CostCandidateList


class Participant:
    def __init__(
            self,
            participant_id: str,
            network: Network,
            entrypoints: List[Entrypoint],
            predecessor_ranking: List[Tuple[str, float]],
            predecessor_map: Dict[str, float],
            max_depth: Optional[int] = None,
            pruning: bool = True,
            predecessor_threshold: float = 1.0,
    ):
        self.id = participant_id
        self.entrypoints = entrypoints
        self.alignment_cache: Dict[Tuple, DecisionPath] = {}
        self.network = network
        self.local_events: Dict[Tuple[str, int], str] = {}
        self.max_depth = max_depth
        self.pruning = pruning
        self.predecessor_threshold = predecessor_threshold
        self._predecessors = PredecessorIterator(
            participant_id, predecessor_ranking, predecessor_map, predecessor_threshold
        )
        self._alignment_builder = AlignmentBuilder()
        self._query_count = 0

    def _get_activity_at(self, case_id: str, log_index: int) -> Optional[str]:
        return self.local_events.get((case_id, log_index))

    def process_event(self, event: Event, log_index: int) -> Alignment:
        self.local_events[(event.case_id, log_index)] = event.activity

        if not self.entrypoints:
            return self._align_unseen_entrypoint(event, log_index)
        best_alignment_result = AlignmentResult.max()

        result = self.compute_best_cost(log_index, event.case_id).with_participant(self)
        if result < best_alignment_result:
            best_alignment_result = result
        self._predecessors.reset()

        while self._predecessors.has_next():
            participant_id = self._predecessors.next()
            participant = self.network.get_participant(participant_id, self.id)
            if participant is not None:
                result = participant.compute_best_cost(log_index, event.case_id).with_participant(participant)
                if result < best_alignment_result:
                    best_alignment_result = result

        if best_alignment_result.is_valid():
            return best_alignment_result.participant.reconstruct_alignment(
                best_alignment_result.entrypoint, log_index, event.case_id
            )

        return Alignment([(None, "?") for _ in range(log_index + 1)])

    def _align_unseen_entrypoint(self, event: Event, log_index: int) -> Alignment:
        if log_index == 0:
            return Alignment([(None, event.activity)])

        best_result = AlignmentResult.max()
        for participant in self.network.participants.values():
            if participant is self:
                continue
            result = participant.compute_best_cost(log_index - 1, event.case_id)
            if result < best_result:
                best_result = result

        if best_result.is_valid():
            alignment = best_result.participant.reconstruct_alignment(
                best_result.entrypoint, log_index - 1, event.case_id
            )
        else:
            alignment = Alignment()

        return alignment.add_log_move(event.activity)

    def request_predecessor_cost(
            self,
            entrypoint: Entrypoint | None,
            log_index: int,
            case_id: str
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
            case_id: str
    ) -> int:
        if entrypoint is None:
            return LOG_COST * (log_index + 1)

        key = IndexedEntrypoint(entrypoint, log_index, case_id)
        if key in self.alignment_cache:
            return self.alignment_cache[key].cost

        activity = self._get_activity_at(case_id, log_index)
        candidates = CostCandidateList(entrypoint, log_index, activity, self.pruning)

        best = candidates.best(
            lambda ep, li: self.request_predecessor_cost(ep, li, case_id)
        )

        self.alignment_cache[key] = best
        return best.cost

    def calculate_costs(
        self,
        entries: List[Tuple[Entrypoint | None, int]],
        case_id: str
    ) -> List[int]:
        return [self.calculate_cost(entrypoint, log_index, case_id) for entrypoint, log_index in entries]

    def request_predecessor_alignment(
            self,
            entrypoint: Entrypoint | None,
            log_index: int,
            case_id: str
    ) -> Alignment:
        if entrypoint is None:
            return Alignment([(None, "?") for _ in range(log_index + 1)])

        if self._predecessors.is_blocked(entrypoint.participant_id):
            return Alignment([(None, "?") for _ in range(log_index + 1)])

        return self.network.get_participant(entrypoint.participant_id, self.id).reconstruct_alignment(
            entrypoint, log_index, case_id
        )

    def compute_best_cost(self, log_index: int, case_id: str) -> AlignmentResult:
        if not self.entrypoints:
            return AlignmentResult.max()

        if not self.pruning:
            best_cost = sys.maxsize
            best_ep = None
            for ep in self.entrypoints:
                cost = self.calculate_cost(ep, log_index, case_id)
                if cost < best_cost:
                    best_cost = cost
                    best_ep = ep
            return AlignmentResult(best_cost, best_ep)

        sorted_eps = sorted(self.entrypoints, key=lambda ep: -ep.frequency)

        first_ep = sorted_eps[0]
        best_cost = self.calculate_cost(first_ep, log_index, case_id)
        best_ep = first_ep

        if best_cost > 0:
            for ep in sorted_eps[1:]:
                lb = abs(ep.depth - log_index)
                if lb >= best_cost:
                    continue
                cost = self.calculate_cost(ep, log_index, case_id)
                if cost < best_cost:
                    best_cost = cost
                    best_ep = ep
                    if best_cost == 0:
                        break

        return AlignmentResult(best_cost, best_ep)

    def reconstruct_alignment(
            self,
            entrypoint: Entrypoint | None,
            log_index: int,
            case_id: str
    ) -> Alignment:
        if entrypoint is None:
            return Alignment([(None, "?") for _ in range(log_index + 1)])

        key = IndexedEntrypoint(entrypoint, log_index, case_id)
        if key not in self.alignment_cache:
            self.calculate_cost(entrypoint, log_index, case_id)

        decision = self.alignment_cache[key]
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