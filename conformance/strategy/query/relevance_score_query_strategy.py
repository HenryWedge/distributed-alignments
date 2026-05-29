from typing import Dict, Tuple

from conformance.strategy.query.abstract_query_strategy import QueryStrategy


class RelevanceScoreStrategy(QueryStrategy):
    def __init__(
        self,
        weight_match: float = 10.0,
        weight_local: float = 5.0,
        weight_entrypoints: float = 1.0,
        weight_prev_cost: float = 3.0,
        min_score_ratio: float = 0.0,
        min_score_threshold: float = 0.0,
    ):
        self.weight_match = weight_match
        self.weight_local = weight_local
        self.weight_entrypoints = weight_entrypoints
        self.weight_prev_cost = weight_prev_cost
        self.min_score_ratio = min_score_ratio
        self.min_score_threshold = min_score_threshold
        self._scores: Dict[str, float] = {}
        self._prev_best_cost: Dict[Tuple[str, str], int] = {}

    def rank_participants(self, activity, log_index, case_id, network):
        self._scores.clear()
        scored = []
        for participant in network.participants.values():
            score = 0.0
            if activity in network.activity_index and participant in network.activity_index.get(activity, []):
                score += self.weight_match
            case_counts = network.participant_event_counts.get(case_id, {})
            local_n = case_counts.get(participant.id, 0)
            total_n = sum(case_counts.values())
            if total_n > 0:
                score += (local_n / total_n) * self.weight_local
            score += len(participant.entrypoints) * self.weight_entrypoints / (1 + len(participant.entrypoints))
            prev = self._prev_best_cost.get((participant.id, case_id), 0)
            score -= (prev / 100.0) * self.weight_prev_cost
            self._scores[participant.id] = score
            scored.append((score, participant))
        scored.sort(key=lambda x: -x[0])
        return [p for _, p in scored]

    def notify_computed(self, participant_id, case_id, cost):
        self._prev_best_cost[(participant_id, case_id)] = cost

    def should_skip_participant(self, participant, best_so_far):
        return self._scores.get(participant.id, 0) < self.min_score_threshold

    def should_stop_early(self, best_so_far, remaining):
        if not remaining:
            return True
        next_score = self._scores.get(remaining[0].id, 0)
        best_score = max(self._scores.values()) if self._scores else 0
        return best_score > 0 and next_score < best_score * self.min_score_ratio
