from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Tuple

if TYPE_CHECKING:
    from .participant import Participant
    from .network import Network


# ---------------------------------------------------------------------------
# Query strategies
# ---------------------------------------------------------------------------

class QueryStrategy(ABC):
    @abstractmethod
    def rank_participants(
        self,
        activity: str,
        log_index: int,
        case_id: str,
        network: Network,
    ) -> List[Participant]:
        ...

    @abstractmethod
    def should_skip_participant(self, participant: Participant, best_so_far: int) -> bool:
        ...

    @abstractmethod
    def should_stop_early(self, best_so_far: int, remaining: List[Participant]) -> bool:
        ...

    def notify_computed(self, participant_id: str, case_id: str, cost: int) -> None:
        pass


class FullScanStrategy(QueryStrategy):
    def rank_participants(self, activity, log_index, case_id, network):
        return list(network.participants.values())

    def should_skip_participant(self, participant, best_so_far):
        return False

    def should_stop_early(self, best_so_far, remaining):
        return False


class ActivityFilteredStrategy(QueryStrategy):
    def __init__(self, skip_non_matching_if_best_below: int = 10**9):
        self.skip_non_matching_if_best_below = skip_non_matching_if_best_below
        self._has_match: Dict[str, bool] = {}

    def rank_participants(self, activity, log_index, case_id, network):
        self._has_match.clear()
        matching = []
        non_matching = []
        for p in network.participants.values():
            candidate = activity in network.activity_index and p in network.activity_index[activity]
            self._has_match[p.id] = candidate
            if candidate:
                matching.append(p)
            else:
                non_matching.append(p)
        return matching + non_matching

    def should_skip_participant(self, participant, best_so_far):
        return (not self._has_match.get(participant.id, False)
                and best_so_far < self.skip_non_matching_if_best_below)

    def should_stop_early(self, best_so_far, remaining):
        if not remaining:
            return True
        remaining_have_match = any(self._has_match.get(p.id, False) for p in remaining)
        return (not remaining_have_match
                and best_so_far < self.skip_non_matching_if_best_below)


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
        for p in network.participants.values():
            s = 0.0
            if activity in network.activity_index and p in network.activity_index.get(activity, []):
                s += self.weight_match
            case_counts = network.participant_event_counts.get(case_id, {})
            local_n = case_counts.get(p.id, 0)
            total_n = sum(case_counts.values())
            if total_n > 0:
                s += (local_n / total_n) * self.weight_local
            s += len(p.entrypoints) * self.weight_entrypoints / (1 + len(p.entrypoints))
            prev = self._prev_best_cost.get((p.id, case_id), 0)
            s -= (prev / 100.0) * self.weight_prev_cost
            self._scores[p.id] = s
            scored.append((s, p))
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


class TopKStrategy(QueryStrategy):
    def __init__(self, inner_strategy: QueryStrategy, k: int = 3):
        self.inner = inner_strategy
        self.k = k

    def rank_participants(self, activity, log_index, case_id, network):
        return self.inner.rank_participants(activity, log_index, case_id, network)[:self.k]

    def should_skip_participant(self, participant, best_so_far):
        return self.inner.should_skip_participant(participant, best_so_far)

    def should_stop_early(self, best_so_far, remaining):
        return len(remaining) == 0

    def notify_computed(self, participant_id, case_id, cost):
        self.inner.notify_computed(participant_id, case_id, cost)


# ---------------------------------------------------------------------------
# Cache strategies
# ---------------------------------------------------------------------------

class CacheStrategy(ABC):
    @abstractmethod
    def on_event(self, log_index: int) -> None:
        ...

    @abstractmethod
    def after_insertion(self, cache: Dict, key: Tuple) -> None:
        ...

    def max_predecessor_depth(self, entrypoint, log_index: int) -> int:
        return 0


class InfiniteCacheStrategy(CacheStrategy):
    def on_event(self, log_index):
        pass

    def after_insertion(self, cache, key):
        pass

    def max_predecessor_depth(self, entrypoint, log_index):
        return 0


class DepthLimitedCacheStrategy(CacheStrategy):
    """Wraps a CacheStrategy and limits entrypoint chain backtracking depth.

    When the predecessor chain depth exceeds *max_depth*, the DP returns an
    estimated cost instead of recursing further.  This bounds recomputation
    cost when the cache evicts entries near the head of the chain.
    """

    def __init__(self, inner: CacheStrategy, max_depth: int = 5):
        self.inner = inner
        self._max_depth = max_depth

    def on_event(self, log_index):
        self.inner.on_event(log_index)

    def after_insertion(self, cache, key):
        self.inner.after_insertion(cache, key)

    def max_predecessor_depth(self, entrypoint, log_index):
        return self._max_depth
