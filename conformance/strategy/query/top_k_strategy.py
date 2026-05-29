from conformance.strategy.query.abstract_query_strategy import QueryStrategy


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
