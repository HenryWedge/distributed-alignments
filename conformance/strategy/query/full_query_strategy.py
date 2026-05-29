from conformance.strategy.query.abstract_query_strategy import QueryStrategy


class FullScanStrategy(QueryStrategy):
    def rank_participants(self, activity, log_index, case_id, network):
        return list(network.participants.values())

    def should_skip_participant(self, participant, best_so_far):
        return False

    def should_stop_early(self, best_so_far, remaining):
        return False