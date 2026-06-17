from __future__ import annotations

import heapq


class InstrumentedNetwork:
    def __init__(self, network):
        self._net = network
        self.route_calls = 0
        self.remote_calls = 0

    def get_participant(self, participant_id, caller_id=None):
        self.route_calls += 1
        if caller_id is not None and caller_id != participant_id:
            self.remote_calls += 1
        return self._net.participants.get(participant_id)

    def __getattr__(self, name):
        return getattr(self._net, name)

    def reset(self):
        self.route_calls = 0
        self.remote_calls = 0


class InstrumentedParticipant:
    def __init__(self, participant):
        self.participant = participant
        self.compute_calls = 0

    def compute_best_cost(self, log_index, case_id):
        self.compute_calls += 1
        return self.participant.compute_best_cost(log_index, case_id)

    def __getattr__(self, name):
        return getattr(self.participant, name)

    @property
    def id(self):
        return self.participant.id


class InstrumentedQueryStrategy:
    def __init__(self, strategy):
        self._strategy = strategy
        self.total_queries = 0

    def reset(self):
        self._strategy.reset()

    def has_next(self):
        return self._strategy.has_next()

    def get_next_predecessor(self):
        p = self._strategy.get_next_predecessor()
        if p is not None:
            self.total_queries += 1
        return p

    def update_predecessor(self, predecessor):
        self._strategy.update_predecessor(predecessor)

    def should_query_participant(self, participant_id):
        return self._strategy.should_query_participant(participant_id)


def instrument_network(network):
    from .strategy.query.rank_predecessors_query_strategy import RankPredecessorsQueryStrategy
    from .strategy.query.rank_predecessors_query_strategy import Entry

    inst_net = InstrumentedNetwork(network)

    for pid, p in list(network.participants.items()):
        wrapped = InstrumentedParticipant(p)
        network.participants[pid] = wrapped
        p.network = inst_net

    all_wrapped = list(network.participants.values())
    for wrapped in all_wrapped:
        orig = wrapped.participant
        old = orig.query_strategy
        new_strat = RankPredecessorsQueryStrategy(all_wrapped, old.query_limit, old.warmup)
        orig.query_strategy = InstrumentedQueryStrategy(new_strat)

    return inst_net
