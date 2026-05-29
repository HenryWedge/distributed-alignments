from __future__ import annotations

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


def instrument_network(network):
    inst_net = InstrumentedNetwork(network)
    for pid, p in list(network.participants.items()):
        wrapped = InstrumentedParticipant(p)
        network.participants[pid] = wrapped
        p.network = inst_net
    return inst_net
