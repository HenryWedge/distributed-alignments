from __future__ import annotations

from typing import Dict, List, Tuple


class InstrumentedNetwork:
    """Wraps a pure Network and counts get_participant calls.

    ``__getattr__`` delegates everything else (participants, activity_index, …)
    to the wrapped Network so the wrapper is a drop-in replacement.
    """

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
    """Wraps a Participant and counts compute_best_cost calls.

    All other attribute access (entrypoints, decision_cache, …) is delegated
    to the real participant via __getattr__.
    """

    def __init__(self, participant):
        self._p = participant
        self.compute_calls = 0

    def compute_best_cost(self, log_index, case_id):
        self.compute_calls += 1
        return self._p.compute_best_cost(log_index, case_id)

    def __getattr__(self, name):
        return getattr(self._p, name)

    @property
    def id(self):
        return self._p.id


def instrument_network(network):
    """Replace every Participant in *network* with an InstrumentedParticipant
    and update their ``.network`` reference to an InstrumentedNetwork.

    Returns the InstrumentedNetwork (callers can read ``.route_calls``,
    ``.remote_calls`` from it and ``.reset()`` it).
    """
    inst_net = InstrumentedNetwork(network)
    for pid, p in list(network.participants.items()):
        wrapped = InstrumentedParticipant(p)
        network.participants[pid] = wrapped
        p.network = inst_net
    return inst_net
