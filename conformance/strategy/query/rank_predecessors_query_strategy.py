import heapq
from typing import List

from conformance.participant import Participant

class RankPredecessorsQueryStrategy:
    def __init__(self, initial_participants: List[Participant], query_limit: int | None = None):
        self.query_limit = query_limit
        self.returned_count = 0
        self.counts = {}
        self.participant_queue = []
        for participant in initial_participants:
            self.counts[participant] = 0
            self.participant_queue.append((0, participant))

    def get_next_predecessor(self) -> Participant | None:
        if self.query_limit is not None and self.returned_count >= self.query_limit:
            return None
        while self.participant_queue:
            neg_count, participant = heapq.heappop(self.participant_queue)
            if -neg_count == self.counts[participant]:
                self.returned_count += 1
                return participant
        return None

    def update_predecessor(self, predecessor: Participant):
        self.counts[predecessor] = self.counts.get(predecessor, 0) + 1
        heapq.heappush(
            self.participant_queue,
            (-self.counts[predecessor], predecessor)
        )

    def reset(self):
        self.returned_count = 0
        self.participant_queue = []
        for participant, count in self.counts.items():
            heapq.heappush(self.participant_queue, (-count, participant))
