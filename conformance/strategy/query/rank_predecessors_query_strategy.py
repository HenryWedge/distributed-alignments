import heapq
from typing import Dict, List, Optional

class Entry:
    def __init__(self, priority: int, participant: 'Participant'):
        self.priority = priority
        self.participant = participant

    def __lt__(self, other: 'Entry') -> bool:
        return self.priority < other.priority

class RankPredecessorsQueryStrategy:
    def __init__(self, initial_participants: 'List[Participant]', query_limit: Optional[int] = None):
        self.query_limit = query_limit
        self.returned_count = 0
        self.counts: 'Dict[Participant, int]' = {}
        self.queue: 'List[Entry]' = []
        self.entry_map: 'Dict[Participant, Entry]' = {}
        for participant in initial_participants:
            self.counts[participant] = 0
            entry = Entry(0, participant)
            self.entry_map[participant] = entry
            self.queue.append(entry)
        heapq.heapify(self.queue)

    def has_next(self) -> bool:
        return bool(self.queue) and (self.query_limit is None or self.returned_count < self.query_limit)

    def get_next_predecessor(self) -> 'Optional[Participant]':
        if not self.has_next():
            return None

        entry = heapq.heappop(self.queue)
        del self.entry_map[entry.participant]
        self.returned_count += 1
        return entry.participant

    def update_predecessor(self, predecessor: 'Participant'):
        self.counts[predecessor] = self.counts.get(predecessor, 0) + 1
        if predecessor in self.entry_map:
            self.entry_map[predecessor].priority = -self.counts[predecessor]
            heapq.heapify(self.queue)
        else:
            entry = Entry(-self.counts[predecessor], predecessor)
            self.entry_map[predecessor] = entry
            heapq.heappush(self.queue, entry)

    def reset(self):
        self.returned_count = 0
        self.queue = []
        self.entry_map = {}
        for participant, count in self.counts.items():
            entry = Entry(-count, participant)
            self.entry_map[participant] = entry
            self.queue.append(entry)
        heapq.heapify(self.queue)
