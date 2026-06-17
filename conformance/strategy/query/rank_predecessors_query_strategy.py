import heapq
from typing import Dict, List, Optional

class Entry:
    def __init__(self, priority: int, participant: 'Participant'):
        self.priority = priority
        self.participant = participant

    def __lt__(self, other: 'Entry') -> bool:
        return self.priority < other.priority

class RankPredecessorsQueryStrategy:
    def __init__(self, initial_participants: 'List[Participant]', query_limit: Optional[int] = None, warmup: Optional[int] = None):
        self.query_limit = query_limit
        self.warmup = warmup
        self.warmup_counter = 0
        self.predecessor_count = 0
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
        if not self.queue or (self.query_limit is not None and self.returned_count >= self.query_limit):
            return False
        if self.warmup is None or self.warmup_counter < self.warmup or self.predecessor_count == 0:
            return True
        return any(self.counts[e.participant] > 0 for e in self.queue)

    def get_next_predecessor(self) -> 'Optional[Participant]':
        if self.query_limit is not None and self.returned_count >= self.query_limit:
            return None

        while self.queue:
            entry = heapq.heappop(self.queue)
            del self.entry_map[entry.participant]

            if self.warmup is not None and self.warmup_counter >= self.warmup and self.predecessor_count > 0 and self.counts[entry.participant] == 0:
                continue

            self.returned_count += 1
            if self.warmup is not None and self.warmup_counter < self.warmup:
                self.warmup_counter += 1
            return entry.participant

        return None

    def update_predecessor(self, predecessor: 'Participant'):
        old_count = self.counts.get(predecessor, 0)
        self.counts[predecessor] = old_count + 1
        if old_count == 0:
            self.predecessor_count += 1
        if predecessor in self.entry_map:
            self.entry_map[predecessor].priority = -self.counts[predecessor]
            heapq.heapify(self.queue)
        else:
            entry = Entry(-self.counts[predecessor], predecessor)
            self.entry_map[predecessor] = entry
            heapq.heappush(self.queue, entry)

    def reset(self):
        self.returned_count = 0
        self.warmup_counter = 0
        self.queue = []
        self.entry_map = {}
        for participant, count in self.counts.items():
            entry = Entry(-count, participant)
            self.entry_map[participant] = entry
            self.queue.append(entry)
        heapq.heapify(self.queue)
