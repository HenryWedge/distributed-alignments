import hashlib
from typing import Dict, List, Tuple, Optional

from .constants import LOG_COST, MODEL_COST


class Event:
    def __init__(self, activity: str, case_id: str, location: str, timestamp):
        self.activity = activity
        self.case_id = case_id
        self.location = location
        self.timestamp = timestamp

    def __str__(self) -> str:
        return self.activity


class Entrypoint:
    def __init__(self, previous_entrypoint: 'Entrypoint | None', activity: str, participant_id: str):
        self.previous_entrypoint = previous_entrypoint
        self.activity = activity
        self.participant_id = participant_id
        self.hash = self.compute_hash()

    def compute_hash(self) -> str:
        if self.previous_entrypoint is None:
            return hashlib.sha256(self.activity.encode()).hexdigest()
        return hashlib.sha256((self.previous_entrypoint.hash + self.activity).encode()).hexdigest()


class DecisionPath:
    def __init__(self, cost, move_type, predecessor_entrypoint, predecessor_log_index):
        self.cost = cost
        self.move_type = move_type
        self.predecessor_entrypoint = predecessor_entrypoint
        self.predecessor_log_index = predecessor_log_index
        self.alignment: Optional['Alignment'] = None

    def __lt__(self, other: 'DecisionPath') -> bool:
        return self.cost < other.cost


class Alignment:
    def __init__(self, moves: Optional[List[Tuple[Optional[str], Optional[str]]]] = None):
        self._moves: List[Tuple[Optional[str], Optional[str]]] = moves if moves is not None else []

    @property
    def moves(self) -> List[Tuple[Optional[str], Optional[str]]]:
        return self._moves

    @property
    def cost(self) -> int:
        c = 0
        for m, l in self._moves:
            if m is None:
                c += LOG_COST
            elif l is None:
                c += MODEL_COST
        return c

    def __str__(self) -> str:
        lines = []
        for m, l in self._moves:
            left = m if m is not None else ">>"
            right = l if l is not None else ">>"
            lines.append(f"({left:20s}, {right})")
        return f"Alignment(cost={self.cost}, len={len(self._moves)})\n" + "\n".join(lines)

    def add_sync_move(self, activity: str) -> 'Alignment':
        return Alignment(self._moves + [(activity, activity)])

    def add_log_move(self, activity: str) -> 'Alignment':
        return Alignment(self._moves + [(None, activity)])

    def add_model_move(self, activity: str) -> 'Alignment':
        return Alignment(self._moves + [(activity, None)])
