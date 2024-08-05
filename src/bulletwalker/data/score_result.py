from dataclasses import dataclass
from typing import Dict


@dataclass
class ScoreResult:
    scores: Dict[str, float]
    contributions: Dict[str, Dict[str, float]]
    total_score: float
