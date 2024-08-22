from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass
class ScoreResult:
    """Container for a score function result.

    Args:
        scores: Dictionary containing the scores for each tracked model.
        contributions: Dictionary containing the contributions of each score function to the total score.
        total_score: The total score of the step.
    """

    scores: Dict[str, float]
    contributions: Dict[str, Dict[str, float]]

    def __post_init__(self) -> None:
        self.total_score = np.mean(list(self.scores.values()))
