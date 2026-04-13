import numpy as np
import math

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

@dataclass
class FilterState:
    x: np.ndarray #shape (n,)
    P: np.ndarray #shape (n,n)
    t: Optional[float] = None

@dataclass
class PredictResult:
    x_pred: np.ndarray
    P_pred: np.ndarray

@dataclass
class UpdateResult:
    x_upd: np.ndarray
    P_upd: np.ndarray
    innovation: np.ndarray
    S: np.ndarray
    K: Optional[np.ndarray] = None

@dataclass
class StepResult:
    predicted: PredictResult
    updated: Optional[UpdateResult]
