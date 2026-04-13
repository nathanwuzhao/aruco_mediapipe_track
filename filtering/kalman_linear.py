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

@dataclass
class LinearKalmanConfig:
    state_dim: int
    measurement_dim: int
    control_dim: int = 0
    enforce_symmetry: bool = True
    use_joseph_form: bool = True

class LinearKalmanFilter:
    def __init__(self, config: LinearKalmanConfig):
        pass

    def set_state(self, x0: np.ndarray, P0: np.ndarray, t0: Optional[float] = None) -> None:
        pass

    def predict(self, F: np.ndarray, Q: np.ndarray, B: Optional[np.ndarray] = None,
                u: Optional[np.ndarray] = None, dt: Optional[float] = None) -> PredictResult:
        pass

    def update(self, z: np.ndarray, H: np.ndarray, R: np.ndarray) -> UpdateResult:
        pass

    def step(self, z: Optional[np.ndarray], F: np.ndarray, Q: np.ndarray, H: Optional[np.ndarray] = None,
                R: Optional[np.ndarray] = None, B: Optional[np.ndarray] = None, u: Optional[np.ndarray] = None,
                dt: Optional[float] = None) -> StepResult:
        pass

    def reset(self) -> None:
        pass

    @property
    def state(self) -> FilterState:
        pass
        