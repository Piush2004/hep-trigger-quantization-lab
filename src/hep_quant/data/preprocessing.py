"""Feature preprocessing: standardization fit on train, applied to all splits."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler


@dataclass
class Preprocessor:
    scaler: StandardScaler

    @classmethod
    def fit(cls, X_train: np.ndarray) -> "Preprocessor":
        scaler = StandardScaler()
        scaler.fit(X_train)
        return cls(scaler=scaler)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(X).astype(np.float32)
