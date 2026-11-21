"""Majority classifier with ties resolved by the smallest label."""

from collections.abc import Sized

import numpy as np


class MajorityClassifier:
    def fit(self, y: np.ndarray) -> "MajorityClassifier":
        labels, counts = np.unique(y, return_counts=True)
        if not labels.size:
            raise ValueError("Training labels must not be empty")
        self.label = labels[counts.argmax()]
        return self

    def predict(self, documents: Sized) -> np.ndarray:
        return np.full(len(documents), self.label)
