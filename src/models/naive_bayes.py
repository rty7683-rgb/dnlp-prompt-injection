"""Multinomial Naive Bayes for nonnegative count vectors."""

from pathlib import Path

import numpy as np


class MultinomialNaiveBayes:
    def __init__(self, alpha: float = 1.0):
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultinomialNaiveBayes":
        if X.ndim != 2 or not all(X.shape) or y.shape != (len(X),):
            raise ValueError("Expected a nonempty count matrix and one label per row")
        self.classes_, self.class_count_ = np.unique(y, return_counts=True)
        self.feature_count_ = np.stack([X[y == label].sum(axis=0) for label in self.classes_])
        self._compute_log_probabilities()
        return self

    def _compute_log_probabilities(self) -> None:
        self.class_log_prior_ = np.log(self.class_count_ / self.class_count_.sum())
        smoothed = self.feature_count_ + self.alpha
        self.feature_log_prob_ = np.log(smoothed) - np.log(smoothed.sum(axis=1, keepdims=True))

    def class_log_scores(self, X: np.ndarray) -> np.ndarray:
        return X @ self.feature_log_prob_.T + self.class_log_prior_

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        scores = self.class_log_scores(X)
        return scores - np.logaddexp.reduce(scores, axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.class_log_scores(X).argmax(axis=1)]

    def save(self, path: Path) -> None:
        with path.open("wb") as file:
            np.savez_compressed(
                file, alpha=self.alpha, classes=self.classes_,
                class_count=self.class_count_, feature_count=self.feature_count_,
            )

    @classmethod
    def load(cls, path: Path) -> "MultinomialNaiveBayes":
        with np.load(path, allow_pickle=False) as data:
            model = cls(float(data["alpha"].item()))
            model.classes_ = data["classes"]
            model.class_count_ = data["class_count"]
            model.feature_count_ = data["feature_count"]
        model._compute_log_probabilities()
        return model
