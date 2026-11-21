"""Deterministic word n-gram count features."""

import json
from collections.abc import Iterator, Sequence

import numpy as np


def feature_names(tokens: Sequence[str], ngram_range: tuple[int, int]) -> Iterator[str]:
    for n in range(ngram_range[0], ngram_range[1] + 1):
        for start in range(len(tokens) - n + 1):
            gram = tokens[start:start + n]
            if "<UNK>" in gram:
                continue
            yield "U:" + gram[0] if n == 1 else "B:" + json.dumps(
                gram, ensure_ascii=False, separators=(",", ":")
            )


class NgramCounts:
    def __init__(self, ngram_range: tuple[int, int]):
        if ngram_range not in {(1, 1), (1, 2)}:
            raise ValueError("Expected unigram or unigram + bigram range")
        self.ngram_range = ngram_range

    def fit(self, documents: Sequence[Sequence[str]]) -> "NgramCounts":
        names = sorted({name for doc in documents for name in feature_names(doc, self.ngram_range)})
        self.vocabulary = {name: index for index, name in enumerate(names)}
        return self

    def transform(self, documents: Sequence[Sequence[str]]) -> tuple[np.ndarray, int]:
        counts = np.zeros((len(documents), len(self.vocabulary)), dtype=np.int64)
        unseen = 0
        for row, doc in enumerate(documents):
            for name in feature_names(doc, self.ngram_range):
                column = self.vocabulary.get(name)
                if column is None:
                    unseen += 1
                else:
                    counts[row, column] += 1
        return counts, unseen
