"""Report cross-split text similarity candidates for the Phase 1 audit."""

import re
from itertools import combinations

import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer

from src.data import RESULTS_DIR, load_verified_splits


def main() -> None:
    splits = load_verified_splits(include_test=True)
    names = ("train", "validation", "test")
    vectorizer = HashingVectorizer(
        analyzer="char", ngram_range=(3, 5), n_features=2**20,
        alternate_sign=False, lowercase=False,
    )
    features = {}
    tokens = {}
    indices = {}
    for name in names:
        frame = splits[name].sort_values("source_index")
        texts = frame["_norm_text"].str.lower().tolist()
        features[name] = vectorizer.transform(texts)
        tokens[name] = [set(re.findall(r"\w+", text)) for text in texts]
        indices[name] = frame["source_index"].tolist()

    rows = []
    for left, right in combinations(names, 2):
        cosine = (features[left] @ features[right].T).toarray()
        for i, left_tokens in enumerate(tokens[left]):
            for j, right_tokens in enumerate(tokens[right]):
                shared = len(left_tokens & right_tokens)
                union = len(left_tokens) + len(right_tokens) - shared
                jaccard = shared / union if union else 0.0
                if cosine[i, j] >= 0.85 or (shared >= 5 and jaccard >= 0.75):
                    rows.append((
                        left, indices[left][i], right, indices[right][j],
                        min(float(cosine[i, j]), 1.0), jaccard,
                    ))

    columns = [
        "left_split", "left_source_index", "right_split", "right_source_index",
        "char_ngram_cosine", "token_jaccard",
    ]
    pd.DataFrame(rows, columns=columns).to_csv(
        RESULTS_DIR / "near_duplicate_audit.csv", index=False,
        lineterminator="\n", float_format="%.8f",
    )


if __name__ == "__main__":
    main()
