"""Run A/B/C on the fixed development split."""

import argparse
import json

import numpy as np

from src.data import RESULTS_DIR, ROOT, load_config, load_verified_splits
from src.evaluate import evaluate_predictions
from src.features import NgramCounts
from src.models.majority import MajorityClassifier
from src.models.naive_bayes import MultinomialNaiveBayes
from src.preprocessing import normalize_text
from src.tokenizer import tokenize
from src.vocab import write_json


def run_experiments(*, save_models: bool = False) -> dict:
    config = load_config()
    splits = load_verified_splits()
    if set(splits) != {"train", "validation"}:
        raise RuntimeError("Expected only train and validation")
    train, validation = splits["train"], splits["validation"]
    y_train = train["label"].to_numpy()
    y_validation = validation["label"].to_numpy()

    train_docs, validation_docs = (
        [tokenize(normalize_text(text, lowercase=config["lowercase"])) for text in frame["text"]]
        for frame in (train, validation)
    )
    alpha = config["alpha"]
    majority = MajorityClassifier().fit(y_train)
    experiments = {
        "A": {
            "majority_label": int(majority.label),
            **evaluate_predictions(y_validation, majority.predict(validation_docs)),
        },
    }
    vocabularies = {}
    for name in ("B", "C"):
        ngram_range = tuple(config[f"model_{name.lower()}_ngram_range"])
        extractor = NgramCounts(ngram_range).fit(train_docs)
        vocabularies[name] = set(extractor.vocabulary)
        X_train, _ = extractor.transform(train_docs)
        X_validation, unseen = extractor.transform(validation_docs)
        model = MultinomialNaiveBayes(alpha).fit(X_train, y_train)
        experiments[name] = {
            "ngram_range": list(ngram_range),
            "alpha": alpha,
            "feature_vocabulary_size": len(extractor.vocabulary),
            "train_matrix_shape": list(X_train.shape),
            "validation_matrix_shape": list(X_validation.shape),
            "train_feature_count": int(X_train.sum()),
            "train_nonzero_count": int(np.count_nonzero(X_train)),
            "validation_unseen_ngram_count": unseen,
            **evaluate_predictions(y_validation, model.predict(X_validation)),
        }
        if save_models:
            directory = ROOT / "artifacts" / "validation"
            directory.mkdir(parents=True, exist_ok=True)
            model.save(directory / f"model_{name.lower()}.npz")
            write_json(directory / f"features_{name.lower()}.json", extractor.vocabulary)

    if ({feature for feature in vocabularies["C"] if feature.startswith("U:")}
            != vocabularies["B"]):
        raise RuntimeError("B and C must share the same unigram features")
    results = {
        "evaluation_split": "validation",
        "train_rows": len(train),
        "validation_rows": len(validation),
        "experiments": experiments,
    }
    write_json(RESULTS_DIR / "validation_experiments.json", results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-models", action="store_true", help="Save validation-only B/C artifacts")
    args = parser.parse_args()
    print(json.dumps(run_experiments(save_models=args.save_models), indent=2))


if __name__ == "__main__":
    main()
