"""Evaluate the frozen A/B/C comparison on official test."""

import json

import numpy as np
import pandas as pd

from src.data import RESULTS_DIR, ROOT, load_config, load_verified_splits
from src.evaluate import evaluate_predictions
from src.features import NgramCounts
from src.models.majority import MajorityClassifier
from src.models.naive_bayes import MultinomialNaiveBayes
from src.preprocessing import normalize_text
from src.tokenizer import tokenize
from src.vocab import write_json


def main() -> None:
    report_path = RESULTS_DIR / "test_experiments.json"
    predictions_path = RESULTS_DIR / "test_predictions.csv"
    if report_path.exists() or predictions_path.exists():
        raise SystemExit("Official test outputs already exist.")

    config = load_config()
    frozen_settings = {
        "lowercase": True, "min_freq": 1, "alpha": 1.0,
        "model_b_ngram_range": [1, 1], "model_c_ngram_range": [1, 2],
        "normalize_urls": False, "normalize_numbers": False,
    }
    if any(config[key] != value for key, value in frozen_settings.items()):
        raise RuntimeError("Configuration differs from the frozen settings")
    validation_report = json.loads(
        (RESULTS_DIR / "validation_experiments.json").read_text(encoding="utf-8")
    )
    splits = load_verified_splits(include_test=True)
    if {name: len(frame) for name, frame in splits.items()} != {
        "train": 432, "validation": 109, "test": 116,
    }:
        raise RuntimeError("Unexpected frozen split sizes")

    train, test = splits["train"], splits["test"]
    train_docs, test_docs = (
        [tokenize(normalize_text(text, lowercase=config["lowercase"])) for text in frame["text"]]
        for frame in (train, test)
    )
    y_train = train["label"].to_numpy()
    majority = MajorityClassifier().fit(y_train)
    fitted = {}
    for name, expected_size in (("B", 2057), ("C", 7116)):
        frozen = validation_report["experiments"][name]
        ngram_range = config[f"model_{name.lower()}_ngram_range"]
        extractor = NgramCounts(tuple(ngram_range)).fit(train_docs)
        X_train, _ = extractor.transform(train_docs)
        if (
            len(extractor.vocabulary) != expected_size
            or expected_size != frozen["feature_vocabulary_size"]
            or ngram_range != frozen["ngram_range"]
            or config["alpha"] != frozen["alpha"]
            or list(X_train.shape) != frozen["train_matrix_shape"]
            or int(X_train.sum()) != frozen["train_feature_count"]
            or int(np.count_nonzero(X_train)) != frozen["train_nonzero_count"]
        ):
            raise RuntimeError(f"{name} differs from the validation training setup")
        fitted[name] = extractor, MultinomialNaiveBayes(config["alpha"]).fit(X_train, y_train)

    y_test = test["label"].to_numpy()
    experiments = {
        "A": {
            "majority_label": int(majority.label),
            **evaluate_predictions(y_test, majority.predict(test_docs)),
        },
    }
    for name, (extractor, model) in fitted.items():
        X_test, unseen = extractor.transform(test_docs)
        scores = model.class_log_scores(X_test)
        predicted = model.classes_[scores.argmax(axis=1)]
        experiments[name] = {
            "ngram_range": list(extractor.ngram_range),
            "alpha": model.alpha,
            "feature_vocabulary_size": len(extractor.vocabulary),
            "test_unseen_ngram_count": unseen,
            **evaluate_predictions(y_test, predicted),
        }
        if name == "B":
            final = ROOT / "artifacts" / "final"
            final.mkdir(parents=True, exist_ok=True)
            model_path = final / "model_b.npz"
            features_path = final / "features_b.json"
            model.save(model_path)
            write_json(features_path, extractor.vocabulary)
            restored_extractor = NgramCounts(extractor.ngram_range)
            restored_extractor.vocabulary = json.loads(features_path.read_text(encoding="utf-8"))
            restored_matrix, _ = restored_extractor.transform(test_docs)
            restored = MultinomialNaiveBayes.load(model_path)
            if not np.array_equal(restored.predict(restored_matrix), predicted):
                raise RuntimeError("Reloaded Model B predictions differ")
            predictions = pd.DataFrame({
                "source_index": test["source_index"].to_numpy(),
                "gold_label": y_test,
                "predicted_label": predicted,
                "log_score_0": scores[:, 0],
                "log_score_1": scores[:, 1],
                "correct": predicted == y_test,
            })

    predictions.to_csv(predictions_path, index=False, lineterminator="\n")
    write_json(report_path, {
        "evaluation_split": "test",
        "train_rows": len(train),
        "test_rows": len(test),
        "frozen_selected_model": "B",
        "note": "Model B was frozen using validation only. Test results were not used for model selection.",
        "model_b_reload_predictions_identical": True,
        "experiments": experiments,
    })


if __name__ == "__main__":
    main()
