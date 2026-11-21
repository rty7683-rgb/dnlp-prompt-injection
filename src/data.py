"""Phase 1: dataset loading, inspection, cleaning, and split creation.

This module intentionally focuses on data quality and reproducibility before
any tokenizer, vocabulary, or model is implemented.

Run:
    python -m src.data

Outputs:
    results/data_stats.json
    results/split_manifest.csv

Raw prompt text is not written to the repository outputs.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
CONFIG_PATH = ROOT / "config.json"


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_for_duplicate_check(text: str) -> str:
    """Conservative normalization used only for duplicate checking.

    We normalize Unicode and whitespace, but do not lowercase or remove
    punctuation here because those choices belong to the later preprocessing
    experiment design.
    """
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text.strip())
    return text


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def length_summary(values: pd.Series) -> dict[str, float | int]:
    if len(values) == 0:
        return {"min": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0}
    arr = values.to_numpy(dtype=float)
    return {
        "min": int(np.min(arr)),
        "mean": round(float(np.mean(arr)), 2),
        "median": round(float(np.median(arr)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
        "max": int(np.max(arr)),
    }


def prepare_frame(dataset_split, split_name: str) -> pd.DataFrame:
    df = dataset_split.to_pandas().copy()

    required = {"text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{split_name}: required columns are missing: {sorted(missing)}"
        )

    df = df[["text", "label"]].copy()
    df["_source_index"] = np.arange(len(df))
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)
    df["_norm_text"] = df["text"].map(normalize_for_duplicate_check)
    return df


def inspect_frame(df: pd.DataFrame) -> dict[str, Any]:
    blank_mask = df["_norm_text"].eq("")
    nonblank = df.loc[~blank_mask].copy()

    duplicate_raw = int(nonblank.duplicated(subset=["text"], keep=False).sum())
    duplicate_norm = int(
        nonblank.duplicated(subset=["_norm_text"], keep=False).sum()
    )

    label_counts = {
        str(int(k)): int(v)
        for k, v in nonblank["label"].value_counts().sort_index().items()
    }

    conflicts = (
        nonblank.groupby("_norm_text")["label"].nunique().gt(1)
        if len(nonblank)
        else pd.Series(dtype=bool)
    )

    char_len = nonblank["_norm_text"].str.len()
    whitespace_token_len = nonblank["_norm_text"].str.split().str.len()

    return {
        "rows": int(len(df)),
        "blank_rows": int(blank_mask.sum()),
        "nonblank_rows": int(len(nonblank)),
        "label_distribution_nonblank": label_counts,
        "raw_duplicate_rows": duplicate_raw,
        "normalized_duplicate_rows": duplicate_norm,
        "conflicting_normalized_text_groups": int(conflicts.sum()),
        "character_length": length_summary(char_len),
        "whitespace_token_length": length_summary(whitespace_token_len),
    }


def clean_frame(df: pd.DataFrame, split_name: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove blank rows and same-label normalized duplicates.

    If the same normalized prompt has conflicting labels, stop instead of
    silently deciding which label is correct.
    """
    before = len(df)
    cleaned = df.loc[~df["_norm_text"].eq("")].copy()
    after_blank = len(cleaned)

    conflicts = cleaned.groupby("_norm_text")["label"].nunique()
    conflicting_texts = conflicts[conflicts > 1].index.tolist()
    if conflicting_texts:
        raise RuntimeError(
            f"{split_name}: found {len(conflicting_texts)} normalized text group(s) "
            "with conflicting labels. Review them before continuing."
        )

    cleaned = cleaned.drop_duplicates(subset=["_norm_text"], keep="first").copy()

    return cleaned, {
        "removed_blank": int(before - after_blank),
        "removed_same_text_duplicates": int(after_blank - len(cleaned)),
        "remaining_rows": int(len(cleaned)),
    }


def make_manifest(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_name, frame in [
        ("train", train_df),
        ("validation", valid_df),
        ("test", test_df),
    ]:
        for row in frame.itertuples(index=False):
            rows.append(
                {
                    "split": split_name,
                    "source_index": int(row._source_index),
                    "label": int(row.label),
                    "text_sha256": sha256_text(row._norm_text),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    config = load_config()
    seed = int(config["seed"])
    validation_ratio = float(config["validation_ratio"])
    dataset_name = str(config["dataset_name"])

    print(f"[Phase 1] Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)

    if "train" not in dataset or "test" not in dataset:
        raise ValueError(
            f"Expected official train/test splits, found: {list(dataset.keys())}"
        )

    train_raw = prepare_frame(dataset["train"], "train")
    test_raw = prepare_frame(dataset["test"], "test")

    raw_stats = {
        "train": inspect_frame(train_raw),
        "test": inspect_frame(test_raw),
    }

    train_clean, train_cleaning = clean_frame(train_raw, "train")
    test_clean, test_cleaning = clean_frame(test_raw, "test")

    train_texts = set(train_clean["_norm_text"])
    test_texts = set(test_clean["_norm_text"])
    cross_split_overlap = train_texts & test_texts

    train_df, valid_df = train_test_split(
        train_clean,
        test_size=validation_ratio,
        random_state=seed,
        stratify=train_clean["label"],
    )

    train_df = train_df.sort_values("_source_index").copy()
    valid_df = valid_df.sort_values("_source_index").copy()
    test_df = test_clean.sort_values("_source_index").copy()

    split_counts = {
        "train": int(len(train_df)),
        "validation": int(len(valid_df)),
        "test": int(len(test_df)),
    }
    split_label_distribution = {
        name: {
            str(int(k)): int(v)
            for k, v in frame["label"].value_counts().sort_index().items()
        }
        for name, frame in [
            ("train", train_df),
            ("validation", valid_df),
            ("test", test_df),
        ]
    }

    stats = {
        "dataset_name": dataset_name,
        "seed": seed,
        "validation_ratio": validation_ratio,
        "official_raw_split_sizes": {
            "train": int(len(train_raw)),
            "test": int(len(test_raw)),
        },
        "raw_inspection": raw_stats,
        "cleaning": {
            "train": train_cleaning,
            "test": test_cleaning,
        },
        "cross_split_normalized_overlap_count": int(len(cross_split_overlap)),
        "derived_split_sizes": split_counts,
        "derived_split_label_distribution": split_label_distribution,
        "notes": [
            "Official test membership is preserved; only blank/same-text duplicate cleaning is applied.",
            "Vocabulary has not been built yet.",
            "Near-duplicate/semantic-variant leakage is not automatically removed in Phase 1.",
            "Raw prompt text is not stored in result files.",
        ],
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with (RESULTS_DIR / "data_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    manifest = make_manifest(train_df, valid_df, test_df)
    manifest.to_csv(RESULTS_DIR / "split_manifest.csv", index=False)

    print("\n=== Raw dataset inspection ===")
    print(json.dumps(raw_stats, ensure_ascii=False, indent=2))

    print("\n=== Cleaning ===")
    print(json.dumps(stats["cleaning"], ensure_ascii=False, indent=2))

    print("\n=== Derived splits ===")
    print(json.dumps(split_counts, ensure_ascii=False, indent=2))
    print(json.dumps(split_label_distribution, ensure_ascii=False, indent=2))

    if cross_split_overlap:
        print(
            f"\n[WARNING] {len(cross_split_overlap)} normalized text(s) occur in "
            "both official train and test. Do not finalize evaluation design "
            "until this overlap is reviewed."
        )
    else:
        print("\n[OK] No exact normalized text overlap between official train/test.")

    print("\nSaved:")
    print(f"- {RESULTS_DIR / 'data_stats.json'}")
    print(f"- {RESULTS_DIR / 'split_manifest.csv'}")
    print("\nPhase 1 inspection complete. Review the outputs before Phase 2.")


if __name__ == "__main__":
    main()
