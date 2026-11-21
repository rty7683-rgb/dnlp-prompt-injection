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
import tempfile
import unicodedata
from importlib.metadata import version
from pathlib import Path
from platform import python_version
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
    df = dataset_split.to_pandas()

    required = {"text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{split_name}: required columns are missing: {sorted(missing)}"
        )

    df = df[["text", "label"]].copy()
    df["source_index"] = np.arange(len(df))
    df["text"] = df["text"].fillna("").astype(str)
    if not df["label"].isin([0, 1]).all():
        raise ValueError(f"{split_name}: expected binary labels 0 or 1")
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
        columns = ["source_index", "label", "_norm_text"]
        for source_index, label, norm_text in frame[columns].itertuples(
            index=False, name=None
        ):
            rows.append(
                {
                    "split": split_name,
                    "source_index": int(source_index),
                    "label": int(label),
                    "text_sha256": sha256_text(norm_text),
                }
            )
    return pd.DataFrame(rows)


def validate_disjoint_splits(splits: dict[str, pd.DataFrame], lowercase: bool) -> None:
    seen: set[str] = set()
    for name, frame in splits.items():
        keys = frame["_norm_text"]
        if lowercase:
            keys = keys.str.lower()
        overlap = seen.intersection(keys)
        if overlap:
            raise ValueError(f"{name}: {len(overlap)} text(s) overlap an earlier split")
        seen.update(keys)


def load_source_frames(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    dataset = load_dataset(config["dataset_name"], revision=config["dataset_revision"])
    if not {"train", "test"}.issubset(dataset):
        raise ValueError(f"Expected official train/test splits, found: {list(dataset)}")
    return {name: prepare_frame(dataset[name], name) for name in ("train", "test")}


def load_verified_splits(*, include_test: bool = False) -> dict[str, pd.DataFrame]:
    """Reuse saved membership and verify it against the pinned source rows."""
    config = load_config()
    sources = load_source_frames(config)
    train, _ = clean_frame(sources["train"], "train")
    sources["train"] = train
    manifest_path = RESULTS_DIR / "split_manifest.csv"
    stats = json.loads((RESULTS_DIR / "data_stats.json").read_text(encoding="utf-8"))
    split_config_keys = (
        "dataset_name", "dataset_revision", "seed", "validation_ratio", "lowercase"
    )
    if any(stats[key] != config[key] for key in split_config_keys):
        raise ValueError("Saved split configuration differs from config.json")
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != stats["manifest_sha256"]:
        raise ValueError("Split manifest does not match data_stats.json")
    manifest = pd.read_csv(manifest_path, dtype={"source_index": int, "label": int})
    if set(manifest["split"]) != {"train", "validation", "test"}:
        raise ValueError("Manifest must contain train, validation, and test")
    splits = {}
    for name in ("train", "validation", "test"):
        rows = manifest.loc[manifest["split"].eq(name)]
        source_name = "test" if name == "test" else "train"
        source = sources[source_name].set_index("source_index", drop=False)
        indices = rows["source_index"]
        if indices.duplicated().any() or not indices.isin(source.index).all():
            raise ValueError(f"{name}: invalid manifest source indices")
        frame = source.loc[indices].reset_index(drop=True)
        labels = frame["label"].tolist()
        hashes = frame["_norm_text"].map(sha256_text).tolist()
        if labels != rows["label"].tolist() or hashes != rows["text_sha256"].tolist():
            raise ValueError(f"{name}: manifest labels or hashes differ from the source")
        splits[name] = frame
    if set(splits["test"]["source_index"]) != set(sources["test"]["source_index"]):
        raise ValueError("Manifest does not preserve official test membership")
    train_ids = set(splits["train"]["source_index"])
    validation_ids = set(splits["validation"]["source_index"])
    if train_ids & validation_ids or (train_ids | validation_ids) != set(train["source_index"]):
        raise ValueError("Manifest does not partition the cleaned official train split")
    validate_disjoint_splits(splits, config["lowercase"])
    if not include_test:
        del splits["test"]
    return splits


def main() -> None:
    config = load_config()
    seed = int(config["seed"])
    validation_ratio = float(config["validation_ratio"])
    dataset_name = str(config["dataset_name"])

    print(f"[Phase 1] Loading dataset: {dataset_name}")
    sources = load_source_frames(config)
    train_raw = sources["train"]
    test_raw = sources["test"]

    raw_stats = {
        "train": inspect_frame(train_raw),
        "test": inspect_frame(test_raw),
    }

    train_clean, train_cleaning = clean_frame(train_raw, "train")
    test_stats = raw_stats["test"]
    if test_stats["blank_rows"] or test_stats["normalized_duplicate_rows"]:
        raise ValueError("Official test contains blank or duplicate rows; review before proceeding")

    train_df, valid_df = train_test_split(
        train_clean,
        test_size=validation_ratio,
        random_state=seed,
        stratify=train_clean["label"],
    )

    train_df = train_df.sort_values("source_index").copy()
    valid_df = valid_df.sort_values("source_index").copy()
    test_df = test_raw
    validate_disjoint_splits(
        {"train": train_df, "validation": valid_df, "test": test_df},
        config["lowercase"],
    )

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
        "dataset_revision": config["dataset_revision"],
        "python_version": python_version(),
        "package_versions": {
            package: version(package)
            for package in ("datasets", "numpy", "pandas", "scikit-learn", "matplotlib")
        },
        "seed": seed,
        "validation_ratio": validation_ratio,
        "lowercase": config["lowercase"],
        "official_raw_split_sizes": {
            "train": int(len(train_raw)),
            "test": int(len(test_raw)),
        },
        "raw_inspection": raw_stats,
        "cleaning": {
            "train": train_cleaning,
            "test": {
                "removed_blank": 0,
                "removed_same_text_duplicates": 0,
                "remaining_rows": len(test_df),
            },
        },
        "cross_split_normalized_overlap_count": 0,
        "derived_split_sizes": split_counts,
        "derived_split_label_distribution": split_label_distribution,
        "notes": [
            "Official test is inspected and preserved without filtering.",
            "Cross-split checks include configured lowercasing.",
            "Vocabulary has not been built yet.",
            "Near-duplicate/semantic-variant leakage is not automatically removed in Phase 1.",
            "Raw prompt text is not stored in result files.",
        ],
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest = make_manifest(train_df, valid_df, test_df)
    with tempfile.TemporaryDirectory(dir=RESULTS_DIR) as directory:
        staged = Path(directory)
        manifest_path = staged / "split_manifest.csv"
        stats_path = staged / "data_stats.json"
        manifest.to_csv(manifest_path, index=False, lineterminator="\n", encoding="utf-8")
        stats["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        stats_path.write_bytes(
            (json.dumps(stats, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        )
        manifest_path.replace(RESULTS_DIR / manifest_path.name)
        stats_path.replace(RESULTS_DIR / stats_path.name)

    print("\n=== Raw dataset inspection ===")
    print(json.dumps(raw_stats, ensure_ascii=False, indent=2))

    print("\n=== Cleaning ===")
    print(json.dumps(stats["cleaning"], ensure_ascii=False, indent=2))

    print("\n=== Derived splits ===")
    print(json.dumps(split_counts, ensure_ascii=False, indent=2))
    print(json.dumps(split_label_distribution, ensure_ascii=False, indent=2))

    print("\n[OK] No normalized text overlap between train, validation, and test.")

    print("\nSaved:")
    print(f"- {RESULTS_DIR / 'data_stats.json'}")
    print(f"- {RESULTS_DIR / 'split_manifest.csv'}")
    print("\nPhase 1 inspection complete. Review the outputs before Phase 2.")


if __name__ == "__main__":
    main()
