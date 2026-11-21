"""Build a train-only word vocabulary and report validation OOV coverage."""

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from src.data import RESULTS_DIR, ROOT, load_config, load_verified_splits
from src.preprocessing import normalize_text
from src.tokenizer import tokenize

UNK_TOKEN = "<UNK>"
UNK_ID = 0


def build_vocabulary(
    train_documents: Iterable[list[str]], min_freq: int
) -> tuple[dict[str, int], Counter[str]]:
    if min_freq < 1:
        raise ValueError("min_freq must be positive")
    counts = Counter(token for document in train_documents for token in document)
    vocabulary = {UNK_TOKEN: UNK_ID}
    for token in sorted(counts, key=lambda token: (-counts[token], token)):
        if token != UNK_TOKEN and counts[token] >= min_freq:
            vocabulary[token] = len(vocabulary)
    return vocabulary, counts


def encode_tokens(tokens: Iterable[str], vocabulary: dict[str, int]) -> list[int]:
    return [vocabulary.get(token, UNK_ID) for token in tokens]


def write_json(path: Path, value: object) -> None:
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def main() -> None:
    config = load_config()
    splits = load_verified_splits()
    if set(splits) != {"train", "validation"}:
        raise RuntimeError("Phase 2 requires exactly train and validation")

    def documents(split: str) -> list[dict]:
        rows = []
        for row in splits[split].itertuples(index=False):
            normalized = normalize_text(row.text, lowercase=config["lowercase"])
            rows.append({
                "split": split,
                "source_index": int(row.source_index),
                "label": row.label,
                "original_text": row.text,
                "normalized_text": normalized,
                "tokens": tokenize(normalized),
            })
        return rows

    train = documents("train")
    min_freq = config["min_freq"]
    vocabulary, counts = build_vocabulary((row["tokens"] for row in train), min_freq)
    validation = documents("validation")
    for row in train + validation:
        row["token_ids"] = encode_tokens(row["tokens"], vocabulary)

    valid_ids = set(vocabulary.values())
    if vocabulary.get(UNK_TOKEN) != 0 or len(valid_ids) != len(vocabulary):
        raise RuntimeError("Invalid vocabulary IDs")
    if any(len(row["tokens"]) != len(row["token_ids"])
           or not valid_ids.issuperset(row["token_ids"]) for row in validation):
        raise RuntimeError("Invalid validation encoding")

    validation_total = sum(len(row["tokens"]) for row in validation)
    oov = sum(row["token_ids"].count(UNK_ID) for row in validation)
    stats = {
        "train_token_count": counts.total(),
        "train_unique_token_count": len(counts),
        "min_freq": min_freq,
        "vocabulary_size": len(vocabulary),
        "vocabulary_size_includes_unk": True,
        "validation_total_token_count": validation_total,
        "validation_oov_token_count": oov,
        "validation_oov_rate": oov / validation_total if validation_total else 0.0,
    }

    def normal_examples(rows: list[dict]) -> list[dict]:
        return sorted(
            (row for row in rows if row["label"] == 0),
            key=lambda row: (len(row["original_text"]), row["source_index"]),
        )

    validation_examples = normal_examples(validation)
    example = next(
        (row for row in validation_examples if UNK_ID in row["token_ids"]),
        validation_examples[0],
    )
    examples = normal_examples(train)[:2] + [example]
    for row in examples:
        del row["label"]

    write_json(ROOT / "artifacts" / "vocab.json", vocabulary)
    write_json(RESULTS_DIR / "vocab_stats.json", stats)
    write_json(RESULTS_DIR / "token_id_examples.json", examples)
    print(json.dumps(stats, indent=2))
    print(json.dumps(examples, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
