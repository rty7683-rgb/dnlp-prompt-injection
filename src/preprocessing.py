"""Text normalization shared by the word feature pipelines."""

import re
import unicodedata


def normalize_text(text: str, *, lowercase: bool) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = re.sub(r"\s+", " ", normalized.strip())
    return normalized.lower() if lowercase else normalized
