"""Deterministic Unicode word tokenizer for normalized text."""

import re

# Keep apostrophes and hyphens between word characters. Preserve other
# punctuation and symbols as individual tokens.
TOKEN_PATTERN = re.compile(r"\w+(?:['’\-‐‑]\w+)*|[^\w\s]")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text)
