from __future__ import annotations

import re
from typing import Iterable


_NON_ALPHA_NUMERIC_PATTERN = re.compile(r"[^a-z0-9\s]")
_EXTRA_SPACES_PATTERN = re.compile(r"\s+")


def preprocess_text(text: str) -> str:
    cleaned = text.lower().strip()
    cleaned = _NON_ALPHA_NUMERIC_PATTERN.sub(" ", cleaned)
    cleaned = _EXTRA_SPACES_PATTERN.sub(" ", cleaned)
    return cleaned.strip()


def preprocess_corpus(texts: Iterable[str]) -> list[str]:
    return [preprocess_text(text) for text in texts]

