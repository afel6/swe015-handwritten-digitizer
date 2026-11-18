"""Cloze sentence normalisation shared by the n-gram corrector and the BERT re-ranker."""

from __future__ import annotations

import re

from models.errors import UserInputError

MASK_TOKEN = "[MASK]"
MAX_SENTENCE_CHARS = 500
_UNDERSCORE_BLANK_RE = re.compile(r"_{3,}")
_ANY_CASE_MASK_RE = re.compile(re.escape(MASK_TOKEN), re.IGNORECASE)

__all__ = ["MASK_TOKEN", "MAX_SENTENCE_CHARS", "normalize_masked_sentence"]


def normalize_masked_sentence(sentence: str) -> str:
    """Rewrite '___' (3+ underscores) and any-case '[mask]' to '[MASK]' and require exactly one blank.

    The rest of the sentence is returned unchanged (no lower-casing). Raises TypeError for
    non-strings and ValueError for sentences over MAX_SENTENCE_CHARS or without exactly one blank.
    """
    if not isinstance(sentence, str):
        raise TypeError("sentence must be a string.")
    if len(sentence) > MAX_SENTENCE_CHARS:
        raise UserInputError(f"sentence must be at most {MAX_SENTENCE_CHARS} characters.")
    normalized = _ANY_CASE_MASK_RE.sub(MASK_TOKEN, _UNDERSCORE_BLANK_RE.sub(MASK_TOKEN, sentence))
    if normalized.count(MASK_TOKEN) != 1:
        raise UserInputError("sentence must contain exactly one [MASK] (or '___') blank.")
    return normalized
