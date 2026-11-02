"""Text helpers: split text into synthesis chunks, and score a transcript against the text
it should contain. Pure Python, so the tests run without torch or a model."""

from __future__ import annotations

import difflib
import re
import unicodedata

_SENTENCE_END = re.compile(r"(?<=[.!?;:])\s+")
_CLAUSE_END = re.compile(r"(?<=,)\s+")
_WORD = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*")


def split_chunks(text: str, max_chars: int) -> list[str]:
    """Split text into chunks of whole sentences, each at most max_chars where possible.

    Long passages are synthesized one chunk at a time because quality drifts over a very
    long single generation. A sentence longer than max_chars is split at commas, then at
    spaces; a single word longer than max_chars is kept whole.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be at least 1")
    chunks: list[str] = []
    current = ""
    for piece in _pieces(" ".join(text.split()), max_chars):
        if current and len(current) + 1 + len(piece) > max_chars:
            chunks.append(current)
            current = piece
        else:
            current = f"{current} {piece}" if current else piece
    if current:
        chunks.append(current)
    return chunks


def _pieces(text: str, max_chars: int) -> list[str]:
    out: list[str] = []
    for sentence in _SENTENCE_END.split(text):
        if len(sentence) <= max_chars:
            out.append(sentence)
            continue
        for clause in _CLAUSE_END.split(sentence):
            if len(clause) <= max_chars:
                out.append(clause)
            else:
                out.extend(clause.split(" "))
    return [p for p in out if p]


def words(text: str) -> list[str]:
    """Lower-case words with punctuation removed; apostrophes inside a word are kept."""
    text = unicodedata.normalize("NFKC", text).lower().replace("’", "'")
    return _WORD.findall(text)


def score(expected: str, heard: str) -> dict:
    """Compare a transcript with the text it should contain.

    Returns the word error rate (substitutions, deletions and insertions over the expected
    word count, from a difflib alignment) and the number of expected words missing at the
    end. A missing tail is the usual sign of a truncated generation, and it can hide under
    a low overall error rate on a long text, so it is reported on its own.
    """
    ref, hyp = words(expected), words(heard)
    matcher = difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False)
    errors = 0
    missing_tail = 0
    opcodes = matcher.get_opcodes()
    for tag, i1, i2, j1, j2 in opcodes:
        if tag != "equal":
            errors += max(i2 - i1, j2 - j1)
    if opcodes and opcodes[-1][0] in ("delete", "replace"):
        _, i1, i2, j1, j2 = opcodes[-1]
        missing_tail = max(0, (i2 - i1) - (j2 - j1))
    return {
        "expected_words": len(ref),
        "heard_words": len(hyp),
        "word_errors": errors,
        "wer": round(errors / len(ref), 4) if ref else (0.0 if not hyp else 1.0),
        "missing_tail_words": missing_tail,
    }
