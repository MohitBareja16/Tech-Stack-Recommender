"""
data_loader.py — Ingestion stage of the IPO pipeline.

Responsibility:
  - Parse raw_skills.csv into ItemRecord dataclasses.
  - Validate every row; skip and warn on malformed rows.
  - Normalize all text (lowercase + strip) before it reaches the vectorizer.

Per ARCHITECTURE.md §3.1 and PRD FR-1 / Edge Cases table.
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class DataLoadError(Exception):
    """Raised when the dataset file is missing, unreadable, or contains no
    usable rows after validation (ARCHITECTURE.md §7)."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ItemRecord:
    """Represents one job role in the item corpus.

    Attributes:
        role_name:       Human-readable job title (e.g. "Data Scientist").
        skills:          Normalized, ordered skill token list for this role.
                         Tokens are already lowercased/stripped; duplicates
                         within the row are preserved (each mention counts
                         for TF purposes — PRD §10.1 applies to items too).
        popularity_rank: Integer rank used exclusively for Cold Start fallback
                         ordering (lower = higher default priority).
    """
    role_name: str
    skills: list[str] = field(default_factory=list)
    popularity_rank: int = 0


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def normalize_text(s: str) -> str:
    """Lowercase, strip leading/trailing whitespace, and collapse internal
    whitespace sequences to a single space.

    Use this for **role names** and display strings that should keep spaces.
    For skill tokens, use ``normalize_skill()``.

    Examples:
        >>> normalize_text("  Python  ")
        'python'
        >>> normalize_text("MACHINE LEARNING")
        'machine learning'
    """
    if not isinstance(s, str):
        s = str(s)
    return re.sub(r"\s+", " ", s.strip()).lower()


def normalize_skill(s: str) -> str:
    """Normalize a skill token: lowercase + strip + replace whitespace with ``_``.

    This is the correct normalization for **skill vocabulary terms** and
    user-input skills, because the item dataset stores multi-word skills as
    underscore-joined tokens (e.g. ``machine_learning``, not ``machine learning``).
    Applying this to user input ensures that "Machine Learning" → "machine_learning"
    which then matches the corpus vocabulary (ALGORITHM_SPEC.md §3.7).

    Examples:
        >>> normalize_skill("  Machine Learning  ")
        'machine_learning'
        >>> normalize_skill("AWS")
        'aws'
    """
    if not isinstance(s, str):
        s = str(s)
    return re.sub(r"\s+", "_", s.strip()).lower()


# ---------------------------------------------------------------------------
# Row-level validation
# ---------------------------------------------------------------------------

def validate_row(row: dict) -> bool:
    """Return True if *row* has all required, non-empty fields.

    Validation rules (per DATA_SCHEMA.md §5):
      - ``role_name`` must be a non-empty string after stripping.
      - ``skills`` column must be present and non-empty.
      - At least one skill token must be non-blank after splitting on ``|``.
      - ``popularity_rank`` must parse as a positive integer.

    A failing row is logged as a warning and excluded from the corpus;
    the pipeline continues (ARCHITECTURE.md §7).
    """
    role_name = row.get("role_name", "").strip()
    if not role_name:
        logger.warning("Skipping row with blank/missing role_name: %r", row)
        return False

    raw_skills = row.get("skills", "").strip()
    if not raw_skills:
        logger.warning("Skipping row '%s': blank skills column.", role_name)
        return False

    tokens = [t.strip() for t in raw_skills.split("|") if t.strip()]
    if not tokens:
        logger.warning("Skipping row '%s': no valid skill tokens after splitting.", role_name)
        return False

    raw_rank = row.get("popularity_rank", "").strip()
    try:
        rank = int(raw_rank)
        if rank <= 0:
            raise ValueError
    except (ValueError, TypeError):
        logger.warning(
            "Skipping row '%s': popularity_rank '%s' is not a positive integer.",
            role_name, raw_rank,
        )
        return False

    return True


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_items(path: str) -> list[ItemRecord]:
    """Read *path* (a CSV file) and return a list of validated ItemRecord
    objects.

    The CSV must have columns: ``role_name``, ``skills``, ``popularity_rank``.
    Skills are pipe-delimited (``|``) within the ``skills`` column.

    Raises:
        DataLoadError: If the file does not exist, cannot be read, or yields
                       zero usable rows after validation.

    Returns:
        A non-empty list of ``ItemRecord`` objects with normalized text.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise DataLoadError(
            f"Dataset file not found: '{path}'. "
            "Check the --dataset flag or ensure 'data/raw_skills.csv' exists."
        )

    items: list[ItemRecord] = []
    skipped = 0

    try:
        with file_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if not validate_row(row):
                    skipped += 1
                    continue

                role_name = normalize_text(row["role_name"])
                raw_skills = row["skills"]
                tokens = [
                    normalize_skill(t)
                    for t in raw_skills.split("|")
                    if t.strip()
                ]
                popularity_rank = int(row["popularity_rank"].strip())

                items.append(ItemRecord(
                    role_name=role_name,
                    skills=tokens,
                    popularity_rank=popularity_rank,
                ))

    except (OSError, UnicodeDecodeError) as exc:
        raise DataLoadError(f"Failed to read dataset file '{path}': {exc}") from exc

    if not items:
        raise DataLoadError(
            f"Dataset at '{path}' contains no usable rows "
            f"({skipped} row(s) skipped due to validation errors). "
            "Ensure the file has a header and at least one valid data row."
        )

    if skipped:
        logger.info("Loaded %d item(s); %d row(s) skipped due to validation issues.", len(items), skipped)
    else:
        logger.debug("Loaded %d item(s) from '%s'.", len(items), path)

    return items
