"""LLM-based canonical name suggester for EEG datasets.

Given a dataset's metadata (title, description, readme, paper abstract),
asks an OpenRouter.ai model to propose one to three canonical / community
names for the dataset, suitable for use as an import alias in
``eegdash.dataset`` (e.g. ``BrainTreeBank``, ``SleepEDF``).

The suggester emits, per dataset:

- ``canonical_name`` — list of valid Python-identifier names (may be empty)
- ``name_source`` — ``"canonical"`` (well-known community name), ``"author_year"``
  (FirstAuthorSurnameYear fallback), or ``"none"`` (nothing proposable)
- ``name_confidence`` — float in [0, 1]
- ``reasoning`` — short free-text from the model

Identifier validity is enforced client-side so the registry can trust the
output without re-validating.
"""

from __future__ import annotations

import json
import keyword
import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Optional, TypedDict

import requests

logger = logging.getLogger(__name__)


class NameSuggestion(TypedDict, total=False):
    """Structured name-suggestion output for a single dataset."""

    dataset_id: str
    canonical_name: list[str]
    name_source: str  # "canonical" | "author_year" | "none"
    name_confidence: float
    reasoning: str


NAMING_SYSTEM_PROMPT = """You are an expert EEG dataset curator for the \
EEGDash catalog.

Your task: given the metadata of ONE EEG dataset, propose up to THREE
canonical "import names" under which the dataset is (or would reasonably
be) known in the neuroscience / BCI community.

Each proposed name MUST be:
- A valid Python identifier (letters, digits, underscore; may not start with
  a digit; not a Python reserved keyword like ``class``, ``None``, ``True``).
- Short, human-readable, and unambiguous.

**Priority of sources (pick the FIRST that applies):**
1. **canonical** — The dataset is known by a widely-used short name or
   acronym in the literature. Examples: ``BrainTreeBank``, ``SleepEDF``,
   ``SleepEDFPlus``, ``TUAB``, ``TUEG``, ``BCICIV2a``, ``MASS``. Only use
   this source when the name is established in publications / community
   usage, NOT when you are inventing a tidy label from the title.
   - Patterns like ``BNCI 20XX-YYY`` in a title SHOULD be surfaced as
     ``BNCI20XX_YYY`` (e.g. ``BNCI2015_012``) — these are canonical
     BNCI Horizon benchmark identifiers.
   - The HBN / Healthy Brain Network releases are canonical under
     ``HBN_r<N>`` (and ``HBN_r<N>mini`` when the dataset is the mini cut).
2. **author_year** — If no canonical community name exists but you can
   identify a first author and a publication year, return ONE name of
   the form ``<FirstAuthorSurname><Year>`` (e.g. ``Smith2023``). Sources
   of author info to USE actively when present:
     • An ``Authors:`` line anywhere in ``dataset_description`` or
       ``readme`` — take the FIRST name, strip accents/punctuation, keep
       only the surname.
     • The title itself when it is already of the form ``Surname<Year>[-suffix]``
       (e.g. ``Mainsah2025-C`` → ``Mainsah2025``; ``Lee2019-SSVEP`` → ``Lee2019``).
   Sources of year to USE actively when present:
     • A 4-digit year inside a reference / citation in ``readme``.
     • A 4-digit year inside the title (e.g. ``2015`` in
       ``BNCI 2015-012``).
     • The DOI's publication year if encoded (most dataset-repo DOIs
       like ``10.18112/openneuro.dsXXXXXX.v1.0.0`` do NOT encode a year —
       do not guess).
   If you have a first author but no year, return ``none`` — do NOT
   fabricate a year.
3. **none** — If you cannot determine either, return an empty list with
   source ``none``. Do NOT invent names.

**Hard rules:**
- NEVER return CamelCase versions of the free-text title
  (e.g. do NOT turn "Single-pulse open-loop TMS-EEG dataset" into
  ``SinglePulseOpenLoopTMSEEGDataset``).
- NEVER return the dataset ID itself (no ``ds002001``, no ``nm000xxx``).
- NEVER return names shorter than 3 characters.
- Prefer ``none`` with low confidence over a guess.

**Output format:** Return strict JSON, no prose, no markdown fences.
Schema:
```
{
  "canonical_name": [<string>, ...],
  "name_source": "canonical" | "author_year" | "none",
  "name_confidence": <float in [0, 1]>,
  "reasoning": "<one or two sentences explaining the choice>"
}
```"""


# Identifier rules here are a strict SUPERSET of the registry's
# ``_is_valid_alias`` (eegdash/dataset/registry.py): we additionally
# require length >= 3 so suggestions like ``MI`` don't reach the
# catalog. The registry itself is intentionally more permissive so
# curator-added short aliases (e.g. ``TUH``) stay legal.
_MIN_IDENTIFIER_LEN = 3


def _is_valid_identifier(name: str) -> bool:
    """Non-keyword Python identifier of at least :data:`_MIN_IDENTIFIER_LEN` chars."""
    if not isinstance(name, str):
        return False
    name = name.strip()
    if len(name) < _MIN_IDENTIFIER_LEN:
        return False
    if not name.isidentifier():
        return False
    if keyword.iskeyword(name):
        return False
    if getattr(keyword, "issoftkeyword", lambda _s: False)(name):
        return False
    return True


class NameSuggester:
    """OpenRouter.ai-backed canonical-name suggester.

    Parallels :class:`OpenRouterTagger` but targets a different output
    schema (canonical names) and uses an inline system prompt rather than
    ``prompt.md``.
    """

    ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

    # Whitelist of metadata fields passed to the model. Keep tight — the
    # goal is a compact, disambiguating context, not a full dataset dump.
    RELEVANT_METADATA_KEYS = {
        "title",
        "dataset_description",
        "readme",
        "paper_abstract",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "openai/gpt-5.2",
        verbose: bool = False,
        max_tokens: int = 600,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY "
                "environment variable or pass api_key parameter."
            )
        self.model = model
        self.verbose = verbose
        self.max_tokens = max_tokens

    def _filter_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in metadata.items():
            if k not in self.RELEVANT_METADATA_KEYS:
                continue
            if v in (None, "", [], {}):
                continue
            # Truncate very long text fields to keep prompt bounded.
            if isinstance(v, str) and len(v) > 4000:
                v = v[:4000] + " …[truncated]"
            out[k] = v
        return out

    def _build_user_message(self, metadata: dict[str, Any]) -> str:
        payload = {"dataset": self._filter_metadata(metadata)}
        return json.dumps(payload, indent=2, ensure_ascii=False)

    def _call_api(self, user_message: str) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": NAMING_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        response = requests.post(
            self.ENDPOINT, headers=headers, json=payload, timeout=120
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _clean_json_response(content: str) -> str:
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines).strip()
        return content

    def _parse_response(
        self, response_data: dict[str, Any], dataset_id: str
    ) -> NameSuggestion:
        content = response_data["choices"][0]["message"]["content"]
        cleaned = self._clean_json_response(content)
        llm_out = json.loads(cleaned)

        raw_names = llm_out.get("canonical_name", [])
        if isinstance(raw_names, str):
            raw_names = [raw_names]
        if not isinstance(raw_names, list):
            raw_names = []

        # Client-side identifier validation — the registry will reject
        # invalid aliases anyway, but filtering here keeps the persisted
        # JSON clean and the source field truthful.
        filtered: list[str] = []
        seen: set[str] = set()
        for name in raw_names:
            if not isinstance(name, str):
                continue
            clean = name.strip()
            if _is_valid_identifier(clean) and clean not in seen:
                filtered.append(clean)
                seen.add(clean)

        source = llm_out.get("name_source")
        if source not in {"canonical", "author_year", "none"}:
            source = "none" if not filtered else "canonical"
        # If the model claimed a source but we filtered everything out,
        # downgrade to "none" so downstream consumers are not misled.
        if not filtered:
            source = "none"

        try:
            conf = float(llm_out.get("name_confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        conf = max(0.0, min(1.0, conf))
        if not filtered:
            conf = 0.0

        return NameSuggestion(
            dataset_id=dataset_id,
            canonical_name=filtered,
            name_source=source,
            name_confidence=conf,
            reasoning=str(llm_out.get("reasoning", ""))[:500],
        )

    def suggest(
        self, metadata: dict[str, Any], dataset_id: str = "unknown"
    ) -> NameSuggestion:
        """Suggest canonical names for a single dataset.

        Always returns a :class:`NameSuggestion`; on error it returns an
        empty ``canonical_name`` with ``name_source="none"`` and the error
        in ``reasoning`` so batch runs stay robust.
        """
        try:
            user_message = self._build_user_message(metadata)
            response = self._call_api(user_message)
            return self._parse_response(response, dataset_id)
        except Exception as exc:  # requests + json errors caught here
            logger.warning("name suggestion failed for %s: %s", dataset_id, exc)
            return NameSuggestion(
                dataset_id=dataset_id,
                canonical_name=[],
                name_source="none",
                name_confidence=0.0,
                reasoning=f"Error: {exc}",
            )


def get_default_metadata_path() -> Path:
    """Default input file — same one the tagger reads."""
    return (
        Path(__file__).parent.parent.parent.parent
        / "data"
        / "processed"
        / "incomplete_metadata.json"
    )


# ---------------------------------------------------------------------------
# Deterministic author_year fallback
# ---------------------------------------------------------------------------
#
# The LLM sometimes conservatively returns ``none`` even when an
# ``Authors:`` line and a 4-digit year are sitting in the description or
# readme. These helpers offer a last-resort extraction that never
# hallucinates — it only fires when both pieces of information are
# unambiguously present.

_AUTHORS_LINE_RE = re.compile(
    r"^\s*Authors?\s*[:\-]\s*(.+)$", re.IGNORECASE | re.MULTILINE
)
# Match ``Surname, First`` OR ``First Surname`` OR ``F. Surname``. We
# only keep the surname — the capitalised word that is NOT a single
# initial and NOT a lowercase connector like ``de``/``van``.
_NAME_TOKEN_RE = re.compile(r"[A-ZÀ-Ý][A-Za-zÀ-ÿ\-]+")
_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")


def _strip_accents(s: str) -> str:
    """Collapse diacritics so ``Mikulán`` becomes ``Mikulan``."""
    return "".join(
        c
        for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )


def _is_initials(tok: str) -> bool:
    """True for all-caps short tokens (``WH``, ``JMR``) that follow a surname."""
    return tok.isupper() and len(tok) <= 4


def _first_surname_from_authors_blob(authors_blob: str) -> str | None:
    """Extract the first author's surname from a free-text author list.

    Handles ``Last, First``, ``First Last``, ``F. Last``, and ASCII-fold
    accented surnames. Returns the surname as a Python-identifier-safe
    string, or ``None`` if nothing usable is found.
    """
    # Split on delimiters between authors: comma that isn't part of "Last,
    # First", semicolons, or " and ".
    first = re.split(r"\s+and\s+|;|\s*\*", authors_blob, maxsplit=1)[0]

    # "Last, First" vs "First Last" — if a single comma with text on
    # both sides, left side is surname.
    if "," in first:
        left, _right = [p.strip() for p in first.split(",", 1)]
        # If left has a capitalised multi-char token, treat as surname.
        left_tokens = _NAME_TOKEN_RE.findall(_strip_accents(left))
        left_tokens = [t for t in left_tokens if len(t) > 1 and not _is_initials(t)]
        if left_tokens:
            surname = re.sub(r"[^A-Za-z]", "", left_tokens[-1])
            if len(surname) >= 2:
                return surname
    # Otherwise: last capitalised non-initials token in the string.
    tokens = _NAME_TOKEN_RE.findall(_strip_accents(first))
    tokens = [t for t in tokens if len(t) > 1 and not _is_initials(t)]
    if not tokens:
        return None
    surname = re.sub(r"[^A-Za-z]", "", tokens[-1])
    return surname if len(surname) >= 2 else None


def extract_author_year(metadata: dict[str, Any]) -> tuple[str, str] | None:
    """Last-resort deterministic ``<FirstAuthorSurname><Year>`` extractor.

    Returns ``(name, reasoning)`` or ``None``. Scans, in order:

    1. ``Authors:`` / ``Author:`` line in ``dataset_description`` or
       ``readme`` for the surname.
    2. Any 4-digit year in readme → description → doi.
    """
    blob_desc = str(metadata.get("dataset_description") or "")
    blob_readme = str(metadata.get("readme") or "")
    blob_doi = str(metadata.get("doi") or "")

    surname: str | None = None
    for blob in (blob_desc, blob_readme):
        m = _AUTHORS_LINE_RE.search(blob)
        if not m:
            continue
        surname = _first_surname_from_authors_blob(m.group(1))
        if surname:
            break
    if not surname:
        return None

    year: str | None = None
    # Prefer a year that sits near a citation-like context; otherwise
    # accept the first 4-digit year we find.
    for blob in (blob_readme, blob_desc, blob_doi):
        match = _YEAR_RE.search(blob)
        if match:
            year = match.group(1)
            break
    if not year:
        return None

    candidate = f"{surname}{year}"
    # Identifier safety is re-enforced here since the surname logic
    # above is defensive but not perfect.
    if not _is_valid_identifier(candidate):
        return None
    return candidate, (
        f"author_year fallback: surname={surname!r} from Authors line, "
        f"year={year!r} from metadata"
    )
