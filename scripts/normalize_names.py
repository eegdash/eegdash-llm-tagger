#!/usr/bin/env python3
"""Normalize the LLM name suggestions.

Starts from ``name_suggestions_all.json`` and, for every name that
collides across datasets:

- Appends a disambiguator derived from the dataset itself — HBN release /
  mini from the ``dataset_id``, title suffix after a trailing ``-X``, a
  paradigm keyword (``P300`` / ``MMN`` / ``MI`` / ``SSVEP`` / ...), the
  OpenNeuro experiment number, or the dataset_id itself as a last resort.
- Strips canonical-name entries for datasets explicitly flagged in
  ``nemar_dup_status.json`` as missing from nemar (stale uploads).
- Deletes known LLM mis-associations where the proposed name does not
  match the dataset's own title (e.g. ``Lee2024`` assigned to a
  stereo-electrode paper).

Writes ``name_suggestions_normalized.json`` alongside the input.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eegdash_tagger.naming.name_suggester import extract_author_year  # noqa: E402

# Resolve paths relative to the llm-tagger repo root so the script runs
# correctly from any CWD.
_REPO_ROOT = Path(__file__).resolve().parent.parent
BASE = _REPO_ROOT / "data" / "processed"
INPUT = BASE / "name_suggestions_all.json"
ALL_META = BASE / "all_metadata.json"
OUTPUT = BASE / "name_suggestions_normalized.json"

# Confidence assigned to names produced by the deterministic Authors:+year
# extractor. Lower than LLM-backed matches because no semantic check is
# applied.
FALLBACK_CONFIDENCE = 0.55


# --- manual strips: (dataset_id, canonical_name_to_strip) ---------------
# These are clear LLM mis-associations where the name doesn't belong to
# the dataset at all. Each entry has the dataset_id and the bad name
# that was attached to it, with a reason.
MANUAL_STRIPS: list[tuple[str, str, str]] = [
    ("ds004819", "Lee2024", "stereo-electrode paper, not appliance-control Lee2024"),
    ("ds006802", "Moerel2025", "collaborative rule-learning, not Moerel2025"),
    ("ds007521", "Moerel2025", "hunger/food neural processing, not Moerel2025"),
    ("ds004625", "Liu2024", "walking-terrain dataset, not Liu2024 motor imagery"),
    ("nm000158", "Liu2024", "motor-imagery Liu2024 kept on its canonical; strip shared alias"),
    ("ds005083", "Yang2025", "pediatric stereo-EEG, not Yang2025"),
    ("ds005489", "Herrema2025", "open-loop stim free recall, not Herrema2025"),
    ("ds005522", "Herrema2025", "spatial-navigation memory, not Herrema2025"),
    ("nm000187", "Mainsah2025", "BigP3BCI-N study, only BigP3BCI is canonical here"),
    ("nm000248", "Mainsah2025", "BigP3BCI-L study, only BigP3BCI is canonical here"),
]


# --- datasets that are missing / deleted on nemar ----------------------
# Names are stripped; `name_source` becomes `stale_upload` so consumers
# can see why nothing is there.
STALE_UPLOADS: set[str] = {
    "nm000349",  # "Mainsah2025-B" — 'Dataset not found' on nemar
    # Deleted from eegdash DB (duplicate uploads; keep-DOI-sibling rule):
    "nm000202",  # dup of nm000260 (BI2012); deleted 2026-04-14
    "nm000203",  # dup of nm000266 (Sosulski2019); deleted 2026-04-14
    "nm000156",  # dup of nm000347 (HEFMI Shi2025); deleted 2026-04-14
}


# --- HBN release extraction --------------------------------------------
HBN_ID_RE = re.compile(r"^EEG2025r(\d+)(mini)?$", re.IGNORECASE)
HBN_TITLE_RELEASE_RE = re.compile(r"Release\s+(\d+)", re.IGNORECASE)
HBN_TITLE_BDF_RE = re.compile(r"BDF Converted", re.IGNORECASE)


# --- OpenNeuro release family markers ----------------------------------
ONL_ID_RE = re.compile(r"^ds(\d+)$", re.IGNORECASE)


# --- paradigm keywords (order matters — first match wins) --------------
PARADIGM_KEYWORDS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bN400\b", re.IGNORECASE), "N400"),
    (re.compile(r"\bP300\b", re.IGNORECASE), "P300"),
    (re.compile(r"\bN2pc\b", re.IGNORECASE), "N2pc"),
    (re.compile(r"\bMMN\b", re.IGNORECASE), "MMN"),
    (re.compile(r"\bLRP\b", re.IGNORECASE), "LRP"),
    (re.compile(r"\bERN\b", re.IGNORECASE), "ERN"),
    (re.compile(r"\bERP\b", re.IGNORECASE), "ERP"),
    (re.compile(r"\bSSVEP\b", re.IGNORECASE), "SSVEP"),
    (re.compile(r"\bMI\s*/\s*ME\b|multimodal MI/ME|MIME\b", re.IGNORECASE), "MIME"),
    (re.compile(r"Motor Execution|\bME\b", re.IGNORECASE), "ME"),
    (re.compile(r"\bMotor Imagery\b|\bMI\b", re.IGNORECASE), "MI"),
    (re.compile(r"\bBurstVEP\s*100\b", re.IGNORECASE), "BurstVEP100"),
    (re.compile(r"\bBurstVEP\s*40\b", re.IGNORECASE), "BurstVEP40"),
    (re.compile(r"\bCVEP\s*100\b", re.IGNORECASE), "CVEP100"),
    (re.compile(r"\bCVEP\s*40\b", re.IGNORECASE), "CVEP40"),
]


TRAILING_SUFFIX_RE = re.compile(r"-\s*([A-Za-z0-9]+)\s*(?:-\s*NEMAR Dataset)?\s*$")


def _sanitize_suffix(s: str) -> str:
    """Keep only identifier-safe chars."""
    return re.sub(r"[^A-Za-z0-9]", "_", s).strip("_")


def _hbn_disambiguator(dataset_id: str, title: str) -> str | None:
    m = HBN_ID_RE.match(dataset_id)
    if m:
        release = m.group(1)
        mini = m.group(2) or ""
        return f"r{release}_bdf{('_' + mini) if mini else ''}".rstrip("_")
    # OpenNeuro HBN release datasets (ds005505 → r1, ds005506 → r2, …)
    m = HBN_TITLE_RELEASE_RE.search(title)
    if m:
        bdf = HBN_TITLE_BDF_RE.search(title) is not None
        return f"r{m.group(1)}" + ("_bdf" if bdf else "")
    return None


def _paradigm_suffix(title: str) -> str | None:
    for pat, tag in PARADIGM_KEYWORDS:
        if pat.search(title):
            return tag
    return None


def _trailing_suffix(title: str) -> str | None:
    """Title endings like ``Mainsah2025-C`` -> ``C``."""
    m = TRAILING_SUFFIX_RE.search(title)
    if m:
        return _sanitize_suffix(m.group(1))
    return None


def _dataset_id_suffix(dataset_id: str) -> str:
    """Last-resort: suffix with the dataset_id itself."""
    return _sanitize_suffix(dataset_id)


def _short_dataset_suffix(dataset_id: str) -> str:
    """Compact data-derived disambiguator — just the digit tail of the
    dataset_id so collision-breakers stay short (``_4520`` not
    ``_ds004520``). Falls back to the full id if no digits are present.
    """
    m = re.search(r"(\d+)\s*$", dataset_id)
    if not m:
        return _sanitize_suffix(dataset_id)
    return m.group(1).lstrip("0") or m.group(1)


def _pick_suffix(name: str, dataset_id: str, title: str) -> str:
    """Return a stable disambiguator for ``name`` on a given dataset."""
    if name in ("HBN", "HealthyBrainNetwork", "HBN_EEG"):
        hbn = _hbn_disambiguator(dataset_id, title)
        if hbn:
            return hbn
    # 1. explicit trailing suffix in title (Mainsah2025-C, GuttmannFlury2025-MI)
    if s := _trailing_suffix(title):
        return s
    # 2. paradigm keyword
    if s := _paradigm_suffix(title):
        return s
    # 3. OpenNeuro ds-id (short form)
    if m := ONL_ID_RE.match(dataset_id):
        return f"ds{m.group(1)}"
    # 4. fall back to full dataset_id
    return _dataset_id_suffix(dataset_id)


def _concat(name: str, suffix: str) -> str:
    """Join name + suffix as a clean Python identifier."""
    out = f"{name}_{suffix}"
    return re.sub(r"__+", "_", out).strip("_")


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


def _build_name_index(results: list[dict]) -> dict[str, list[str]]:
    """Build a ``canonical_name -> [dataset_id, ...]`` index."""
    index: dict[str, list[str]] = defaultdict(list)
    for r in results:
        for n in r["canonical_name"]:
            index[n].append(r["dataset_id"])
    return index


def _clear_entry(r: dict, source: str) -> None:
    """Wipe the suggestion for ``r`` and tag it with ``source``."""
    r["canonical_name"] = []
    r["name_source"] = source
    r["name_confidence"] = 0.0


def _apply_manual_strips(results: list[dict]) -> None:
    """Remove LLM mis-associations listed in :data:`MANUAL_STRIPS`."""
    strips: dict[str, set[str]] = defaultdict(set)
    for did, bad, _reason in MANUAL_STRIPS:
        strips[did].add(bad)
    for r in results:
        bad = strips.get(r["dataset_id"])
        if not bad:
            continue
        r["canonical_name"] = [n for n in r["canonical_name"] if n not in bad]
        if not r["canonical_name"]:
            # Stripped the last name — consumers should see this as a non-result.
            _clear_entry(r, "none")


def _apply_stale_uploads(results_by_id: dict[str, dict]) -> None:
    """Zero canonical names for datasets known missing on nemar."""
    for did in STALE_UPLOADS:
        r = results_by_id.get(did)
        if r and r["canonical_name"]:
            _clear_entry(r, "stale_upload")


def _apply_author_year_fallback(
    results: list[dict], meta: dict[str, dict]
) -> int:
    """Deterministic ``Surname<Year>`` pass for remaining ``none`` entries.

    Scans each still-empty entry's metadata for an ``Authors:`` line plus
    any 4-digit year. Only fires when both are unambiguously present —
    never invents a year. Returns the number of entries upgraded.
    """
    count = 0
    for r in results:
        if r.get("name_source") != "none" or r["canonical_name"]:
            continue
        if r["dataset_id"] in STALE_UPLOADS:
            continue
        md = meta.get(r["dataset_id"], {}).get("metadata", {})
        extracted = extract_author_year(md)
        if not extracted:
            continue
        name, reasoning = extracted
        r["canonical_name"] = [name]
        r["name_source"] = "author_year"
        r["name_confidence"] = FALLBACK_CONFIDENCE
        r["reasoning"] = reasoning
        count += 1
    return count


def _rewrite_collision(
    r: dict, colliding_name: str, new_name: str
) -> None:
    """Replace ``colliding_name`` with ``new_name`` in r, order-preserving and deduped."""
    replaced: list[str] = []
    seen: set[str] = set()
    for existing in r["canonical_name"]:
        candidate = new_name if existing == colliding_name else existing
        if candidate not in seen:
            replaced.append(candidate)
            seen.add(candidate)
    r["canonical_name"] = replaced


def _resolve_collisions(
    results: list[dict],
    results_by_id: dict[str, dict],
    meta: dict[str, dict],
) -> tuple[int, int]:
    """Disambiguate every colliding canonical name.

    Returns ``(renamed_slots, colliding_groups)``.
    """
    renamed = 0
    groups = 0
    for name, ids in _build_name_index(results).items():
        if len(ids) < 2:
            continue
        groups += 1
        for did in ids:
            r = results_by_id[did]
            title = (meta.get(did, {}).get("metadata", {}).get("title") or "").strip()
            new_name = _concat(name, _pick_suffix(name, did, title))
            _rewrite_collision(r, name, new_name)
            renamed += 1
    return renamed, groups


def _apply_residual_suffix(
    results: list[dict], results_by_id: dict[str, dict]
) -> None:
    """Suffix with the dataset_id digit tail if any collision survives the
    primary disambiguation pass. Applies to every owner so each final
    name reads unambiguously (``Smith2023_4520``, ``Smith2023_7137``).
    """
    remaining = {n: ids for n, ids in _build_name_index(results).items() if len(ids) > 1}
    for name, ids in remaining.items():
        for did in ids:
            tiebreak = _concat(name, _short_dataset_suffix(did))
            _rewrite_collision(results_by_id[did], name, tiebreak)


def main() -> int:
    with INPUT.open() as f:
        data = json.load(f)
    with ALL_META.open() as f:
        meta = {e["dataset_id"]: e for e in json.load(f)}

    results = data["results"]
    results_by_id = {r["dataset_id"]: r for r in results}

    _apply_manual_strips(results)
    _apply_stale_uploads(results_by_id)
    fallback_count = _apply_author_year_fallback(results, meta)
    renamed, colliding_groups = _resolve_collisions(results, results_by_id, meta)
    _apply_residual_suffix(results, results_by_id)

    with OUTPUT.open("w") as f:
        json.dump({"results": results}, f, indent=2, ensure_ascii=False)

    print(f"Wrote {OUTPUT}")
    print(f"Renamed {renamed} name-slot(s) across {colliding_groups} colliding groups")
    print(f"Stale uploads zeroed: {sorted(STALE_UPLOADS)}")
    print(f"Manual strips applied: {len(MANUAL_STRIPS)}")
    print(f"Author_year fallback filled: {fallback_count}")

    final_dups = [(n, ids) for n, ids in _build_name_index(results).items() if len(ids) > 1]
    if final_dups:
        print(f"\n!! {len(final_dups)} unresolved collisions remain:")
        for n, ids in final_dups[:10]:
            print(f"   {n}: {ids}")
        return 1
    print("\n✓ All canonical names are unique across datasets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
