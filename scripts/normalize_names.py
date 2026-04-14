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

BASE = Path("data/processed")
INPUT = BASE / "name_suggestions_all.json"
NEMAR_STATUS = BASE / "nemar_dup_status.json"
ALL_META = BASE / "all_metadata.json"
OUTPUT = BASE / "name_suggestions_normalized.json"


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
    out = re.sub(r"__+", "_", out).strip("_")
    return out


def main() -> int:
    with INPUT.open() as f:
        data = json.load(f)
    with ALL_META.open() as f:
        meta = {e["dataset_id"]: e for e in json.load(f)}

    results = data["results"]
    results_by_id = {r["dataset_id"]: r for r in results}

    # 1. Apply manual strips before computing collisions — these entries
    #    were wrong upstream and shouldn't influence the collision map.
    manual_strip_map: dict[str, set[str]] = defaultdict(set)
    for did, bad_name, _ in MANUAL_STRIPS:
        manual_strip_map[did].add(bad_name)

    for r in results:
        bad = manual_strip_map.get(r["dataset_id"])
        if not bad:
            continue
        before = r["canonical_name"]
        r["canonical_name"] = [n for n in before if n not in bad]
        if not r["canonical_name"]:
            # We stripped the last name — downgrade metadata so consumers
            # see this as a non-result.
            r["name_source"] = "none"
            r["name_confidence"] = 0.0

    # 2. Strip names for datasets missing on nemar.
    for did in STALE_UPLOADS:
        if did in results_by_id:
            r = results_by_id[did]
            if r["canonical_name"]:
                r["canonical_name"] = []
                r["name_source"] = "stale_upload"
                r["name_confidence"] = 0.0

    # 2b. Deterministic author_year fallback: for everything still
    # ``name_source == 'none'`` (and not a stale upload), scan the
    # metadata for an ``Authors:`` line + a year and synthesise
    # ``<Surname><Year>``. Only fires when both pieces are unambiguously
    # present — it never invents a year.
    fallback_count = 0
    for r in results:
        if r.get("name_source") != "none":
            continue
        if r["dataset_id"] in STALE_UPLOADS:
            continue
        if r["canonical_name"]:
            continue
        md = meta.get(r["dataset_id"], {}).get("metadata", {})
        extracted = extract_author_year(md)
        if not extracted:
            continue
        name, reasoning = extracted
        r["canonical_name"] = [name]
        r["name_source"] = "author_year"
        # Lower confidence than LLM-backed matches — this is a purely
        # mechanical extraction, no semantic check.
        r["name_confidence"] = 0.55
        r["reasoning"] = reasoning
        fallback_count += 1

    # 3. Compute collision map after strips.
    name_to_ids: dict[str, list[str]] = defaultdict(list)
    for r in results:
        for n in r["canonical_name"]:
            name_to_ids[n].append(r["dataset_id"])

    # 4. For each colliding name, rewrite each dataset's entry with a
    #    disambiguator. We don't touch single-owner names.
    rewritten_count = 0
    collisions_resolved: list[tuple[str, list[tuple[str, str]]]] = []
    for name, ids in list(name_to_ids.items()):
        if len(ids) < 2:
            continue
        changes: list[tuple[str, str]] = []
        for did in ids:
            r = results_by_id[did]
            title = (meta.get(did, {}).get("metadata", {}).get("title") or "").strip()
            suffix = _pick_suffix(name, did, title)
            new_name = _concat(name, suffix)
            # Replace the colliding name in-place, preserve order, dedupe.
            replaced: list[str] = []
            seen: set[str] = set()
            for existing in r["canonical_name"]:
                candidate = new_name if existing == name else existing
                if candidate not in seen:
                    replaced.append(candidate)
                    seen.add(candidate)
            r["canonical_name"] = replaced
            changes.append((did, new_name))
            rewritten_count += 1
        collisions_resolved.append((name, changes))

    # 5. Re-check for any remaining collisions (suffix function could
    #    still produce collisions if two datasets share both the same
    #    name AND the same derived suffix). For each still-colliding
    #    name, append a short data-derived suffix (digit tail of the
    #    dataset_id) to every owner — so ``Smith2023`` on ds004520 and
    #    ds007137 becomes ``Smith2023_4520`` / ``Smith2023_7137``.
    new_name_to_ids: dict[str, list[str]] = defaultdict(list)
    for r in results:
        for n in r["canonical_name"]:
            new_name_to_ids[n].append(r["dataset_id"])
    remaining_dups = {n: ids for n, ids in new_name_to_ids.items() if len(ids) > 1}
    if remaining_dups:
        for name, ids in remaining_dups.items():
            # Apply the short-suffix to ALL owners (not just the latecomers)
            # — every entry reads unambiguously on its own.
            for did in ids:
                r = results_by_id[did]
                tiebreak = _concat(name, _short_dataset_suffix(did))
                r["canonical_name"] = [
                    tiebreak if n == name else n for n in r["canonical_name"]
                ]

    # 6. Write output and print summary.
    with OUTPUT.open("w") as f:
        json.dump({"results": results}, f, indent=2, ensure_ascii=False)

    print(f"Wrote {OUTPUT}")
    print(f"Renamed {rewritten_count} name-slot(s) across "
          f"{len(collisions_resolved)} colliding groups")
    print(f"Stale uploads zeroed: {sorted(STALE_UPLOADS)}")
    print(f"Manual strips applied: {len(MANUAL_STRIPS)}")
    print(f"Author_year fallback filled: {fallback_count}")

    # Verify zero unresolved collisions.
    final_name_to_ids: dict[str, list[str]] = defaultdict(list)
    for r in results:
        for n in r["canonical_name"]:
            final_name_to_ids[n].append(r["dataset_id"])
    final_dups = [(n, ids) for n, ids in final_name_to_ids.items() if len(ids) > 1]
    if final_dups:
        print(f"\n!! {len(final_dups)} unresolved collisions remain:")
        for n, ids in final_dups[:10]:
            print(f"   {n}: {ids}")
    else:
        print("\n✓ All canonical names are unique across datasets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
