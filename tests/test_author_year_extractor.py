"""Unit tests for the deterministic ``extract_author_year`` fallback.

Locks in the token-classification behavior (surname vs. initials, ``Last,
First`` vs ``First Last``, accent folding) that the LLM fallback relies
on. Any change to the regexes in ``name_suggester.py`` should either keep
these cases passing or update them explicitly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eegdash_tagger.naming.name_suggester import (  # noqa: E402
    _is_valid_identifier,
    extract_author_year,
)


@pytest.mark.parametrize(
    "desc, readme, expected",
    [
        # Canonical "Authors: First Last" + year in readme.
        (
            "Authors: Janine Mendola, Elizabeth Bock",
            "Mendola et al. (2020). Neural dynamics...",
            "Mendola2020",
        ),
        # "Authors: Last, First" — surname is the left side of the comma.
        (
            "Authors: Smith, John A.",
            "Smith 2023",
            "Smith2023",
        ),
        # Initial sequences like ``WH`` / ``RP`` must NOT be picked as
        # the surname — they follow the surname, not replace it.
        (
            "Authors: Thompson WH*, Nair R*, Oya H*, Esteban O",
            "Thompson et al. 2020 NeuroImage",
            "Thompson2020",
        ),
        # Accent folding: ``Mikulán`` → ``Mikulan``.
        (
            "Authors: Mikulán, Ezequiel",
            "Mikulán 2018",
            "Mikulan2018",
        ),
        # Lowercase connectors (``van den``) must be skipped.
        (
            "Authors: van den Berg, Smith",
            "van den Berg 2021",
            "Berg2021",
        ),
        # Single-word author list still works.
        (
            "Authors: Delorme",
            "2019 publication",
            "Delorme2019",
        ),
    ],
)
def test_extract_author_year_happy_paths(desc, readme, expected):
    md = {"dataset_description": desc, "readme": readme}
    got = extract_author_year(md)
    assert got is not None, f"expected a match, got None for {md!r}"
    name, reasoning = got
    assert name == expected
    assert "author_year fallback" in reasoning


@pytest.mark.parametrize(
    "md",
    [
        # No Authors: line anywhere.
        {"dataset_description": "Some paper.", "readme": "Published 2019."},
        # Authors present but no year anywhere.
        {"dataset_description": "Authors: Smith, J.", "readme": ""},
        # Completely empty metadata.
        {},
        # ``Author`` (singular) with no year.
        {"dataset_description": "Author: Smith", "readme": ""},
    ],
)
def test_extract_author_year_returns_none_when_incomplete(md):
    assert extract_author_year(md) is None


def test_extract_author_year_refuses_to_invent_year():
    """Even if a 4-digit number is in the doi, it must not be paired with
    an author if the number is not actually a year."""
    # 2-digit / 3-digit tokens shouldn't upgrade to a year.
    md = {
        "dataset_description": "Authors: Smith",
        "readme": "version 123; revision 89",
        "doi": "",
    }
    assert extract_author_year(md) is None


def test_doi_year_is_used_as_fallback():
    """When readme/description have no year but the DOI encodes one."""
    md = {
        "dataset_description": "Authors: Smith",
        "readme": "",
        "doi": "doi:10.1038/nature.2022.1234",
    }
    got = extract_author_year(md)
    assert got is not None
    name, _ = got
    assert name == "Smith2022"


def test_is_valid_identifier_length_floor():
    # Registry permits short names, but the suggester floor is 3.
    assert not _is_valid_identifier("MI")
    assert _is_valid_identifier("MMN")


def test_is_valid_identifier_rejects_keywords_and_non_strings():
    assert not _is_valid_identifier("class")
    assert not _is_valid_identifier("None")
    assert not _is_valid_identifier("True")
    assert not _is_valid_identifier(None)  # type: ignore[arg-type]
    assert not _is_valid_identifier(123)  # type: ignore[arg-type]
    assert not _is_valid_identifier("3bad")
    assert not _is_valid_identifier("has-dash")
