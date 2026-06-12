"""Resolve a dataset's source-paper DOI through a multi-source cascade.

Extends ``abstract_fetcher`` (which extracts DOIs from reference text) with the
full discovery cascade established in the EEGDash paper analysis:

    1. text          declared DOIs in how_to_acknowledge / references / readme
    2. openneuro      OpenNeuro GraphQL ``metadata.associatedPaperDOI`` (authoritative)
    3. nemar          NEMAR DataCite ``relatedIdentifiers`` (IsDerivedFrom), for nm/on
    4. search         OpenAlex -> Crossref -> PubMed -> Semantic Scholar -> CORE
                      (title + author, gated by author-surname overlap)
    5. oa             Unpaywall Open-Access status for the chosen DOI

Declared/authoritative links are trusted; inferred (search) links are accepted
only when the paper's authors overlap the dataset's authors. Data-repository and
methodology DOIs are excluded everywhere.

Keys are read from the environment (``.env`` via python-dotenv):
    SEMANTIC_SCHOLAR_API_KEY, CORE_API_KEY, CONTACT_EMAIL
"""
from __future__ import annotations

import os
import re
import time
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # pragma: no cover
    pass

CONTACT_EMAIL = os.environ.get("CONTACT_EMAIL", "")
S2_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
CORE_KEY = os.environ.get("CORE_API_KEY")
UA = {"User-Agent": f"EEGDash-paper-resolver/1.0 (mailto:{CONTACT_EMAIL})"}

# --- exclusions -------------------------------------------------------------
DATA_REPO_PREFIXES = (
    "10.18112/openneuro", "10.82901", "10.13026", "10.5281/zenodo",
    "10.5061/dryad", "10.17605/osf", "10.6084/m9.figshare", "10.7910/dvn",
    "10.12751/g-node",
)
METHODOLOGY_DOIS = {
    "10.1038/sdata.2016.44", "10.1038/s41597-019-0104-8", "10.1038/s41597-019-0105-7",
    "10.1038/sdata.2018.110", "10.21105/joss.01896", "10.1016/j.jneumeth.2003.10.009",
    "10.3389/fnins.2013.00267", "10.1016/j.neuroimage.2013.10.027", "10.1155/2011/156869",
    "10.7554/elife.71774", "10.1038/s41592-018-0235-4", "10.1371/journal.pcbi.1005209",
}
PREPRINT_PREFIXES = ("10.48550/arxiv", "10.1101/", "10.31234/", "10.31219/", "10.21203/")
DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"'<>)\]}\\]+", re.I)
_STOP = set("the a an of for and to in on with study dataset data eeg meg ieeg recording "
            "recordings task experiment human brain during using based via from this".split())


def _clean_doi(s: str) -> str:
    return s.strip().lower().rstrip(".,);]'\"")


def is_excluded_doi(doi: str) -> bool:
    doi = (doi or "").lower()
    return (not doi) or doi in METHODOLOGY_DOIS or any(doi.startswith(p) for p in DATA_REPO_PREFIXES)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z]", "", unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower())


def dataset_surnames(authors: Optional[List[str]]) -> Set[str]:
    out: Set[str] = set()
    for a in authors or []:
        a = str(a)
        parts = [_norm(p) for p in re.split(r"[,\s]+", a) if len(_norm(p)) > 2]
        if parts:
            out.add(parts[-1])
            if "," in a:
                out.add(parts[0])
    return {x for x in out if len(x) > 2}


def _title_tokens(s: Optional[str]) -> Set[str]:
    norm = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return {w for w in re.findall(r"[a-z0-9]+", norm) if len(w) > 2 and w not in _STOP}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    return len(a & b) / max(1, len(a | b)) if a and b else 0.0


def _get(url: str, headers: Optional[dict] = None, timeout: int = 30) -> Optional[dict]:
    try:
        r = requests.get(url, headers=headers or UA, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


# --- channel 1: declared DOIs in dataset text -------------------------------
def paper_dois_from_text(dataset: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Return [(doi, field)] of non-excluded paper DOIs, by field priority."""
    out: List[Tuple[str, str]] = []
    seen: Set[str] = set()
    for field in ("how_to_acknowledge", "references_and_links", "readme", "acknowledgements"):
        v = dataset.get(field)
        if not v:
            continue
        text = "\n".join(map(str, v)) if isinstance(v, list) else str(v)
        for m in DOI_RE.findall(text):
            doi = _clean_doi(m)
            if not is_excluded_doi(doi) and doi not in seen:
                seen.add(doi)
                out.append((doi, field))
    return out


# --- channel 2: OpenNeuro authoritative metadata ----------------------------
def openneuro_associated_paper(accession: str) -> Optional[str]:
    q = "query($id:ID!){dataset(id:$id){metadata{associatedPaperDOI openneuroPaperDOI}}}"
    try:
        r = requests.post("https://openneuro.org/crn/graphql",
                          json={"query": q, "variables": {"id": accession}}, headers=UA, timeout=30)
        meta = ((r.json().get("data") or {}).get("dataset") or {}).get("metadata") or {}
    except Exception:
        return None
    for key in ("associatedPaperDOI", "openneuroPaperDOI"):
        raw = meta.get(key)
        if raw:
            m = DOI_RE.search(re.sub(r"^.*?10\.", "10.", raw))
            doi = _clean_doi(m.group(0)) if m else None
            if doi and not is_excluded_doi(doi):
                return doi
    return None


# --- channel 3: NEMAR DataCite relatedIdentifiers ---------------------------
def nemar_source_doi(dataset_id: str) -> Optional[str]:
    doi_url = f"https://api.datacite.org/dois/10.82901/nemar.{dataset_id}"
    d = _get(doi_url)
    if not d:
        return None
    attrs = d.get("data", {}).get("attributes", {})
    src_rel = {"isderivedfrom", "issupplementto", "references"}
    for r in attrs.get("relatedIdentifiers") or []:
        if (r.get("relatedIdentifierType", "").lower() == "doi"
                and r.get("relationType", "").lower() in src_rel):
            doi = _clean_doi(r.get("relatedIdentifier", ""))
            if not is_excluded_doi(doi):
                return doi
    return None


# --- channels 4: inferred search (author-gated) -----------------------------
def _score(ov: int, jac: float) -> Optional[str]:
    # Inferred (search) channels REQUIRE >=1 author-surname match; title similarity
    # only boosts confidence. Title-only matches (ov==0) are rejected -- a 2-token
    # dataset name can spuriously hit a high Jaccard against an unrelated work.
    if ov >= 2 or (ov >= 1 and jac >= 0.5):
        return "high"
    if ov >= 1:
        return "medium"
    return None


def _eval_candidate(doi: str, paper_surnames: Set[str], title: str,
                    dsa: Set[str], nt: Set[str]) -> Optional[Tuple[str, int, float, str]]:
    doi = _clean_doi(doi)
    if is_excluded_doi(doi):
        return None
    ov = len(dsa & paper_surnames)
    jac = _jaccard(nt, _title_tokens(title))
    conf = _score(ov, jac)
    return (doi, ov, round(jac, 2), conf) if conf else None


def search_openalex(name: str, dsa: Set[str]) -> Optional[Tuple[str, int, float, str]]:
    nt = _title_tokens(name)
    d = _get("https://api.openalex.org/works?" + requests.compat.urlencode(
        {"search": name, "per_page": 6, "mailto": CONTACT_EMAIL,
         "select": "doi,title,authorships,type"}))
    best = None
    for w in (d or {}).get("results", []):
        if w.get("type") in ("dataset", "grant", "peer-review"):
            continue
        pa = {_norm((a.get("author") or {}).get("display_name", "").split()[-1])
              for a in w.get("authorships", [])}
        c = _eval_candidate((w.get("doi") or "").replace("https://doi.org/", ""),
                            {x for x in pa if len(x) > 2}, w.get("title") or "", dsa, nt)
        if c and (best is None or c[1] > best[1]):
            best = c
    return best


def search_crossref(name: str, dsa: Set[str]) -> Optional[Tuple[str, int, float, str]]:
    nt = _title_tokens(name)
    d = _get("https://api.crossref.org/works?" + requests.compat.urlencode(
        {"query.bibliographic": name, "rows": 5, "mailto": CONTACT_EMAIL}))
    best = None
    for it in (d or {}).get("message", {}).get("items", []):
        pa = {_norm(a.get("family", "")) for a in it.get("author", []) if a.get("family")}
        c = _eval_candidate(it.get("DOI", ""), {x for x in pa if len(x) > 2},
                            (it.get("title") or [""])[0], dsa, nt)
        if c and (best is None or c[1] > best[1]):
            best = c
    return best


def search_semantic_scholar(name: str, dsa: Set[str]) -> Optional[Tuple[str, int, float, str]]:
    if not S2_KEY:
        return None
    nt = _title_tokens(name)
    headers = {**UA, "x-api-key": S2_KEY}
    try:
        r = requests.get("https://api.semanticscholar.org/graph/v1/paper/search",
                         params={"query": name, "limit": 6, "fields": "title,externalIds,authors"},
                         headers=headers, timeout=30)
        data = r.json().get("data", []) if r.status_code == 200 else []
    except Exception:
        return None
    best = None
    for p in data:
        pa = {_norm((a.get("name") or "").split()[-1]) for a in (p.get("authors") or [])}
        c = _eval_candidate((p.get("externalIds") or {}).get("DOI", ""),
                            {x for x in pa if len(x) > 2}, p.get("title") or "", dsa, nt)
        if c and (best is None or c[1] > best[1]):
            best = c
    return best


def search_core(name: str, dsa: Set[str]) -> Optional[Tuple[str, int, float, str]]:
    if not CORE_KEY:
        return None
    nt = _title_tokens(name)
    d = _get("https://api.core.ac.uk/v3/search/works?" + requests.compat.urlencode({"q": name, "limit": 8}),
             headers={**UA, "Authorization": f"Bearer {CORE_KEY}"})
    best = None
    for w in (d or {}).get("results", []):
        pa = set()
        for a in w.get("authors") or []:
            nm = a.get("name") if isinstance(a, dict) else str(a)
            pa.add(_norm((nm or "").split(",")[0] if "," in (nm or "") else (nm or "").split()[-1] if nm else ""))
        doi = w.get("doi") or ""
        if not doi:
            for ident in w.get("identifiers") or []:
                s = str(ident.get("identifier") if isinstance(ident, dict) else ident)
                m = DOI_RE.search(s)
                if m:
                    doi = m.group(0)
                    break
        c = _eval_candidate(doi.replace("https://doi.org/", ""), {x for x in pa if len(x) > 2},
                            w.get("title") or "", dsa, nt)
        if c and (best is None or c[1] > best[1]):
            best = c
    return best


# --- channel 5: Open Access status ------------------------------------------
def oa_status(doi: str) -> Dict[str, Any]:
    if any(doi.startswith(p) for p in PREPRINT_PREFIXES):
        return {"is_oa": True, "oa_status": "preprint", "oa_url": f"https://doi.org/{doi}"}
    d = _get(f"https://api.unpaywall.org/v2/{doi}?email={CONTACT_EMAIL}")
    if not d:
        return {"is_oa": None, "oa_status": "unknown", "oa_url": ""}
    loc = d.get("best_oa_location") or {}
    return {"is_oa": bool(d.get("is_oa")), "oa_status": d.get("oa_status"), "oa_url": loc.get("url") or ""}


# --- orchestrator -----------------------------------------------------------
def _openneuro_accession(dataset: Dict[str, Any]) -> Optional[str]:
    did = dataset.get("dataset_id", "")
    if str(did).startswith("ds"):
        return did
    blob = (dataset.get("dataset_doi") or "") + ((dataset.get("external_links") or {}).get("source_url") or "")
    m = re.search(r"(ds\d{6})", blob)
    return m.group(1) if m else None


def _search_evidence(channel: str, ov, jac, partial: bool = False) -> str:
    """Human-readable evidence for an inferred (search-based) dataset->paper match."""
    qual = " (best partial match)" if partial else ""
    j = f", title Jaccard {jac:.2f}" if isinstance(jac, (int, float)) else ""
    return f"{channel} title+author search{qual}: {ov} author surname(s) matched{j}"


def resolve_source_paper(dataset: Dict[str, Any], with_oa: bool = True) -> Dict[str, Any]:
    """Resolve one dataset (an EEGDash dataset document) to its source paper.

    Returns {source_doi, channel, confidence, author_overlap, title_jaccard, is_oa, oa_status, oa_url}
    or {source_doi: None, ...} when nothing is found.
    """
    dsa = dataset_surnames(dataset.get("authors"))
    name = dataset.get("name") or dataset.get("computed_title") or ""
    result: Dict[str, Any] = {"source_doi": None, "channel": None, "confidence": None,
                              "author_overlap": None, "title_jaccard": None,
                              "match_evidence": None}

    def finish(doi, channel, conf, ov=None, jac=None, evidence=None):
        result.update(source_doi=doi, channel=channel, confidence=conf,
                      author_overlap=ov, title_jaccard=jac, match_evidence=evidence)
        if with_oa and doi:
            result.update(oa_status(doi))
        return result

    # 1. declared text (how_to_acknowledge first = authoritative)
    text_dois = paper_dois_from_text(dataset)
    for doi, field in text_dois:
        if field == "how_to_acknowledge":
            return finish(doi, "text/how_to_acknowledge", "high",
                          evidence="DOI declared in the dataset's 'how_to_acknowledge' field")
    # 2. OpenNeuro authoritative metadata
    acc = _openneuro_accession(dataset)
    if acc:
        doi = openneuro_associated_paper(acc)
        if doi:
            return finish(doi, "openneuro/associatedPaperDOI", "high",
                          evidence=f"OpenNeuro metadata (associatedPaperDOI) for accession {acc}")
    # 3. NEMAR DataCite
    if str(dataset.get("dataset_id", "")).startswith(("nm", "on")):
        doi = nemar_source_doi(dataset["dataset_id"])
        if doi:
            return finish(doi, "nemar/IsDerivedFrom", "high",
                          evidence="NEMAR/DataCite related identifier (IsDerivedFrom / IsSupplementTo / References)")
    # 4. remaining declared text (references/readme) -> verify author overlap
    for doi, field in text_dois:
        au = _openalex_authors(doi)
        ov = len(dsa & au) if au is not None else -1
        ev = (f"DOI found in the dataset's '{field}' field; "
              + ("author surnames not verifiable via OpenAlex" if ov == -1
                 else f"{ov} dataset author surname(s) match the paper"))
        return finish(doi, f"text/{field}",
                      "high" if (ov >= 1 or ov == -1) else "medium", ov, evidence=ev)
    # 5. inferred search, author-gated; stop at first high
    if len(name) >= 6 and dsa:
        for fn, ch in ((search_openalex, "openalex"), (search_crossref, "crossref"),
                       (search_semantic_scholar, "semanticscholar"), (search_core, "core")):
            hit = fn(name, dsa)
            if hit:
                doi, ov, jac, conf = hit
                if conf == "high":
                    return finish(doi, ch, conf, ov, jac,
                                  evidence=_search_evidence(ch, ov, jac))
                result["_pending"] = (doi, ch, conf, ov, jac)  # remember best medium
    pend = result.pop("_pending", None)
    if pend:
        doi, ch, conf, ov, jac = pend
        return finish(doi, ch, conf, ov, jac,
                      evidence=_search_evidence(ch, ov, jac, partial=True))
    return result


def _openalex_authors(doi: str) -> Optional[Set[str]]:
    d = _get(f"https://api.openalex.org/works/doi:{doi}?select=authorships&mailto={CONTACT_EMAIL}")
    if d is None:
        return None
    return {_norm((a.get("author") or {}).get("display_name", "").split()[-1])
            for a in d.get("authorships", [])} or set()


__all__ = ["resolve_source_paper", "oa_status", "is_excluded_doi", "paper_dois_from_text"]
