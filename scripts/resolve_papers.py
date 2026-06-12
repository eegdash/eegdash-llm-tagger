#!/usr/bin/env python3
"""Resolve each EEGDash dataset to its source-paper DOI (+ Open-Access status).

Runs the multi-source cascade in ``eegdash_tagger.scraping.paper_resolver`` over
the live EEGDash catalog and writes ``resolved_links.csv``.

    python3 scripts/resolve_papers.py [--limit N] [--out resolved_links.csv] [--workers 8]

Keys (Semantic Scholar, CORE, contact email) are read from ``.env``. See
``.env.example``. Only the OpenAlex/Crossref/PubMed/Unpaywall channels run
without keys; Semantic Scholar and CORE are skipped when their key is absent.
"""
import argparse
import concurrent.futures as cf
import csv
import json
import sys
import urllib.request

from eegdash_tagger.scraping.paper_resolver import resolve_source_paper

API = "https://data.eegdash.org/api/eegdash/datasets?limit=1000"  # 1000 = page cap


def fetch_catalog():
    req = urllib.request.Request(API, headers={"User-Agent": "EEGDash-paper-resolver/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["data"]


CHANNEL_DESC = {
    "text/how_to_acknowledge": "DOI declared in the dataset's how_to_acknowledge field",
    "openneuro/associatedPaperDOI": "OpenNeuro associatedPaperDOI metadata",
    "nemar/IsDerivedFrom": "NEMAR/DataCite related identifier",
    "text/references_and_links": "DOI found in the references_and_links field",
    "text/readme": "DOI found in the dataset README",
    "text/acknowledgements": "DOI found in the acknowledgements field",
    "openalex": "OpenAlex title+author search",
    "crossref": "Crossref title+author search",
    "semanticscholar": "Semantic Scholar title+author search",
    "core": "CORE title+author search",
}


def write_report(rows, path):
    """Write a Markdown listing of HOW each dataset was matched to its source paper."""
    from collections import Counter, defaultdict

    by_channel = Counter()
    conf = defaultdict(Counter)
    example = {}
    unresolved = []
    for did, r in rows:
        if r.get("source_doi"):
            ch = r.get("channel")
            by_channel[ch] += 1
            conf[ch][r.get("confidence") or "?"] += 1
            example.setdefault(ch, (did, r.get("source_doi")))
        else:
            unresolved.append(did)
    resolved = sum(by_channel.values())
    total = resolved + len(unresolved)

    out = [
        "# Dataset → source-paper match provenance\n",
        f"Resolved **{resolved}/{total}** datasets to a source paper "
        f"({len(unresolved)} unresolved). Each match records the *channel* (how the "
        f"paper was found), a confidence, and human-readable evidence. Channels are "
        f"tried in order of authority: declared DOIs first, then archive metadata "
        f"(OpenNeuro/NEMAR), then author-gated literature search.\n",
        "## How matches were made (by channel)\n",
        "| channel | what it means | matches | high | medium | example |",
        "|---|---|--:|--:|--:|---|",
    ]
    for ch, n in by_channel.most_common():
        desc = CHANNEL_DESC.get(ch, ch or "")
        hi = conf[ch].get("high", 0)
        me = conf[ch].get("medium", 0)
        ex_did, ex_doi = example.get(ch, ("", ""))
        out.append(f"| `{ch}` | {desc} | {n} | {hi} | {me} | `{ex_did}` → {ex_doi} |")
    out.append(f"| *(unresolved)* | no source paper found | {len(unresolved)} | | | |\n")

    out += [
        "## Per-dataset matches\n",
        "| dataset | source DOI | channel | confidence | evidence |",
        "|---|---|---|---|---|",
    ]
    for did, r in sorted(rows):
        if not r.get("source_doi"):
            continue
        ev = (r.get("match_evidence") or "").replace("|", "\\|")
        out.append(f"| `{did}` | {r.get('source_doi')} | `{r.get('channel')}` | "
                   f"{r.get('confidence') or ''} | {ev} |")
    if unresolved:
        out.append("\n## Unresolved datasets (no source paper found)\n")
        out.append(", ".join(f"`{d}`" for d in sorted(unresolved)))

    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only first N datasets (debug)")
    ap.add_argument("--out", default="resolved_links.csv")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-oa", action="store_true", help="skip Unpaywall OA lookup")
    ap.add_argument("--report", nargs="?", const="paper_match_report.md", default=None,
                    help="also write a Markdown listing of how each dataset matched its paper")
    args = ap.parse_args()

    datasets = fetch_catalog()
    if args.limit:
        datasets = datasets[: args.limit]
    print(f"resolving {len(datasets)} datasets ...", file=sys.stderr)

    rows = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(resolve_source_paper, d, not args.no_oa): d for d in datasets}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            d = futs[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = {"source_doi": None, "channel": f"error:{type(e).__name__}"}
            rows.append((d["dataset_id"], res))
            if i % 50 == 0:
                print(f"  ...{i}/{len(datasets)}", file=sys.stderr)

    resolved = [r for _, r in rows if r.get("source_doi")]
    hi = sum(1 for _, r in rows if r.get("confidence") == "high")
    me = sum(1 for _, r in rows if r.get("confidence") == "medium")
    oa = sum(1 for _, r in rows if r.get("is_oa") is True)
    print(f"resolved {len(resolved)}/{len(rows)} "
          f"(high {hi}, medium {me}); open-access {oa}", file=sys.stderr)

    cols = ["dataset_id", "source_doi", "channel", "confidence",
            "author_overlap", "title_jaccard", "match_evidence",
            "is_oa", "oa_status", "oa_url"]
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for did, r in sorted(rows):
            if r.get("source_doi"):
                w.writerow([did] + [r.get(c) for c in cols[1:]])
    print(f"wrote {args.out}", file=sys.stderr)

    if args.report:
        write_report(rows, args.report)
        print(f"wrote {args.report}", file=sys.stderr)


if __name__ == "__main__":
    main()
