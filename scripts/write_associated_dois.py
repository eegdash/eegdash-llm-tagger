#!/usr/bin/env python3
"""Write resolved source-paper DOIs into the EEGDash database (admin API).

PATCHes ``associated_paper_doi`` (and optional provenance) onto each dataset via
``PATCH /admin/{database}/datasets/{dataset_id}``.

SAFETY:
  * Dry-run by default -- prints what WOULD change, sends nothing. Pass --execute to write.
  * Targets the ``eegdash_dev`` database by default. Pass --database eegdash for production.
  * Only writes datasets whose confidence meets --min-confidence (default: high).

Auth: set EEGDASH_ADMIN_TOKEN in .env (Authorization: Bearer <token>).
      EEGDASH_BASE_URL defaults to https://data.eegdash.org.

    python3 scripts/write_associated_dois.py --csv resolved_links.csv            # dry-run, dev
    python3 scripts/write_associated_dois.py --csv resolved_links.csv --execute   # write to dev
    python3 scripts/write_associated_dois.py --csv resolved_links.csv --database eegdash --execute
"""
import argparse
import csv
import os
import sys

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

BASE = os.environ.get("EEGDASH_BASE_URL", "https://data.eegdash.org").rstrip("/")
TOKEN = os.environ.get("EEGDASH_ADMIN_TOKEN")
RANK = {"high": 2, "medium": 1, "low": 0}


def _as_bool(v):
    return {"true": True, "false": False}.get(str(v).strip().lower())


def _as_int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def build_payload(row, with_provenance):
    payload = {"associated_paper_doi": row["source_doi"]}
    if with_provenance:
        payload["associated_paper_meta"] = {
            "channel": row.get("channel"),
            "confidence": row.get("confidence"),
            "author_overlap": _as_int(row.get("author_overlap")),
            "match_evidence": row.get("match_evidence"),
            "is_oa": _as_bool(row.get("is_oa")),
            "oa_status": row.get("oa_status"),
            "source": "paper_resolver",
        }
        if row.get("oa_url"):
            payload["external_links"] = {"paper_url": row["oa_url"]}
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="resolved_links.csv")
    ap.add_argument("--database", default="eegdash_dev",
                    help="target DB (default eegdash_dev; use 'eegdash' for production)")
    ap.add_argument("--min-confidence", choices=["high", "medium"], default="high")
    ap.add_argument("--with-provenance", action="store_true",
                    help="also write associated_paper_meta + external_links.paper_url")
    ap.add_argument("--execute", action="store_true", help="actually send PATCHes (default: dry-run)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    with open(args.csv) as f:
        rows = [r for r in csv.DictReader(f) if r.get("source_doi")]
    rows = [r for r in rows if RANK.get(r.get("confidence"), 0) >= RANK[args.min_confidence]]
    if args.limit:
        rows = rows[: args.limit]

    mode = "EXECUTE" if args.execute else "DRY-RUN"
    print(f"[{mode}] {len(rows)} datasets -> {BASE}/admin/{args.database}/datasets/<id> "
          f"(min-confidence={args.min_confidence}, provenance={args.with_provenance})", file=sys.stderr)
    if args.database == "eegdash" and args.execute:
        print("  *** WRITING TO PRODUCTION 'eegdash' ***", file=sys.stderr)

    if args.execute and not TOKEN:
        sys.exit("ERROR: EEGDASH_ADMIN_TOKEN not set in .env -- cannot authenticate writes.")

    sess = requests.Session()
    if TOKEN:
        sess.headers["Authorization"] = f"Bearer {TOKEN}"
    sess.headers["User-Agent"] = "EEGDash-paper-resolver/1.0"

    ok = err = 0
    for i, r in enumerate(rows, 1):
        did = r["dataset_id"]
        payload = build_payload(r, args.with_provenance)
        url = f"{BASE}/admin/{args.database}/datasets/{did}"
        body = {"update": payload}  # admin API: {"update": {<$set fields>}}
        if not args.execute:
            if i <= 5 or i == len(rows):
                print(f"  would PATCH {did}: {body}")
            continue
        try:
            resp = sess.patch(url, json=body, timeout=30)
            if resp.status_code < 300:
                ok += 1
            else:
                err += 1
                if err <= 5:
                    print(f"  FAIL {did}: HTTP {resp.status_code} {resp.text[:120]}", file=sys.stderr)
        except Exception as e:
            err += 1
            if err <= 5:
                print(f"  FAIL {did}: {e}", file=sys.stderr)
        if i % 100 == 0:
            print(f"  ...{i}/{len(rows)} (ok={ok}, err={err})", file=sys.stderr)

    if args.execute:
        print(f"DONE: {ok} written, {err} failed", file=sys.stderr)
    else:
        print(f"DRY-RUN complete: {len(rows)} datasets ready. Re-run with --execute to write.", file=sys.stderr)


if __name__ == "__main__":
    main()
