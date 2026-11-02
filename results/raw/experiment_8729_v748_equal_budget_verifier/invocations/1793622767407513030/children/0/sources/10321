"""Citation-following sweep helper.

The keyword-based study-phase sweep saturated 2026-05-15 at 81% dedupe
filter rate across 2 successive fires — the 4 fixed-cluster queries
have fully mapped the recent-window arxiv state at current SOTA.
Citation-following replicates the highest-yield literature-discovery
pattern: start from a high-value anchor paper, fetch papers citing
it AND papers it cites, then dedupe + score.

Uses the Semantic Scholar Graph API (free; no API key needed for
read-only paper-graph queries; rate limit ~100 req/sec). Returns
ONLY arXiv IDs of candidate papers — keyword filtering happens
downstream in the sweep workflow.

Usage::

    # Fetch both directions from one anchor; print arXiv IDs
    python scripts/sweep_citations.py 2605.12484

    # Citations only (papers that cite the anchor — forward in time)
    python scripts/sweep_citations.py 2605.12484 --direction citations

    # References only (papers the anchor cites — backward in time)
    python scripts/sweep_citations.py 2605.12484 --direction references

    # Pipe through the dedupe filter for new candidates only
    python scripts/sweep_citations.py 2605.12484 | \
        python scripts/sweep_dedupe.py --filter

Filed by: outer-loop 2026-05-15 (operator confirmed citation-following
as the citation-depth alternative after 2 successive saturation
sweeps with 0 promotions; routine keyword rotation had exhausted the
SOTA window).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request

SS_BASE = "https://api.semanticscholar.org/graph/v1/paper"
# Nested-field format for the S2 graph endpoint: each subfield needs the
# full dotted path, comma-separated. Just-`externalIds,title,year` would
# mean "externalIds prefixed by parent, but title/year top-level" — a
# silent footgun that returned empty externalIds in the first deploy.
CITED_FIELDS = "citedPaper.externalIds,citedPaper.title,citedPaper.year"
CITING_FIELDS = "citingPaper.externalIds,citingPaper.title,citingPaper.year"
# Request both citations + references in one call where possible (S2
# graph endpoints) using paginated `offset`/`limit` parameters.
PAGE_LIMIT = 100  # S2 max for these endpoints
USER_AGENT = "carnot-ebm-outer-loop/1.0 (https://github.com/Carnot-EBM/carnot-ebm)"


def _fetch_json(url: str, retries: int = 3, backoff_s: float = 2.0) -> dict:
    """Fetch a JSON document with simple retry-on-429/5xx backoff."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                # Rate-limit or transient; sleep and retry
                time.sleep(backoff_s * (2**attempt))
                continue
            raise
        except urllib.error.URLError:
            if attempt < retries - 1:
                time.sleep(backoff_s * (2**attempt))
                continue
            raise
    raise RuntimeError(f"Failed to fetch {url} after {retries} attempts")


def _arxiv_id_from_paper(paper: dict) -> str | None:
    """Pull the bare arXiv ID (YYMM.NNNNN) from a S2 paper record."""
    ext_ids = paper.get("externalIds") or {}
    arx = ext_ids.get("ArXiv")
    if not arx:
        return None
    # ArXiv field is typically just "2605.14449" or "2605.14449v1".
    # Strip version suffix.
    return re.sub(r"v\d+$", "", str(arx))


def fetch_citations(anchor: str) -> list[str]:
    """Papers that CITE the anchor (forward in time)."""
    out: list[str] = []
    offset = 0
    while True:
        url = (
            f"{SS_BASE}/ARXIV:{anchor}/citations"
            f"?fields={CITING_FIELDS}&limit={PAGE_LIMIT}&offset={offset}"
        )
        data = _fetch_json(url)
        items = data.get("data") or []
        if not items:
            break
        for item in items:
            citing = item.get("citingPaper") or {}
            arx = _arxiv_id_from_paper(citing)
            if arx:
                out.append(arx)
        offset += PAGE_LIMIT
        if len(items) < PAGE_LIMIT:
            break
    return out


def fetch_references(anchor: str) -> list[str]:
    """Papers that the anchor CITES (backward in time)."""
    out: list[str] = []
    offset = 0
    while True:
        url = (
            f"{SS_BASE}/ARXIV:{anchor}/references"
            f"?fields={CITED_FIELDS}&limit={PAGE_LIMIT}&offset={offset}"
        )
        data = _fetch_json(url)
        items = data.get("data") or []
        if not items:
            break
        for item in items:
            cited = item.get("citedPaper") or {}
            arx = _arxiv_id_from_paper(cited)
            if arx:
                out.append(arx)
        offset += PAGE_LIMIT
        if len(items) < PAGE_LIMIT:
            break
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Citation-following sweep helper for Carnot's literature queue."
    )
    parser.add_argument(
        "anchor",
        help="Anchor arXiv ID (e.g., 2605.12484; version suffix optional).",
    )
    parser.add_argument(
        "--direction",
        choices=("citations", "references", "both"),
        default="both",
        help="Which direction(s) to fetch. Default: both.",
    )
    args = parser.parse_args()

    # Normalize anchor (strip version suffix; S2 accepts either form).
    anchor = re.sub(r"v\d+$", "", args.anchor.strip())

    ids: set[str] = set()
    try:
        if args.direction in ("citations", "both"):
            ids.update(fetch_citations(anchor))
        if args.direction in ("references", "both"):
            ids.update(fetch_references(anchor))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(
                f"# anchor {anchor} not found in Semantic Scholar (arxiv ID may not be indexed yet)",
                file=sys.stderr,
            )
            return 3
        print(f"# error fetching S2 graph: HTTP {e.code}", file=sys.stderr)
        return 2
    except Exception as e:  # pragma: no cover
        print(f"# error: {e}", file=sys.stderr)
        return 2

    # Drop the anchor itself if it shows up via self-citation noise.
    ids.discard(anchor)

    for arxiv_id in sorted(ids):
        print(arxiv_id)
    print(
        f"# anchor: {anchor} direction: {args.direction} found: {len(ids)} unique arxiv IDs",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
