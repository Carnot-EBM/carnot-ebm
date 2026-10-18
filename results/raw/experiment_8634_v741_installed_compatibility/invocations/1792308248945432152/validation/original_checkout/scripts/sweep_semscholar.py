"""Semantic Scholar keyword-search alternative channel.

The 4 arxiv-cluster queries saturated at ~81% dedupe filter rate
across 2 successive sweeps 2026-05-15. Semantic Scholar's
``/graph/v1/paper/search`` indexes a wider venue space than arXiv
(NeurIPS / ICLR / ACL / non-arxiv journals + arxiv preprints) and
its full-text relevance ranking sometimes surfaces papers that
narrow ``abs:"<phrase>"`` queries miss.

This helper does keyword search on S2 and prints arxiv IDs (skipping
S2-only papers that don't have an arxiv mirror). Output is piped
through ``sweep_dedupe.py --filter`` like the citation helper.

Usage::

    # Search for one topic; print arxiv IDs only
    python scripts/sweep_semscholar.py "fast-slow training LLM verifier"

    # Multiple keyword phrases (one per line via stdin)
    printf "process reward model\\ntranscoder LLM\\n" | \\
        python scripts/sweep_semscholar.py -

    # Pipe through dedupe
    python scripts/sweep_semscholar.py "verifier ensemble null space" | \\
        python scripts/sweep_dedupe.py --filter

Uses the free Semantic Scholar Graph API (~100 req/sec rate limit;
no API key needed for the search endpoint).

Filed by: outer-loop 2026-05-15 alongside sweep_citations.py /
sweep_paginate.py / sweep_clusters.py as the 4-pronged response to
the saturation finding. Channel-switching complements both the
breadth-broadening (sweep_clusters) and the depth-following
(sweep_citations) approaches.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

SS_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "externalIds,title,year,venue"
USER_AGENT = "carnot-ebm-outer-loop/1.0 (https://github.com/Carnot-EBM/carnot-ebm)"


def _fetch_json(url: str, retries: int = 3, backoff_s: float = 2.0) -> dict:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(backoff_s * (2**attempt))
                continue
            raise
        except urllib.error.URLError:
            if attempt < retries - 1:
                time.sleep(backoff_s * (2**attempt))
                continue
            raise
    raise RuntimeError(f"Failed to fetch {url} after {retries} attempts")


def search(query: str, limit: int = 20) -> list[str]:
    """Return arxiv IDs from S2 keyword search; skip non-arxiv hits."""
    q = urllib.parse.quote(query)
    url = f"{SS_SEARCH}?query={q}&limit={limit}&fields={FIELDS}"
    data = _fetch_json(url)
    out: list[str] = []
    for paper in data.get("data") or []:
        ext_ids = paper.get("externalIds") or {}
        arx = ext_ids.get("ArXiv")
        if not arx:
            continue
        out.append(re.sub(r"v\d+$", "", str(arx)))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Semantic Scholar keyword-search alternative channel."
    )
    parser.add_argument(
        "query",
        help="Keyword query string OR '-' to read queries from stdin (one per line).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Max results per query (S2 max is 100; default 20 balances cost vs novelty).",
    )
    args = parser.parse_args()

    queries: list[str] = []
    if args.query == "-":
        queries = [ln.strip() for ln in sys.stdin if ln.strip()]
    else:
        queries = [args.query]

    seen: set[str] = set()
    for q in queries:
        try:
            for arx in search(q, args.limit):
                if arx not in seen:
                    seen.add(arx)
                    print(arx)
        except urllib.error.HTTPError as e:
            print(f"# {q!r}: HTTP {e.code}", file=sys.stderr)
            continue

    print(
        f"# semscholar: {len(queries)} query(ies) → {len(seen)} unique arxiv IDs",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
