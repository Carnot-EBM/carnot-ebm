"""Pagination helper for the existing 4 cron-prompt cluster queries.

The keyword-rotation sweep (cron) uses ``max_results=8`` with no offset,
returning the same 5-8 recent papers per cluster across rotations once
saturation hits. This helper emits arXiv API URLs paginated at any
offset so the sweep can extend the recent-window backward when the
front page has saturated.

Usage::

    # Print the cluster-0 URL at offset 8 (next page after the cron default)
    python scripts/sweep_paginate.py 0 --start 8

    # All 4 clusters at offset 16
    python scripts/sweep_paginate.py all --start 16

    # All 4 clusters at multiple offsets — sweep deeper window
    for s in 8 16 24; do python scripts/sweep_paginate.py all --start $s; done

The cron prompt itself stays unchanged; this helper is invoked from
the outer-loop sweep workflow when the dedupe filter rate exceeds
~70% (saturation signal), to extend the window before declaring 0
promotions.

Filed by: outer-loop 2026-05-15 alongside sweep_citations.py /
sweep_clusters.py / sweep_semscholar.py as the 4-pronged response to
the 2-successive-sweep saturation finding.
"""

from __future__ import annotations

import argparse
import sys

# The 4 fixed cluster queries from the cron prompt, expressed as the
# bare search_query string (without URL-prefix). Pagination is just
# `&start=N` appended at the end of the existing URL.
CLUSTERS: dict[int, str] = {
    0: 'abs:"verifier+ensemble"+OR+abs:"null+space+attack"+OR+abs:"specification+gaming"',
    1: 'abs:"energy+based+model"+AND+(abs:"reasoning"+OR+abs:"verification"+OR+abs:"LLM")',
    2: 'abs:"sparse+autoencoder"+OR+abs:"white+box+probe"+OR+abs:"reconstruction+error"+AND+abs:"LLM"',
    3: 'abs:"active+inference"+OR+abs:"free+energy"+AND+abs:"LLM"',
}

API_BASE = "http://export.arxiv.org/api/query"


def cluster_url(cluster: int, start: int, max_results: int) -> str:
    """Build the cluster-N URL at the given offset."""
    if cluster not in CLUSTERS:
        raise ValueError(f"cluster must be 0-3, got {cluster}")
    return (
        f"{API_BASE}?search_query={CLUSTERS[cluster]}"
        f"&start={start}&max_results={max_results}"
        f"&sortBy=submittedDate&sortOrder=descending"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pagination helper for the 4 cron-prompt cluster queries."
    )
    parser.add_argument(
        "cluster",
        help="Cluster index 0|1|2|3 or 'all' for every cluster.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=8,
        help="Pagination offset. Cron default is 0; this helper defaults to 8 "
        "(the next page). For deeper sweeps use 16, 24, 32.",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=8,
        help="Result count per page (cron default is 8; matches the existing window).",
    )
    args = parser.parse_args()

    if args.cluster == "all":
        clusters = [0, 1, 2, 3]
    else:
        try:
            clusters = [int(args.cluster)]
        except ValueError:
            print(
                f"cluster must be 0-3 or 'all', got {args.cluster!r}",
                file=sys.stderr,
            )
            return 2

    for c in clusters:
        print(cluster_url(c, args.start, args.max_results))

    return 0


if __name__ == "__main__":
    sys.exit(main())
