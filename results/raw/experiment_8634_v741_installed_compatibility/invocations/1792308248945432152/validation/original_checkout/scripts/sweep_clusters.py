"""Broader cluster queries — fixes the precedence bug + adds OR-expansions.

The cron prompt has 4 hardcoded cluster URLs that:
  1. Use narrow ``abs:"<phrase>"`` exact-quote matching, missing adjacent
     literature that uses synonyms (e.g., "process reward model" PRM is
     verifier-adjacent but never matches "verifier ensemble").
  2. Have an operator-precedence bug in clusters 2 + 3:
     ``A OR B OR C AND D`` parses as ``A OR B OR (C AND D)``, so the AND
     constraint binds only to the last term. This surfaces EEG-domain SAE
     papers (cluster 2) and UAV-swarm active-inference papers (cluster 3)
     that don't actually mention "LLM" at all.

This helper emits the same 4 clusters with both fixes:
  * Each topical OR-chain is wrapped in explicit parens so AND-binding
    is unambiguous.
  * Adjacent terms added per the 12:48Z sweep proposal.

Usage::

    # Print cluster 0 (broadened) — verifier ensembles + PRM + reward hacking + deliberative alignment
    python scripts/sweep_clusters.py 0

    # Print all 4 broadened cluster URLs
    python scripts/sweep_clusters.py all

    # With pagination
    python scripts/sweep_clusters.py 0 --start 16 --max-results 20

Operator can use this output to replace the hardcoded URLs in the
cron prompt for /study-sweep. The outer-loop sweep workflow can also
invoke this helper directly when the dedupe filter rate exceeds the
saturation threshold (currently ~70%).

Filed by: outer-loop 2026-05-15 alongside sweep_citations.py /
sweep_paginate.py / sweep_semscholar.py as the 4-pronged response to
the saturation finding. The actual cron-prompt edit remains
operator-owned.
"""

from __future__ import annotations

import argparse
import sys

# Broadened cluster queries with parens-wrapped OR-chains and added
# adjacent terms per the 2026-05-15T12:48Z sweep recommendation.
# Cluster 4 added 2026-05-16 (operator-approved) for the hardware /
# sampling-acceleration topic area that the original 4 clusters missed.
BROADENED_CLUSTERS: dict[int, str] = {
    0: (
        '(abs:"verifier+ensemble"+OR+abs:"verifier+ensembles"+OR+'
        'abs:"null+space"+OR+abs:"specification+gaming"+OR+'
        'abs:"process+reward+model"+OR+abs:"deliberative+alignment"+OR+'
        'abs:"reward+hacking")'
    ),
    1: (
        '(abs:"energy+based+model"+OR+abs:"energy-based+model"+OR+'
        'abs:"energy+guided+decoding"+OR+abs:"token+energy"+OR+'
        'abs:"EBT")+AND+'
        '(abs:"reasoning"+OR+abs:"verification"+OR+abs:"LLM"+OR+'
        'abs:"language+model")'
    ),
    2: (
        '(abs:"sparse+autoencoder"+OR+abs:"white+box+probe"+OR+'
        'abs:"reconstruction+error"+OR+abs:"transcoder"+OR+'
        'abs:"crosscoder"+OR+abs:"feature+attribution")+AND+'
        '(abs:"LLM"+OR+abs:"language+model"+OR+abs:"transformer")'
    ),
    3: (
        '(abs:"active+inference"+OR+abs:"free+energy"+OR+'
        'abs:"free+energy+principle"+OR+abs:"predictive+coding"+OR+'
        'abs:"world+model")+AND+'
        '(abs:"LLM"+OR+abs:"language+model"+OR+abs:"reasoning")'
    ),
    # Cluster 4 — hardware acceleration / probabilistic sampling
    # Covers FPGA, photonic, Ising machines, thermodynamic computing,
    # analog ML — the topic area that BRAIN, Extropic Z1, KV260, GateMate,
    # and PolarFire all live in but which the original 4 clusters missed.
    # Constraint: AND with sampling / verification / inference / LLM to
    # filter out general-purpose RTL papers.
    4: (
        '(abs:"FPGA"+OR+abs:"Ising+machine"+OR+'
        'abs:"thermodynamic+computing"+OR+abs:"photonic+computing"+OR+'
        'abs:"analog+ML"+OR+abs:"analog+inference"+OR+'
        'abs:"in-memory+computing"+OR+abs:"probabilistic+computing")+AND+'
        '(abs:"sampling"+OR+abs:"verification"+OR+abs:"inference"+OR+'
        'abs:"LLM"+OR+abs:"reasoning")'
    ),
    # Cluster 5 — ARC-AGI-3 interactive-agent action efficiency (added
    # 2026-06-20, outer-loop, for the ARC-AGI-3 submission-sprint pivot).
    # The leaderboard scoring metric is action efficiency
    # ((human_actions/agent_actions)^2), so the live wall is "explore with
    # FEWER actions": affordance / clickability / frame-change prediction
    # (the StochasticGoose lever) + directed/novelty-seeking exploration.
    # The original 5 clusters had NO ARC / interactive-agent coverage —
    # they fed the pre-pivot verifier-moat / EBM / paper-v6 program.
    # Constraint: AND with agent / RL / exploration / interactive env so
    # this surfaces interactive-agent work, not generic vision affordances.
    5: (
        '(abs:"affordance"+OR+abs:"action+effect"+OR+abs:"clickability"+OR+'
        'abs:"frame+prediction"+OR+abs:"intrinsic+motivation"+OR+'
        'abs:"directed+exploration"+OR+abs:"novelty+search")+AND+'
        '(abs:"reinforcement+learning"+OR+abs:"agent"+OR+abs:"exploration"+OR+'
        'abs:"interactive+environment"+OR+abs:"ARC")'
    ),
    # Cluster 6 — neural-guided search / learned heuristics + goal/world-model
    # induction for agents (added 2026-06-20, outer-loop). The other ARC
    # lever besides affordance: prune/guide the explorer's search with a
    # learned heuristic (value-guided best-first), and induce the game's
    # goal / transition model (the Family-A/Family-B winning architectures).
    # Constraint: AND with planning / agent / reasoning / RL.
    6: (
        '(abs:"neural+guided+search"+OR+abs:"learned+heuristic"+OR+'
        'abs:"value+guided+search"+OR+abs:"program+induction"+OR+'
        'abs:"world+model"+OR+abs:"goal+induction")+AND+'
        '(abs:"planning"+OR+abs:"agent"+OR+abs:"reasoning"+OR+'
        'abs:"reinforcement+learning")'
    ),
}

API_BASE = "http://export.arxiv.org/api/query"


def cluster_url(cluster: int, start: int, max_results: int) -> str:
    if cluster not in BROADENED_CLUSTERS:
        raise ValueError(f"cluster must be 0-6, got {cluster}")
    return (
        f"{API_BASE}?search_query={BROADENED_CLUSTERS[cluster]}"
        f"&start={start}&max_results={max_results}"
        f"&sortBy=submittedDate&sortOrder=descending"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Emit broadened cluster URLs for the /study-sweep cron."
    )
    parser.add_argument("cluster", help="0|1|2|3|4|5|6 or 'all'")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument(
        "--max-results",
        type=int,
        default=20,
        help="Default 20 (cron default 8); broader queries + dedupe filter "
        "make the higher count cheap.",
    )
    args = parser.parse_args()

    if args.cluster == "all":
        clusters = [0, 1, 2, 3, 4, 5, 6]
    else:
        try:
            clusters = [int(args.cluster)]
        except ValueError:
            print(f"cluster must be 0-4 or 'all', got {args.cluster!r}", file=sys.stderr)
            return 2

    for c in clusters:
        print(cluster_url(c, args.start, args.max_results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
