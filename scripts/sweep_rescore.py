"""Periodic low-score rescore — sample stale low-scoring entries.

The sweep scoring rubric (relevance × novelty × feasibility × urgency)
gives papers a snapshot judgment at ingest time. As Carnot's active
research questions evolve, a paper that scored 30-100 last week may
score 200+ this week if it newly applies to an open question. The
filter has no automatic mechanism to catch this — once scored, an
entry sits in research-studying.md indefinitely.

This helper samples N low-scoring entries (Score 30-150 band) from
research-studying.md and emits their arXiv IDs + score + one-line
description for re-evaluation against current Carnot priorities.

Usage::

    # Sample 5 low-scoring entries for re-evaluation
    python scripts/sweep_rescore.py

    # Sample N entries
    python scripts/sweep_rescore.py --sample 10

    # Score band override (default 30-150)
    python scripts/sweep_rescore.py --min-score 50 --max-score 200

The intent is one rescore-pass every 4-5 keyword sweeps (or whenever
the keyword rotation returns 100% dedupe-skip — the saturation
signal). Output lines are arXiv IDs + brief context that the
sweep workflow can re-evaluate against current research-studying.md
"Current Focus" header.

Concrete example trigger: exp1709 .175 near-critical sampler limit
opened a new open question. arXiv:2601.02594 ALPS (annealed Langevin
for image inverse problems) had been scored 54 — out-of-domain.
After exp1709 surfaced the open question, ALPS is suddenly a
cross-cite candidate worth Score 200+. Periodic rescore catches that.

Filed by: outer-loop 2026-05-16 (operator-approved after asking whether
the filter catches things that could help the project).
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE = PROJECT_ROOT / "research-studying.md"

# Match score lines like:
#   - **Score:** 3×3×2×3 = **54**
#   - **Score:** 4×4×3×4 = **192**
SCORE_PATTERN = re.compile(r"\*?\*?Score:\*?\*?\s*\d+×\d+×\d+×\d+\s*=\s*\*?\*?(\d+)")
# Match arXiv ID near a score block
ARXIV_PATTERN = re.compile(r"(?:arxiv)\s*[:#]?\s*\**\s*(\d{4}\.\d{4,5})(v\d+)?", re.IGNORECASE)


def parse_entries(source: Path = SOURCE) -> list[dict]:
    """Walk research-studying.md and emit per-entry (arxiv_id, score, header) dicts."""
    if not source.exists():
        return []
    text = source.read_text()
    # Split on level-3 headers; the first chunk is the file preamble (skip)
    chunks = re.split(r"\n###\s+", text)[1:]
    entries: list[dict] = []
    for chunk in chunks:
        lines = chunk.split("\n", 1)
        header = lines[0].strip()
        body = lines[1] if len(lines) > 1 else ""

        score_m = SCORE_PATTERN.search(body)
        arxiv_m = ARXIV_PATTERN.search(header + " " + body)
        if not score_m or not arxiv_m:
            continue

        entries.append(
            {
                "arxiv_id": arxiv_m.group(1),
                "score": int(score_m.group(1)),
                "header": header[:120],
            }
        )
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sample low-scoring entries from research-studying.md "
        "for periodic re-evaluation against current Carnot priorities."
    )
    parser.add_argument("--sample", type=int, default=5,
                        help="Number of low-scoring entries to surface (default: 5).")
    parser.add_argument("--min-score", type=int, default=30,
                        help="Minimum score for rescore-eligible band (default: 30).")
    parser.add_argument("--max-score", type=int, default=150,
                        help="Maximum score for rescore-eligible band (default: 150).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible sampling (default: random).")
    args = parser.parse_args()

    entries = parse_entries()
    eligible = [e for e in entries if args.min_score <= e["score"] <= args.max_score]

    if not eligible:
        print(
            f"# no entries in score band [{args.min_score}, {args.max_score}]; "
            f"total parsed: {len(entries)}",
            file=sys.stderr,
        )
        return 0

    if args.seed is not None:
        random.seed(args.seed)
    sample_size = min(args.sample, len(eligible))
    sampled = random.sample(eligible, sample_size)
    sampled.sort(key=lambda e: e["score"])

    print(f"# rescore candidates (sample from band [{args.min_score}, {args.max_score}]; "
          f"{len(eligible)} eligible, {sample_size} sampled):")
    for e in sampled:
        print(f"arXiv:{e['arxiv_id']}  # score={e['score']:>4}  {e['header']}")
    print(
        f"# eligible_total={len(eligible)} sampled={sample_size}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
