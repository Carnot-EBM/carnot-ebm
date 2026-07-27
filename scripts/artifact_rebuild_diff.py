#!/usr/bin/env python3
"""Diff two versions of the same artifact and classify EVERY leaf that moved.

WHY THIS EXISTS (2026-07-27)
============================
The freshness lint ends with "Then DIFF the rebuild against the committed version and report
exactly which numbers moved". That instruction was previously followed by hand, and the resulting
claim was overstated: a lane reported "only `git_head` moved" when in fact `duration_s`,
`analysis_duration_s`, `run_date`, the provenance `sha256`/`bytes`/`mtime_utc` fields and the
derived `reproducibility_checksum` had all moved too. The SUBSTANCE was right -- no measurement
number changed -- but the phrasing was not, and a reader checking it would find it false.

So the distinction is mechanised. Every leaf is put in exactly one bucket:

  EXPECTED_ON_ANY_REBUILD  clocks (`duration_s`, `*_duration_s`, `run_date`, `build_timestamp_utc`),
                           provenance fingerprints (`sha256`, `bytes`, `mtime_utc`, `git_head`) and
                           anything derived from them (`reproducibility_checksum`).
  MEASUREMENT_BEARING      everything else that changed value. A non-empty bucket here is a
                           CORRECTION OWED, not a formality.
  ADDED / REMOVED          keys present in only one side. REMOVED is a never-prune violation.

Usage:
  artifact_rebuild_diff.py OLD.json NEW.json [--json]
Exit code: 0 if MEASUREMENT_BEARING and REMOVED are both empty, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Leaf key names whose movement on a rebuild is expected and carries no measurement meaning.
CLOCK_KEYS = {
    "duration_s",
    "analysis_duration_s",
    "analyser_duration_s",
    "run_date",
    "build_timestamp_utc",
    "generated_at",
    "timestamp",
}
PROVENANCE_KEYS = {"sha256", "bytes", "mtime_utc", "git_head", "reproducibility_checksum"}
# Wall clocks of the MEASUREMENT itself. On an ANALYSER rebuild over fixed persisted rows these must
# NOT move (the rows carry them), so they stay measurement-bearing by default. On a LIVE
# RE-MEASUREMENT they necessarily move, and `--live-remeasure` moves them into the expected bucket --
# explicitly, so a reader knows which comparison was made rather than being handed a silently
# widened definition of "clean".
LIVE_CLOCK_KEYS = {"wall_s", "measurement_wall_s", "sum_per_cell_wall_s", "elapsed_s"}


def _leaves(node: Any, path: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(node, dict):
        for k, v in node.items():
            out.update(_leaves(v, f"{path}.{k}" if path else str(k)))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            out.update(_leaves(v, f"{path}[{i}]"))
    else:
        out[path] = node
    return out


def _is_expected(path: str, live_remeasure: bool = False) -> bool:
    last = path.split(".")[-1].split("[")[0]
    if last in CLOCK_KEYS or last in PROVENANCE_KEYS:
        return True
    if live_remeasure and last in LIVE_CLOCK_KEYS:
        return True
    # nested provenance blocks: anything under `provenance` that is a fingerprint-ish leaf
    if ".provenance." in f".{path}." and last in PROVENANCE_KEYS:
        return True
    return False


def diff(old: dict, new: dict, live_remeasure: bool = False) -> dict:
    lo, ln = _leaves(old), _leaves(new)
    added = sorted(set(ln) - set(lo))
    removed = sorted(set(lo) - set(ln))
    changed = [k for k in sorted(set(lo) & set(ln)) if lo[k] != ln[k]]
    expected = [k for k in changed if _is_expected(k, live_remeasure)]
    bearing = [k for k in changed if not _is_expected(k, live_remeasure)]
    return {
        "mode": "live_remeasure" if live_remeasure else "analyser_rebuild",
        "n_leaves_old": len(lo),
        "n_leaves_new": len(ln),
        "EXPECTED_ON_ANY_REBUILD": expected,
        "n_expected": len(expected),
        "MEASUREMENT_BEARING": [{"path": k, "old": lo[k], "new": ln[k]} for k in bearing[:200]],
        "n_measurement_bearing": len(bearing),
        "ADDED": added[:400],
        "n_added": len(added),
        "REMOVED": removed[:400],
        "n_removed": len(removed),
        "clean": not bearing and not removed,
        "honest_phrasing": (
            f"{len(bearing)} measurement-bearing diffs; {len(expected)} expected-on-rebuild leaves "
            f"moved (clocks, provenance fingerprints, derived checksum); {len(added)} keys added, "
            f"{len(removed)} removed"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--json", action="store_true")
    ap.add_argument(
        "--live-remeasure",
        action="store_true",
        help=(
            "the 'rebuild' re-RAN a live measurement, so per-cell and total wall clocks are expected "
            "to move. Never use this for an analyser rebuild over fixed rows."
        ),
    )
    a = ap.parse_args(argv)
    old = json.loads(Path(a.old).read_text())
    new = json.loads(Path(a.new).read_text())
    d = diff(old, new, live_remeasure=a.live_remeasure)
    if a.json:
        print(json.dumps(d, indent=1, default=str))
    else:
        print(f"{Path(a.new).name}: {d['honest_phrasing']}")
        if d["MEASUREMENT_BEARING"]:
            print("  MEASUREMENT-BEARING (a correction is owed):")
            for e in d["MEASUREMENT_BEARING"][:40]:
                print(f"    {e['path']}: {e['old']!r} -> {e['new']!r}")
        if d["REMOVED"]:
            print("  REMOVED (never-prune violation):")
            for k in d["REMOVED"][:40]:
                print(f"    {k}")
    return 0 if d["clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
