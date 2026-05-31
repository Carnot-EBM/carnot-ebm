#!/usr/bin/env python3
"""summarize_artifact.py — the misread-proof way to read a result artifact.

WHY THIS EXISTS
---------------
Three times in one outer-loop session the operator's agent misread a result
artifact and almost reported a wrong conclusion:

  1. Read a degenerate `flip_count=0` reranker as a real "P0.1 verdict"
     (caught only by running adversarial_verify and reading the artifact's
     own g2 gate).
  2. Counted a stale log line and reported phantom looping transition tasks.
  3. Flagged identifier/seed TAUTOLOGY false-positives as a "coverage gap."

The common root cause: eyeballing a 200-field JSON blob and latching onto the
first number that looks like a headline, without reading (a) what the artifact
itself says its acceptance gates concluded, and (b) what the adversarial
verifier flags about it. This tool forces both into view in a fixed order so
the reader cannot skip them.

READING-RESULTS DISCIPLINE (the contract this tool enforces)
------------------------------------------------------------
Before citing ANY number from an artifact as a conclusion, you MUST see:
  1. honest_verdict          — what the experiment claims happened
  2. flagged_adversarial     — was it quarantined? (stamped + LIVE re-check)
  3. every acceptance_gate_* — the experiment's OWN pass/fail self-report;
                               a False gate overrides a celebratory verdict
  4. duration_s + substrate  — plausibility floor for the claim
  5. headline metrics        — only AFTER 1-4, and annotated with any flag

A null/negative claim (e.g. "X does not beat Y") is NOT a finding unless a
positive control passed — this tool surfaces FALSE_NEGATIVE_RISK flags
prominently for exactly that reason.

USAGE
-----
    python3 scripts/summarize_artifact.py results/experiment_3507_*.json
    python3 scripts/summarize_artifact.py 3507          # by experiment id
    python3 scripts/summarize_artifact.py --recent 10   # last N by mtime
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path
from typing import Any

# Import the sibling adversarial_verify module robustly whether this file is
# run as `scripts/summarize_artifact.py` (script dir on sys.path) or imported
# as `scripts.summarize_artifact` (package context).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import adversarial_verify as av  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Field-name fragments that usually carry the headline number of an experiment.
_HEADLINE_HINTS = (
    "auroc", "accuracy", "acc", "solve_rate", "pass_rate", "tpr", "fpr",
    "f1", "precision", "recall", "delta", "lift", "improvement", "score",
    "energy", "kl", "flip", "n_samples", "n_problems", "n_completed",
)


def _resolve(arg: str) -> list[Path]:
    """Resolve a CLI arg to artifact paths: a glob, a path, or a bare exp id."""
    if Path(arg).exists():
        return [Path(arg)]
    hits = sorted(glob.glob(arg))
    if hits:
        return [Path(h) for h in hits]
    if arg.isdigit():
        hits = sorted(RESULTS_DIR.glob(f"experiment_{arg}_*.json")) + sorted(
            RESULTS_DIR.glob(f"experiment_{arg}.json")
        )
        return hits
    return []


def _live_flags(path: Path) -> list[dict[str, Any]]:
    try:
        rep = av.verify_artifact(path)
    except Exception as e:  # never let the verifier crash the reader
        return [{"kind": "VERIFY_ERROR", "severity": "warn", "detail": str(e)}]
    return rep.get("flags", [])


def _sev(f: dict[str, Any]) -> str:
    return str(f.get("severity", "")).lower()


def summarize(path: Path) -> int:
    """Print the disciplined summary. Return 2 if any LIVE critical flag, 1 if
    any warn (incl. FALSE_NEGATIVE_RISK), else 0 — usable as an exit signal."""
    try:
        d = json.loads(path.read_text())
    except Exception as e:
        print(f"!! could not load {path}: {e}")
        return 2
    if not isinstance(d, dict):
        print(f"-- {path.name}: non-dict top-level artifact (list?), skipping")
        return 0

    flags = _live_flags(path)
    crit = [f for f in flags if _sev(f) == "critical"]
    warn = [f for f in flags if _sev(f) == "warn"]
    fnr = [f for f in flags if f.get("kind") == "FALSE_NEGATIVE_RISK"]

    print("=" * 78)
    print(f"ARTIFACT  {path.name}")
    print("-" * 78)

    # 1. verdict
    print(f"  verdict          : {d.get('honest_verdict')}")
    # 2. quarantine status — stamped AND live
    stamped = d.get("flagged_adversarial")
    live = "CRITICAL" if crit else ("warn" if warn else "clean")
    print(f"  flagged_adversarial (stamped): {stamped}   |   LIVE re-check: {live}")
    if stamped and not crit:
        print("    note: stamped-flagged but live re-check is clean "
              "(rule may have been fixed since; verify the corrigendum).")
    if crit and not stamped:
        print("    *** GAP: live verifier flags CRITICAL but artifact is NOT "
              "stamped flagged_adversarial. Do not cite as clean. ***")

    # 3. acceptance gates — the experiment's own self-report
    gate_fields = {
        k: v for k, v in d.items()
        if "acceptance_gate" in k.lower() or k.lower().startswith("gate_")
        or k.lower().endswith("_gate")
    }
    if gate_fields:
        print("  acceptance gates :")
        for k, v in sorted(gate_fields.items()):
            mark = "PASS" if v is True else ("FAIL" if v is False else "?")
            print(f"      [{mark:4}] {k} = {v!r}")
        if any(v is False for v in gate_fields.values()):
            print("    >> a FAILED gate overrides any celebratory verdict above.")
    else:
        print("  acceptance gates : (none found — claim has no self-reported gate)")

    # 4. plausibility floor
    print(f"  duration_s       : {d.get('duration_s')}   "
          f"substrate: {d.get('inference_substrate')}")

    # 5. headline metrics (only now), annotated with any flag touching them
    flag_fields = {
        seg.split("=")[0].strip()
        for f in flags
        for seg in str(f.get("detail", "")).replace(" and ", " ").split()
        if "=" in seg
    }
    heads = {
        k: v for k, v in d.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
        and any(h in k.lower() for h in _HEADLINE_HINTS)
    }
    if heads:
        print("  headline metrics :")
        for k, v in sorted(heads.items()):
            tag = "  <-- flagged" if k in flag_fields else ""
            print(f"      {k} = {v}{tag}")

    # FALSE-NEGATIVE prominence — a null claim is not a finding without control
    if fnr:
        print("  !! FALSE_NEGATIVE_RISK — this NULL/negative claim may be a")
        print("     degenerate test, not real evidence the method fails:")
        for f in fnr:
            print(f"       - {f.get('detail')}")

    # all flags, last, in severity order
    if flags:
        print("  adversarial flags:")
        _rank = {"info": 0, "warn": 1, "critical": 2}
        for f in sorted(flags, key=lambda x: -_rank.get(_sev(x), 0)):
            print(f"      [{_sev(f):8}] {f.get('kind')}: {str(f.get('detail'))[:96]}")
    else:
        print("  adversarial flags: none")

    return 2 if crit else (1 if warn else 0)


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    paths: list[Path] = []
    if argv[0] == "--recent":
        n = int(argv[1]) if len(argv) > 1 else 10
        paths = sorted(
            RESULTS_DIR.glob("experiment_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:n]
    else:
        for a in argv:
            r = _resolve(a)
            if not r:
                print(f"!! no artifact matched: {a}")
            paths.extend(r)
    if not paths:
        return 1
    worst = 0
    for p in paths:
        worst = max(worst, summarize(p))
    print("=" * 78)
    return worst


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
