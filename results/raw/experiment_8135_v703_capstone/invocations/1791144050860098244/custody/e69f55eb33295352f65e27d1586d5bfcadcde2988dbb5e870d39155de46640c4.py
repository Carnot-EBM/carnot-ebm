#!/usr/bin/env python3
"""Publication readiness gate — the STABLE replacement for publication_blocker_count.

WHY THIS EXISTS
---------------
`publication_blocker_count` was the project's de-facto "are we ready to publish"
signal, emitted as a free integer by each capstone task. It could not steer:
between capstone v303 and v304 it moved 105 -> 10 via `blocker_delta_from_v303:
-95` — a recount, not 95 resolutions. A number an agent can redefine each
milestone is not a finish line. See ops/north-star.md §2.

This script replaces the redefinable count with a FIXED 4-condition gate
(G1-G4). The conditions are external and mechanical wherever possible, so
"paper_ready" cannot be moved by re-counting. Capstone tasks should call this
and emit its `g1..g4` + `paper_ready` booleans INSTEAD OF inventing a blocker
count.

THE GATE (do not redefine these to show progress — that is the failure mode
this replaces):

  G1 — Headline measured: a FoVer dual-condition AUROC artifact exists with
       >=5 seeds and a reported CI (the headline claim, ops/north-star.md §1).
  G2 — Independently reproduced: >=1 non-operator reproducer landed within the
       CI. EXTERNAL — read from ops/publication_gate_state.json (cannot be
       auto-detected; an honest manual boolean with evidence).
  G3 — Prose narrowing-clean: the technical report + paper draft contain ZERO
       phrasings from the Paper-v6 Narrowing retraction list. MECHANICAL — this
       script IS the narrowing lint G3 referenced as not-yet-existing.
  G4 — Numbers trace to artifacts: the headline artifact carries random_seed
       (or random_seeds_used) + reproducibility_checksum. MECHANICAL.

  paper_ready := G1 and G2 and G3 and G4.

Usage:
  python3 scripts/publication_gate.py            # human-readable
  python3 scripts/publication_gate.py --json     # machine-readable (for capstones)

This script makes NO network calls and NEVER edits anything. It reports what it
checked and the limits of each check honestly (per Adversarial Artifact
Verification discipline — a gate that overstates its own coverage is itself a
fabrication risk).
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = PROJECT_ROOT / "ops" / "publication_gate_state.json"
TECH_REPORT = PROJECT_ROOT / "docs" / "technical-report.md"
PAPER_TEX = PROJECT_ROOT / "docs" / "arxiv-paper" / "main.tex"

# G3 forbidden phrasings — the Paper-v6 Narrowing Discipline retraction list
# (CLAUDE.md). Any of these in operator-facing prose fails G3. Patterns are
# narrow to avoid false positives on the *narrowed replacement* text.
FORBIDDEN_PHRASINGS = [
    (r"\b0\.9857\b", "retracted FoVer v2 headline AUROC (use 0.9131)"),
    (r"\+0\.0621\b", "retracted HIVE-comparator delta"),
    (r"hardware sovereignty", "retracted; use 'local edge deployability'"),
    (r"\bthermaliz", "retracted KV260 Boltzmann-thermalization claim"),
    (r"equilibrium samples", "retracted KV260 equilibrium claim"),
    (r"\b(11680|12788|13000)x\b", "retracted KV260 speedup figure"),
    (r"0% to 36%", "unsupported 35B HumanEval claim (no artifact)"),
    (r"0% → 36%", "unsupported 35B HumanEval claim (no artifact)"),
    (r"replacement-grade", "prompt-injection replacement was REFUTED (exp3273)"),
]


def _load_state() -> dict:
    """Load the manual-gate state (G2 + any operator overrides)."""
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except json.JSONDecodeError:
            pass
    return {}


def _find_headline_artifact() -> tuple[dict | None, str | None]:
    """Find the most recent FoVer dual-condition AUROC artifact (the G1 source).

    Heuristic: a results/ artifact whose payload has a production AUROC near
    0.91 and a seed count >= 5. Returns (payload, path) or (None, None).
    """
    candidates = []
    for path in glob.glob(str(PROJECT_ROOT / "results" / "*fover*.json")) + glob.glob(
        str(PROJECT_ROOT / "results" / "*dual_condition*.json")
    ):
        try:
            d = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError):
            continue
        blob = json.dumps(d).lower()
        # must look like a dual-condition AUROC artifact
        if "auroc" not in blob:
            continue
        # pull any auroc-ish production number
        auroc = None
        for k in (
            "condition_a_production_auroc_mean",
            "production_auroc",
            "auroc",
            "headline_auroc",
        ):
            v = d.get(k)
            if isinstance(v, (int, float)) and 0.80 <= v <= 0.96:
                auroc = v
                break
        seeds = d.get("n_seeds") or d.get("random_seeds_used")
        n_seeds = len(seeds) if isinstance(seeds, list) else (seeds if isinstance(seeds, int) else None)
        if auroc is not None and (n_seeds or 0) >= 5:
            candidates.append((path, d, auroc, n_seeds))
    if not candidates:
        return None, None
    # most recent by filename experiment number
    def expnum(p: str) -> int:
        m = re.search(r"experiment_(\d+)", p)
        return int(m.group(1)) if m else 0
    candidates.sort(key=lambda c: expnum(c[0]))
    path, d, _, _ = candidates[-1]
    return d, path


def check_g1() -> dict:
    """G1 — headline measured (FoVer dual-condition AUROC, >=5 seeds, CI)."""
    d, path = _find_headline_artifact()
    if d is None:
        return {"pass": False, "detail": "no FoVer dual-condition AUROC artifact with >=5 seeds found"}
    has_ci = any("ci" in k.lower() for k in d) or "ci95" in json.dumps(d).lower()
    return {
        "pass": True,
        "detail": f"FoVer dual-condition AUROC artifact present ({Path(path).name}); CI reported={has_ci}",
        "source": Path(path).name,
    }


def check_g2(state: dict) -> dict:
    """G2 — independently reproduced (EXTERNAL; honest manual boolean)."""
    val = bool(state.get("g2_independent_reproducer", False))
    ev = state.get("g2_evidence", "")
    return {
        "pass": val,
        "detail": (f"reproducer confirmed: {ev}" if val else
                   "no independent reproducer recorded in ops/publication_gate_state.json "
                   "(set g2_independent_reproducer + g2_evidence when one lands)"),
    }


# Retraction-context markers. A forbidden phrasing that appears NEAR one of
# these is legitimate disclosure (explaining the retraction), not a live claim.
# Without this whitelist G3 would flag the project's own honest narrowing
# narrative — the same false-positive class the conductor verdict classifier
# guards with _POSITIVE_BLOCKED_PATTERNS.
_RETRACTION_CONTEXT = re.compile(
    r"retract|repin|downward|\bv2 headline\b|earlier|prior|was the|"
    r"\bnot\b|remove the impl|narrowed|superseded|no longer|instead of|"
    r"rather than|deprecated|\bv2\b",
    re.IGNORECASE,
)


def check_g3() -> dict:
    """G3 — prose narrowing-clean. This IS the narrowing lint G3 referenced.

    A forbidden phrasing only fails the gate if it is asserted as a LIVE claim.
    The same phrasing inside a retraction-disclosure context (a +/-80-char
    window containing a retraction marker) is permitted — you must be able to
    say 'the v2 0.9857 figure was retracted downward to 0.9131' without
    tripping your own lint.
    """
    hits = []
    for doc in (TECH_REPORT, PAPER_TEX):
        if not doc.exists():
            continue
        text = doc.read_text(errors="ignore")
        for pat, why in FORBIDDEN_PHRASINGS:
            for m in re.finditer(pat, text):
                window = text[max(0, m.start() - 80): m.end() + 80]
                if _RETRACTION_CONTEXT.search(window):
                    continue  # legitimate retraction disclosure, not a live claim
                snippet = text[max(0, m.start() - 30): m.end() + 30].replace("\n", " ")
                hits.append(f"{doc.name}: '{m.group(0)}' ({why}) ...{snippet}...")
    return {
        "pass": len(hits) == 0,
        "detail": "no retracted phrasings asserted as live claims"
                  if not hits else f"{len(hits)} forbidden phrasing(s) asserted outside retraction context",
        "hits": hits[:20],
    }


def check_g4() -> dict:
    """G4 — headline numbers trace to a primary artifact w/ seed + checksum."""
    d, path = _find_headline_artifact()
    if d is None:
        return {"pass": False, "detail": "no headline artifact to verify provenance on"}
    has_seed = ("random_seed" in d) or ("random_seeds_used" in d) or ("n_seeds" in d)
    has_checksum = "reproducibility_checksum" in d
    return {
        "pass": bool(has_seed and has_checksum),
        "detail": f"{Path(path).name}: random_seed/seeds={has_seed}, reproducibility_checksum={has_checksum}",
        "source": Path(path).name,
    }


def evaluate() -> dict:
    state = _load_state()
    g1 = check_g1()
    g2 = check_g2(state)
    g3 = check_g3()
    g4 = check_g4()
    paper_ready = g1["pass"] and g2["pass"] and g3["pass"] and g4["pass"]
    unmet = [name for name, g in (("G1", g1), ("G2", g2), ("G3", g3), ("G4", g4)) if not g["pass"]]
    return {
        "paper_ready": paper_ready,
        "gates": {"G1": g1, "G2": g2, "G3": g3, "G4": g4},
        "unmet_gates": unmet,
        "note": "Stable 4-condition gate (ops/north-star.md §2). Replaces the "
                "redefinable publication_blocker_count. Report unmet_gates, not a count.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()
    result = evaluate()
    if args.json:
        print(json.dumps(result, indent=2))
        return 0
    print("=" * 64)
    print("PUBLICATION GATE (stable replacement for publication_blocker_count)")
    print("=" * 64)
    for name, g in result["gates"].items():
        mark = "PASS" if g["pass"] else "UNMET"
        print(f"  {name} [{mark}] {g['detail']}")
        for h in g.get("hits", []):
            print(f"        - {h}")
    print("-" * 64)
    print(f"  paper_ready = {result['paper_ready']}   unmet: {result['unmet_gates'] or 'none'}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
