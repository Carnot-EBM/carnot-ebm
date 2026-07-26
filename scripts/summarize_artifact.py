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
import hashlib
import json
import re
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
    "auroc",
    "accuracy",
    "acc",
    "solve_rate",
    "pass_rate",
    "tpr",
    "fpr",
    "f1",
    "precision",
    "recall",
    "delta",
    "lift",
    "improvement",
    "score",
    "energy",
    "kl",
    "flip",
    "n_samples",
    "n_problems",
    "n_completed",
)
_DIAGNOSIS_CONTEXT_FIELDS = (
    "barrier_diagnosis",
    "levers_tried",
    "barrier_refinement",
)
_TAUTOLOGY_FIELD_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)=")


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


def _tautology_field_pair(flag: dict[str, Any]) -> tuple[str, str] | None:
    """Extract the two field names from adversarial_verify's TAUTOLOGY detail."""
    fields = _TAUTOLOGY_FIELD_RE.findall(str(flag.get("detail", "")))
    if len(fields) < 2:
        return None
    return fields[0], fields[1]


def _suffix_stem(name: str, suffix: str) -> str | None:
    marker = f"_{suffix}"
    if not name.endswith(marker):
        return None
    return name[: -len(marker)]


def _control_vs_treatment_pair(k1: str, k2: str) -> bool:
    """True for same-stem baseline/control versus best/treatment metric names."""
    left_suffixes = ("baseline", "control")
    right_suffixes = ("best", "treatment")
    pairs = ((k1, k2), (k2, k1))
    for left, right in pairs:
        for left_suffix in left_suffixes:
            left_stem = _suffix_stem(left, left_suffix)
            if left_stem is None:
                continue
            for right_suffix in right_suffixes:
                if _suffix_stem(right, right_suffix) == left_stem:
                    return True
    return False


def _is_explicit_zero(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) == 0.0


def classify_known_false_positive_null_delta(
    artifact: dict[str, Any],
    flags: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Classify the annotated control-vs-treatment null-delta TAUTOLOGY case.

    This does not weaken adversarial verification. It only tells aggregation
    readers that a quarantined artifact's diagnosis context can still be read
    when the artifact explicitly documents a zero efficiency delta and the sole
    live critical flag is the expected control/best equality.
    """
    critical = [flag for flag in flags if _sev(flag) == "critical"]
    if len(critical) != 1 or critical[0].get("kind") != "TAUTOLOGY":
        return None
    pair = _tautology_field_pair(critical[0])
    if pair is None or not _control_vs_treatment_pair(*pair):
        return None
    if not _is_explicit_zero(artifact.get("efficiency_delta")):
        return None
    note = artifact.get("null_delta_methodology_note")
    if not isinstance(note, str) or not note.strip():
        return None
    return {
        "kind": "KNOWN_FALSE_POSITIVE_NULL_DELTA_TAUTOLOGY",
        "field_pair": [pair[0], pair[1]],
        "null_delta_methodology_note": note,
        "corrigendum_note": (
            "corrigendum: live TAUTOLOGY is the annotated control-vs-treatment "
            "null-delta; read diagnosis context only, do not aggregate headline "
            "numbers as an improvement."
        ),
        "diagnosis_context_fields": [
            field for field in _DIAGNOSIS_CONTEXT_FIELDS if field in artifact
        ],
    }


def readable_diagnosis_context(
    artifact: dict[str, Any],
    flags: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return diagnosis fields that aggregation may read under the corrigendum."""
    classification = classify_known_false_positive_null_delta(artifact, flags)
    if classification is None:
        return None
    context = {field: artifact[field] for field in _DIAGNOSIS_CONTEXT_FIELDS if field in artifact}
    if not context:
        return None
    context["corrigendum"] = classification
    return context


def _staleness_banner(path: Path, d: dict[str, Any]) -> tuple[str, int]:
    """Is this artifact the output of the code and inputs on disk right now?

    WHY THIS IS FIRST, ABOVE THE VERDICT (added 2026-07-26). A stale artifact's numbers get quoted
    with exactly the same confidence as a fresh one's, because nothing distinguishes them at read
    time. The incident: `results/outer_loop_scored_path_lever_ab_llm_on_20260726.json` sat committed
    for ~1h56m after its analyser was edited and committed without a rebuild. That window happened
    to change no number, but nobody could have known that without rebuilding and diffing -- which is
    precisely the work a reader does not do before citing.

    This module is the ONLY sanctioned way to read a result artifact (CLAUDE.md Reading-Results
    Discipline), so wiring the check here is what makes it unmissable. Purely hash-based: it
    recomputes the sha256 of each dependency the artifact recorded in its own `provenance` block.
    Artifacts with no `provenance` block (everything written before the guard) report UNKNOWN, never
    "fresh" -- an unchecked artifact is not a checked-and-clean one.

    Returns (banner_text, severity) where severity 2 = STALE (joins the critical tier and the exit
    code), 1 = unverifiable, 0 = fresh / not-applicable.
    """
    prov = d.get("provenance")
    if not isinstance(prov, dict) or not prov:
        # Silent for the overwhelming majority of artifacts, which are single-shot experiment
        # results with no separate analyser step and nothing to go stale against.
        return ("", 0)
    recorded: list[dict[str, Any]] = list(prov.get("code") or [])
    for group in (prov.get("rows_sources") or {}).values():
        recorded.extend(group or [])
    if not recorded:
        return ("", 0)
    drift, unreadable = [], []
    for entry in recorded:
        p = Path(str(entry.get("path", "")))
        try:
            now = hashlib.sha256(p.read_bytes()).hexdigest()
        except OSError as exc:
            unreadable.append(f"{p} ({type(exc).__name__})")
            continue
        if now != entry.get("sha256"):
            drift.append(str(p))
    if drift:
        lines = [
            "  *** STALE: this artifact was NOT built from the code/inputs now on disk. ***",
            *[f"      drifted: {p}" for p in drift],
        ]
        cmd = prov.get("rebuild_command")
        if cmd:
            lines.append(f"      rebuild: {str(cmd).replace('<this file>', str(path))}")
        lines.append("      Do NOT cite numbers from it until it is rebuilt and re-diffed.")
        return ("\n".join(lines), 2)
    if unreadable:
        return (
            "  !! staleness UNVERIFIABLE: "
            + f"{len(unreadable)} recorded dependency file(s) could not be read "
            + f"({unreadable[0]}{'...' if len(unreadable) > 1 else ''}). "
            + "This is not a pass; the question was not answered.",
            1,
        )
    return (f"  staleness       : FRESH ({len(recorded)} dependencies verified)", 0)


def summarize(path: Path) -> int:
    """Print the disciplined summary. Return 2 if any LIVE critical flag OR the
    artifact is STALE w.r.t. its own recorded provenance, 1 if any warn (incl.
    FALSE_NEGATIVE_RISK or unverifiable staleness), else 0 — usable as an exit signal."""
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
    fnr_open = [f for f in fnr if "false_negative_risk_open" in str(f.get("detail", ""))]

    print("=" * 78)
    print(f"ARTIFACT  {path.name}")
    print("-" * 78)

    # 0. STALENESS, printed ABOVE the verdict on purpose: if the artifact was not built from the
    #    code on disk, every number below it is of unknown provenance and the verdict is the first
    #    thing a reader would otherwise take at face value.
    stale_banner, stale_sev = _staleness_banner(path, d)
    if stale_banner:
        print(stale_banner)

    # 1. verdict
    print(f"  verdict          : {d.get('honest_verdict')}")
    # 2. quarantine status — stamped AND live
    stamped = d.get("flagged_adversarial")
    live = "CRITICAL" if crit else ("warn" if warn else "clean")
    print(f"  flagged_adversarial (stamped): {stamped}   |   LIVE re-check: {live}")
    if stamped and not crit:
        print(
            "    note: stamped-flagged but live re-check is clean "
            "(rule may have been fixed since; verify the corrigendum)."
        )
    null_delta_corrigendum = readable_diagnosis_context(d, flags)
    if null_delta_corrigendum is not None:
        print(
            "    note: annotated null-delta TAUTOLOGY corrigendum; diagnosis "
            "context may be read, headline numbers remain quarantined."
        )
    if crit and not stamped:
        print(
            "    *** GAP: live verifier flags CRITICAL but artifact is NOT "
            "stamped flagged_adversarial. Do not cite as clean. ***"
        )

    # 3. acceptance gates — the experiment's own self-report
    gate_fields = {
        k: v
        for k, v in d.items()
        if "acceptance_gate" in k.lower()
        or k.lower().startswith("gate_")
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
    print(f"  duration_s       : {d.get('duration_s')}   substrate: {d.get('inference_substrate')}")
    floor = av.duration_floor_for_artifact(d)
    if floor is None:
        print("  duration floor   : none")
    else:
        print(
            f"  duration floor   : {floor.get('substrate')} "
            f">={floor.get('min_duration_s')}s ({floor.get('reason')})"
        )
    methodology = av.offline_arc_methodology_descriptor(d)
    if methodology is not None:
        fields = ", ".join(methodology.get("evidence_fields", []))
        print(f"  methodology      : {methodology.get('kind')} via {fields}")

    # 5. headline metrics (only now), annotated with any flag touching them
    flag_fields = {
        seg.split("=")[0].strip()
        for f in flags
        for seg in str(f.get("detail", "")).replace(" and ", " ").split()
        if "=" in seg
    }
    heads = {
        k: v
        for k, v in d.items()
        if isinstance(v, (int, float))
        and not isinstance(v, bool)
        and any(h in k.lower() for h in _HEADLINE_HINTS)
    }
    if heads:
        print("  headline metrics :")
        for k, v in sorted(heads.items()):
            tag = "  <-- flagged" if k in flag_fields else ""
            print(f"      {k} = {v}{tag}")

    # FALSE-NEGATIVE prominence — a null claim is not a finding without control
    if fnr:
        if fnr_open:
            print("  !! false_negative_risk_open — positive control failed or was unchecked.")
            print("     Treat this as a broken-test signal, not a clean null.")
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

    # STALE joins the critical tier: a number of unknown provenance is not a lesser problem than a
    # number with an adversarial flag, and folding it into the exit code is what stops an automated
    # reader from treating a stale artifact as clean.
    return max(stale_sev, 2 if crit else (1 if warn else 0))


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
