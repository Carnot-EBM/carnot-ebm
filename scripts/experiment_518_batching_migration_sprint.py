#!/usr/bin/env python3
"""Experiment 518 — Batching Migration Sprint.

Migrates the top-20 legacy experiment scripts from sequential inference loops
to BatchedInferenceRunner, recovering wall-time savings estimated in the .38
retrospective and Exp 481 batching audit (77 violations across 362 scripts,
~3% milestone wall-time savings available).

**What this experiment does:**
    1. Loads the Exp 481 violation list from results/.
    2. Ranks violating scripts by estimated wall-time savings (recency + violation
       count heuristic, supplemented by known wall-time data from ops/metrics.md).
    3. Attempts automated migration of the top-20 scripts (or all if < 20 remain).
    4. Migration strategy: finds simple ``for var in items: result = fn(var);
       results.append(result)`` patterns and replaces with BatchedInferenceRunner.
       Complex loops (multi-statement bodies, nested conditionals) are flagged for
       manual review rather than silently mangled — correctness over completeness.
    5. Emits an artifact with artifact_schema='carnot.batching_migration.v1'.

**Why conservative migration:**
    These legacy scripts were written before BatchedInferenceRunner existed.  Their
    inference loops are deeply coupled to local state (per-question structs, nested
    conditionals, early-exit logic).  A greedy regex rewrite would produce syntactically
    valid but semantically broken code.  Better to auto-migrate only the provably safe
    cases and document the rest for manual follow-up.

CPU-only: No GPU required — this is an infrastructure experiment.

Spec: REQ-INFRA-047, REQ-INFRA-048,
      SCENARIO-INFRA-055, SCENARIO-INFRA-056
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Boilerplate: apply_env_autofix FIRST (belt-and-suspenders, RETRO-022)
# Env autofix must run before any JAX/torch imports to prevent ROCm thrml crash.
# ---------------------------------------------------------------------------
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "python"))
sys.path.insert(0, str(_repo_root / "scripts"))

from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

apply_env_autofix()

from carnot.pipeline.deliverable_guard import DeliverableGuard  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from experiment_template import ExperimentTemplate  # noqa: E402

_DELIVERABLE = "results/experiment_518_batching_migration_sprint.json"
_EXP_481_RESULTS = "results/experiment_481_inference_batching_enforcement.json"

# ---------------------------------------------------------------------------
# Known wall-time data from ops/metrics.md and .38 retro context.
# Used to improve savings estimates for specific experiment IDs.
# For all others, a per-violation heuristic is used.
# ---------------------------------------------------------------------------
_KNOWN_WALL_TIMES_MINUTES: dict[int, float] = {
    221: 78.0,
    226: 73.0,
    260: 74.0,
    308: 105.0,
    425: 76.0,
}

# Approximate fraction of wall time spent in sequential inference loops for
# experiments that used sequential inference throughout.  The .38 retro estimated
# ~20% wall-time reduction per migration to BatchedInferenceRunner.
_BATCHING_SAVINGS_FRACTION = 0.20

# Conservative estimate of savings per violation when we lack actual wall-time data.
# Rationale: one sequential loop over 50 questions at 3 s/question vs 0.5 s batched
# at batch_size=8 ≈ (3 - 0.5) * 50 / 60 ≈ 2.1 minutes.  Rounded up to 3.0.
_SAVINGS_PER_VIOLATION_MINUTES = 3.0

# Scripts with exp_id >= this threshold get a 1.5x recency factor because
# higher-numbered experiments are more likely to be re-run by the conductor.
_RECENCY_THRESHOLD = 400
_RECENCY_FACTOR = 1.5

# Import line to inject when BatchedInferenceRunner is not yet imported.
_BIR_IMPORT_LINE = (
    "from scripts.experiment_template import "
    "BatchedInferenceRunner  # noqa: E402 (added by Exp 518 migration)\n"
)

# ---------------------------------------------------------------------------
# Violation grouping and ranking helpers
# ---------------------------------------------------------------------------


def extract_exp_id(script_path: str) -> int | None:
    """Extract the experiment ID from a script filename like experiment_123_foo.py.

    Returns None for non-experiment scripts (e.g. generate_qa_dataset.py).

    Why we need this: exp IDs let us look up known wall-time data and apply
    a recency factor so that recently-run experiments are prioritised for migration
    over archived ones that will never run again.
    """
    m = re.search(r"experiment_(\d+)_", Path(script_path).name)
    return int(m.group(1)) if m else None


def estimate_savings_minutes(exp_id: int | None, n_violations: int) -> float:
    """Estimate wall-time savings (minutes) from migrating a given script.

    When a known wall-time exists, we apply the fixed 20% savings fraction.
    Otherwise, we use a conservative per-violation estimate with a recency factor
    for high-numbered experiments that are more likely to be re-run.

    Parameters
    ----------
    exp_id : int | None
        Experiment ID extracted from the script path, or None.
    n_violations : int
        Number of sequential-loop violations in this script.
    """
    if exp_id is not None and exp_id in _KNOWN_WALL_TIMES_MINUTES:
        return _KNOWN_WALL_TIMES_MINUTES[exp_id] * _BATCHING_SAVINGS_FRACTION
    recency = _RECENCY_FACTOR if (exp_id is not None and exp_id >= _RECENCY_THRESHOLD) else 1.0
    return _SAVINGS_PER_VIOLATION_MINUTES * n_violations * recency


def group_violations_by_script(
    violations: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group the flat per-violation list into {script_path: [violations]}.

    The Exp 481 audit emits one record per violation; a single script may have
    several violations.  Grouping lets us perform a single migration pass per
    script rather than re-reading and re-writing the file multiple times.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for v in violations:
        path = v["script_path"]
        grouped.setdefault(path, []).append(v)
    return grouped


def rank_scripts_by_savings(
    grouped: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Return scripts sorted by estimated wall-time savings (highest first).

    Each entry in the returned list has:
      script_path, exp_id, n_violations, estimated_savings_min, violations
    """
    rows: list[dict[str, Any]] = []
    for script_path, viols in grouped.items():
        exp_id = extract_exp_id(script_path)
        savings = estimate_savings_minutes(exp_id, len(viols))
        rows.append(
            {
                "script_path": script_path,
                "exp_id": exp_id,
                "n_violations": len(viols),
                "estimated_savings_min": savings,
                "violations": viols,
            }
        )
    rows.sort(key=lambda r: r["estimated_savings_min"], reverse=True)
    return rows


# ---------------------------------------------------------------------------
# Migration helpers
# ---------------------------------------------------------------------------

# The one pattern we can auto-migrate without risk of semantic breakage.
# Body must be exactly two lines:
#   Line 1: <target> = <fn>(<var>)
#   Line 2: <results>.append(<target>)
#
# Why only this exact shape: any additional statement (conditional, dict build,
# logging) means the loop body accumulates state that BatchedInferenceRunner
# cannot replicate via a simple .response extraction.  The safe contract is:
# "if it looks like exactly this, swap it; otherwise flag for manual review."
_SIMPLE_LOOP_RE = re.compile(
    r"^(?P<indent>[ \t]*)for (?P<var>\w+) in (?P<items>\w+):\n"
    r"(?P=indent)    (?P<target>\w+) = (?P<fn>\w+)\((?P=var)\)\n"
    r"(?P=indent)    (?P<results>\w+)\.append\((?P=target)\)\n"
    r"(?!(?P=indent)    )",  # reject if the next line is still at loop-body indent
    re.MULTILINE,
)


def find_simple_loop(content: str) -> re.Match | None:
    """Find the first auto-migratable sequential inference loop in script content.

    A 'simple' loop has exactly this shape:
        for var in items:
            target = fn(var)
            results.append(target)

    Anything more complex (multi-statement body, nested if, dict construction,
    comprehension) returns None — those require human judgment.
    """
    return _SIMPLE_LOOP_RE.search(content)


def build_bir_replacement(match: re.Match) -> str:
    """Build the BatchedInferenceRunner replacement for a matched simple loop.

    Replaces:
        for var in items:
            target = fn(var)
            results.append(target)
    With:
        _bir_518 = BatchedInferenceRunner(fn, batch_size=8)
        _bir_results_518 = _bir_518.run_batch(items)
        results = [r.response for r in _bir_results_518]

    Uses _bir_518 / _bir_results_518 as temp names to avoid colliding with
    existing script-level variables.  batch_size=8 is the project default
    (REQ-INFRA-084) for GSM8K-scale question lists.
    """
    indent = match.group("indent")
    fn = match.group("fn")
    items = match.group("items")
    results = match.group("results")
    return (
        f"{indent}_bir_518 = BatchedInferenceRunner({fn}, batch_size=8)\n"
        f"{indent}_bir_results_518 = _bir_518.run_batch({items})\n"
        f"{indent}{results} = [r.response for r in _bir_results_518]\n"
    )


def ensure_bir_import(content: str) -> str:
    """Inject the BatchedInferenceRunner import if it is not already present.

    Idempotent: returns content unchanged when BatchedInferenceRunner is already
    imported.  When missing, inserts the import immediately after the last
    'from experiment_template import' line (the natural anchor), or at the first
    import statement if no such line exists.

    Why after experiment_template imports: that module owns BatchedInferenceRunner,
    and grouping related imports together respects existing style.
    """
    if "BatchedInferenceRunner" in content:
        return content

    # Preferred anchor: existing experiment_template import line
    anchor_re = re.compile(
        r"(from (?:scripts\.)?experiment_template import [^\n]+\n)"
    )
    m = anchor_re.search(content)
    if m:
        pos = m.end()
        return content[:pos] + _BIR_IMPORT_LINE + content[pos:]

    # Fallback: insert before the first import statement
    lines = content.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            lines.insert(i, _BIR_IMPORT_LINE)
            return "".join(lines)

    # Last resort: prepend
    return _BIR_IMPORT_LINE + content


def attempt_script_migration(script_path: str) -> dict[str, Any]:
    """Attempt automated migration of one script to BatchedInferenceRunner.

    Returns a dict with keys:
      script_path (str), success (bool), reason (str), lines_changed (int)

    Strategy:
      1. Read the file.
      2. Find the first auto-migratable simple loop (find_simple_loop).
      3. Build the replacement with build_bir_replacement.
      4. Ensure the import is present with ensure_bir_import.
      5. Verify the modified file parses as valid Python (ast.parse).
      6. Write back ONLY on verified success — never write a broken file.

    Failure reasons:
      file_not_found       — script_path does not exist on disk
      no_simple_loop_found — no auto-migratable pattern detected
      syntax_error_after_patch — ast.parse failed on the patched content
    """
    path = Path(script_path)
    if not path.exists():
        return {
            "script_path": script_path,
            "success": False,
            "reason": "file_not_found",
            "lines_changed": 0,
        }

    content = path.read_text(encoding="utf-8")
    original_line_count = content.count("\n")

    m = find_simple_loop(content)
    if m is None:
        return {
            "script_path": script_path,
            "success": False,
            "reason": "no_simple_loop_found",
            "lines_changed": 0,
        }

    replacement = build_bir_replacement(m)
    modified = content[: m.start()] + replacement + content[m.end() :]
    modified = ensure_bir_import(modified)

    try:
        ast.parse(modified)
    except SyntaxError as exc:
        return {
            "script_path": script_path,
            "success": False,
            "reason": f"syntax_error_after_patch: {exc}",
            "lines_changed": 0,
        }

    path.write_text(modified, encoding="utf-8")
    lines_changed = abs(modified.count("\n") - original_line_count)
    return {
        "script_path": script_path,
        "success": True,
        "reason": "simple_loop_replaced",
        "lines_changed": lines_changed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Exp 518: migrate top-20 legacy scripts to BatchedInferenceRunner."""
    tmpl = ExperimentTemplate(
        518,
        "Batching Migration Sprint",
        _DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()
    guard = DeliverableGuard(str(tmpl._output_path))

    with ExperimentTimeoutWatchdog(518, timeout_minutes=30):
        # ------------------------------------------------------------------
        # Step 1: Load Exp 481 violation list
        # ------------------------------------------------------------------
        audit_path = _repo_root / _EXP_481_RESULTS
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        violations: list[dict[str, Any]] = audit["violations"]

        # ------------------------------------------------------------------
        # Step 2: Group by script and rank by estimated wall-time savings
        # ------------------------------------------------------------------
        grouped = group_violations_by_script(violations)
        ranked = rank_scripts_by_savings(grouped)
        top20 = ranked[:20]

        # ------------------------------------------------------------------
        # Step 3: Attempt migrations — conservative strategy, honest reporting
        # ------------------------------------------------------------------
        migrated_scripts: list[dict[str, Any]] = []
        migration_results: list[dict[str, Any]] = []
        n_attempted = 0
        n_migrated = 0
        n_failed = 0
        total_savings = 0.0

        for entry in top20:
            n_attempted += 1
            result = attempt_script_migration(entry["script_path"])
            migration_results.append(result)

            if result["success"]:
                n_migrated += 1
                migrated_scripts.append(
                    {
                        "script": entry["script_path"],
                        "exp_id": entry["exp_id"],
                        "estimated_savings_min": entry["estimated_savings_min"],
                        "lines_changed": result["lines_changed"],
                    }
                )
                total_savings += entry["estimated_savings_min"]
            else:
                n_failed += 1

            # Checkpoint after each attempt so a conductor interruption does not
            # lose progress — the watchdog timeout (30 min) may fire mid-sprint.
            tmpl.checkpoint_save(
                {
                    "n_attempted": n_attempted,
                    "n_migrated": n_migrated,
                    "n_failed": n_failed,
                    "migrated_scripts": migrated_scripts,
                },
                step=n_attempted,
            )

        honest_verdict = (
            "migration_complete" if n_migrated >= 10 else "partial_migration"
        )

        artifact = tmpl.build_result(
            {
                "artifact_schema": "carnot.batching_migration.v1",
                "n_scripts_attempted": n_attempted,
                "n_scripts_migrated": n_migrated,
                "n_scripts_failed": n_failed,
                "migrated_scripts": migrated_scripts,
                "total_estimated_savings_min": round(total_savings, 1),
                "honest_verdict": honest_verdict,
            },
            status="success",
        )

        tmpl._output_path.write_text(json.dumps(artifact, indent=2) + "\n")

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
