#!/usr/bin/env python3
"""Experiment 793 — Manifest Full-Scope Audit.

PURPOSE
-------
Exp 527 appeared in the slowest-5 for 7+ consecutive milestones after being added to the
conductor exclusion manifest.  The retro (RETRO-MANIFEST-FULL-SCOPE) identified the root
cause: the ExclusionManifest check is placed inside pick_next_task() but NOT within 5
source lines of the for-loop that iterates RESEARCH_TASKS.  This creates a fragile guard
that future refactors can accidentally bypass.

This script reads scripts/research_conductor.py as plain text, locates every "dequeue
site" (any line where a task is selected for execution), and records whether a manifest
check appears within 5 lines of that site.  It does NOT modify research_conductor.py —
it only writes a structured patch spec so a human can apply the fix.

DELIVERABLE
-----------
results/experiment_793_manifest_full_scope_audit.json

REFERENCES
----------
REQ-INFRA-058 — manifest check MUST be at ALL dequeue sites (Exp 793)
REQ-INFRA-059 — excluded tasks MUST be logged at WARNING level before skip (Exp 793)
SCENARIO-INFRA-067 — Exp 527 excluded at dequeue, WARNING emitted, task skipped
SCENARIO-INFRA-068 — Exp 793 not in manifest, runs normally
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Resolve project root so this script works whether invoked from any CWD.
PROJECT_ROOT = Path(__file__).parent.parent

sys.path.insert(0, str(PROJECT_ROOT / "python"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# apply_env_autofix MUST be called before any JAX or CUDA import.  Even though
# this experiment never touches JAX, the template may import libraries that probe
# for CUDA at import time.  Calling it first ensures CARNOT_FORCE_LIVE is set
# before any downstream code has a chance to pick the wrong compute backend.
from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

apply_env_autofix()

from experiment_template import ExperimentTemplate  # noqa: E402

EXPERIMENT_ID = 793
EXPERIMENT_TITLE = "Manifest Full-Scope Audit — Identify All Unguarded Dequeue Sites"
DELIVERABLE = "results/experiment_793_manifest_full_scope_audit.json"

# These are the patterns that indicate a dequeue event — a point in the code
# where a task_id is being selected for execution from a queue or list.
# The patterns are applied as regular expressions against each source line.
DEQUEUE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"task_queue"),
    re.compile(r"\.pop\(\)"),
    re.compile(r"\.popleft\(\)"),
    re.compile(r"next\(iter\("),
    re.compile(r"random\.choice"),
    re.compile(r"queue\.get\(\)"),
    # Matches "for task in <anything>" — the primary execution iteration pattern
    re.compile(r"for\s+task\s+in\s+"),
]

# The recommended patch code to inject immediately after a dequeue pattern line
# when no manifest guard is present.  This is what a human applying REQ-INFRA-058
# should insert at each unguarded site.
_RECOMMENDED_PATCH = (
    "        # REQ-INFRA-058 (Exp 793): manifest guard must be adjacent to dequeue.\n"
    "        # REQ-INFRA-059: emit WARNING so exclusion events are auditable.\n"
    "        _excluded, _exclusion_reason = _task_is_excluded(task)\n"
    "        if _excluded:\n"
    '            logger.warning(\n'
    '                "EXCLUDED task \'%s\' by manifest (%s) — skipping without dispatch",\n'
    '                task.get("title", "?")[:60], _exclusion_reason,\n'
    '            )\n'
    "            continue"
)

_NO_PATCH_NEEDED = "# Already guarded — manifest check found within 5 lines. No patch required."


def scan_conductor_text(conductor_text: str) -> list[dict]:
    """Scan the conductor source for dequeue patterns and classify each match.

    For each matching line the function records:
    - line_number: 1-based line number in the file
    - code_snippet: the matched line, stripped of leading whitespace
    - pattern_matched: the regex pattern that triggered the match
    - is_manifest_checked: True if "manifest" appears in any of the 5 lines
      before or after the matched line (the 11-line window centred on the match)
    - patch_required: True when is_manifest_checked is False
    - recommended_patch_code: the patch string to insert after the matched line

    The 5-line proximity rule is intentionally tight.  Placing the manifest check
    10+ lines into a loop body is functionally equivalent to a missing guard from a
    code-review standpoint — a future refactor inserting code between the dequeue
    and the check can accidentally reorder them.
    """
    lines = conductor_text.splitlines()
    results: list[dict] = []
    seen_lines: set[int] = set()  # Avoid double-counting a line that matches >1 pattern

    for line_idx, line in enumerate(lines):
        if line_idx in seen_lines:
            continue
        stripped = line.strip()
        if stripped.startswith("#"):
            continue  # Skip pure comment lines — they can't be dequeue sites

        for pattern in DEQUEUE_PATTERNS:
            if not pattern.search(line):
                continue

            line_number = line_idx + 1
            seen_lines.add(line_idx)

            # Build the context window: 2 lines before + match + 5 lines after.
            # We show more lines after because the guard typically follows the dequeue.
            ctx_start = max(0, line_idx - 2)
            ctx_end = min(len(lines), line_idx + 6)
            context_lines = lines[ctx_start:ctx_end]

            # Manifest proximity check: scan the 5 lines immediately before and
            # the 5 lines immediately after.  "manifest" must appear somewhere in
            # that window for the site to count as guarded.  We use a case-
            # insensitive check because the variable name (_EXCLUSION_MANIFEST),
            # the function call (_task_is_excluded), and comments all use different
            # capitalizations.
            window_start = max(0, line_idx - 5)
            window_end = min(len(lines), line_idx + 6)
            window_text = "\n".join(lines[window_start:window_end])
            is_manifest_checked = "manifest" in window_text.lower()

            results.append(
                {
                    "line_number": line_number,
                    "code_snippet": stripped,
                    "pattern_matched": pattern.pattern,
                    "context_lines": context_lines,
                    "is_manifest_checked": is_manifest_checked,
                    "patch_required": not is_manifest_checked,
                    "recommended_patch_code": (
                        _NO_PATCH_NEEDED if is_manifest_checked else _RECOMMENDED_PATCH
                    ),
                }
            )
            break  # One match per line is sufficient; avoid duplicate entries

    return results


def build_patch_sites(patch_sites: list[dict]) -> list[dict]:
    """Strip internal context_lines from patch sites to keep artifact lean."""
    return [
        {
            "line_number": s["line_number"],
            "code_snippet": s["code_snippet"],
            "pattern_matched": s["pattern_matched"],
            "is_manifest_checked": s["is_manifest_checked"],
            "patch_required": s["patch_required"],
            "recommended_patch_code": s["recommended_patch_code"],
        }
        for s in patch_sites
    ]


def main() -> None:
    """Run the manifest full-scope audit and write the deliverable artifact."""
    tmpl = ExperimentTemplate(
        EXPERIMENT_ID,
        EXPERIMENT_TITLE,
        DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()

    conductor_path = PROJECT_ROOT / "scripts" / "research_conductor.py"

    # Read the conductor as plain text.  We never modify it — only inspect it.
    if not conductor_path.exists():
        artifact = tmpl.build_result(
            {
                "patch_sites": [],
                "n_dequeue_sites_total": 0,
                "n_dequeue_sites_guarded": 0,
                "n_dequeue_sites_unguarded": 0,
                "estimated_waste_min_per_milestone": 0,
                "excluded_experiment_ids": [],
                "honest_verdict": "conductor_not_readable",
            },
            status="blocked",
        )
        import json

        (PROJECT_ROOT / DELIVERABLE).write_text(json.dumps(artifact, indent=2))
        tmpl.assert_deliverable_written()
        return

    conductor_text = conductor_path.read_text(encoding="utf-8")
    raw_sites = scan_conductor_text(conductor_text)
    patch_sites = build_patch_sites(raw_sites)

    n_total = len(patch_sites)
    n_unguarded = sum(1 for s in patch_sites if s["patch_required"])
    n_guarded = n_total - n_unguarded

    # Load the excluded experiment IDs from the manifest so we can report which
    # specific experiments are at risk from unguarded dequeue sites.
    manifest_path = PROJECT_ROOT / "scripts" / "conductor_exclusion_manifest.json"
    excluded_ids: list[int | str] = []
    if manifest_path.exists():
        import json

        try:
            manifest_data = json.loads(manifest_path.read_text())
            excluded_ids = [
                e["experiment_id"] for e in manifest_data.get("excluded", [])
            ]
        except Exception:
            excluded_ids = []

    # Estimate wasted time per milestone from unguarded sites.  Each unguarded
    # site can dispatch a retired experiment.  The average retired experiment
    # ran for ~55 min (weighted mean of the slowest-5 data from Exp 575 analysis:
    # Exps 308/260/309/425/410 averaged 53.7 min per milestone appearance).
    avg_retired_exp_duration_min = 55
    estimated_waste = n_unguarded * avg_retired_exp_duration_min

    if n_total == 0:
        honest_verdict = "conductor_not_readable"
    elif n_unguarded == 0:
        honest_verdict = "all_dequeue_sites_guarded"
    else:
        honest_verdict = f"{n_unguarded}_unguarded_sites_found"

    result_payload = {
        "patch_sites": patch_sites,
        "n_dequeue_sites_total": n_total,
        "n_dequeue_sites_guarded": n_guarded,
        "n_dequeue_sites_unguarded": n_unguarded,
        "estimated_waste_min_per_milestone": estimated_waste,
        "excluded_experiment_ids": excluded_ids,
        "honest_verdict": honest_verdict,
    }

    artifact = tmpl.build_result(result_payload, status="success")

    import json

    (PROJECT_ROOT / DELIVERABLE).write_text(json.dumps(artifact, indent=2))
    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
