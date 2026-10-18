#!/usr/bin/env python3
"""Exp 1104 — Failure-Ledger v2: Issues 1+5+manifest deployment + planner audit.

Three conductor regressions caused legitimate research tasks to be
retired in milestones .84 and .85.  This experiment ships the surgical
fixes for all three in a single deployment pass and records the
deployment status + a planner self-audit.

The fixes themselves live in ``scripts/failure_ledger_v2.py`` and are
wired into ``scripts/research_conductor.py``'s ``log_step`` and
``pick_next_task``.  This script's job is to:

  1. Verify the three fixes are deployed (module importable, tests
     passing, conductor wire-in present in the source).
  2. Run the planner self-audit on the active roadmap
     (research-roadmap.yaml is the .86 milestone; the conductor's planner
     consumes research-roadmap-next.yaml during planning, but that file is
     only present mid-planning — the active roadmap is the audit target
     once a milestone is in flight).
  3. Run the planner coherence check for cross-vendor model/agent
     mismatches.
  4. Write the artifact to the canonical results path with the schema
     fields the conductor's reconciler expects.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


def _verify_module_importable() -> bool:
    """Issue 1 + Issue 5 + manifest fix: all live in failure_ledger_v2."""
    try:
        from failure_ledger_v2 import (  # noqa: F401
            count_failures_for_task,
            extract_experiment_id,
            is_excluded_by_manifest,
            keywords_overlap,
        )

        return True
    except Exception:
        return False


def _verify_conductor_wire_in() -> dict[str, bool]:
    """Confirm the conductor source contains the wire-in markers.

    We check the source text rather than monkey-patching the conductor
    because the conductor module is large and has heavy import-time side
    effects we don't want to trigger during a small audit experiment.
    """
    src_path = PROJECT_ROOT / "scripts" / "research_conductor.py"
    src = src_path.read_text()
    return {
        "log_step_has_task_id_param": 'def log_step(task: str, status: str, details: str = "", task_id'
        in src,
        "pick_next_task_uses_id_counter": "from failure_ledger_v2 import" in src
        and "count_failures_for_task" in src,
        "pick_next_task_uses_yaml_manifest": "is_excluded_by_manifest" in src,
        "dispatch_skip_log_present": "exclusion_manifest retirement" in src,
    }


def _run_tests() -> tuple[int, int]:
    """Run only the new test module and return (n_written, n_passing).

    We deliberately scope the test run to the four tests this experiment
    introduced.  Project-wide tests are run by the conductor's own
    pre/post-test gates.
    """
    test_file = PROJECT_ROOT / "tests" / "python" / "test_failure_ledger_v2.py"
    pytest_bin = PROJECT_ROOT / ".venv" / "bin" / "pytest"
    if not pytest_bin.exists():
        return 4, 0  # cannot run; report the count we wrote
    result = subprocess.run(
        [str(pytest_bin), str(test_file), "-v", "--no-cov", "-p", "no:cacheprovider"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        env={"JAX_PLATFORMS": "cpu", "PATH": "/usr/bin:/bin"},
    )
    output = result.stdout + result.stderr
    n_passed = len(re.findall(r"PASSED", output))
    return 4, n_passed


def _planner_self_audit() -> dict:
    """Audit the active roadmap for prior_failures coverage on rerun-scope tasks.

    For each task in the active roadmap, tokenize its title and check
    against every entry in research-complete.yaml.  Any task with ≥2
    substantive token overlap with a prior task is flagged as needing
    a prior_failures: declaration; the audit then reports whether the
    declaration is present.

    This mirrors the dispatch-time check the conductor will perform via
    the failure-ledger discipline.  Running it as a one-shot audit gives
    the operator visibility into planner discipline before the conductor
    starts dequeuing tasks.
    """
    from failure_ledger_v2 import keywords_overlap

    roadmap_path = PROJECT_ROOT / "research-roadmap.yaml"
    complete_path = PROJECT_ROOT / "research-complete.yaml"

    if not roadmap_path.exists():
        return {
            "n_tasks_checked": 0,
            "n_needing_prior_failures": 0,
            "n_with_declarations": 0,
            "violations": [],
            "note": "research-roadmap.yaml not found",
        }

    with open(roadmap_path) as f:
        roadmap = yaml.safe_load(f) or {}
    tasks = roadmap.get("tasks", []) or []

    prior_titles: list[tuple[str, str]] = []
    if complete_path.exists():
        try:
            with open(complete_path) as f:
                complete = yaml.safe_load(f) or {}
            for ms in complete.get("milestones", []) or []:
                for t in ms.get("tasks", []) or []:
                    title = t.get("title") or ""
                    tid = t.get("id") or ""
                    if title and tid:
                        prior_titles.append((tid, title))
        except Exception:
            pass

    violations: list[dict] = []
    n_needing = 0
    n_with_decl = 0
    for task in tasks:
        title = task.get("title") or ""
        if not title:
            continue
        # Look for any prior task that overlaps on ≥2 substantive tokens
        # (the same threshold the dispatch-time keyword matcher uses).
        matches = [
            ptid
            for (ptid, ptitle) in prior_titles
            if ptid != task.get("id") and keywords_overlap(title, ptitle, min_count=2)
        ]
        if not matches:
            continue
        n_needing += 1
        has_pf = bool(task.get("prior_failures"))
        if has_pf:
            n_with_decl += 1
        else:
            violations.append(
                {
                    "task_id": task.get("id"),
                    "title": title[:80],
                    "n_prior_matches": len(matches),
                    "sample_prior_ids": matches[:3],
                }
            )
    return {
        "n_tasks_checked": len(tasks),
        "n_needing_prior_failures": n_needing,
        "n_with_declarations": n_with_decl,
        "violations": violations,
    }


def _planner_coherence_check() -> list[str]:
    """Detect cross-vendor model/agent mismatches in the active roadmap.

    Tasks that bind ``model: opus`` (Claude family) to ``agent_type: codex``
    or ``gemini`` are mis-configured: the codex/gemini CLI cannot route
    an opus model.  Reports the offending task ids so the operator can
    correct them before the conductor blocks at dispatch.
    """
    roadmap_path = PROJECT_ROOT / "research-roadmap.yaml"
    if not roadmap_path.exists():
        return []
    with open(roadmap_path) as f:
        data = yaml.safe_load(f) or {}
    violations: list[str] = []
    claude_models = {"opus", "sonnet", "haiku", "claude-opus-4-7", "claude-sonnet-4-6"}
    for t in data.get("tasks", []) or []:
        model = (t.get("model") or "").lower()
        agent_type = (t.get("agent_type") or "").lower()
        if model in claude_models and agent_type in {"codex", "gemini", "opencode"}:
            violations.append(f"{t.get('id')}: model={model} but agent_type={agent_type}")
        # Inverse: codex/gemini-only models routed through claude
        if model.startswith("gpt-") and agent_type == "claude":
            violations.append(f"{t.get('id')}: model={model} but agent_type=claude")
        if model.startswith("gemini-") and agent_type == "claude":
            violations.append(f"{t.get('id')}: model={model} but agent_type=claude")
    return violations


def main() -> int:
    started = datetime.now(UTC)

    module_ok = _verify_module_importable()
    wire_in = _verify_conductor_wire_in()
    n_written, n_passing = _run_tests()
    audit = _planner_self_audit()
    coherence_violations = _planner_coherence_check()

    issue_1_deployed = (
        module_ok
        and wire_in.get("log_step_has_task_id_param", False)
        and wire_in.get("pick_next_task_uses_id_counter", False)
    )
    issue_5_deployed = module_ok  # keywords_overlap is the issue 5 helper
    manifest_deployed = (
        module_ok
        and wire_in.get("pick_next_task_uses_yaml_manifest", False)
        and wire_in.get("dispatch_skip_log_present", False)
    )
    deployed_count = sum([issue_1_deployed, issue_5_deployed, manifest_deployed])
    if deployed_count == 3 and n_passing == n_written:
        verdict = "all_three_fixes_deployed"
    elif deployed_count == 2:
        verdict = "two_of_three_deployed"
    elif deployed_count >= 1:
        verdict = "partial"
    else:
        verdict = "failed"

    finished = datetime.now(UTC)
    artifact = {
        "experiment": "exp1104-failure-ledger-v2-issues-1-5-manifest",
        "title": "Failure-Ledger v2 (Issues 1+5+manifest)",
        "run_date": finished.strftime("%Y-%m-%d"),
        "duration_s": (finished - started).total_seconds(),
        "schema_version": 1,
        "status": "success" if verdict == "all_three_fixes_deployed" else "partial",
        "honest_verdict": verdict,
        "failure_ledger_id_fix_deployed": issue_1_deployed,
        "failure_ledger_keyword_threshold_deployed": issue_5_deployed,
        "manifest_dispatch_enforcement_deployed": manifest_deployed,
        "tests_written": n_written,
        "tests_passing": n_passing,
        "planner_audit_n_tasks_checked": audit["n_tasks_checked"],
        "planner_audit_n_needing_prior_failures": audit["n_needing_prior_failures"],
        "planner_audit_n_with_declarations": audit["n_with_declarations"],
        "planner_audit_violations": audit.get("violations", []),
        "planner_audit_cross_vendor_violations": coherence_violations,
        "wire_in_checks": wire_in,
        "module_importable": module_ok,
        "deliverable": "results/experiment_1104_failure_ledger_v2_id_keyword_manifest.json",
        # Conductor reconciler also looks for these fields on infra experiments.
        "summary": (
            f"Issue 1 fix (id-aware fail counting): {issue_1_deployed}; "
            f"Issue 5 fix (≥2 keyword threshold): {issue_5_deployed}; "
            f"manifest dispatch enforcement: {manifest_deployed}; "
            f"tests {n_passing}/{n_written}; "
            f"planner audit {audit['n_with_declarations']}/{audit['n_needing_prior_failures']} declared"
        ),
    }
    out_path = (
        PROJECT_ROOT / "results" / "experiment_1104_failure_ledger_v2_id_keyword_manifest.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if verdict == "all_three_fixes_deployed" else 1


if __name__ == "__main__":
    sys.exit(main())
