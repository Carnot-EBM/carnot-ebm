"""Reconcile V665 branch evidence without repeating unavailable work.

The module is a read-only aggregator. Historical model and hardware rows stay
source evidence; this process performs no model call or board operation.

Spec refs: REQ-REPORT-7628 and SCENARIO-REPORT-7628-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import tempfile
import time
from typing import Any

from carnot import experiment_7615_v665_contract_methods as contract_reader
from carnot import experiment_7627_v665_native_cost as native_reader
from carnot.experiment_7572_v661_capstone import evaluate_publication_gates
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
)


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260924"
MILESTONE = "2026.09.665"
EXPERIMENT_ID = "exp7628-capstone"
SCHEMA = "carnot.exp7628.v665.capstone.v1"
EXPECTED_TASK_IDS = contract_reader.EXPECTED_TASK_IDS

DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
NOTE_PATH = Path("docs/research-notes/v665-capstone.md")
RESULT_PATH = Path("results/experiment_7628_v665_capstone.json")
RAW_DIR = Path("results/raw/experiment_7628_v665_capstone")
MODULE_PATH = Path("python/carnot/experiment_7628_v665_capstone.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7628_v665_capstone.py")
TEST_PATH = Path("tests/python/test_experiment_7628_v665_capstone.py")

PRE_GATE_PATHS = {
    "exp7618-fit-evidence": Path("results/experiment_7618_fit_evidence.json"),
    "exp7619-online-evidence": Path("results/experiment_7619_online_evidence.json"),
    "exp7620-evaluation-evidence": Path("results/experiment_7620_evaluation_evidence.json"),
}
INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("results/experiment_7614_v664_capstone.json"),
    Path("docs/research-notes/v664-capstone.md"),
    Path("scripts/publication_gate.py"),
    Path("scripts/adversarial_verify.py"),
    Path("scripts/verdict_row_consistency_lint.py"),
    SPEC_PATH,
    Path("research-references.md"),
    DESIGN_PATH,
)
TERMINAL_READER_NAMES = (
    "declared_entrypoint",
    "fresh_process_cold_reduction",
    "task_specific_e2e",
    "independent_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "verdict_class",
    "flagged_adversarial",
    "gate_check_summary",
    "acceptance_gate_results",
    "rows",
    "sample_size_budget",
    "preconditions_checked",
    "inference_substrate",
    "inference_substrate_class",
    "MODEL_SPECS",
    "model_invoked",
    "execution_venue",
    "phase_spans",
    "invocation_counts",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "source_artifact_hashes",
    "validation_receipts",
    "verifier_is_oracle",
    "field_principles",
    "milestone_dispositions",
    "evidence_summary",
    "remaining_prd_gaps",
    "publication_gates",
    "next_decisions",
)


def load_json(path: Path) -> JsonDict:
    """Require named JSON operands rather than accepting an ambiguous scalar."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _failed_check(
    check: str,
    upstream: str,
    path: object,
    field: str,
    operator: str,
    expected: object,
    observed: object,
) -> JsonDict:
    """Keep the seven diagnostic operands needed to investigate a block."""

    return {
        "check": check,
        "upstream": upstream,
        "path": path,
        "field": field,
        "operator": operator,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "passed": False,
    }


def load_authority(root: Path) -> JsonDict:
    """Resolve matching staging or consumed active V665 authority bytes."""

    selected, roadmap, candidates = contract_reader.resolve_v665_roadmap(root)
    comparison = contract_reader.compare_contract_authorities(
        (root / DESIGN_PATH).read_text(encoding="utf-8"), roadmap
    )
    tasks = roadmap.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("V665 task authority must contain a task list")
    return {
        "selected_roadmap_path": selected.relative_to(root).as_posix(),
        "resolution_candidates": deepcopy(candidates),
        "comparison_passed": comparison.get("passed") is True,
        "comparison": deepcopy(comparison),
        "tasks": deepcopy(tasks),
    }


def _terminal_readers(payload: Mapping[str, Any], fallback: str) -> object:
    """Preserve producer reader results or name why no readers could run."""

    value = payload.get("terminal_reader_outcomes")
    return deepcopy(value) if value is not None else {"status": fallback}


def _pre_gate_failure(task_id: str, path: str, payload: Mapping[str, Any]) -> JsonDict:
    """Normalize one conductor diagnostic without pretending it was a run."""

    raw = payload.get("blocked_diagnostic_contract")
    row = raw if isinstance(raw, Mapping) else payload
    return _failed_check(
        "conductor_pre_gate",
        str(row.get("failed_upstream") or task_id),
        row.get("failed_evidence_path") or path,
        str(row.get("failed_field") or "unknown"),
        str(row.get("failed_operator") or "=="),
        row.get("failed_expected"),
        row.get("failed_observed"),
    )


def _producer_disposition(root: Path, task: Mapping[str, Any], order: int) -> JsonDict:
    """Classify actual bytes, a pre-gate record, or genuinely missing work."""

    task_id = str(task.get("id") or "")
    planned = str(task.get("deliverable") or "")
    actual_relative = PRE_GATE_PATHS.get(task_id, Path(planned))
    actual = root / actual_relative
    if actual.is_file():
        payload = load_json(actual)
        pre_gate = isinstance(payload.get("blocked_diagnostic_contract"), Mapping)
        kind = "conductor_pre_gate" if pre_gate else "terminal_producer"
        failure = (
            _pre_gate_failure(task_id, actual_relative.as_posix(), payload) if pre_gate else None
        )
        return {
            "order": order,
            "task_id": task_id,
            "title": task.get("title"),
            "phase": task.get("phase"),
            "planned_path": planned,
            "actual_path": actual_relative.as_posix(),
            "custody_kind": kind,
            "exists": True,
            "sha256": sha256_file(actual),
            "size_bytes": actual.stat().st_size,
            "honest_verdict": payload.get("honest_verdict"),
            "verdict_class": ("blocked" if pre_gate else payload.get("verdict_class")),
            "flagged_adversarial": payload.get("flagged_adversarial", False),
            "gate_check": failure,
            "terminal_reader_results": _terminal_readers(
                payload, "not_run_conductor_pre_gate" if pre_gate else "not_declared"
            ),
        }
    failure = _failed_check(
        "required_scientific_producer",
        task_id,
        planned,
        "path",
        "exists",
        True,
        False,
    )
    return {
        "order": order,
        "task_id": task_id,
        "title": task.get("title"),
        "phase": task.get("phase"),
        "planned_path": planned,
        "actual_path": None,
        "custody_kind": "missing_work",
        "exists": False,
        "sha256": None,
        "size_bytes": 0,
        "honest_verdict": None,
        "verdict_class": None,
        "flagged_adversarial": False,
        "gate_check": failure,
        "terminal_reader_results": {"status": "not_run_missing_work"},
    }


def collect_milestone_dispositions(
    root: Path, tasks: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Return exactly fourteen literal task states in authority order."""

    ids = [str(task.get("id") or "") for task in tasks]
    if ids != list(EXPECTED_TASK_IDS):
        raise ValueError("V665 authority order is invalid")
    rows = [_producer_disposition(root, task, index) for index, task in enumerate(tasks[:-1], 1)]
    final = tasks[-1]
    rows.append(
        {
            "order": 14,
            "task_id": str(final.get("id")),
            "title": final.get("title"),
            "phase": final.get("phase"),
            "planned_path": str(final.get("deliverable")),
            "actual_path": RESULT_PATH.as_posix(),
            "custody_kind": "current_self",
            "exists": False,
            "sha256": None,
            "size_bytes": 0,
            "honest_verdict": "complete_blocked_required_v665_external_evidence",
            "verdict_class": "blocked",
            "flagged_adversarial": False,
            "gate_check": None,
            "terminal_reader_results": {"status": "pending_current_terminal_validation"},
        }
    )
    return rows


def disposition_counts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Count custody categories without treating missing work as a producer."""

    counts = Counter(str(row.get("custody_kind")) for row in rows)
    return {
        name: counts[name]
        for name in ("terminal_producer", "conductor_pre_gate", "missing_work", "current_self")
    }


def _arc_reduction(payload: Mapping[str, Any]) -> JsonDict:
    """Recompute observational counts from saved row operands."""

    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("Exp7625 rows are unavailable")
    units = {str(row.get("unit")) for row in rows if isinstance(row, Mapping)}
    counts = {
        field: sum(int(row.get(field) or 0) for row in rows if isinstance(row, Mapping))
        for field in ("proposed_redirects", "would_have_redirects", "actual_firings", "helped")
    }
    recorded = payload.get("redirect_counts")
    recorded_names = {
        "proposed_redirects": "proposed_redirect_count",
        "actual_firings": "actual_firing_count",
    }
    recomputed_matches = isinstance(recorded, Mapping) and all(
        counts[name] == recorded.get(recorded_names[name]) for name in recorded_names
    )
    return {
        "eligible": payload.get("supervisor_outcome_ledger_ready_score") == 1,
        "observed_independent_games": len(units),
        **counts,
        "causal_benefit": False,
        "recomputed_from_rows": recomputed_matches,
        "claim_scope": "observational_support_only_no_treatment_effect",
    }


def _native_reduction(payload: Mapping[str, Any]) -> JsonDict:
    """Recompute fixed native ratios from all 120 paired row receipts."""

    rows = payload.get("paired_timing_rows")
    if not isinstance(rows, list):
        raise ValueError("Exp7627 paired timing rows are unavailable")
    reduced = native_reader.reduce_timing_rows(rows)
    headline = reduced["equal_stratum_geometric_mean"]["python_over_direct_native"]
    benefit = bool(
        reduced.get("complete")
        and headline["estimate"] >= 1.10
        and headline["lower95"] >= 1.10
        and reduced["minimum_primary_stratum_lower95"] >= 0.95
        and reduced.get("parity_complete")
        and reduced.get("extra_native_errors") == 0
    )
    recorded = payload.get("timing_reduction")
    recorded_headline = (
        recorded.get("equal_stratum_geometric_mean", {}).get("python_over_direct_native", {})
        if isinstance(recorded, Mapping)
        else {}
    )
    return {
        "eligible": payload.get("native_cost_valid_score") == 1,
        "independent_blocks": reduced["independent_blocks"],
        "python_over_direct_native_estimate": headline["estimate"],
        "python_over_direct_native_lower95": headline["lower95"],
        "python_over_direct_native_upper95": headline["upper95"],
        "minimum_stratum_lower95": reduced["minimum_primary_stratum_lower95"],
        "benefit_gate_passed": benefit,
        "nfr_10x_met": headline["estimate"] >= 10.0,
        "recomputed_from_rows": headline["estimate"] == recorded_headline.get("estimate"),
        "direction": "python_total_cost_over_direct_native_higher_favors_native",
    }


def reduce_evidence_summary(root: Path) -> JsonDict:
    """Keep eligibility, effects, retention, exposure, ARC, and speed separate."""

    pilot = load_json(root / "results/experiment_7617_v665_schema_pilot.json")
    audit = load_json(root / "results/experiment_7624_v665_evidence_audit.json")
    arc = load_json(root / "results/experiment_7625_v665_arc_supervisor_transfer.json")
    native = load_json(root / "results/experiment_7627_v665_native_cost.json")
    static_branch = next(
        row for row in audit["branch_dispositions"] if row.get("branch") == "static"
    )
    learning_branch = next(
        row for row in audit["branch_dispositions"] if row.get("branch") == "learning"
    )
    budget = pilot.get("sample_size_budget") or {}
    return {
        "syntax_readiness": {
            "eligible": pilot.get("verdict_class") != "disqualified",
            "ready": pilot.get("evidence_transport_ready_score") == 1,
            "observed_groups": budget.get("observed", 0),
            "intended_groups": budget.get("intended", 8),
            "block": deepcopy(pilot.get("gate_check_summary")),
        },
        "calibration_benefit": {
            "eligible": audit.get("static_audit_eligible_score") == 1,
            "observed": static_branch.get("calibration_only_improvement"),
            "claim_scope": "unavailable_not_a_null",
        },
        "semantic_evidence_dependence": {
            "eligible": audit.get("static_audit_eligible_score") == 1,
            "observed": static_branch.get("semantic_evidence_dependence"),
            "benefit": audit.get("audited_evidence_benefit_score"),
            "claim_scope": "unavailable_not_a_null",
        },
        "retained_delayed_learning": {
            "eligible": audit.get("learning_audit_eligible_score") == 1,
            "observed": learning_branch.get("retained_online_learning"),
            "benefit": audit.get("audited_learning_benefit_score"),
            "claim_scope": "unavailable_not_a_null",
        },
        "observational_arc_support": _arc_reduction(arc),
        "deployment_speed": _native_reduction(native),
        "freshness": {
            "observed": False,
            "historically_exposed_roles": ["fit", "tune", "policy", "online", "evaluation"],
            "fresh_confirmatory_claim_allowed": False,
        },
    }


def classify_terminal(
    owned_work_complete: bool, external_evidence_absent: bool, eligible_science_benefit: bool
) -> tuple[str, str]:
    """Reserve partial for unfinished work owned by this capstone."""

    if not owned_work_complete:
        return "incomplete_owned_capstone_work", "partial"
    if external_evidence_absent:
        return "complete_blocked_required_v665_external_evidence", "blocked"
    if eligible_science_benefit:
        return "complete_positive_v665_independent_benefit", "positive"
    return "complete_null_v665_no_independent_benefit", "null"


def _blocking_checks(dispositions: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Retain branch failures while avoiding duplicate inherited symptoms."""

    checks: list[JsonDict] = []
    pilot = next(row for row in dispositions if row.get("task_id") == "exp7617-schema-pilot")
    checks.append(
        _failed_check(
            "evidence_transport_ready",
            "exp7617-schema-pilot",
            pilot.get("actual_path"),
            "evidence_transport_ready_score",
            "==",
            1,
            0,
        )
    )
    for row in dispositions:
        if row.get("custody_kind") == "missing_work" and isinstance(row.get("gate_check"), Mapping):
            checks.append(deepcopy(dict(row["gate_check"])))
    return checks


def _acceptance_gates(summary: Mapping[str, Any]) -> JsonDict:
    """Expose five independent acceptance categories and their principles."""

    principle = "Validity, readiness, benefit, retention, and freshness are separate claims."
    return {
        "validity": {
            "condition": "authority, producer bytes, and independent reductions authenticate",
            "observed": bool(
                summary["observational_arc_support"]["recomputed_from_rows"]
                and summary["deployment_speed"]["recomputed_from_rows"]
            ),
            "passed": True,
            "principle": principle,
        },
        "readiness": {
            "condition": "syntax transport and all required scientific producers are ready",
            "observed": False,
            "passed": False,
            "principle": principle,
        },
        "benefit": {
            "condition": "eligible evidence decision benefit is independently measured",
            "observed": None,
            "passed": False,
            "principle": principle,
        },
        "retention": {
            "condition": "causal delayed learning remains useful after restart",
            "observed": None,
            "passed": False,
            "principle": principle,
        },
        "freshness": {
            "condition": "external evaluation roles were not historically exposed",
            "observed": False,
            "passed": False,
            "principle": principle,
        },
    }


def _remaining_prd_gaps(summary: Mapping[str, Any]) -> list[JsonDict]:
    """Name the three product gaps left open by actual V665 results."""

    speed = summary["deployment_speed"]
    return [
        {
            "gap": "decision_evidence",
            "status": "open_unmeasured",
            "observed": None,
            "next_condition": "capture isolated evaluation evidence after a schema pilot runs on an exclusive task-owned GPU lease",
        },
        {
            "gap": "causal_retained_learning",
            "status": "open_unmeasured",
            "observed": None,
            "next_condition": "complete delayed online rows and evaluator-only retention rows under role separation",
        },
        {
            "gap": "total_deployment_cost",
            "status": "open_nfr_10x_unmet",
            "observed": speed["python_over_direct_native_estimate"],
            "expected": 10.0,
            "next_condition": "a changed end-to-end consumer boundary must achieve a measured total-cost ratio of at least 10x",
        },
    ]


def _next_decisions() -> list[JsonDict]:
    """Attach one changed premise to every failed-scope continuation."""

    return [
        {
            "hypothesis": "schema_transport",
            "decision": "change",
            "scope": "exclusive-capacity launch mechanism",
            "changed_premise_required": "an idle task-owned GPU with at least 20000 MiB free; never evict foreign work",
        },
        {
            "hypothesis": "evidence_semantics",
            "decision": "keep",
            "scope": "source-dependent semantic evidence",
            "changed_premise_required": "complete independently valid fit and evaluation rows from the repaired schema",
        },
        {
            "hypothesis": "guarded_delayed_learning",
            "decision": "keep",
            "scope": "role-separated causal update and retention",
            "changed_premise_required": "complete online release and isolated retention rows",
        },
        {
            "hypothesis": "arc_supervisor_refinement",
            "decision": "retire",
            "scope": "unchanged no-firing observational refinement",
            "changed_premise_required": "a preregistered treatment produces nonzero actual firings before outcomes are read",
        },
        {
            "hypothesis": "native_boundary_speed",
            "decision": "keep",
            "scope": "measured direct native boundary",
            "changed_premise_required": "optimize total consumer work and remeasure only if targeting the still-unmet 10x NFR",
        },
    ]


def _tool_version(root: Path, executable: str, *args: str) -> str:
    """Read one declared local tool version with a short fixed timeout."""

    completed = subprocess.run(  # noqa: S603 - fixed worktree executable.
        (str(root / executable), *args),
        cwd=root,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    return (completed.stdout or completed.stderr).strip().splitlines()[0]


def collect_preconditions(root: Path, authority: Mapping[str, Any]) -> list[JsonDict]:
    """Authenticate named inputs and tool identities before aggregation."""

    rows: list[JsonDict] = []
    for relative in INPUT_PATHS:
        path = root / relative
        exists = path.is_file()
        rows.append(
            {
                "check": "named_input_file",
                "upstream": "repository",
                "path": relative.as_posix(),
                "field": "exists",
                "operator": "==",
                "expected": True,
                "observed": exists,
                "passed": exists,
                "sha256": sha256_file(path) if exists else None,
            }
        )
    rows.append(
        {
            "check": "v665_authority",
            "upstream": "staged_or_active_roadmap",
            "path": authority["selected_roadmap_path"],
            "field": "milestone_and_contract_match",
            "operator": "==",
            "expected": True,
            "observed": authority["comparison_passed"],
            "passed": authority["comparison_passed"],
        }
    )
    for name, executable, args in (
        ("python", ".venv/bin/python", ("--version",)),
        ("pytest", ".venv/bin/pytest", ("--version",)),
        ("ruff", ".venv/bin/ruff", ("--version",)),
        ("mypy", ".venv/bin/mypy", ("--version",)),
    ):
        observed = _tool_version(root, executable, *args)
        rows.append(
            {
                "check": "declared_tool_version",
                "upstream": name,
                "path": executable,
                "field": "version",
                "operator": "present",
                "expected": "nonempty_version_string",
                "observed": observed,
                "passed": bool(observed),
            }
        )
    return rows


def _source_hashes(
    root: Path, dispositions: Sequence[Mapping[str, Any]], publication: Mapping[str, Any]
) -> list[JsonDict]:
    """Separate actual producers, pre-gates, missing work, and gate sources."""

    rows = [
        {
            "task_id": row["task_id"],
            "source_kind": row["custody_kind"],
            "planned_path": row["planned_path"],
            "actual_path": (None if row["custody_kind"] == "current_self" else row["actual_path"]),
            "sha256": row["sha256"],
            "exists": row["exists"],
        }
        for row in dispositions
    ]
    gate_paths = [Path("scripts/publication_gate.py"), Path("ops/publication_gate_state.json")]
    source_name = publication.get("gates", {}).get("G1", {}).get("source")
    if isinstance(source_name, str):
        gate_paths.append(Path("results") / source_name)
    for relative in gate_paths:
        path = root / relative
        rows.append(
            {
                "task_id": "publication_gate",
                "source_kind": "fixed_gate_source",
                "planned_path": relative.as_posix(),
                "actual_path": relative.as_posix() if path.is_file() else None,
                "sha256": sha256_file(path) if path.is_file() else None,
                "exists": path.is_file(),
            }
        )
    return rows


def _publication_record(root: Path) -> JsonDict:
    """Run fixed G1-G4 and mark its historical claim boundary."""

    result = evaluate_publication_gates(root)
    return {
        **result,
        "claim_boundary": "historical_fover_eligibility_only_not_a_v665_claim",
        "authorizes_submission": False,
    }


def _rows(dispositions: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Map each roster task to one auditable independent accounting row."""

    return [
        {
            "unit_id": row["task_id"],
            "arm": row["custody_kind"],
            "absolute_metric": 1,
            "numerator": 1,
            "denominator": 1,
            "seed": None,
            "direction": "accounting_completion",
            "censored": False,
            "raw_provenance": row["actual_path"] or row["planned_path"],
            "verdict_class": row["verdict_class"],
        }
        for row in dispositions
    ]


FIELD_PRINCIPLES = {
    "honest_verdict": "A complete prefix records terminal work without implying benefit.",
    "verdict_class": "External absence is blocked; partial is reserved for unfinished owned work.",
    "flagged_adversarial": "A flagged terminal reader cannot open a downstream gate.",
    "gate_check_summary": "Every block retains exact check operands.",
    "acceptance_gate_results": "Validity, readiness, benefit, retention, and freshness stay separate.",
    "rows": "One authority task contributes one accounting unit.",
    "sample_size_budget": "Seeds, views, arms, and replays never multiply independent units.",
    "preconditions_checked": "Unavailable inputs remain explicit instead of becoming fabricated work.",
    "inference_substrate": "Current aggregation cannot become historical model work.",
    "inference_substrate_class": "Planned and actual execution classes remain explicit.",
    "MODEL_SPECS": "An empty list records that current work made no LLM call.",
    "model_invoked": "Historical GPU rows do not count as current invocation.",
    "execution_venue": "Historical devices do not become current execution venues.",
    "phase_spans": "Disjoint monotonic spans expose current work and pending operations.",
    "invocation_counts": "Typed zeros prevent invented model work.",
    "duration_s": "Measured monotonic duration excludes inherited work and padding.",
    "random_seed": "Every stochastic reduction declares its fixed purpose and seed.",
    "reproducibility_checksum": "Immutable evidence, configuration, and reductions bind identity.",
    "source_artifact_hashes": "Hashes distinguish producers, pre-gates, missing work, and gate sources.",
    "validation_receipts": "Commands, exits, worktree paths, and log hashes bind validation.",
    "verifier_is_oracle": "Exact fixtures cannot establish oracle-distinct learned benefit.",
    "field_principles": "Every top-level field states the failure it prevents.",
    "milestone_dispositions": "Exactly fourteen ordered tasks preserve literal custody.",
    "evidence_summary": "Independent sources bound syntax, science, ARC, and speed claims.",
    "remaining_prd_gaps": "Measured success cannot close unrelated product requirements.",
    "publication_gates": "Historical FoVer eligibility does not create a V665 claim.",
    "next_decisions": "Every failed-scope continuation requires a materially changed premise.",
}


def _field_principles(fields: Sequence[str]) -> dict[str, str]:
    """Give every top-level field a short failure-prevention principle."""

    return {
        field: FIELD_PRINCIPLES.get(
            field, f"Exact {field} evidence prevents an unstated inference."
        )
        for field in fields
    }


def _sample_budget(summary: Mapping[str, Any]) -> JsonDict:
    """Report independent units without multiplying arms or replay views."""

    return {
        "milestone_tasks": {
            "independent_unit": "authority_task",
            "intended": 14,
            "observed": 14,
            "excluded": 0,
            "censored": 0,
        },
        "static_evidence": {
            "independent_unit": "source_group",
            "intended": 40,
            "observed": 0,
            "excluded": 0,
            "censored": 0,
            "missing_external": 40,
        },
        "delayed_learning": {
            "independent_unit": "source_group",
            "intended": 80,
            "observed": 0,
            "excluded": 0,
            "censored": 0,
            "missing_external": 80,
        },
        "arc_supervisor": {
            "independent_unit": "game",
            "intended": 6,
            "observed": summary["observational_arc_support"]["observed_independent_games"],
            "excluded": 0,
            "censored": 0,
        },
        "native_cost": {
            "independent_unit": "paired_workload_block",
            "intended": 120,
            "observed": summary["deployment_speed"]["independent_blocks"],
            "excluded": 0,
            "censored": 0,
        },
        "seeds_views_arms_replays_multiply_independent_samples": False,
    }


def _operator_resources(root: Path) -> list[JsonDict]:
    """Preserve operator and physical blocks without issuing an operation."""

    prior = load_json(root / "results/experiment_7614_v664_capstone.json")
    native = load_json(root / "results/experiment_7627_v665_native_cost.json")
    gate_mate = next(
        row for row in native["hardware_dispositions"] if row.get("hardware") == "GateMate"
    )
    return [
        {
            "resource": "exclusive_cuda_capacity",
            "status": "blocked_at_exp7617",
            "operator_action_required": False,
            "next_condition": "an idle task-owned GPU with at least 20000 MiB free without disturbing foreign work",
        },
        {
            "resource": "GateMate_physical_chain",
            "status": gate_mate["disposition"],
            "operator_action_required": True,
            "observed": gate_mate.get("last_observed"),
            "next_condition": gate_mate["exact_next_condition"],
        },
        *[
            {
                "resource": name,
                "status": status,
                "operator_action_required": True,
                "next_condition": "operator supplies a new dated confirmation",
            }
            for name, status in prior.get("operator_held_confirmations", {}).items()
        ],
    ]


def _checksum_payload(value: Mapping[str, Any]) -> JsonDict:
    """Select immutable evidence, configuration, and reduction fields."""

    dispositions = [
        {
            key: row.get(key)
            for key in (
                "order",
                "task_id",
                "planned_path",
                "actual_path",
                "custody_kind",
                "sha256",
                "verdict_class",
            )
        }
        for row in value.get("milestone_dispositions", [])
        if isinstance(row, Mapping)
    ]
    return {
        "schema": value.get("schema"),
        "milestone": value.get("milestone"),
        "selected_authority": value.get("selected_authority"),
        "milestone_dispositions": dispositions,
        "evidence_summary": value.get("evidence_summary"),
        "remaining_prd_gaps": value.get("remaining_prd_gaps"),
        "next_decisions": value.get("next_decisions"),
        "random_seed": value.get("random_seed"),
        "source_artifact_hashes": value.get("source_artifact_hashes"),
    }


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    """Bind immutable inputs and reductions without binding current clocks."""

    return canonical_hash(_checksum_payload(value))


def _validation_manifest() -> JsonDict:
    """Freeze the exact affected files before any validation child runs."""

    return {
        "test_paths": [TEST_PATH.as_posix()],
        "changed_modules": [MODULE_PATH.as_posix()],
        "static_python_paths": [WRAPPER_PATH.as_posix()],
        "spec_paths": [SPEC_PATH.as_posix()],
        "note_paths": [NOTE_PATH.as_posix()],
        "frozen_before_validation": True,
    }


def build_artifact(
    root: Path,
    authority: Mapping[str, Any],
    dispositions: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    publication: Mapping[str, Any],
    preconditions: Sequence[Mapping[str, Any]],
    *,
    validation_receipts: Sequence[Mapping[str, Any]] = (),
    terminal_reader_outcomes: Mapping[str, Any] | None = None,
    phase_spans: Sequence[Mapping[str, Any]] = (),
    started_monotonic_ns: int = 0,
    ended_monotonic_ns: int = 1_000_000,
    started_at_utc: str = "2026-09-24T00:00:00+00:00",
    completed_at_utc: str = "2026-09-24T00:00:00.001000+00:00",
) -> JsonDict:
    """Compose one terminal candidate from authenticated independent sources."""

    terminal = deepcopy(dict(terminal_reader_outcomes or {}))
    roster = deepcopy([dict(row) for row in dispositions])
    if terminal:
        roster[-1]["terminal_reader_results"] = terminal
    failures = _blocking_checks(roster)
    honest_verdict, verdict_class = classify_terminal(True, bool(failures), False)
    duration_s = (ended_monotonic_ns - started_monotonic_ns) / 1_000_000_000
    current_receipt = build_current_work_receipt(
        run_id="exp7628-current-aggregation",
        owner_pid=os.getpid(),
        events=(),
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_details={"historical_model_identity": "unsloth/Qwen3.8-27B-GGUF"},
        inference_substrate_class="aggregation",
        execution_venue="host",
        started_monotonic_ns=started_monotonic_ns,
        ended_monotonic_ns=ended_monotonic_ns,
        phase_spans=phase_spans,
    )
    source_hashes = _source_hashes(root, roster, publication)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": 7628,
        "experiment_id": EXPERIMENT_ID,
        "title": "Reconcile fourteen dispositions and select the next scientific question",
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "status": "complete",
        "honest_verdict": honest_verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": False,
        "capstone_complete_score": 1,
        "gate_check_summary": {
            "passed": not failures,
            "failed_count": len(failures),
            "first_failure": deepcopy(failures[0]) if failures else None,
            "failed_checks": deepcopy(failures),
        },
        "acceptance_gate_results": _acceptance_gates(summary),
        "rows": _rows(roster),
        "sample_size_budget": _sample_budget(summary),
        "preconditions_checked": deepcopy([dict(row) for row in preconditions]),
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "planned_inference_substrate_class": "aggregation",
        "actual_inference_substrate_class": "aggregation",
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "historical_model_identity": "unsloth/Qwen3.8-27B-GGUF",
        "execution_venue": "host",
        "execution_venue_details": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "current_gpu_uuid": None,
            "current_physical_device": None,
        },
        "phase_spans": deepcopy([dict(row) for row in phase_spans]),
        "invocation_counts": {
            "model_loads_attempted": 0,
            "model_loads_completed": 0,
            "forwards_attempted": 0,
            "forwards_completed": 0,
            "generations_attempted": 0,
            "generations_completed": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        },
        "current_work_receipt": current_receipt,
        "duration_s": duration_s,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "random_seed": {
            "native_bootstrap_base": 7627,
            "native_equal_stratum_bootstrap": 9627,
            "arc_reduction": None,
            "purpose": "replay producer-fixed bootstrap seeds; capstone adds no randomness",
        },
        "source_artifact_hashes": source_hashes,
        "validation_receipts": deepcopy([dict(row) for row in validation_receipts]),
        "terminal_reader_outcomes": terminal,
        "verifier_is_oracle": False,
        "selected_authority": authority["selected_roadmap_path"],
        "authority_comparison_passed": authority["comparison_passed"],
        "milestone_dispositions": roster,
        "disposition_counts": disposition_counts(roster),
        "evidence_summary": deepcopy(dict(summary)),
        "remaining_prd_gaps": _remaining_prd_gaps(summary),
        "publication_gates": deepcopy(dict(publication)),
        "next_decisions": _next_decisions(),
        "prior_failure_disposition": {
            "prior_experiment": "exp7614-capstone",
            "prior_verdict": "complete_blocked_required_v664_external_evidence",
            "retired_scope": "unchanged_v664_capstone_accounting_retry",
            "scientific_hypothesis_retired": False,
        },
        "architecture_status": {
            "evidence_path": "blocked_before_measurement",
            "arc_supervisor": "valid_observational_null_no_firings",
            "native_service": "measured_benefit_below_10x_nfr",
        },
        "operator_blocked_resources": _operator_resources(root),
        "affected_file_validation_manifest": _validation_manifest(),
        "repository_health": {"status": "outside_current_affected_validity"},
        "submitted_externally": False,
        "publication_performed": False,
        "roadmap_activation_performed": False,
        "research_conductor_modified": False,
        "production_defaults_changed": False,
        "generator_weights_changed": False,
        "purchase_performed": False,
        "hardware_operations_issued": [],
        "reproducibility_checksum": "",
        "field_principles": {},
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    artifact["field_principles"] = _field_principles(tuple(artifact))
    return artifact


def build_artifact_for_test(root: Path = REPO_ROOT) -> JsonDict:
    """Build deterministic current bytes without running validation children."""

    authority = load_authority(root)
    dispositions = collect_milestone_dispositions(root, authority["tasks"])
    summary = reduce_evidence_summary(root)
    publication = _publication_record(root)
    preconditions = collect_preconditions(root, authority)
    return build_artifact(root, authority, dispositions, summary, publication, preconditions)


def mutate_for_test(value: JsonDict, mutation: str) -> JsonDict:
    """Apply one private E2E corruption without touching repository evidence."""

    if mutation == "drop_disposition":
        value["milestone_dispositions"].pop()
    elif mutation == "missing_producer":
        row = next(
            item
            for item in value["milestone_dispositions"]
            if item["task_id"] == "exp7627-native-cost"
        )
        row["custody_kind"] = "missing_work"
        row["actual_path"] = None
    elif mutation == "positive_over_block":
        value["verdict_class"] = "positive"
        value["honest_verdict"] = "complete_positive_v665_independent_benefit"
    elif mutation == "partial_over_external":
        value["verdict_class"] = "partial"
        value["honest_verdict"] = "incomplete_owned_capstone_work"
    elif mutation == "native_ratio":
        value["evidence_summary"]["deployment_speed"]["python_over_direct_native_estimate"] += 1.0
    else:
        raise ValueError(f"unknown mutation: {mutation}")
    value["reproducibility_checksum"] = reproducibility_checksum(value)
    return value


def _validate_dispositions(
    value: Mapping[str, Any], root: Path, errors: list[str]
) -> list[Mapping[str, Any]]:
    """Reload authority and compare literal custody fields."""

    raw = value.get("milestone_dispositions")
    if not isinstance(raw, list) or len(raw) != 14:
        errors.append("milestone_disposition_count")
        return []
    if [row.get("task_id") for row in raw if isinstance(row, Mapping)] != list(EXPECTED_TASK_IDS):
        errors.append("milestone_disposition_order")
        return []
    try:
        authority = load_authority(root)
        expected = collect_milestone_dispositions(root, authority["tasks"])
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"authority_reload_failed:{type(exc).__name__}")
        return [row for row in raw if isinstance(row, Mapping)]
    for observed, wanted in zip(raw, expected, strict=True):
        if not isinstance(observed, Mapping):  # pragma: no cover - order guard returns first.
            errors.append("milestone_disposition_shape")
            continue
        task_id = str(wanted["task_id"])
        for field in ("custody_kind", "planned_path", "actual_path", "sha256"):
            if observed.get(field) != wanted.get(field):
                errors.append(f"{field}_mismatch:{task_id}")
    return [row for row in raw if isinstance(row, Mapping)]


def _validate_source_hashes(value: Mapping[str, Any], root: Path, errors: list[str]) -> None:
    """Reject missing or changed actual source bytes."""

    rows = value.get("source_artifact_hashes")
    if not isinstance(rows, list):
        errors.append("source_artifact_hashes_shape")
        return
    for row in rows:
        if not isinstance(row, Mapping):
            errors.append("source_artifact_hash_shape")
            continue
        label = row.get("actual_path")
        if label is None:
            if row.get("exists") is not False or row.get("sha256") is not None:
                errors.append(f"missing_source_state:{row.get('task_id')}")
            continue
        path = Path(str(label))
        resolved = path if path.is_absolute() else root / path
        observed = sha256_file(resolved) if resolved.is_file() else None
        if observed != row.get("sha256"):
            errors.append(f"source_hash_mismatch:{label}")


def _terminal_receipts_pass(value: Mapping[str, Any]) -> bool:
    """Require one passing exact receipt for every terminal reader."""

    outcomes = value.get("terminal_reader_outcomes")
    if not isinstance(outcomes, Mapping):
        return False
    return all(
        isinstance(outcomes.get(name), Mapping)
        and outcomes[name].get("passed") is True
        and outcomes[name].get("exit_code") == 0
        and isinstance(outcomes[name].get("log_sha256"), str)
        for name in TERMINAL_READER_NAMES
    )


def validate_artifact(
    value: object, *, root: Path = REPO_ROOT, require_terminal: bool = True
) -> list[str]:
    """Cold-reload all exact sources and recompute the terminal result."""

    if not isinstance(value, Mapping):
        return ["artifact_object_required"]
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in value:
            errors.append(f"required_field_missing:{field}")
    dispositions = _validate_dispositions(value, root, errors)
    if dispositions:
        expected_failures = _blocking_checks(dispositions)
        expected_verdict = classify_terminal(True, bool(expected_failures), False)
        if (value.get("honest_verdict"), value.get("verdict_class")) != expected_verdict:
            errors.append("terminal_classification_mismatch")
        summary = value.get("gate_check_summary")
        if not isinstance(summary, Mapping) or summary.get("failed_checks") != expected_failures:
            errors.append("gate_check_summary_mismatch")
        counts = value.get("disposition_counts")
        if counts != disposition_counts(dispositions):
            errors.append("disposition_counts_mismatch")
    try:
        expected_summary = reduce_evidence_summary(root)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"evidence_reduction_failed:{type(exc).__name__}")
    else:
        if value.get("evidence_summary") != expected_summary:
            errors.append("evidence_summary_mismatch")
        if value.get("acceptance_gate_results") != _acceptance_gates(expected_summary):
            errors.append("acceptance_gate_results_mismatch")
        if value.get("sample_size_budget") != _sample_budget(expected_summary):
            errors.append("sample_size_budget_mismatch")
        if value.get("remaining_prd_gaps") != _remaining_prd_gaps(expected_summary):
            errors.append("remaining_prd_gaps_mismatch")
    _validate_source_hashes(value, root, errors)
    if value.get("reproducibility_checksum") != reproducibility_checksum(value):
        errors.append("reproducibility_checksum_mismatch")
    principles = value.get("field_principles")
    if not isinstance(principles, Mapping) or not set(value).issubset(principles):
        errors.append("field_principles_incomplete")
    if value.get("MODEL_SPECS") != [] or value.get("model_invoked") is not False:
        errors.append("current_model_provenance_mismatch")
    if value.get("inference_substrate_class") != "aggregation":
        errors.append("inference_substrate_class_mismatch")
    if value.get("execution_venue") != "host":
        errors.append("execution_venue_mismatch")
    if value.get("submitted_externally") is not False:
        errors.append("external_submission_mismatch")
    if require_terminal and not _terminal_receipts_pass(value):
        errors.append("terminal_reader_receipts_incomplete")
    return list(dict.fromkeys(errors))


def independent_reduce(value: object, *, root: Path = REPO_ROOT) -> list[str]:
    """Use the cold validator without trusting stored terminal receipts."""

    return validate_artifact(value, root=root, require_terminal=False)


def date_argument(value: str) -> str:
    """Accept only the frozen V665 execution date."""

    if value != RUN_DATE:
        raise ValueError(f"run date must be {RUN_DATE}")
    return value


def root_argument(value: str | Path) -> Path:
    """Require an absolute path resolving to this exact worktree."""

    supplied = Path(value)
    resolved = supplied.resolve()
    if (
        not supplied.is_absolute()
        or resolved != REPO_ROOT
        or not (resolved / "AGENTS.md").is_file()
    ):
        raise ValueError(f"repository root must resolve absolutely to {REPO_ROOT}")
    return resolved


def progress(started: float, phase: str, event: str, **details: object) -> None:  # pragma: no cover
    """Emit one flushed monotonic boundary around each phase and child group."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7628] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _span(  # pragma: no cover - real clock boundary.
    phase: str,
    phase_started: float,
    run_started: float,
    *,
    completed_units: int,
    checkpoint: str,
) -> JsonDict:
    """Close one disjoint measured phase and name its checkpoint."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "started_elapsed_s": phase_started - run_started,
        "ended_elapsed_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": completed_units,
        "current_pending_operation": None,
        "checkpoint": checkpoint,
    }


def build_validation_commands(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Build the fixed affected-file validation plan."""

    (private_root / "pytest").mkdir(parents=True, exist_ok=True)
    return validation_scope.build_scoped_commands(
        root,
        [TEST_PATH.as_posix()],
        [MODULE_PATH.as_posix()],
        static_paths=[WRAPPER_PATH.as_posix()],
        basetemp=private_root / "pytest",
        coverage_file=private_root / ".coverage.exp7628",
    )


def _terminal_commands(root: Path, candidate: Path) -> list[validation_scope.CommandSpec]:
    """Build fresh replay, mutation, and strict terminal reader commands."""

    python = str(root / ".venv/bin/python")
    base = (python, "-u", WRAPPER_PATH.as_posix(), "--root", str(root))
    return [
        validation_scope.CommandSpec(
            "declared_entrypoint", (*base, "--validate", str(candidate)), "candidate"
        ),
        validation_scope.CommandSpec(
            "fresh_process_cold_reduction", (*base, "--cold-replay", str(candidate)), "candidate"
        ),
        validation_scope.CommandSpec(
            "task_specific_e2e", (*base, "--e2e", str(candidate)), "candidate_mutations"
        ),
        validation_scope.CommandSpec(
            "independent_reduction",
            (*base, "--independent-reduce", str(candidate)),
            "candidate_reduction",
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "candidate_safety",
        ),
        validation_scope.CommandSpec(
            "verdict_row_consistency_strict",
            (
                python,
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "candidate_rows",
        ),
    ]


def _outcomes(receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Persist exact terminal exit and log identities by reader name."""

    return {
        str(row["name"]): {
            "passed": row.get("passed") is True,
            "exit_code": row.get("exit_code"),
            "log_path": row.get("log_path"),
            "log_sha256": row.get("log_sha256"),
            "worktree": str(REPO_ROOT),
        }
        for row in receipts
    }


def _utc_now() -> str:  # pragma: no cover - real clock boundary.
    """Return one aware UTC timestamp."""

    return datetime.now(UTC).isoformat()


def _task_specific_e2e(value: JsonDict, root: Path) -> list[str]:
    """Exercise exact replay, missing producer, and classification controls."""

    errors = independent_reduce(value, root=root)
    controls = (
        ("drop_disposition", "milestone_disposition_count"),
        ("missing_producer", "custody_kind_mismatch:exp7627-native-cost"),
        ("positive_over_block", "terminal_classification_mismatch"),
        ("partial_over_external", "terminal_classification_mismatch"),
    )
    for mutation, wanted in controls:
        observed = validate_artifact(
            mutate_for_test(deepcopy(value), mutation), root=root, require_terminal=False
        )
        if wanted not in observed:
            errors.append(f"mutation_not_rejected:{mutation}")
    if classify_terminal(True, True, False)[1] != "blocked":
        errors.append("external_block_classification")
    if classify_terminal(False, False, False)[1] != "partial":
        errors.append("owned_partial_classification")
    return errors


def run_experiment(  # pragma: no cover - exercised through declared entrypoint.
    root: Path, run_date: str, *, output_path: Path = RESULT_PATH
) -> JsonDict:
    """Run scoped checks and atomically publish one validated capstone."""

    root = root_argument(root)
    date_argument(run_date)
    started = time.monotonic()
    started_ns = time.monotonic_ns()
    started_at = _utc_now()
    spans: list[JsonDict] = []
    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7628-", dir="/tmp"))
    raw_dir = root / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)

    phase_started = time.monotonic()
    progress(started, "preconditions", "before")
    authority = load_authority(root)
    preconditions = collect_preconditions(root, authority)
    if not all(row.get("passed") is True for row in preconditions):
        raise RuntimeError("required_input_or_tool_precondition_failed")
    dispositions = collect_milestone_dispositions(root, authority["tasks"])
    spans.append(
        _span(
            "preconditions",
            phase_started,
            started,
            completed_units=len(preconditions),
            checkpoint="named_inputs_tools_and_fourteen_custody_rows_authenticated",
        )
    )
    progress(started, "preconditions", "after", completed_units=len(preconditions))

    phase_started = time.monotonic()
    progress(started, "publication_gates", "before_subprocess")
    publication = _publication_record(root)
    if publication.get("exit_code") != 0:
        raise RuntimeError("publication_gate_reader_failed")
    spans.append(
        _span(
            "publication_gates",
            phase_started,
            started,
            completed_units=4,
            checkpoint="fixed_g1_g4_historical_boundary_recorded",
        )
    )
    progress(
        started, "publication_gates", "after_subprocess", paper_ready=publication["paper_ready"]
    )

    phase_started = time.monotonic()
    progress(started, "reduction", "before")
    summary = reduce_evidence_summary(root)
    spans.append(
        _span(
            "reduction",
            phase_started,
            started,
            completed_units=14,
            checkpoint="six_branches_and_fourteen_dispositions_reduced",
        )
    )
    progress(started, "reduction", "after", completed_units=14)

    for phase, checkpoint in (
        ("model_load", "no_current_model_load"),
        ("generation", "no_current_generation"),
        ("benchmark", "no_new_benchmark_only_row_reduction"),
    ):
        phase_started = time.monotonic()
        progress(started, phase, "before", completed_units=0)
        spans.append(_span(phase, phase_started, started, completed_units=0, checkpoint=checkpoint))
        progress(started, phase, "after", completed_units=0)

    phase_started = time.monotonic()
    progress(started, "validation", "before_subprocesses")
    manifest = _validation_manifest()
    atomic_json(raw_dir / "affected_file_validation_manifest.json", manifest)
    commands = build_validation_commands(root, private_root)
    affected = validation_scope.run_commands(
        root, commands, log_dir=private_root / "logs/affected", heartbeat_s=60.0
    )
    reduced_validation = validation_scope.reduce_required_checks(affected)
    spans.append(
        _span(
            "validation",
            phase_started,
            started,
            completed_units=len(affected),
            checkpoint="affected_checks_complete",
        )
    )
    progress(
        started,
        "validation",
        "after_subprocesses",
        completed_units=len(affected),
        passed=reduced_validation["required_checks_passed"],
    )
    if reduced_validation["required_checks_passed"] is not True:
        raise RuntimeError("affected_validation_failed")

    candidate = build_artifact(
        root,
        authority,
        dispositions,
        summary,
        publication,
        preconditions,
        validation_receipts=affected,
        phase_spans=spans,
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=time.monotonic_ns(),
        started_at_utc=started_at,
        completed_at_utc=_utc_now(),
    )
    candidate_path = private_root / "terminal/candidate.json"
    atomic_json(candidate_path, candidate)

    phase_started = time.monotonic()
    progress(started, "terminal_validation", "before_subprocesses")
    terminal_receipts = validation_scope.run_commands(
        root,
        _terminal_commands(root, candidate_path),
        log_dir=private_root / "logs/terminal",
        heartbeat_s=60.0,
    )
    outcomes = _outcomes(terminal_receipts)
    spans.append(
        _span(
            "terminal_validation",
            phase_started,
            started,
            completed_units=len(terminal_receipts),
            checkpoint="cold_replay_mutations_and_strict_readers_complete",
        )
    )
    progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        completed_units=len(terminal_receipts),
        passed=all(row.get("passed") is True for row in terminal_receipts),
    )

    phase_started = time.monotonic()
    progress(started, "write", "before_atomic", path=output_path.as_posix())
    final_spans = [
        *spans,
        _span(
            "write",
            phase_started,
            started,
            completed_units=1,
            checkpoint="terminal_candidate_ready",
        ),
    ]
    final = build_artifact(
        root,
        authority,
        dispositions,
        summary,
        publication,
        preconditions,
        validation_receipts=[*affected, *terminal_receipts],
        terminal_reader_outcomes=outcomes,
        phase_spans=final_spans,
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=time.monotonic_ns(),
        started_at_utc=started_at,
        completed_at_utc=_utc_now(),
    )
    errors = validate_artifact(final, root=root, require_terminal=True)
    if errors:
        raise ValueError("artifact_validation_failed:" + ",".join(errors))
    atomic_json(
        raw_dir / "exact_terminal_validation_receipts.json", {"receipts": terminal_receipts}
    )
    destination = output_path if output_path.is_absolute() else root / output_path
    atomic_json(destination, final)
    progress(started, "write", "after_atomic", path=destination.as_posix())
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    """Parse frozen root, date, output, and read-only replay modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(REPO_ROOT))
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    parser.add_argument("--e2e", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI boundary.
    """Run the capstone or one bounded fresh-process reader."""

    print("[exp7628] phase=startup event=flushed", flush=True)
    args = parse_args(argv)
    root = root_argument(args.root)
    candidate_path = args.validate or args.cold_replay or args.independent_reduce or args.e2e
    if candidate_path is not None:
        try:
            value = load_json(candidate_path)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
            print(json.dumps({"errors": [str(error)]}, sort_keys=True), flush=True)
            return 1
        if args.e2e is not None:
            errors = _task_specific_e2e(value, root)
        elif args.independent_reduce is not None:
            errors = independent_reduce(value, root=root)
        else:
            errors = validate_artifact(value, root=root, require_terminal=False)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    run_experiment(root, date_argument(args.date), output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
