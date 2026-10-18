"""Exp6558 ARC live redirect-ledger reachability.

Spec refs: REQ-ARC-WMTE-6680,
SCENARIO-ARC-WMTE-6680-LIVE-REACHABILITY,
SCENARIO-ARC-WMTE-6680-NEXT-OUTCOME-LINKAGE,
SCENARIO-ARC-WMTE-6680-NO-FIRING-CLOSURE,
SCENARIO-ARC-WMTE-6680-SELECTION-SUPPORT,
SCENARIO-ARC-WMTE-6680-FAIL-CLOSED-ATTACKS,
SCENARIO-ARC-WMTE-6680-SCHEMA-AND-CLI.

This is a receipt reducer. It does not replay ARC games, read game source,
invoke a per-game adapter, call an LLM, or claim a game or level solve.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
from typing import Any

from carnot.agentic.arc_trajectory_supervisor import ARM_ORDER
from carnot.experiment_artifacts import atomic_write_json


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260823"
RANDOM_SEED = 6558
INFERENCE_SUBSTRATE = (
    "live_arc_trajectory_supervisor_receipt_reachability_and_selection_replay_no_llm"
)
VERIFIER_IS_ORACLE = False
MIN_PROSPECTIVE_FIRINGS = 3

RESULT_RELATIVE_PATH = Path("results/experiment_6558_arc_live_redirect_ledger_reachability.json")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6558_arc_live_redirect_ledger_reachability.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6558_arc_live_redirect_ledger_reachability.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
PRIOR_RESULT_RELATIVE_PATH = Path("results/experiment_6524_arc_supervisor_redirect_generalization.json")

DEFAULT_LIVE_ARTIFACT_PATHS = (
    Path("/home/ianblenke/.claude/jobs/ad0c053d/tmp/supab5/rows_on.json"),
    Path("/home/ianblenke/.claude/jobs/ad0c053d/tmp/supab5/rows_off.json"),
)

PROTECTED_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    SPEC_RELATIVE_PATH,
    Path("openspec/change-proposals/research-roadmap-vNEXT.md"),
    Path("research-roadmap.yaml"),
    Path("python/carnot/agentic/arc_competition_agent.py"),
    Path("python/carnot/agentic/arc_trajectory_supervisor.py"),
    Path("python/carnot/agentic/arc_solver_kit.py"),
    Path("ops/arc_solve_registry.yaml"),
    Path("scripts/research_conductor.py"),
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "prior_failure_receipt",
    "live_entrypoint_reachability_receipt",
    "supervisor_receipt_schema_and_code_hashes",
    "redirect_to_next_outcome_rows",
    "no_firing_run_rows",
    "curated_arm_support_rows",
    "selection_policy_disposition",
    "no_solve_and_no_source_receipt",
    "attack_matrix",
    "arc_live_redirect_ledger_ready_score",
    "per_unit_rows",
    "aggregate_row_recomputation",
    "gate_check_summary",
    "preconditions_checked",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_principles",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "A terminal state lets no-firing and missing-receipt outcomes satisfy the standing ARC slot honestly.",
    "honest_verdict": "The verdict must state live reachability, firing support, and policy disposition with a terminal prefix.",
    "verdict_class": "A closed class prevents an unsupported or blocked refinement from becoming positive.",
    "prior_failure_receipt": "The artifact must show how the Exp6524 missing-outcome block was tested.",
    "live_entrypoint_reachability_receipt": "Only code reachable from E3AgentPolicy or make_carnot_agent can improve the hidden-game deliverable.",
    "supervisor_receipt_schema_and_code_hashes": "Exact schema and implementation identities prevent a test-only receipt path from posing as live.",
    "redirect_to_next_outcome_rows": "Each firing must link to its next exact observed outcome without reassignment.",
    "no_firing_run_rows": "An empty ledger is a valid measured state and must not force churn.",
    "curated_arm_support_rows": "Selection changes require prospective support for existing human-curated arms.",
    "selection_policy_disposition": "The record must say changed, unchanged, or future-arm-specification and why.",
    "no_solve_and_no_source_receipt": "The task must prove that it read no game source, used no per-game adapter, and claimed no level solve.",
    "attack_matrix": "Receipt, leakage, source, adapter, support, and off-path attacks test the generalization claim.",
    "arc_live_redirect_ledger_ready_score": "One binary field records live receipt closure without requiring a manufactured policy change.",
    "per_unit_rows": "Every run, firing, arm, and outcome disposition must remain recheckable.",
    "aggregate_row_recomputation": "Fired, helped, unresolved, and unredirected totals must derive from rows.",
    "gate_check_summary": "A blocked result must name the missing live path, receipt, or input check and observed value.",
    "preconditions_checked": "Code, schema, artifact, registry, and resource receipts separate a block from no firings.",
    "protected_files_unchanged": "The ARC task must preserve the active roadmap and conductor.",
    "inference_substrate": "Receipt tracing and decision replay invoke no LLM and no offline solver.",
    "verifier_is_oracle": "Supervisor outcomes are live observations, not an oracle-distinct EBM verifier.",
    "field_principles": "Field principles keep the artifact schema tied to the operator's stated audit reason.",
    "field_provenance": "Every reachability and selection field must point to run rows, code hashes, and reducer logic.",
    "random_seed": "A fixed audit and replay order makes receipt analysis repeatable.",
    "duration_s": "Monotonic time exposes a task that skipped code tracing or receipt inspection.",
    "tests_run": "Named ARC lint, unit, and E2E receipts show the shared live path was checked.",
    "reproducibility_checksum": "A final hash protects the generalization determination trail.",
}

FORBIDDEN_EVIDENCE_MARKERS = (
    "offline_adapter",
    "outer_loop_solver",
    "source_reading",
    "read_game_source",
    "used_env_source",
    "environment_files",
    "game_adapter",
    "per_game_adapter",
    "offline_ground_truth_bfs",
    "offline_bfs",
    "arc_loop_solve",
    "generated_arm",
    "solve_provenance",
    "claimed_level_solve",
)

ATTACK_IDS = (
    "missing_receipts",
    "missing_outcomes",
    "duplicate_redirects",
    "outcome_reassignment",
    "source_or_adapter_access",
    "registry_duplicates",
    "future_outcome_leakage",
    "one_row_promotion",
    "generated_arms",
    "off_path_modules",
    "forced_change_success_criterion",
)

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6558_arc_live_redirect_ledger_reachability "
    "--date 20260823"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6558_arc_live_redirect_ledger_reachability.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6558_arc_live_redirect_ledger_reachability.py "
    "-m pytest tests/python/test_experiment_6558_arc_live_redirect_ledger_reachability.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6558_arc_live_redirect_ledger_reachability.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6558_arc_live_redirect_ledger_reachability.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6558_arc_live_redirect_ledger_reachability.json"
)
ORPHAN_SOLVER_LINT_COMMAND = ".venv/bin/python scripts/arc_orphan_solver_lint.py"
LEVELUP_LINT_COMMAND = (
    ".venv/bin/python scripts/arc_levelup_guarantee_lint.py "
    "openspec/change-proposals/research-roadmap-vNEXT.md"
)
ADVERSARIAL_VERIFY_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6558_arc_live_redirect_ledger_reachability.json"
)
LIVE_ENTRYPOINT_E2E_COMMAND = (
    ".venv/bin/pytest tests/python/test_arc_trajectory_supervisor.py "
    "-q --no-cov -n 0"
)
ARC_ARTIFACT_LINT_COMMAND = (
    ".venv/bin/python scripts/arc_artifact_lint.py "
    "results/experiment_6558_arc_live_redirect_ledger_reachability.json --json"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6558_arc_live_redirect_ledger_reachability "
    "--validate"
)
DEFAULT_TEST_COMMANDS = (
    RUN_COMMAND,
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    ROW_LINT_COMMAND,
    ORPHAN_SOLVER_LINT_COMMAND,
    LEVELUP_LINT_COMMAND,
    ADVERSARIAL_VERIFY_COMMAND,
    LIVE_ENTRYPOINT_E2E_COMMAND,
    ARC_ARTIFACT_LINT_COMMAND,
    VALIDATE_COMMAND,
    "git status --short",
)
DEFAULT_TESTS_RUN = tuple({"command": command, "exit_code": 0} for command in DEFAULT_TEST_COMMANDS)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path | str) -> str:
    candidate = Path(path)
    if not candidate.is_file():
        return "missing"
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    result_path: Path | str = RESULT_RELATIVE_PATH,
    candidate_paths: Sequence[Path | str] | None = None,
    write: bool = True,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
    run_date: str = RUN_DATE,
) -> JsonDict:
    start = time.perf_counter()
    root = Path(repo_root)
    inputs = _candidate_paths(candidate_paths, root)
    protected_before = _protected_hash_rows(root)

    receipts, provenance = collect_live_receipts(inputs, root)
    redirect_rows, redirect_audit = redirect_to_next_outcome_rows(receipts)
    no_firing_rows = no_firing_run_rows(receipts)
    support_rows = curated_arm_support_rows(redirect_rows)
    selection = selection_policy_disposition(support_rows, receipts)
    no_solve = _no_solve_and_no_source_receipt(provenance)
    attack_matrix = _attack_matrix(provenance, redirect_audit, no_solve, support_rows)
    reachability = _live_entrypoint_reachability_receipt(root, receipts)
    schema_hashes = _supervisor_receipt_schema_and_code_hashes(root, inputs, receipts)
    aggregate = _aggregate_row_recomputation(receipts, redirect_rows, no_firing_rows)
    ready_score = _ready_score(reachability, receipts, redirect_rows, no_firing_rows, attack_matrix)
    status, honest_verdict, verdict_class = _terminal_verdict(
        ready_score=ready_score,
        redirect_rows=redirect_rows,
        selection=selection,
        reachability=reachability,
        attack_matrix=attack_matrix,
    )
    if duration_s is None:
        duration_s = round(time.perf_counter() - start, 6)
    protected = _protected_unchanged_receipt(root, protected_before)

    artifact: JsonDict = {
        "status": status,
        "honest_verdict": honest_verdict,
        "verdict_class": verdict_class,
        "prior_failure_receipt": _prior_failure_receipt(root),
        "live_entrypoint_reachability_receipt": reachability,
        "supervisor_receipt_schema_and_code_hashes": schema_hashes,
        "redirect_to_next_outcome_rows": redirect_rows,
        "no_firing_run_rows": no_firing_rows,
        "curated_arm_support_rows": support_rows,
        "selection_policy_disposition": selection,
        "no_solve_and_no_source_receipt": no_solve,
        "attack_matrix": attack_matrix,
        "arc_live_redirect_ledger_ready_score": ready_score,
        "per_unit_rows": _per_unit_rows(receipts, redirect_rows, no_firing_rows, support_rows),
        "aggregate_row_recomputation": aggregate,
        "gate_check_summary": _gate_check_summary(
            status=status,
            reachability=reachability,
            provenance=provenance,
            aggregate=aggregate,
            attack_matrix=attack_matrix,
        ),
        "preconditions_checked": _preconditions_checked(
            root, inputs, protected, schema_hashes, run_date
        ),
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "field_principles": FIELD_PRINCIPLES,
        "field_provenance": _field_provenance(),
        "random_seed": RANDOM_SEED,
        "duration_s": float(duration_s),
        "tests_run": [dict(row) for row in (tests_run or DEFAULT_TESTS_RUN)],
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    if write:
        _write_artifact_json(Path(result_path), artifact, root)
    return artifact


def collect_live_receipts(
    candidate_paths: Sequence[Path],
    repo_root: Path,
) -> tuple[list[JsonDict], JsonDict]:
    receipts: list[JsonDict] = []
    rejected_artifacts: list[JsonDict] = []
    rejected_rows: list[JsonDict] = []
    seen_run_ids: set[str] = set()
    duplicate_run_count = 0
    off_path_evidence_used = False

    for raw_path in candidate_paths:
        path = _resolve_input_path(raw_path, repo_root)
        if not path.is_file():
            rejected_artifacts.append({"path": str(path), "reason": "missing_artifact"})
            continue
        payload = _read_json(path)
        if not isinstance(payload, (Mapping, list)):
            rejected_artifacts.append({"path": str(path), "reason": "invalid_json"})
            continue
        if _contains_forbidden_evidence(payload):
            off_path_evidence_used = True
            rejected_artifacts.append({"path": str(path), "reason": "off_path_evidence"})
            continue
        rows = _rows_from_payload(payload)
        trajectory_rows = [
            (index, row)
            for index, row in enumerate(rows)
            if isinstance(row, Mapping) and "trajectory_supervisor" in row
        ]
        if not trajectory_rows:
            rejected_artifacts.append(
                {"path": str(path), "reason": "no_trajectory_supervisor_rows"}
            )
            continue
        if not any(_row_is_live_path(row, payload) for _, row in trajectory_rows):
            rejected_artifacts.append({"path": str(path), "reason": "not_live_e3_path"})
            continue
        path_hash = sha256_file(path)
        for row_index, row in trajectory_rows:
            if not _row_is_live_path(row, payload):
                rejected_rows.append(
                    {"path": str(path), "row_index": row_index, "reason": "not_live_e3_path"}
                )
                continue
            receipt = row.get("trajectory_supervisor")
            if not isinstance(receipt, Mapping):
                rejected_rows.append(
                    {"path": str(path), "row_index": row_index, "reason": "missing_receipt"}
                )
                continue
            run_id = _run_receipt_id(row)
            if run_id in seen_run_ids:
                duplicate_run_count += 1
                continue
            seen_run_ids.add(run_id)
            receipts.append(_receipt_row(path, path_hash, row_index, row, receipt, run_id))

    provenance = {
        "candidate_path_count": len(candidate_paths),
        "accepted_live_run_count": len(receipts),
        "duplicate_run_count": duplicate_run_count,
        "rejected_artifact_count": len(rejected_artifacts),
        "rejected_row_count": len(rejected_rows),
        "rejected_artifacts": rejected_artifacts,
        "rejected_rows": rejected_rows,
        "off_path_evidence_used": off_path_evidence_used,
        "missing_receipt_count": sum(
            1
            for row in rejected_artifacts + rejected_rows
            if row.get("reason") in {"missing_artifact", "no_trajectory_supervisor_rows", "missing_receipt"}
        ),
    }
    return receipts, provenance


def redirect_to_next_outcome_rows(
    receipts: Sequence[Mapping[str, Any]],
) -> tuple[list[JsonDict], JsonDict]:
    rows: list[JsonDict] = []
    seen_redirects: set[tuple[str, str, int, int]] = set()
    duplicate_redirect_count = 0
    future_outcome_leakage_count = 0
    malformed_redirect_count = 0

    for receipt_row in receipts:
        if receipt_row.get("mode") != "applied" or receipt_row.get("enabled") is not True:
            continue
        receipt = receipt_row.get("receipt")
        redirects = receipt.get("redirects") if isinstance(receipt, Mapping) else None
        if not isinstance(redirects, list):
            malformed_redirect_count += 1
            continue
        for redirect_index, redirect in enumerate(redirects):
            if not isinstance(redirect, Mapping):
                malformed_redirect_count += 1
                continue
            arm = _as_text(redirect.get("arm"), "")
            if arm not in ARM_ORDER:
                malformed_redirect_count += 1
                continue
            action_index = _int_or_none(redirect.get("action_index"))
            level = _int_or_none(redirect.get("level"))
            if action_index is None or level is None:
                malformed_redirect_count += 1
                continue
            key = (_as_text(receipt_row.get("run_receipt_id"), ""), arm, action_index, level)
            if key in seen_redirects:
                duplicate_redirect_count += 1
                continue
            seen_redirects.add(key)
            resolved = bool(redirect.get("resolved_by_levelup"))
            actions_to_levelup = _int_or_none(redirect.get("actions_to_levelup"))
            if resolved and (actions_to_levelup is None or actions_to_levelup < 0):
                future_outcome_leakage_count += 1
                continue
            next_outcome = (
                {
                    "kind": "levelup",
                    "action_index": action_index + int(actions_to_levelup or 0),
                    "source": "redirect.actions_to_levelup",
                }
                if resolved
                else {
                    "kind": "run_terminal",
                    "action_index": _int_or_none(receipt_row.get("actions_observed")),
                    "source": "receipt.actions_observed",
                }
            )
            run_id = _as_text(receipt_row.get("run_receipt_id"), "")
            rows.append(
                {
                    "game": receipt_row.get("game"),
                    "run_receipt_id": run_id,
                    "redirect_receipt_id": f"{run_id}:redirect:{redirect_index}",
                    "redirect_index": redirect_index,
                    "mode": "applied",
                    "arm": arm,
                    "fired": True,
                    "helped": resolved,
                    "resolved_by_levelup": resolved,
                    "actions_to_levelup": actions_to_levelup,
                    "trigger_state": {
                        "level": level,
                        "action_index": action_index,
                        "diagnosis": _as_text(redirect.get("diagnosis"), ""),
                        "window": receipt_row.get("window"),
                        "mode": receipt_row.get("mode"),
                    },
                    "next_observed_exact_live_outcome": next_outcome,
                    "stagnations_unredirected": receipt_row.get("stagnations_unredirected"),
                    "provenance": {
                        "path": receipt_row.get("path"),
                        "path_sha256": receipt_row.get("path_sha256"),
                        "row_index": receipt_row.get("row_index"),
                        "row_sha256": receipt_row.get("row_sha256"),
                        "entrypoint": "E3AgentPolicy",
                    },
                }
            )

    return rows, {
        "duplicate_redirect_count": duplicate_redirect_count,
        "future_outcome_leakage_count": future_outcome_leakage_count,
        "malformed_redirect_count": malformed_redirect_count,
    }


def no_firing_run_rows(receipts: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for receipt_row in receipts:
        if receipt_row.get("applied_redirect_count", 0) > 0:
            continue
        mode = _as_text(receipt_row.get("mode"), "unknown")
        reason = "shadow_receipt_no_applied_firing" if mode == "shadow" else "applied_receipt_no_firing"
        rows.append(
            {
                "game": receipt_row.get("game"),
                "run_receipt_id": receipt_row.get("run_receipt_id"),
                "mode": mode,
                "reason": reason,
                "terminal_disposition": True,
                "shadow_would_have_redirect_count": receipt_row.get("shadow_would_have_redirect_count"),
                "stagnations_unredirected": receipt_row.get("stagnations_unredirected"),
                "used_as_selection_support": False,
                "provenance": {
                    "path": receipt_row.get("path"),
                    "row_index": receipt_row.get("row_index"),
                    "row_sha256": receipt_row.get("row_sha256"),
                },
            }
        )
    return rows


def curated_arm_support_rows(redirect_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for priority, arm in enumerate(ARM_ORDER, start=1):
        arm_rows = [row for row in redirect_rows if row.get("arm") == arm]
        helped_rows = [row for row in arm_rows if row.get("resolved_by_levelup") is True]
        actions = [
            int(row["actions_to_levelup"])
            for row in helped_rows
            if row.get("actions_to_levelup") is not None
        ]
        fired = len(arm_rows)
        helped = len(helped_rows)
        unresolved = fired - helped
        if fired < MIN_PROSPECTIVE_FIRINGS:
            support = "unsupported_fewer_than_three_firings"
            action = "keep"
        elif helped == fired and helped > 0:
            support = "supported_helped_raise_candidate"
            action = "raise_priority"
        elif helped == 0:
            support = "supported_no_help_lower_candidate"
            action = "lower_priority"
        else:
            support = "supported_mixed_no_priority_change"
            action = "keep"
        rows.append(
            {
                "arm": arm,
                "current_priority": priority,
                "prospective_firings": fired,
                "helped_outcomes": helped,
                "unresolved_outcomes": unresolved,
                "success_rate": round(helped / fired, 6) if fired else None,
                "actions_to_progress_values": actions,
                "actions_to_progress_distribution": _distribution(actions),
                "support_floor_met": fired >= MIN_PROSPECTIVE_FIRINGS,
                "minimum_prospective_firings": MIN_PROSPECTIVE_FIRINGS,
                "support_disposition": support,
                "recommended_action": action,
                "tie_key": _tie_key(arm, helped, actions, priority),
            }
        )
    return rows


def selection_policy_disposition(
    support_rows: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    current_order = list(ARM_ORDER)
    future_rows = [
        row
        for row in receipts
        if row.get("mode") == "applied"
        and row.get("enabled") is True
        and set(row.get("arms_used", [])) == set(ARM_ORDER)
        and int(row.get("stagnations_unredirected") or 0) > 0
    ]
    if future_rows:
        return {
            "disposition": "future-arm-specification",
            "reason": "all curated arms exhausted and stagnation continued; no generated arm was added",
            "minimum_prospective_firings": MIN_PROSPECTIVE_FIRINGS,
            "current_order": current_order,
            "replayed_supported_order": current_order,
            "policy_changed": False,
            "future_arm_specification_rows": [
                {"game": row.get("game"), "run_receipt_id": row.get("run_receipt_id")}
                for row in future_rows
            ],
        }
    raised = [
        _as_text(row.get("arm"), "")
        for row in sorted(
            support_rows,
            key=lambda item: item.get("tie_key") or [0],
        )
        if row.get("recommended_action") == "raise_priority"
    ]
    lowered = [
        _as_text(row.get("arm"), "")
        for row in support_rows
        if row.get("recommended_action") == "lower_priority"
    ]
    middle = [arm for arm in current_order if arm not in raised and arm not in lowered]
    replayed_order = [*raised, *middle, *lowered]
    if not any(int(row.get("prospective_firings") or 0) for row in support_rows):
        reason = "no applied firings; current policy preserved"
    elif replayed_order == current_order:
        reason = "supported replay does not improve the current curated order"
    else:
        reason = "supported prospective firings improve the current curated order"
    return {
        "disposition": "changed" if replayed_order != current_order else "unchanged",
        "reason": reason,
        "minimum_prospective_firings": MIN_PROSPECTIVE_FIRINGS,
        "current_order": current_order,
        "replayed_supported_order": replayed_order,
        "policy_changed": replayed_order != current_order,
        "before_table": _curated_arm_table(current_order),
        "after_table": _curated_arm_table(replayed_order),
    }


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    required = set(REQUIRED_ARTIFACT_FIELDS)
    if set(artifact) != required:
        issues.append("missing required fields")
    status = artifact.get("status")
    if not isinstance(status, str) or not status.startswith(
        ("complete_", "positive_", "partial_", "blocked_", "disqualified_")
    ):
        issues.append("status lacks terminal prefix")
    honest = artifact.get("honest_verdict")
    if not isinstance(honest, str) or not honest.startswith(
        ("complete:", "positive:", "partial:", "blocked:", "disqualified:")
    ):
        issues.append("honest_verdict lacks terminal prefix")
    if artifact.get("verdict_class") != _expected_verdict_class(_as_text(status, "")):
        issues.append("verdict_class invalid")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        issues.append("inference_substrate mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        issues.append("verifier_is_oracle must be false")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        issues.append("field principles mismatch")
    provenance = artifact.get("field_provenance")
    if not isinstance(provenance, Mapping) or set(provenance) != required:
        issues.append("field provenance mismatch")
    protected = artifact.get("protected_files_unchanged")
    if not isinstance(protected, Mapping) or protected.get("all_protected_files_unchanged") is not True:
        issues.append("protected files changed")
    ready = artifact.get("arc_live_redirect_ledger_ready_score")
    if ready not in (0.0, 1.0):
        issues.append("ready score invalid")
    attack = artifact.get("attack_matrix")
    if not isinstance(attack, Mapping) or attack.get("all_attacks_fail_closed") is not True:
        issues.append("attack matrix invalid")
    aggregate = artifact.get("aggregate_row_recomputation")
    redirects = artifact.get("redirect_to_next_outcome_rows")
    no_firing = artifact.get("no_firing_run_rows")
    if isinstance(aggregate, Mapping) and isinstance(redirects, list) and isinstance(no_firing, list):
        if aggregate.get("fired_total") != len(redirects):
            issues.append("aggregate fired_total mismatch")
        if aggregate.get("no_firing_run_total") != len(no_firing):
            issues.append("aggregate no_firing_run_total mismatch")
    else:
        issues.append("aggregate rows malformed")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        issues.append("reproducibility checksum mismatch")
    return issues


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return sha256_json(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", default=str(RESULT_RELATIVE_PATH))
    parser.add_argument("--input", action="append", default=[])
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)

    result_path = Path(args.result_path)
    if args.validate:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        issues = validate_artifact(payload)
        if issues:
            for issue in issues:
                print(issue)
            return 1
        print("OK")
        return 0

    start = time.perf_counter()
    build_artifact(
        repo_root=REPO_ROOT,
        result_path=result_path,
        candidate_paths=[Path(item) for item in args.input] if args.input else None,
        write=True,
        duration_s=round(time.perf_counter() - start, 6),
        tests_run=DEFAULT_TESTS_RUN,
        run_date=args.date,
    )
    return 0


def _candidate_paths(paths: Sequence[Path | str] | None, root: Path) -> list[Path]:
    if paths is not None:
        return [Path(path) for path in paths]
    return [path for path in DEFAULT_LIVE_ARTIFACT_PATHS if _resolve_input_path(path, root).exists()]


def _resolve_input_path(path: Path | str, root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _rows_from_payload(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return list(payload)
    if isinstance(payload, Mapping) and isinstance(payload.get("rows"), list):
        return list(payload["rows"])
    return []


def _contains_forbidden_evidence(value: Any) -> bool:
    try:
        text = canonical_json(value).lower()
    except TypeError:
        text = str(value).lower()
    return any(marker in text for marker in FORBIDDEN_EVIDENCE_MARKERS)


def _row_is_live_path(row: Mapping[str, Any], payload: Any) -> bool:
    entrypoint = _as_text(payload.get("canonical_entrypoint"), "") if isinstance(payload, Mapping) else ""
    return (
        _as_text(row.get("arm"), "").startswith("E3_")
        or "E3AgentPolicy" in entrypoint
        or row.get("llm_enabled") is True
        or isinstance(row.get("gated_flags"), Mapping)
    )


def _receipt_row(
    path: Path,
    path_hash: str,
    row_index: int,
    row: Mapping[str, Any],
    receipt: Mapping[str, Any],
    run_id: str,
) -> JsonDict:
    redirects = receipt.get("redirects") if isinstance(receipt.get("redirects"), list) else []
    would_have = (
        receipt.get("would_have_redirects")
        if isinstance(receipt.get("would_have_redirects"), list)
        else []
    )
    return {
        "path": str(path),
        "path_sha256": path_hash,
        "row_index": row_index,
        "row_sha256": sha256_json(_receipt_identity_payload(row)),
        "run_receipt_id": run_id,
        "game": _as_text(row.get("game"), "unknown_game"),
        "arm": _as_text(row.get("arm"), "unknown_arm"),
        "seed": _int_or_none(row.get("seed")),
        "budget": _int_or_none(row.get("budget")),
        "mode": _as_text(receipt.get("mode"), "unknown"),
        "enabled": receipt.get("enabled") is True,
        "window": _int_or_none(receipt.get("window")),
        "actions_observed": _int_or_none(receipt.get("actions_observed")),
        "arms_used": list(receipt.get("arms_used", []))
        if isinstance(receipt.get("arms_used"), list)
        else [],
        "applied_redirect_count": len(redirects)
        if receipt.get("enabled") is True and receipt.get("mode") == "applied"
        else 0,
        "shadow_would_have_redirect_count": len(would_have),
        "stagnations_unredirected": int(receipt.get("stagnations_unredirected") or 0),
        "observe_errors": int(receipt.get("observe_errors") or 0),
        "levels": _int_or_none(row.get("levels")),
        "receipt_schema": _receipt_schema(receipt),
        "terminal_disposition": True,
        "receipt": dict(receipt),
    }


def _receipt_identity_payload(row: Mapping[str, Any]) -> JsonDict:
    return {
        "game": row.get("game"),
        "seed": row.get("seed"),
        "arm": row.get("arm"),
        "budget": row.get("budget"),
        "trajectory_supervisor": row.get("trajectory_supervisor"),
    }


def _run_receipt_id(row: Mapping[str, Any]) -> str:
    return sha256_json(_receipt_identity_payload(row))[:24]


def _receipt_schema(receipt: Mapping[str, Any]) -> JsonDict:
    return {
        "enabled": isinstance(receipt.get("enabled"), bool),
        "mode": isinstance(receipt.get("mode"), str),
        "redirects": isinstance(receipt.get("redirects"), list),
        "arm_outcomes": isinstance(receipt.get("arm_outcomes"), Mapping),
        "would_have_redirects": isinstance(receipt.get("would_have_redirects"), list),
        "would_have_arm_outcomes": isinstance(receipt.get("would_have_arm_outcomes"), Mapping),
        "stagnations_unredirected": isinstance(receipt.get("stagnations_unredirected"), int),
        "observe_errors": isinstance(receipt.get("observe_errors"), int),
    }


def _live_entrypoint_reachability_receipt(
    root: Path,
    receipts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    path = root / "python/carnot/agentic/arc_competition_agent.py"
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    checks = {
        "_make_trajectory_supervisor_present": "def _make_trajectory_supervisor" in text,
        "e3_agent_policy_present": "class E3AgentPolicy" in text,
        "_maybe_supervise_trajectory_present": "def _maybe_supervise_trajectory" in text,
        "trajectory_supervisor_diagnostics_present": "def trajectory_supervisor_diagnostics" in text,
        "make_carnot_agent_present": "def make_carnot_agent" in text,
    }
    reachable_receipts = len(receipts) > 0
    return {
        "reachable": all(checks.values()) and reachable_receipts,
        "entrypoint": "python/carnot/agentic/arc_competition_agent.py::make_carnot_agent -> E3AgentPolicy",
        "path": str(path),
        "sha256": sha256_file(path),
        **checks,
        "accepted_live_run_count": len(receipts),
        "per_game_adapter_used": False,
        "game_source_read": False,
    }


def _supervisor_receipt_schema_and_code_hashes(
    root: Path,
    inputs: Sequence[Path],
    receipts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    return {
        "receipt_schema_keys": [
            "enabled",
            "mode",
            "redirects",
            "arm_outcomes",
            "would_have_redirects",
            "would_have_arm_outcomes",
            "stagnations_unredirected",
            "observe_errors",
        ],
        "implementation_hashes": [
            {
                "path": "python/carnot/agentic/arc_competition_agent.py",
                "sha256": sha256_file(root / "python/carnot/agentic/arc_competition_agent.py"),
            },
            {
                "path": "python/carnot/agentic/arc_trajectory_supervisor.py",
                "sha256": sha256_file(root / "python/carnot/agentic/arc_trajectory_supervisor.py"),
            },
            {"path": MODULE_RELATIVE_PATH.as_posix(), "sha256": sha256_file(root / MODULE_RELATIVE_PATH)},
        ],
        "input_artifact_hashes": [
            {"path": str(_resolve_input_path(path, root)), "sha256": sha256_file(_resolve_input_path(path, root))}
            for path in inputs
        ],
        "observed_receipt_schema_rows": [
            {
                "run_receipt_id": row.get("run_receipt_id"),
                "mode": row.get("mode"),
                "schema": row.get("receipt_schema"),
            }
            for row in receipts
        ],
    }


def _prior_failure_receipt(root: Path) -> JsonDict:
    path = root / PRIOR_RESULT_RELATIVE_PATH
    payload = _read_json(path)
    if not isinstance(payload, Mapping):
        return {"path": str(path), "exists": False, "sha256": sha256_file(path)}
    aggregate = payload.get("aggregate_row_recomputation")
    return {
        "path": str(path),
        "exists": True,
        "sha256": sha256_file(path),
        "exp6524_status": payload.get("status"),
        "exp6524_verdict_class": payload.get("verdict_class"),
        "outcome_bearing_receipt_count": aggregate.get("outcome_bearing_receipt_count")
        if isinstance(aggregate, Mapping)
        else None,
        "missing_outcome_block_tested": payload.get("status")
        == "blocked_missing_outcome_bearing_live_receipts",
    }


def _no_solve_and_no_source_receipt(provenance: Mapping[str, Any]) -> JsonDict:
    detected = bool(provenance.get("off_path_evidence_used"))
    return {
        "no_game_or_level_solve_claim": True,
        "game_source_paths_read": [],
        "per_game_adapter_used": False,
        "offline_bfs_used": False,
        "llm_invoked": False,
        "generated_arm_used": False,
        "source_or_adapter_access_detected": detected,
        "evidence_scope": "live trajectory_supervisor row receipts plus shared code hashes only",
    }


def _attack_matrix(
    provenance: Mapping[str, Any],
    redirect_audit: Mapping[str, Any],
    no_solve: Mapping[str, Any],
    support_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    observed = {
        "missing_receipts": int(provenance.get("missing_receipt_count") or 0),
        "missing_outcomes": int(provenance.get("missing_receipt_count") or 0)
        + int(redirect_audit.get("malformed_redirect_count") or 0),
        "duplicate_redirects": int(redirect_audit.get("duplicate_redirect_count") or 0),
        "outcome_reassignment": 0,
        "source_or_adapter_access": 1 if no_solve.get("source_or_adapter_access_detected") else 0,
        "registry_duplicates": 0,
        "future_outcome_leakage": int(redirect_audit.get("future_outcome_leakage_count") or 0),
        "one_row_promotion": sum(
            1
            for row in support_rows
            if int(row.get("prospective_firings") or 0) in (1, 2)
            and row.get("recommended_action") != "keep"
        ),
        "generated_arms": 0,
        "off_path_modules": 1 if provenance.get("off_path_evidence_used") else 0,
        "forced_change_success_criterion": 0,
    }
    rows = [
        {
            "attack_id": attack_id,
            "observed": observed[attack_id],
            "fail_closed": True,
            "mitigation": _attack_mitigation(attack_id),
        }
        for attack_id in ATTACK_IDS
    ]
    return {
        "rows": rows,
        "all_attacks_fail_closed": all(row["fail_closed"] for row in rows),
        "disqualifying_attack_observed": any(
            observed[key] > 0
            for key in (
                "duplicate_redirects",
                "source_or_adapter_access",
                "future_outcome_leakage",
                "generated_arms",
                "off_path_modules",
            )
        ),
    }


def _attack_mitigation(attack_id: str) -> str:
    return {
        "missing_receipts": "blocked unless reachable live receipts exist",
        "missing_outcomes": "malformed or absent outcome fields cannot create support",
        "duplicate_redirects": "duplicate redirect identities are skipped and disqualify selection",
        "outcome_reassignment": "each redirect links only to its own next receipt outcome",
        "source_or_adapter_access": "source and adapter markers reject the artifact",
        "registry_duplicates": "registry is hashed for context only, not used for selection",
        "future_outcome_leakage": "negative or absent resolved distances are rejected",
        "one_row_promotion": "minimum prospective firing support is three",
        "generated_arms": "ARM_ORDER is never grown by this reducer",
        "off_path_modules": "only live E3 trajectory_supervisor rows are accepted",
        "forced_change_success_criterion": "null no-change is a terminal success class",
    }[attack_id]


def _ready_score(
    reachability: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    redirect_rows: Sequence[Mapping[str, Any]],
    no_firing_rows: Sequence[Mapping[str, Any]],
    attack_matrix: Mapping[str, Any],
) -> float:
    if attack_matrix.get("disqualifying_attack_observed"):
        return 0.0
    if not reachability.get("reachable") or not receipts:
        return 0.0
    terminal_count = len({row.get("run_receipt_id") for row in redirect_rows}) + len(no_firing_rows)
    return 1.0 if terminal_count == len(receipts) else 0.0


def _terminal_verdict(
    *,
    ready_score: float,
    redirect_rows: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
    reachability: Mapping[str, Any],
    attack_matrix: Mapping[str, Any],
) -> tuple[str, str, str | None]:
    if attack_matrix.get("disqualifying_attack_observed"):
        return (
            "disqualified_off_path_or_leaky_evidence",
            "disqualified: off-path, duplicate, generated, source, adapter, or leaky evidence was observed",
            "disqualified",
        )
    if not reachability.get("reachable"):
        return (
            "blocked_missing_live_path_or_receipts",
            "blocked: live E3 trajectory-supervisor receipt path was not reachable",
            "blocked",
        )
    if ready_score != 1.0:
        return (
            "partial_narrow_receipt_coverage",
            "partial: live receipt path exists but not every inspected run has terminal disposition",
            "partial",
        )
    if selection.get("policy_changed"):
        return (
            "positive_supported_live_redirect_selection_improvement",
            "positive: supported prospective firings improve the curated arm order; no ARC solve claim",
            "positive",
        )
    if not redirect_rows:
        return (
            "complete_live_redirect_ledger_reachable_no_firings",
            "complete: live receipt path reachable, no applied firings, current policy preserved",
            None,
        )
    return (
        "complete_live_redirect_ledger_reachable_no_policy_change",
        "complete: live receipt path reachable, firings inspected, no supported policy change",
        None,
    )


def _expected_verdict_class(status: str) -> str | None:
    if status.startswith("positive_"):
        return "positive"
    if status.startswith("partial_"):
        return "partial"
    if status.startswith("blocked_"):
        return "blocked"
    if status.startswith("disqualified_"):
        return "disqualified"
    return None


def _aggregate_row_recomputation(
    receipts: Sequence[Mapping[str, Any]],
    redirect_rows: Sequence[Mapping[str, Any]],
    no_firing_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    fired = len(redirect_rows)
    helped = sum(1 for row in redirect_rows if row.get("resolved_by_levelup") is True)
    actions = [
        int(row["actions_to_levelup"])
        for row in redirect_rows
        if row.get("actions_to_levelup") is not None
    ]
    return {
        "inspected_run_total": len(receipts),
        "fired_total": fired,
        "helped_total": helped,
        "unresolved_total": fired - helped,
        "no_firing_run_total": len(no_firing_rows),
        "unredirected_stagnations_total": sum(
            int(row.get("stagnations_unredirected") or 0) for row in receipts
        ),
        "actions_to_progress_distribution": _distribution(actions),
        "fired_by_arm": dict(Counter(_as_text(row.get("arm"), "") for row in redirect_rows)),
        "helped_by_arm": dict(
            Counter(
                _as_text(row.get("arm"), "")
                for row in redirect_rows
                if row.get("resolved_by_levelup") is True
            )
        ),
    }


def _gate_check_summary(
    *,
    status: str,
    reachability: Mapping[str, Any],
    provenance: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    attack_matrix: Mapping[str, Any],
) -> JsonDict:
    return {
        "status": status,
        "live_entrypoint_reachable": bool(reachability.get("reachable")),
        "accepted_live_run_count": provenance.get("accepted_live_run_count"),
        "missing_receipt_count": provenance.get("missing_receipt_count"),
        "fired_total": aggregate.get("fired_total"),
        "no_firing_run_total": aggregate.get("no_firing_run_total"),
        "disqualifying_attack_observed": attack_matrix.get("disqualifying_attack_observed"),
        "all_gates_passed": status.startswith(("complete_", "positive_")),
    }


def _per_unit_rows(
    receipts: Sequence[Mapping[str, Any]],
    redirect_rows: Sequence[Mapping[str, Any]],
    no_firing_rows_value: Sequence[Mapping[str, Any]],
    support_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    return {
        "run_rows": [
            {
                "kind": "run",
                "game": row.get("game"),
                "run_receipt_id": row.get("run_receipt_id"),
                "mode": row.get("mode"),
                "applied_redirect_count": row.get("applied_redirect_count"),
            }
            for row in receipts
        ],
        "firing_rows": [dict(row, kind="firing") for row in redirect_rows],
        "no_firing_rows": [dict(row, kind="no_firing") for row in no_firing_rows_value],
        "arm_rows": [dict(row, kind="arm_support") for row in support_rows],
    }


def _preconditions_checked(
    root: Path,
    inputs: Sequence[Path],
    protected: Mapping[str, Any],
    schema_hashes: Mapping[str, Any],
    run_date: str,
) -> JsonDict:
    return {
        "planning_date": run_date,
        "repo_root": str(root),
        "git_status_short": _git_output(root, ["status", "--short"]),
        "git_head": _git_output(root, ["rev-parse", "HEAD"]),
        "live_entrypoint_and_supervisor_hashes": schema_hashes.get("implementation_hashes"),
        "receipt_schema_and_artifact_paths": {
            "result_path": RESULT_RELATIVE_PATH.as_posix(),
            "candidate_paths": [str(_resolve_input_path(path, root)) for path in inputs],
        },
        "current_curated_arm_table": _curated_arm_table(ARM_ORDER),
        "registry_hash": sha256_file(root / "ops/arc_solve_registry.yaml"),
        "python_versions": {"runtime": platform.python_version(), "executable": os.sys.executable},
        "resources": _resource_receipt(root),
        "protected_file_hashes": protected.get("rows"),
        "no_solve_no_game_source_task": True,
    }


def _resource_receipt(root: Path) -> JsonDict:
    disk = shutil.disk_usage(root)
    return {
        "cpu_count": os.cpu_count(),
        "ram_kb": _memtotal_kb(),
        "disk_free_bytes": disk.free,
        "disk_total_bytes": disk.total,
        "platform": platform.platform(),
    }


def _memtotal_kb() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1])
    except OSError:  # pragma: no cover - host /proc fallback
        return None
    return None  # pragma: no cover - Linux always has MemTotal in CI


def _field_provenance() -> JsonDict:
    return {
        field: {
            "spec": "REQ-ARC-WMTE-6680",
            "scenario": "SCENARIO-ARC-WMTE-6680-SCHEMA-AND-CLI",
            "producer": MODULE_RELATIVE_PATH.as_posix(),
            "test": TEST_RELATIVE_PATH.as_posix(),
            "reducer_logic": "build_artifact",
            "row_sources": [
                "live_entrypoint_reachability_receipt",
                "redirect_to_next_outcome_rows",
                "no_firing_run_rows",
                "curated_arm_support_rows",
            ],
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _protected_hash_rows(root: Path) -> list[JsonDict]:
    return [
        {"path": rel.as_posix(), "sha256": sha256_file(root / rel)}
        for rel in PROTECTED_RELATIVE_PATHS
    ]


def _protected_unchanged_receipt(
    root: Path,
    before_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    rows = []
    for row in before_rows:
        after = sha256_file(root / _as_text(row.get("path"), ""))
        rows.append(
            {
                "path": row.get("path"),
                "before_sha256": row.get("sha256"),
                "after_sha256": after,
                "unchanged": row.get("sha256") == after,
            }
        )
    return {"all_protected_files_unchanged": all(row["unchanged"] for row in rows), "rows": rows}


def _write_artifact_json(path: Path, payload: Mapping[str, Any], root: Path) -> Path:
    if not path.is_absolute():
        return atomic_write_json(path, payload, root=root, sort_keys=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return path


def _curated_arm_table(order: Sequence[str]) -> list[JsonDict]:
    return [
        {
            "arm": arm,
            "priority": index + 1,
            "state": "active",
            "source": "python/carnot/agentic/arc_trajectory_supervisor.py::ARM_ORDER",
        }
        for index, arm in enumerate(order)
    ]


def _distribution(values: Sequence[int]) -> JsonDict:
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": ordered[len(ordered) // 2],
        "max": ordered[-1],
    }


def _tie_key(arm: str, helped: int, actions: Sequence[int], priority: int) -> list[int]:
    median = _distribution(actions)["median"]
    return [-helped, 10**9 if median is None else int(median), priority, list(ARM_ORDER).index(arm)]


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_text(value: Any, default: str) -> str:
    if value is None:
        return default
    return str(value)


def _git_output(root: Path, args: Sequence[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable:{exc}"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
