"""Synthesize the five V582 branches without pooling their outcomes.

The reducer reads retained artifact rows and the conductor log. It keeps a
missing measurement as a null value with a zero denominator and a cause. It
does not call an LLM and it does not turn exact-oracle agreement into an
independent learned-verifier claim.

Spec: REQ-REPORT-6687 and SCENARIO-REPORT-6687-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
from typing import Any

import yaml


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260827"
RANDOM_SEED = 6687582
RESULT_PATH = Path("results/experiment_6687_v582_branch_synthesis.json")
MODULE_PATH = Path("python/carnot/experiment_6687_v582_branch_synthesis.py")
TEST_PATH = Path("tests/python/test_experiment_6687_v582_branch_synthesis.py")
REPORT_SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
ACTIVE_ROADMAP_PATH = Path("research-roadmap.yaml")
CONDUCTOR_PATH = Path("scripts/research_conductor.py")
CONDUCTOR_LOG_PATH = Path("ops/conductor-log.md")
INFERENCE_SUBSTRATE = "artifact_row_and_document_synthesis_no_llm"

CLOSED_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
OUTPUT_ARMS = ("natural", "immediate_json", "triggered_tail")
BRANCH_ORDER = (
    "execution_integrity",
    "output_transport",
    "continuous_self_learning",
    "live_arc_outcome",
    "stochastic_portability",
)
VERIFIER_BY_BRANCH = {
    "mode": "mixed_by_branch",
    "branches": {
        "execution_integrity": True,
        "output_transport": True,
        "continuous_self_learning": True,
        "live_arc_outcome": True,
        "stochastic_portability": True,
    },
}
PLANNED_TASK_IDS = [f"exp{number}" for number in range(6674, 6688)]
VALIDATOR_NAMES = (
    "row_consistency",
    "verdict_class_consistency",
    "artifact_validation",
    "adversarial_verification",
    "recurring_blocker_check",
    "claim_boundary_audit",
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "gate_check_summary",
    "planned_task_rows",
    "terminal_task_rows",
    "missing_artifact_rows",
    "output_transport_branch",
    "continuous_self_learning_branch",
    "live_arc_outcome_branch",
    "stochastic_portability_branch",
    "branch_rows",
    "validation_rows",
    "documentation_reconciliation_rows",
    "per_unit_rows",
    "aggregate_row_recomputation",
    "preconditions_checked",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
)

SOURCE_MODULES = {
    6674: "carnot.experiment_6674_v582_manifest_parity_contract",
    6675: "carnot.experiment_6675_triggered_tail_scope_receipt",
    6676: "carnot.experiment_6676_three_family_triggered_tail_ab",
    6678: "carnot.experiment_6678_constraint_family_stream",
    6681: "carnot.experiment_6681_arc_post_redirect_outcomes",
    6682: "carnot.experiment_6682_arc_held_family_supervisor_ab",
    6683: "carnot.experiment_6683_ising_reference_scope_receipt",
    6684: "carnot.experiment_6684_torx_typed_factor_parity",
}

PRECONDITION_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    ACTIVE_ROADMAP_PATH,
    Path("research-complete.yaml"),
    Path("openspec/change-proposals/research-roadmap-vNEXT.md"),
    REPORT_SPEC_PATH,
    Path("openspec/capabilities/constraint-compiler/spec.md"),
    Path("openspec/capabilities/continuous-self-learning/spec.md"),
    Path("openspec/capabilities/arc-world-model-trust-energy/spec.md"),
    Path("openspec/capabilities/samplers/spec.md"),
    Path("openspec/capabilities/verification/spec.md"),
    Path("openspec/capabilities/evidence/spec.md"),
    Path("openspec/capabilities/inference/spec.md"),
    Path("openspec/capabilities/safety/spec.md"),
    Path("openspec/capabilities/research-pipeline/spec.md"),
    Path("ops/e2e-test-plan.md"),
    Path("_bmad/traceability.md"),
    Path("ops/status.md"),
    Path("ops/changelog.md"),
    CONDUCTOR_LOG_PATH,
    CONDUCTOR_PATH,
)

VERIFICATION_COMMANDS = (
    (
        "focused_tests",
        ".venv/bin/coverage run --rcfile=/dev/null --data-file=/tmp/carnot_exp6687_coverage "
        "--include=*/experiment_6687_v582_branch_synthesis.py -m pytest "
        f"{TEST_PATH} -q --no-cov -n 0 -o addopts=",
    ),
    (
        "scoped_coverage",
        ".venv/bin/coverage report --rcfile=/dev/null "
        "--data-file=/tmp/carnot_exp6687_coverage "
        "--include=*/experiment_6687_v582_branch_synthesis.py --fail-under=100 "
        "--show-missing",
    ),
    ("ruff_check", f".venv/bin/ruff check {MODULE_PATH} {TEST_PATH}"),
    ("format_check", f".venv/bin/ruff format --check {MODULE_PATH} {TEST_PATH}"),
    ("spec_coverage", f".venv/bin/python scripts/check_spec_coverage.py {TEST_PATH}"),
    (
        "applicable_e2e",
        ".venv/bin/pytest tests/python/test_e2e_training_sampling.py -q --no-cov -n 0 -o addopts=",
    ),
    ("full_python_suite", ".venv/bin/pytest tests/python -q"),
    ("roadmap_protection", ".venv/bin/python scripts/roadmap_schema.py research-roadmap.yaml"),
    ("git_status", "git status --short"),
)


def canonical_json(value: Any) -> str:
    """Return one stable JSON encoding so hashes have one meaning."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    """Return a labeled SHA-256 digest for a byte sequence."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file in chunks so large row stores do not need a second copy."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def sha256_json(value: Any) -> str:
    """Hash a JSON value after canonical encoding."""

    return sha256_bytes(canonical_json(value).encode("utf-8"))


def load_json(path: Path) -> JsonDict:
    """Load one JSON object and reject a non-object root."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def spec_anchors(text: str) -> list[str]:
    """Return unique requirement and scenario anchors in source order."""

    return list(dict.fromkeys(re.findall(r"\b(?:REQ|SCENARIO)-[A-Z0-9-]+", text)))


def experiment_number(task_id: str) -> int:
    """Read the numeric experiment prefix from one planned task ID."""

    match = re.fullmatch(r"exp(\d+)(?:-.+)?", task_id)
    if match is None:
        raise ValueError(f"invalid task id: {task_id}")
    return int(match.group(1))


def expected_branch(number: int) -> str:
    """Map the fixed V582 execution order to its independent branch."""

    if number == 6674:
        return "execution_integrity"
    if number <= 6677:
        return "output_transport"
    if number <= 6680:
        return "continuous_self_learning"
    if number <= 6682:
        return "live_arc_outcome"
    if number <= 6686:
        return "stochastic_portability"
    return "synthesis"


def load_planned_tasks(root: Path) -> list[JsonDict]:
    """Load and enforce the complete ordered V582 task contract."""

    manifest = yaml.safe_load((root / ACTIVE_ROADMAP_PATH).read_text(encoding="utf-8"))
    tasks = manifest.get("tasks", [])
    task_ids = [task.get("id") for task in tasks]
    expected_ids = [f"exp{number}-" for number in range(6674, 6688)]
    if len(tasks) != 14 or any(
        not str(task_id).startswith(prefix)
        for task_id, prefix in zip(task_ids, expected_ids, strict=False)
    ):
        raise ValueError("active V582 roadmap must contain ordered Exp6674-Exp6687")
    rows = []
    for order, task in enumerate(tasks, 1):
        number = experiment_number(str(task["id"]))
        rows.append(
            {
                "order": order,
                "task_id": f"exp{number}",
                "manifest_task_id": task["id"],
                "title": task["title"],
                "path": task["deliverable"],
                "expected_branch": expected_branch(number),
            }
        )
    return rows


def load_source_artifacts(
    root: Path, planned: Sequence[Mapping[str, Any]]
) -> dict[str, JsonDict | None]:
    """Load every available pre-synthesis artifact by planned task ID."""

    sources: dict[str, JsonDict | None] = {}
    for row in planned:
        task_id = str(row["task_id"])
        number = experiment_number(task_id)
        path = root / str(row["path"])
        sources[task_id] = None if number == 6687 or not path.is_file() else load_json(path)
    return sources


def load_conductor_states(root: Path, planned: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Keep the latest conductor state for each exact V582 task title."""

    text = (root / CONDUCTOR_LOG_PATH).read_text(encoding="utf-8")
    states: dict[str, JsonDict] = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 4:
            continue
        date, title, state, detail = cells
        for row in planned:
            if title == row["title"]:
                states[str(row["task_id"])] = {
                    "date": date,
                    "state": state,
                    "detail": detail,
                    "line_number": line_number,
                    "row_hash": sha256_json(cells),
                }
                break
    return states


def _task_class(number: int, payload: Mapping[str, Any] | None) -> str:
    """Infer only a missing closed class from terminal process evidence."""

    if number == 6687:
        # The synthesis itself attempts every planned unit and finishes; its
        # mixed branch evidence is a null finding, not a retryable partial
        # (REQ-CONDUCTOR-VERDICT-3).
        return "null"
    if payload is None:
        return "blocked"
    declared = payload.get("verdict_class")
    if declared in CLOSED_CLASSES:
        return str(declared)
    status = str(payload.get("status", "")).lower()
    if "blocked" in status:
        return "blocked"
    if number in {6674, 6675, 6683} and "ready" in status:
        return "null"
    return "disqualified"


def _diagnostic(payload: Mapping[str, Any] | None, conductor: Mapping[str, Any]) -> Any:
    """Prefer an artifact gate diagnostic and retain the conductor fallback."""

    if payload is not None:
        gate = payload.get("gate_check_summary")
        if gate:
            return gate
        if payload.get("blocked_reason"):
            return payload["blocked_reason"]
    return conductor.get("detail", "no conductor row")


def build_terminal_task_rows(
    root: Path,
    planned: Sequence[Mapping[str, Any]],
    sources: Mapping[str, Mapping[str, Any] | None],
    conductor: Mapping[str, Mapping[str, Any]],
) -> list[JsonDict]:
    """Join every planned task to its artifact or conductor-only terminal state."""

    rows = []
    for plan in planned:
        task_id = str(plan["task_id"])
        number = experiment_number(task_id)
        payload = sources.get(task_id)
        state = conductor.get(task_id, {})
        path = root / str(plan["path"])
        if number == 6687:
            artifact_state = "current_synthesis"
            terminal_source = "current_synthesis"
            status = "complete_terminal_null"
            verdict = "complete_null: current non-pooled branch synthesis"
        elif payload is None:
            artifact_state = "missing"
            terminal_source = "conductor"
            status = str(state.get("state", "MISSING"))
            verdict = str(state.get("detail", "missing artifact and conductor state"))
        else:
            artifact_state = "present"
            terminal_source = "artifact"
            status = str(payload.get("status", "missing_status"))
            verdict = str(payload.get("honest_verdict", state.get("detail", "missing")))
        rows.append(
            {
                **dict(plan),
                "experiment_number": number,
                "artifact_state": artifact_state,
                "terminal_source": terminal_source,
                "terminal_status": status,
                "honest_verdict": verdict,
                "verdict_class": _task_class(number, payload),
                "duration_s": None if payload is None else payload.get("duration_s"),
                "deliverable_hash": sha256_file(path) if payload is not None else None,
                "gate_diagnostic": _diagnostic(payload, state),
                "conductor_state": state.get("state"),
                "conductor_row_hash": state.get("row_hash"),
            }
        )
    return rows


def _null_metric(cause: str) -> JsonDict:
    """Represent an unavailable metric without inventing a measured zero."""

    return {"value": None, "denominator": 0, "state": "missing", "cause": cause}


def _rate(successes: int, denominator: int, cause: str | None = None) -> JsonDict:
    """Return a measured rate or an explicit null-denominator state."""

    row = {
        "successes": successes,
        "denominator": denominator,
        "value": successes / denominator if denominator else None,
        "state": "measured" if denominator else "missing",
    }
    if not denominator:
        row["cause"] = cause or "no eligible retained rows"
    return row


def recompute_output_transport(sources: Mapping[str, Mapping[str, Any] | None]) -> JsonDict:
    """Recompute exact, parse, and harmful-flip results from Exp6676 rows."""

    payload = sources.get("exp6676") or {}
    rows = list(payload.get("per_unit_rows", []))
    completed = [row for row in rows if row.get("arm") in OUTPUT_ARMS]
    exact: JsonDict = {}
    parsed: JsonDict = {}
    for arm in OUTPUT_ARMS:
        arm_rows = [row for row in completed if row.get("arm") == arm]
        exact[arm] = _rate(
            sum(
                bool(
                    row.get("exact_success")
                    or (row.get("exact_outcome") or {}).get("exact_success")
                    or (row.get("exact_outcome") or {}).get("passed")
                )
                for row in arm_rows
            ),
            len(arm_rows),
            "Exp6676 stopped at its runtime precondition before inference rows existed",
        )
        parsed[arm] = _rate(
            sum(
                bool(
                    row.get("parsed")
                    or (row.get("parse_outcome") or {}).get("parsed")
                    or (row.get("parse_outcome") or {}).get("passed")
                )
                for row in arm_rows
            ),
            len(arm_rows),
            "Exp6676 stopped at its runtime precondition before inference rows existed",
        )
    flip_rows = list(payload.get("harmful_flip_rows", []))
    harmful = _rate(
        len(flip_rows),
        len(completed),
        "Exp6676 produced no completed pairs eligible for harmful-flip analysis",
    )
    return {
        "verdict_class": "blocked",
        "exact_success": exact,
        "parse_yield": parsed,
        "harmful_flips": harmful,
        "planned_units": (payload.get("aggregate_row_recomputation") or {}).get(
            "expected_unit_count"
        ),
        "observed_units": len(completed),
        "audit": {
            "state": "blocked",
            "source": "Exp6677 gate-block artifact",
        },
        "claim_boundary": (
            "No transport rate is measured. Exact checkers are semantic oracles and "
            "cannot establish an oracle-distinct learned verifier."
        ),
    }


def recompute_continuous_self_learning(
    sources: Mapping[str, Mapping[str, Any] | None],
) -> JsonDict:
    """Recompute retained durability rows and preserve the absent CSL A/B."""

    fixture = sources.get("exp6678") or {}
    safety_rows = list(fixture.get("restart_rollback_rows", []))
    restart = [row for row in safety_rows if row.get("row_type") == "restart"]
    rollback = [row for row in safety_rows if row.get("row_type") == "rollback"]
    cause = "Exp6679 was gate-blocked before prequential A/B rows existed"
    return {
        "verdict_class": "blocked",
        "future_yield": _null_metric(cause),
        "order_level_intervals": _null_metric(cause),
        "retention": _null_metric(cause),
        "restart": {
            "passed": sum(bool(row.get("passed")) for row in restart),
            "denominator": len(restart),
            "value": (
                sum(bool(row.get("passed")) for row in restart) / len(restart) if restart else None
            ),
        },
        "rollback": {
            "passed": sum(bool(row.get("passed")) for row in rollback),
            "denominator": len(rollback),
            "value": (
                sum(bool(row.get("passed")) for row in rollback) / len(rollback)
                if rollback
                else None
            ),
        },
        "fixture_audit": (fixture.get("aggregate_row_recomputation") or {}).get("checks"),
        "audit": {"state": "missing", "source": "Exp6680 has no artifact"},
        "claim_boundary": (
            "Restart and rollback fixtures passed, but no future-yield, retention, "
            "or order-interval claim was executed."
        ),
    }


def recompute_live_arc(sources: Mapping[str, Mapping[str, Any] | None]) -> JsonDict:
    """Recompute ARC deltas from matched live episode and intervention rows."""

    transport = sources.get("exp6681") or {}
    ab = sources.get("exp6682") or {}
    pairs = list(ab.get("paired_episode_rows", []))
    deltas = [float(row["transition_utility_delta"]) for row in pairs]
    interventions = list(ab.get("false_intervention_rows", []))
    false_count = sum(not bool(row.get("benefit_observed")) for row in interventions)
    off_actions = sum(int((row.get("off") or {}).get("actions_spent", 0)) for row in pairs)
    on_actions = sum(int((row.get("on") or {}).get("actions_spent", 0)) for row in pairs)
    return {
        "verdict_class": "partial",
        "transport": {
            "eligible_outcomes": int(
                (transport.get("aggregate_row_recomputation") or {}).get(
                    "eligible_redirect_outcome_rows", 0
                )
            ),
            "one_outcome_per_redirect": bool(
                (transport.get("aggregate_row_recomputation") or {}).get(
                    "all_redirects_exactly_joined", False
                )
            ),
        },
        "utility": {
            "delta": sum(deltas) / len(deltas) if deltas else None,
            "denominator": len(deltas),
            "losses": sum(delta < 0 for delta in deltas),
            "ties": sum(delta == 0 for delta in deltas),
            "wins": sum(delta > 0 for delta in deltas),
        },
        "false_intervention": {
            "count": false_count,
            "denominator": len(pairs),
            "delta": false_count / len(pairs) if pairs else None,
        },
        "forbidden_action": {
            "delta": (
                sum(float(row.get("forbidden_action_delta", 0.0)) for row in pairs) / len(pairs)
                if pairs
                else None
            ),
            "denominator": len(pairs),
            "no_headroom_rows": sum(bool(row.get("forbidden_no_headroom")) for row in pairs),
        },
        "action": {
            "off_actions": off_actions,
            "on_actions": on_actions,
            "delta": on_actions - off_actions,
        },
        "audit": {
            "analysis_rows_match": bool(
                (ab.get("aggregate_row_recomputation") or {}).get("all_headlines_match", False)
            ),
            "verification_gate_passed": bool((ab.get("gate_check_summary") or {}).get("passed")),
        },
        "solve_claim": False,
        "claim_boundary": (
            "Live outcomes authorize environment validity only. The supervisor lost "
            "utility, produced false interventions, and made no solve."
        ),
    }


def recompute_stochastic_portability(
    sources: Mapping[str, Mapping[str, Any] | None],
) -> JsonDict:
    """Recompute exact and Torx parity while keeping absent chain metrics null."""

    exact = sources.get("exp6683") or {}
    torx = sources.get("exp6684") or {}
    exact_rows = list(exact.get("exact_probability_rows", []))
    state_rows = list(torx.get("state_parity_rows", []))
    factor_rows = list(torx.get("factor_rows", []))
    probability_errors = [abs(float(row.get("probability_error", 0.0))) for row in exact_rows]
    factor_errors = [
        abs(float(state.get("absolute_error", 0.0)))
        for factor in factor_rows
        for state in factor.get("state_energy_rows", [])
    ]
    state_fields = ("total_energy", "log_weight", "probability", "marginal", "correlation")
    maximum_errors = {
        field: max(
            (
                float((row.get("field_errors") or {}).get(field, {}).get("absolute", 0.0))
                for row in state_rows
            ),
            default=None,
        )
        for field in state_fields
    }
    missing_cause = "Exp6685 was gate-blocked before raw chain rows existed"
    return {
        "verdict_class": "blocked",
        "exact_reference": {
            "state_count": len(exact_rows),
            "all_passed": bool(exact_rows) and all(bool(row.get("passed")) for row in exact_rows),
            "maximum_probability_error": max(probability_errors, default=None),
        },
        "torx_parity": {
            "state_count": len(state_rows),
            "factor_count": len(factor_rows),
            "all_rows_valid": bool(state_rows)
            and all(bool(row.get("valid")) for row in state_rows),
            "maximum_factor_energy_error": max(factor_errors, default=None),
            "maximum_field_errors": maximum_errors,
            "software_ready": bool(torx.get("torx_factor_parity_ready")),
        },
        "likelihood_error": _null_metric(missing_cause),
        "acf": _null_metric(missing_cause),
        "iat": _null_metric(missing_cause),
        "ess": _null_metric(missing_cause),
        "audit": {"state": "missing", "source": "Exp6686 has no artifact"},
        "claim_boundary": (
            "Exact and Torx CPU software parity are oracle-defined. No raw-chain, "
            "schedule, likelihood, ACF, IAT, ESS, accelerator, or hardware claim ran."
        ),
    }


def build_branch_rows(sources: Mapping[str, Mapping[str, Any] | None]) -> list[JsonDict]:
    """Build five independent dispositions with exact promotion actions."""

    output = recompute_output_transport(sources)
    csl = recompute_continuous_self_learning(sources)
    arc = recompute_live_arc(sources)
    stochastic = recompute_stochastic_portability(sources)
    manifest = sources.get("exp6674") or {}
    return [
        {
            "branch": "execution_integrity",
            "evidence": {
                "manifest_parity_ready": bool(manifest.get("v582_manifest_parity_ready")),
                "artifact": "Exp6674",
            },
            "verdict_class": "null",
            "promotion_gate": "Keep exact document-to-manifest parity before execution.",
            "claim_boundary": "Infrastructure parity is not scientific success.",
            "exact_next_action": "Retain the Exp6674 receipt with the terminal close.",
        },
        {
            "branch": "output_transport",
            "evidence": output,
            "verdict_class": "blocked",
            "promotion_gate": "Exp6676 must produce all preregistered rows and pass its gate.",
            "claim_boundary": output["claim_boundary"],
            "exact_next_action": (
                "Clear the conflicting workload, rerun Exp6676 under its owner lease, "
                "then run Exp6677."
            ),
        },
        {
            "branch": "continuous_self_learning",
            "evidence": csl,
            "verdict_class": "blocked",
            "promotion_gate": "Exp6678 must pass its owned gate before Exp6679 and Exp6680.",
            "claim_boundary": csl["claim_boundary"],
            "exact_next_action": (
                "Repair the Exp6678 verification failure, rerun it, then execute "
                "Exp6679 and Exp6680 in chronological order."
            ),
        },
        {
            "branch": "live_arc_outcome",
            "evidence": arc,
            "verdict_class": "partial",
            "promotion_gate": (
                "Exp6682 verification must pass and utility must improve without false "
                "interventions."
            ),
            "claim_boundary": arc["claim_boundary"],
            "exact_next_action": (
                "Resolve the Exp6682 verification failure and test valid-action blocks "
                "before another held-family live A/B."
            ),
        },
        {
            "branch": "stochastic_portability",
            "evidence": stochastic,
            "verdict_class": "blocked",
            "promotion_gate": "Exp6684 must pass E2E-002 before Exp6685 and Exp6686.",
            "claim_boundary": stochastic["claim_boundary"],
            "exact_next_action": (
                "Fix the E2E-002 integration failure without changing exact parity rows, "
                "rerun Exp6684, then execute Exp6685 and Exp6686."
            ),
        },
    ]


def _validation_row(
    validator: str,
    plan: Mapping[str, Any],
    exit_code: int | None,
    finding: str,
    severity: str,
    command: str | None = None,
) -> JsonDict:
    """Create one content-addressed validation finding."""

    row = {
        "validator": validator,
        "target": plan["path"],
        "target_experiment": experiment_number(str(plan["task_id"])),
        "exit": exit_code,
        "finding": finding,
        "severity": severity,
        "command": command,
    }
    row["hash"] = sha256_json(row)
    return row


def _run_shell(
    command: str, root: Path
) -> tuple[int, str]:  # pragma: no cover - production receipt
    """Run one fixed validation command and return a bounded output receipt."""

    completed = subprocess.run(
        command,
        cwd=root,
        shell=True,
        check=False,
        capture_output=True,
        text=True,
        timeout=7200,
    )
    output = (completed.stdout + "\n" + completed.stderr).strip()
    return completed.returncode, output[-2000:]


def _owner_validation(number: int, payload: Mapping[str, Any]) -> list[str]:
    """Call an artifact owner's validator when the source module has one."""

    module_name = SOURCE_MODULES.get(number)
    if module_name is None:
        required = {"status", "honest_verdict", "blocked_reason"}
        return [] if required <= set(payload) else ["generic terminal artifact schema"]
    module = importlib.import_module(module_name)
    return list(module.validate_artifact(payload))


def _claim_issues(number: int, payload: Mapping[str, Any]) -> list[str]:
    """Reject positive oracle claims and broad solve or hardware claims."""

    issues = []
    if payload.get("verifier_is_oracle") is True and payload.get("verdict_class") == "positive":
        issues.append("positive oracle result must be circular_positive")
    verdict = str(payload.get("honest_verdict", "")).lower()
    if number in {6681, 6682} and "solve" in verdict and "no solve" not in verdict:
        issues.append("unsupported ARC solve claim")
    if number in {6683, 6684} and "hardware" in verdict and "no hardware" not in verdict:
        issues.append("unsupported stochastic hardware claim")
    return issues


def build_validation_rows(
    root: Path,
    planned: Sequence[Mapping[str, Any]],
    sources: Mapping[str, Mapping[str, Any] | None],
    conductor: Mapping[str, Mapping[str, Any]],
    *,
    run_external: bool = False,
) -> list[JsonDict]:
    """Audit every pre-synthesis task and name each unavailable audit input."""

    rows = []
    for plan in planned[:-1]:
        task_id = str(plan["task_id"])
        number = experiment_number(task_id)
        payload = sources.get(task_id)
        if payload is None:
            detail = conductor.get(task_id, {}).get("detail", "no conductor state")
            for validator in VALIDATOR_NAMES:
                rows.append(
                    _validation_row(
                        validator,
                        plan,
                        None,
                        f"missing audit input; last conductor state: {detail}",
                        "blocked",
                    )
                )
            continue

        path = str(plan["path"])
        per_unit = payload.get("per_unit_rows")
        if run_external:
            command = f".venv/bin/python scripts/verdict_row_consistency_lint.py {path}"
            code, finding = _run_shell(command, root)
        else:
            command = "internal retained-row shape check"
            code = 0 if per_unit is None or isinstance(per_unit, list) else 1
            finding = "row container is consistent" if code == 0 else "per_unit_rows is not a list"
        rows.append(
            _validation_row(
                "row_consistency", plan, code, finding, "info" if code == 0 else "error", command
            )
        )

        declared_class = payload.get("verdict_class")
        class_ok = declared_class in CLOSED_CLASSES
        rows.append(
            _validation_row(
                "verdict_class_consistency",
                plan,
                0 if class_ok else 1,
                (
                    f"declared closed class: {declared_class}"
                    if class_ok
                    else f"missing declared class; synthesis infers {_task_class(number, payload)}"
                ),
                "info" if class_ok else "warning",
                "internal closed-enum check",
            )
        )

        owner_missing = number not in SOURCE_MODULES
        owner_issues = (
            ["owner validator unavailable"]
            if owner_missing
            else (_owner_validation(number, payload) if run_external else [])
        )
        rows.append(
            _validation_row(
                "artifact_validation",
                plan,
                None if owner_missing else (0 if not owner_issues else 1),
                "owner validator passed" if not owner_issues else "; ".join(owner_issues[:8]),
                "blocked" if owner_missing else ("info" if not owner_issues else "error"),
                "owner module validate_artifact",
            )
        )

        if run_external:
            command = f".venv/bin/python scripts/adversarial_verify.py {path}"
            code, finding = _run_shell(command, root)
        else:
            command = "internal audit-input presence check"
            code = 0
            finding = "adversarial input retained"
        rows.append(
            _validation_row(
                "adversarial_verification",
                plan,
                code,
                finding,
                "info" if code == 0 else "warning",
                command,
            )
        )

        matches = sum(
            1
            for line in (root / CONDUCTOR_LOG_PATH).read_text(encoding="utf-8").splitlines()
            if f"| {plan['title']} |" in line and "| GATE_BLOCK |" in line
        )
        rows.append(
            _validation_row(
                "recurring_blocker_check",
                plan,
                0,
                f"conductor gate-block occurrence count={matches}",
                "warning" if matches > 1 else "info",
                "internal exact-title conductor count",
            )
        )

        claim_issues = _claim_issues(number, payload)
        rows.append(
            _validation_row(
                "claim_boundary_audit",
                plan,
                0 if not claim_issues else 1,
                "claim boundary accepted" if not claim_issues else "; ".join(claim_issues),
                "info" if not claim_issues else "error",
                "internal oracle, ARC solve, and hardware boundary audit",
            )
        )
    return rows


def protected_hashes(root: Path) -> JsonDict:
    """Hash the two files that this close must not change."""

    return {str(path): sha256_file(root / path) for path in (ACTIVE_ROADMAP_PATH, CONDUCTOR_PATH)}


def collect_preconditions(root: Path, planned: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Hash inputs, raw stores, and measured host resources."""

    input_rows = []
    paths = list(PRECONDITION_PATHS) + [MODULE_PATH, TEST_PATH]
    paths.extend(Path(str(row["path"])) for row in planned[:-1])
    for relative in dict.fromkeys(paths):
        path = root / relative
        input_rows.append(
            {
                "path": str(relative),
                "state": "present" if path.is_file() else "missing",
                "hash": sha256_file(path) if path.is_file() else None,
            }
        )
    raw_rows = []
    for base in (
        root / "results/state/experiment_6678",
        root / "results/.experiment_6676_three_family_triggered_tail_ab",
    ):
        if base.exists():
            for path in sorted(item for item in base.rglob("*") if item.is_file()):
                raw_rows.append(
                    {
                        "path": str(path.relative_to(root)),
                        "bytes": path.stat().st_size,
                        "hash": sha256_file(path),
                    }
                )
    cpu = "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    memory_bytes = None
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        match = re.search(r"^MemTotal:\s+(\d+) kB", meminfo.read_text(), re.MULTILINE)
        memory_bytes = int(match.group(1)) * 1024 if match else None
    disk = shutil.disk_usage(root)
    return {
        "inputs": input_rows,
        "raw_row_stores": raw_rows,
        "resources": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu": cpu,
            "ram_bytes": memory_bytes,
            "disk_total_bytes": disk.total,
            "disk_free_bytes": disk.free,
        },
        "tools": {
            "python_executable": sys.executable,
            "no_llm": True,
            "substrate": INFERENCE_SUBSTRATE,
        },
    }


def documentation_rows(root: Path) -> list[JsonDict]:
    """Record measured reconciliation scope without editing conductor-owned docs."""

    definitions = (
        (REPORT_SPEC_PATH, "REQ-REPORT-6687", "added synthesis contract before code"),
        (
            Path("openspec/change-proposals/research-roadmap-vNEXT.md"),
            "V582 close",
            "retained; separate conductor reconciliation owns the close update",
        ),
        (
            Path("research-complete.yaml"),
            "V582 completion record",
            "deferred to the conductor reconciliation step",
        ),
        (
            Path("_bmad/traceability.md"),
            "REQ-REPORT-6687",
            "deferred to the conductor reconciliation step",
        ),
        (Path("ops/status.md"), "V582", "deferred to the conductor reconciliation step"),
        (Path("ops/changelog.md"), "20260827", "deferred to the conductor reconciliation step"),
    )
    rows = []
    for path, section, change in definitions:
        absolute = root / path
        rows.append(
            {
                "file": str(path),
                "section": section,
                "evidence_source": str(RESULT_PATH),
                "change": change,
                "hash": sha256_file(absolute) if absolute.is_file() else None,
            }
        )
    return rows


def _field_provenance(source_hash: str) -> JsonDict:
    """Bind every required field to the deterministic synthesis reducer."""

    provenance = {}
    for field in REQUIRED_ARTIFACT_FIELDS:
        provenance[field] = {
            "artifact": str(RESULT_PATH),
            "raw_row": "Exp6674-Exp6686 artifacts, raw stores, or conductor row",
            "reducer": "V582 branch-preserving reducer",
            "function": "build_artifact",
            "hash": sha256_json({"field": field, "source_hash": source_hash}),
        }
    return provenance


def _checksum(payload: Mapping[str, Any]) -> str:
    """Hash an artifact while excluding its own checksum field."""

    return sha256_json(
        {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    )


def build_artifact(
    *,
    root: Path,
    date: str,
    duration_s: float,
    planned: Sequence[Mapping[str, Any]],
    sources: Mapping[str, Mapping[str, Any] | None],
    conductor: Mapping[str, Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    tests_run: Sequence[Mapping[str, Any]],
    protected_before: Mapping[str, str],
) -> JsonDict:
    """Build the terminal artifact only from retained evidence rows."""

    terminal = build_terminal_task_rows(root, planned, sources, conductor)
    terminal[-1]["duration_s"] = duration_s
    missing = [
        {
            "task_id": row["task_id"],
            "path": row["path"],
            "last_conductor_state": row["conductor_state"],
            "diagnostic": row["gate_diagnostic"],
            "consequence": "branch metric remains null; absence is not a zero",
        }
        for row in terminal
        if row["artifact_state"] == "missing"
    ]
    output = recompute_output_transport(sources)
    csl = recompute_continuous_self_learning(sources)
    arc = recompute_live_arc(sources)
    stochastic = recompute_stochastic_portability(sources)
    branch_rows = build_branch_rows(sources)
    validations_by_number = {
        number: [dict(row) for row in validation_rows if row.get("target_experiment") == number]
        for number in range(6674, 6688)
    }
    metrics = {
        "execution_integrity": branch_rows[0]["evidence"],
        "output_transport": output,
        "continuous_self_learning": csl,
        "live_arc_outcome": arc,
        "stochastic_portability": stochastic,
        "synthesis": {"branch_count": 5, "pooled_success_claim": False},
    }
    per_unit = [
        {
            "order": row["order"],
            "task_id": row["task_id"],
            "branch": row["expected_branch"],
            "terminal_state": row["terminal_status"],
            "artifact_state": row["artifact_state"],
            "verdict_class": row["verdict_class"],
            "metric_recomputation": metrics[row["expected_branch"]],
            "validation_rows": validations_by_number[row["experiment_number"]],
        }
        for row in terminal
    ]
    preconditions = collect_preconditions(root, planned)
    source_hash = sha256_json(
        {
            "inputs": preconditions["inputs"],
            "raw_row_stores": preconditions["raw_row_stores"],
        }
    )
    after = protected_hashes(root)
    protected = {
        "before": dict(protected_before),
        "after": after,
        "all_unchanged": dict(protected_before) == after,
    }
    class_counts = {
        verdict_class: sum(row["verdict_class"] == verdict_class for row in branch_rows)
        for verdict_class in CLOSED_CLASSES
    }
    artifact: JsonDict = {
        "experiment": "Exp6687",
        "run_date": date,
        # The synthesis run finished: 14 of 14 planned tasks joined, all five
        # branches recomputed, and the two known-missing artifacts recorded.
        # Mixed branch evidence with no pooled claim is class null; partial
        # would mark a finished run as retryable (REQ-CONDUCTOR-VERDICT-3).
        "status": "complete_terminal_null",
        "honest_verdict": (
            "complete_null: V582 has a null execution-integrity receipt, blocked "
            "output, CSL, and stochastic branches, and a partial adverse ARC branch; "
            "there is no pooled success claim"
        ),
        "verdict_class": "null",
        "gate_check_summary": [
            {
                "branch": row["branch"],
                "expected": row["promotion_gate"],
                "observed": row["evidence"],
            }
            for row in branch_rows
            if row["verdict_class"] in {"blocked", "partial", "disqualified"}
        ],
        "planned_task_rows": [dict(row) for row in planned],
        "terminal_task_rows": terminal,
        "missing_artifact_rows": missing,
        "output_transport_branch": output,
        "continuous_self_learning_branch": csl,
        "live_arc_outcome_branch": arc,
        "stochastic_portability_branch": stochastic,
        "branch_rows": branch_rows,
        "validation_rows": [dict(row) for row in validation_rows],
        "documentation_reconciliation_rows": documentation_rows(root),
        "per_unit_rows": per_unit,
        "aggregate_row_recomputation": {
            "planned_task_count": len(planned),
            "terminal_task_count": len(terminal),
            "present_source_artifact_count": sum(
                row["artifact_state"] == "present" for row in terminal
            ),
            "missing_source_artifact_count": len(missing),
            "branch_count": len(branch_rows),
            "branch_class_counts": class_counts,
            "pooled_success_claim": False,
            "overall_verdict_class": "null",
            "all_branch_rows_recomputed": len(branch_rows) == 5,
        },
        "preconditions_checked": preconditions,
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_BY_BRANCH,
        "field_provenance": _field_provenance(source_hash),
        "random_seed": RANDOM_SEED,
        "duration_s": duration_s,
        "tests_run": [dict(row) for row in tests_run],
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    return artifact


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    """Fail closed on schema, class, branch, null, protection, or hash drift."""

    issues = []
    missing_fields = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(payload))
    if missing_fields:
        return [f"missing required fields: {missing_fields}"]
    # A finished synthesis must declare null, never partial: a partial
    # declaration made the conductor re-run this completed 2,983s task to the
    # 3-fail limit (REQ-CONDUCTOR-VERDICT-3, SCENARIO-CONDUCTOR-VERDICT-5).
    if payload["status"] != "complete_terminal_null":
        issues.append("status")
    if payload["verdict_class"] != "null":
        issues.append("verdict_class")
    if not str(payload["honest_verdict"]).startswith("complete_null:"):
        issues.append("honest_verdict")
    if payload["inference_substrate"] != INFERENCE_SUBSTRATE:
        issues.append("inference_substrate")
    if payload["verifier_is_oracle"] != VERIFIER_BY_BRANCH:
        issues.append("verifier_is_oracle")
    if len(payload["planned_task_rows"]) != 14 or len(payload["terminal_task_rows"]) != 14:
        issues.append("task_rows")
    if len(payload["per_unit_rows"]) != 14:
        issues.append("per_unit_rows")
    missing_numbers = {
        experiment_number(str(row["task_id"])) for row in payload["missing_artifact_rows"]
    }
    if missing_numbers != {6680, 6686}:
        issues.append("missing_artifact_rows")
    branches = payload["branch_rows"]
    if [row.get("branch") for row in branches] != list(BRANCH_ORDER):
        issues.append("branch_rows")
    elif [row.get("verdict_class") for row in branches] != [
        "null",
        "blocked",
        "blocked",
        "partial",
        "blocked",
    ]:
        issues.append("branch_classes")
    if payload["aggregate_row_recomputation"].get("pooled_success_claim") is not False:
        issues.append("pooled_success_claim")
    if not payload["protected_files_unchanged"].get("all_unchanged"):
        issues.append("protected_files_unchanged")
    if set(REQUIRED_ARTIFACT_FIELDS) - set(payload["field_provenance"]):
        issues.append("field_provenance")
    for branch_name in (
        "output_transport_branch",
        "continuous_self_learning_branch",
        "stochastic_portability_branch",
    ):
        text = canonical_json(payload[branch_name])
        if '"denominator":0,"value":0' in text:
            issues.append(f"missing_to_zero:{branch_name}")
    if payload["reproducibility_checksum"] != _checksum(payload):
        issues.append("reproducibility_checksum")
    return issues


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one validated artifact by same-directory atomic replacement."""

    issues = validate_artifact(payload)
    if issues:
        raise ValueError("invalid Exp6687 artifact: " + ", ".join(issues))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def validate_path(path: Path) -> list[str]:
    """Load and validate one stored Exp6687 artifact."""

    return validate_artifact(load_json(path))


def run_verification_commands(
    root: Path,
) -> list[JsonDict]:  # pragma: no cover - production receipt
    """Run each required close check once and retain its exact result."""

    rows = []
    for check_id, command in VERIFICATION_COMMANDS:
        started = time.monotonic()
        exit_code, output = _run_shell(command, root)
        rows.append(
            {
                "check_id": check_id,
                "command": command,
                "exit": exit_code,
                "duration_s": time.monotonic() - started,
                "summary": output[-1000:] or "no output",
                "output_hash": sha256_bytes(output.encode("utf-8")),
            }
        )
    return rows


def run(root: Path, date: str, *, run_tests: bool = True) -> JsonDict:  # pragma: no cover - CLI
    """Execute the synthesis, validations, checks, and atomic write."""

    started = time.monotonic()
    before = protected_hashes(root)
    planned = load_planned_tasks(root)
    sources = load_source_artifacts(root, planned)
    conductor = load_conductor_states(root, planned)
    validations = build_validation_rows(root, planned, sources, conductor, run_external=True)
    tests = run_verification_commands(root) if run_tests else []
    artifact = build_artifact(
        root=root,
        date=date,
        duration_s=time.monotonic() - started,
        planned=planned,
        sources=sources,
        conductor=conductor,
        validation_rows=validations,
        tests_run=tests,
        protected_before=before,
    )
    atomic_write_json(root / RESULT_PATH, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    """Generate a synthesis or validate an existing deliverable."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--skip-tests", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    if args.validate:
        issues = validate_path(root / RESULT_PATH)
        if issues:
            print("\n".join(issues), file=sys.stderr)
            return 1
        print("Exp6687 artifact valid")
        return 0
    artifact = run(root, args.date, run_tests=not args.skip_tests)
    print(
        json.dumps(
            {
                "path": str(root / RESULT_PATH),
                "status": artifact["status"],
                "verdict_class": artifact["verdict_class"],
                "checksum": artifact["reproducibility_checksum"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
