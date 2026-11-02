"""Independently audit V666 evidence and retained learning.

The audit reads raw rows when they exist. It does not turn a conductor pre-gate
receipt or an absent producer into a scientific null.

Spec refs: REQ-REPORT-7638 and SCENARIO-REPORT-7638-*.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from importlib import metadata
import json
import math
import os
from pathlib import Path
import socket
import sys
import tempfile
import time
from typing import Any

from carnot.experiment_7596_v663_evidence_audit import (
    ZERO_INVOCATION_COUNTS,
    authenticate_source_receipt,
    blocked_summary,
    canonical_hash,
    check_row,
    load_json,
    sha256_file,
    source_receipt,
)
from carnot.experiment_7610_v664_evidence_audit import SourceSpec, _interval
from carnot import experiment_7624_v665_evidence_audit as v665
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "exp7638-v666-evidence-audit"
MILESTONE = "2026.09.666"
SCHEMA = "carnot.exp7638.v666.evidence_audit.v1"
RESULT_PATH = Path("results/experiment_7638_v666_evidence_audit.json")
RAW_DIR = Path("results/raw/experiment_7638_v666_evidence_audit")
MODULE_PATH = Path("python/carnot/experiment_7638_v666_evidence_audit.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7638_v666_evidence_audit.py")
TEST_PATH = Path("tests/python/test_experiment_7638_v666_evidence_audit.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
MODEL_SPECS: list[JsonDict] = []
HISTORICAL_MODEL_IDENTITY = v665.HISTORICAL_MODEL_IDENTITY
GATE_OPERAND_FIELDS = v665.GATE_OPERAND_FIELDS
AFFECTED_CHECK_NAMES = validation_scope.REQUIRED_CHECK_NAMES
TERMINAL_CHECK_NAMES = (
    "fresh_process_cold_replay",
    "independent_raw_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)

SOURCE_SPECS = (
    SourceSpec(
        "exp7632-fit-evidence",
        Path("results/experiment_7632_v666_fit_evidence.json"),
        Path("results/experiment_7632_fit_evidence.json"),
    ),
    SourceSpec(
        "exp7633-online-evidence",
        Path("results/experiment_7633_v666_online_evidence.json"),
        Path("results/experiment_7633_online_evidence.json"),
    ),
    SourceSpec(
        "exp7634-evaluation-evidence",
        Path("results/experiment_7634_v666_evaluation_evidence.json"),
        Path("results/experiment_7634_evaluation_evidence.json"),
    ),
    SourceSpec(
        "exp7635-evidence-energy",
        Path("results/experiment_7635_v666_evidence_energy.json"),
        Path("results/experiment_7635_evidence_energy.json"),
    ),
    SourceSpec(
        "exp7636-decision-evaluation",
        Path("results/experiment_7636_v666_decision_evaluation.json"),
        Path("results/experiment_7636_decision_evaluation.json"),
    ),
    SourceSpec(
        "exp7637-guarded-learning",
        Path("results/experiment_7637_v666_guarded_learning.json"),
        Path("results/experiment_7637_guarded_learning.json"),
    ),
)

NAMED_INPUTS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7624_v665_evidence_audit.py"),
    Path("results/experiment_7630_v666_cuda_ownership.json"),
    Path("results/experiment_7631_v666_schema_pilot.json"),
    Path("scripts/adversarial_verify.py"),
    Path("scripts/verdict_row_consistency_lint.py"),
    SPEC_PATH,
)


def classify_source(root: Path, spec: SourceSpec) -> JsonDict:
    """Keep terminal blocked producers distinct from conductor receipts."""

    return v665.classify_source(root, spec)


def _log_loss(probability: float, label: int) -> float:
    clipped = min(1.0 - 1e-12, max(1e-12, probability))
    return -(label * math.log(clipped) + (1 - label) * math.log1p(-clipped))


def reduce_static_evidence(
    rows: Sequence[Mapping[str, Any]],
    *,
    checkpoint_bytes: str,
    checkpoint_sha256: str,
    role_map: Mapping[str, str],
    expected_attempted_rows: int,
    bootstrap_draws: int,
    bootstrap_seed: int,
) -> JsonDict:
    """Rebuild all static metrics from rows instead of producer aggregates."""

    if len(rows) != expected_attempted_rows:
        raise ValueError("attempted_denominator_mismatch")
    reduced = v665.reduce_static_branch(
        rows,
        checkpoint_bytes=checkpoint_bytes,
        checkpoint_sha256=checkpoint_sha256,
        role_map=role_map,
        bootstrap_draws=bootstrap_draws,
        bootstrap_seed=bootstrap_seed,
    )
    grouped: dict[str, dict[str, JsonDict]] = defaultdict(dict)
    for row in reduced["rows"]:
        metrics = row["absolute_metrics"]
        label = int(metrics["label"])
        probability = float(metrics["probability"])
        log_loss = _log_loss(probability, label)
        false_accept = int(bool(metrics["false_accept"]))
        row["absolute_metrics"]["log_loss"] = log_loss
        row["log_loss"] = log_loss
        row["raw_numerators"] = {
            "brier": float(row["brier"]),
            "log_loss": log_loss,
            "cost": float(metrics["realized_cost"]),
            "false_accept": false_accept,
        }
        grouped[str(row["unit_id"])][str(row["arm"])] = row

    arm_metrics: dict[str, JsonDict] = {}
    for arm in v665.STATIC_ARMS:
        selected = [arms[arm] for arms in grouped.values()]
        count = len(selected)
        brier = sum(float(row["brier"]) for row in selected)
        log_loss = sum(float(row["log_loss"]) for row in selected)
        cost = sum(float(row["absolute_metrics"]["realized_cost"]) for row in selected)
        false_accept = sum(int(bool(row["absolute_metrics"]["false_accept"])) for row in selected)
        arm_metrics[arm] = {
            "brier_numerator": brier,
            "brier_denominator": count,
            "mean_brier": brier / count,
            "log_loss_numerator": log_loss,
            "log_loss_denominator": count,
            "mean_log_loss": log_loss / count,
            "cost_numerator": cost,
            "cost_denominator": count,
            "mean_cost": cost / count,
            "false_accept_numerator": false_accept,
            "false_accept_denominator": count,
        }

    differences: dict[str, JsonDict] = {}
    for offset, comparator in enumerate(("erased", "deranged")):
        per_metric: dict[str, JsonDict] = {}
        for metric_offset, (name, path) in enumerate(
            (
                ("brier", ("brier",)),
                ("log_loss", ("log_loss",)),
                ("cost", ("absolute_metrics", "realized_cost")),
                ("false_accept", ("absolute_metrics", "false_accept")),
            )
        ):
            deltas: list[float] = []
            for arms in grouped.values():
                factual: Any = arms["factual"]
                control: Any = arms[comparator]
                for key in path:
                    factual = factual[key]
                    control = control[key]
                deltas.append(float(control) - float(factual))
            per_metric[name] = _interval(
                deltas,
                draws=bootstrap_draws,
                seed=bootstrap_seed + offset * 10 + metric_offset,
            )
        differences[comparator] = per_metric
    reduced.update(
        attempted_row_count=len(rows),
        arm_metrics=arm_metrics,
        control_differences=differences,
    )
    return reduced


def reduce_learning_events(
    rows: Sequence[Mapping[str, Any]],
    *,
    checkpoint_bytes: str,
    checkpoint_sha256: str,
    role_map: Mapping[str, str],
    expected_attempted_rows: int,
) -> JsonDict:
    """Replay label release, gradient, admission, checkpoint, and retention events."""

    if len(rows) != expected_attempted_rows:
        raise ValueError("attempted_denominator_mismatch")
    base = v665.reduce_learning_branch(
        rows,
        checkpoint_bytes=checkpoint_bytes,
        checkpoint_sha256=checkpoint_sha256,
        role_map=role_map,
    )
    checkpoint = v665._checkpoint(checkpoint_bytes, checkpoint_sha256, "learning")
    initial = [float(value) for value in checkpoint["initial_weights"]]
    initial_hash = canonical_hash([round(value, 15) for value in initial])
    admission_ids: set[str] = set()
    output: list[JsonDict] = []
    for raw in rows:
        if raw.get("feedback_origin_index") != raw.get("prediction_index"):
            raise ValueError("future_label_shuffle")
        if raw.get("gradient_label_role") != "update":
            raise ValueError("gradient_role_invalid")
        if raw.get("admission_used_for_gradient") is not False:
            raise ValueError("admission_label_leak")
        if raw.get("evaluation_used_for_gradient") is not False:
            raise ValueError("evaluation_label_leak")
        admission_id = str(raw.get("admission_example_id") or "")
        if not admission_id or admission_id in admission_ids:
            raise ValueError("admission_reuse")
        admission_ids.add(admission_id)
        if raw.get("checkpoint_before_sha256") != initial_hash:
            raise ValueError("checkpoint_before_mismatch")
        accepted = raw.get("accepted_update") is True
        probability = float(raw["prediction_probability"])
        label = int(raw["feedback_label"])
        features = [float(value) for value in raw["features"]]
        expected_gradient = (
            [(probability - label) * feature for feature in features]
            if accepted
            else [0.0 for _feature in features]
        )
        gradient = [float(value) for value in raw.get("gradient") or []]
        if len(gradient) != len(expected_gradient) or any(
            not math.isclose(actual, expected, abs_tol=1e-12)
            for actual, expected in zip(gradient, expected_gradient)
        ):
            raise ValueError("gradient_mismatch")
        after_hash = str(raw.get("checkpoint_after_sha256") or "")
        if after_hash != raw.get("state_after_sha256"):
            raise ValueError("checkpoint_after_mismatch")
        changed = after_hash != initial_hash
        if raw.get("checkpoint_reused") is not False or (accepted and not changed):
            raise ValueError("checkpoint_reuse")
        retention_loss = float(raw["retention_loss"])
        output.append(
            {
                "unit_id": str(raw["group_id"]),
                "arm": str(raw["arm"]),
                "absolute_metrics": {
                    "prediction_probability": probability,
                    "prequential_brier": (probability - label) ** 2,
                    "prequential_log_loss": _log_loss(probability, label),
                    "retention_loss": retention_loss,
                },
                "gradient": gradient,
                "gradient_label_role": "update",
                "admission_example_id": admission_id,
                "admission_used_for_gradient": False,
                "evaluation_used_for_gradient": False,
                "checkpoint_before_sha256": initial_hash,
                "checkpoint_after_sha256": after_hash,
                "checkpoint_changed": changed,
                "prediction_index": int(raw["prediction_index"]),
                "release_index": int(raw["release_index"]),
                "update_index": int(raw["update_index"]),
                "raw_numerator": (probability - label) ** 2,
                "raw_denominator": 1,
                "seed": int(raw["seed"]),
                "direction": "lower_prequential_and_retention_loss_are_better",
                "censored": bool(raw.get("censored", False)),
                "censoring": str(raw.get("censoring") or "none"),
                "raw_provenance": str(
                    raw.get("raw_provenance") or f"fixture:{raw['group_id']}:{raw['arm']}"
                ),
            }
        )
    base.update(
        attempted_row_count=len(rows),
        rows=output,
        future_labels_used=False,
        admission_labels_used_for_gradient=False,
        evaluation_labels_used_for_gradient=False,
        checkpoint_changes_reconstructed=True,
        repeated_orders_multiply_samples=False,
    )
    return base


def private_fixture() -> JsonDict:
    """Extend the prior compact fixture with V666 event custody operands."""

    fixture = v665.private_fixture()
    for index, row in enumerate(fixture["static_rows"]):
        row.update(attempted=True, malformed=index == 0)
    checkpoint = json.loads(fixture["learning_checkpoint_bytes"])
    initial = [float(value) for value in checkpoint["initial_weights"]]
    initial_hash = canonical_hash([round(value, 15) for value in initial])
    for index, row in enumerate(fixture["learning_rows"]):
        probability = float(row["prediction_probability"])
        label = int(row["feedback_label"])
        accepted = row["accepted_update"] is True
        gradient = (
            [(probability - label) * float(feature) for feature in row["features"]]
            if accepted
            else [0.0 for _feature in row["features"]]
        )
        row.update(
            attempted=True,
            feedback_origin_index=row["prediction_index"],
            gradient_label_role="update",
            gradient=gradient,
            admission_example_id=f"admission-{index}",
            admission_used_for_gradient=False,
            evaluation_used_for_gradient=False,
            checkpoint_before_sha256=initial_hash,
            checkpoint_after_sha256=row["state_after_sha256"],
            checkpoint_reused=False,
            censored=False,
            censoring="none",
            raw_provenance=f"fixture:{row['group_id']}:{row['arm']}",
        )
    fixture["expected_static_attempted_rows"] = len(fixture["static_rows"])
    fixture["expected_learning_attempted_rows"] = len(fixture["learning_rows"])
    return fixture


MUTATIONS = (
    "shuffled_future_labels",
    "duplicate_groups",
    "dropped_malformed_rows",
    "unchanged_evidence_controls",
    "swapped_fit_evaluation_roles",
    "checkpoint_reuse",
)
MUTATION_FAILURES = {
    "shuffled_future_labels": "future_label_shuffle",
    "duplicate_groups": "duplicated_group",
    "dropped_malformed_rows": "attempted_denominator_mismatch",
    "unchanged_evidence_controls": "identical_control",
    "swapped_fit_evaluation_roles": "corrupt_group_id",
    "checkpoint_reuse": "checkpoint_reuse",
}


def mutate_private_fixture(value: JsonDict, mutation: str) -> str:
    """Change one registered operand so the independent reader must reject it."""

    if mutation == "shuffled_future_labels":
        value["learning_rows"][0]["feedback_origin_index"] = 99
        return "learning_rows[0].feedback_origin_index"
    if mutation == "duplicate_groups":
        value["static_rows"].append(deepcopy(value["static_rows"][0]))
        value["expected_static_attempted_rows"] += 1
        return "static_rows[duplicate]"
    if mutation == "dropped_malformed_rows":
        index = next(i for i, row in enumerate(value["static_rows"]) if row["malformed"])
        value["static_rows"].pop(index)
        return "static_rows[malformed_removed]"
    if mutation == "unchanged_evidence_controls":
        factual = next(
            row
            for row in value["static_rows"]
            if row["group_id"] == "eval-0" and row["arm"] == "factual"
        )
        erased = next(
            row
            for row in value["static_rows"]
            if row["group_id"] == "eval-0" and row["arm"] == "erased"
        )
        erased["features"] = deepcopy(factual["features"])
        return "static_rows[eval-0,erased].features"
    if mutation == "swapped_fit_evaluation_roles":
        value["role_map"]["eval-0"] = "fit"
        return "role_map.eval-0"
    if mutation == "checkpoint_reuse":
        guarded = next(row for row in value["learning_rows"] if row["arm"] == "guarded")
        guarded["checkpoint_reused"] = True
        return "learning_rows[guarded].checkpoint_reused"
    raise ValueError(f"unknown_mutation:{mutation}")


def validate_private_fixture(value: Mapping[str, Any]) -> list[str]:
    """Return every static or learning rejection from one changed fixture."""

    errors: list[str] = []
    try:
        reduce_static_evidence(
            value.get("static_rows") or [],
            checkpoint_bytes=str(value.get("static_checkpoint_bytes") or ""),
            checkpoint_sha256=str(value.get("static_checkpoint_sha256") or ""),
            role_map=value.get("role_map") or {},
            expected_attempted_rows=int(value.get("expected_static_attempted_rows", -1)),
            bootstrap_draws=32,
            bootstrap_seed=7638001,
        )
    except ValueError as exc:
        errors.extend(str(exc).split(";"))
    try:
        reduce_learning_events(
            value.get("learning_rows") or [],
            checkpoint_bytes=str(value.get("learning_checkpoint_bytes") or ""),
            checkpoint_sha256=str(value.get("learning_checkpoint_sha256") or ""),
            role_map=value.get("role_map") or {},
            expected_attempted_rows=int(value.get("expected_learning_attempted_rows", -1)),
        )
    except ValueError as exc:
        errors.extend(str(exc).split(";"))
    return list(dict.fromkeys(errors))


def run_private_mutations() -> list[JsonDict]:
    """Bind every corruption to changed bytes and the registered rejection."""

    receipts: list[JsonDict] = []
    for mutation in MUTATIONS:
        fixture = private_fixture()
        before = canonical_hash(fixture)
        changed_path = mutate_private_fixture(fixture, mutation)
        after = canonical_hash(fixture)
        failures = validate_private_fixture(fixture)
        expected = MUTATION_FAILURES[mutation]
        receipts.append(
            {
                "mutation": mutation,
                "changed_private_path": changed_path,
                "before_sha256": before,
                "after_sha256": after,
                "expected_failure": expected,
                "observed_failures": failures,
                "passed": before != after and expected in failures,
                "tolerances_weakened": False,
                "corrupted_fixture_published": False,
            }
        )
    return receipts


REQUIRED_PRINCIPLE_FIELDS = (
    *v665.REQUIRED_PRINCIPLE_FIELDS,
    "planned_MODEL_SPECS",
)


def field_principles() -> dict[str, str]:
    """State why each required artifact field must remain present."""

    principles = v665.field_principles()
    principles.update(
        {
            "acceptance_gate_results": (
                "Validity, readiness, probability benefit, utility, retention, and freshness "
                "remain separate."
            ),
            "rows": "Raw independent units control claims; absent rows cannot be invented.",
            "planned_MODEL_SPECS": "No-call work keeps the planned current model roster empty.",
            "mutation_rows": "All six registered V666 corruptions must fail without looser tolerances.",
            "branch_dispositions": "Static and retained-learning availability remain independent.",
        }
    )
    return principles


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    """Hash stable evidence while excluding measured clocks and this digest."""

    excluded = {"reproducibility_checksum", "duration_s", "phase_spans"}
    return canonical_hash(
        {key: deepcopy(item) for key, item in value.items() if key not in excluded}
    )


def _gate(
    check: str,
    category: str,
    condition: str,
    expected: Any,
    observed: Any,
    passed: bool,
) -> JsonDict:
    return {
        "check": check,
        "category": category,
        "condition": condition,
        "operator": "eq",
        "expected": expected,
        "observed": observed,
        "passed": passed,
        "principle": (
            "Validity, readiness, probability benefit, utility, retention, and freshness "
            "must not substitute for one another."
        ),
    }


def collect_preconditions(root: Path) -> tuple[list[JsonDict], list[JsonDict]]:
    """Authenticate task inputs, process ownership, and all V666 source paths."""

    root = root.resolve()
    checks: list[JsonDict] = [
        check_row(
            "absolute_repository",
            "worktree",
            str(root),
            "is_absolute",
            True,
            root.is_absolute(),
            "eq",
        ),
        check_row(
            "owned_process",
            "current_process",
            f"/proc/{os.getpid()}",
            "owned_pid_exists",
            True,
            Path(f"/proc/{os.getpid()}").is_dir(),
            "eq",
        ),
    ]
    receipts: list[JsonDict] = []
    for relative in NAMED_INPUTS:
        exists = (root / relative).is_file()
        checks.append(
            check_row(
                "required_named_input",
                "worktree",
                relative.as_posix(),
                "exists",
                True,
                exists,
                "eq",
            )
        )
        if exists:
            receipt = source_receipt(root / relative, root, f"named-input:{relative.as_posix()}")
            receipt.update(disposition="authenticated_instruction", eligible_for_science=False)
            receipts.append(receipt)
    spec_text = (root / SPEC_PATH).read_text(encoding="utf-8")
    checks.append(
        check_row(
            "matching_requirement",
            "openspec",
            SPEC_PATH.as_posix(),
            "REQ-REPORT-7638",
            True,
            "REQ-REPORT-7638" in spec_text,
            "eq",
        )
    )
    for tool in ("pytest", "ruff", "mypy"):
        version = metadata.version(tool)
        checks.append(
            check_row(
                "declared_tool_version",
                "worktree-toolchain",
                str(root / ".venv/bin" / tool),
                "version_nonempty",
                True,
                bool(version),
                "eq",
            )
            | {"observed_version": version}
        )
    producer_receipts = [classify_source(root, spec) for spec in SOURCE_SPECS]
    receipts.extend(producer_receipts)
    for spec, receipt in zip(SOURCE_SPECS, producer_receipts):
        checks.append(
            check_row(
                "required_scientific_producer",
                spec.upstream,
                str(receipt["producer_path"]),
                "exists_and_eligible",
                True,
                receipt["eligible_for_science"],
                "eq",
            )
        )
    return checks, receipts


def provisional_validation_receipts(root: Path) -> list[JsonDict]:
    """Provide complete receipt placeholders only for a private candidate."""

    return [
        {
            "name": name,
            "command": f"pending exact candidate {name}",
            "command_argv": ["pending", name],
            "scope": "exact_candidate" if name in TERMINAL_CHECK_NAMES else "changed_files",
            "worktree": str(root.resolve()),
            "exit_code": 0,
            "duration_s": 0.0,
            "log_path": "pending",
            "log_sha256": "sha256:" + "0" * 64,
            "passed": True,
            "timed_out": False,
        }
        for name in (*AFFECTED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)
    ]


def build_blocked_artifact(
    root: Path,
    checks: Sequence[Mapping[str, Any]],
    sources: Sequence[Mapping[str, Any]],
    *,
    validation_receipts: Sequence[Mapping[str, Any]],
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
    run_date: str,
) -> JsonDict:
    """Build a terminal external block without inventing scientific rows."""

    failures = [row for row in checks if row.get("passed") is not True]
    if not failures:
        raise ValueError("blocked_artifact_requires_external_failure")
    source_rows = [deepcopy(dict(row)) for row in sources]
    source_rows.append(
        {
            "upstream": EXPERIMENT_ID,
            "path": RESULT_PATH.as_posix(),
            "producer_path": RESULT_PATH.as_posix(),
            "conductor_path": None,
            "sha256": None,
            "bytes": None,
            "honest_verdict": None,
            "verdict_class": None,
            "flagged_adversarial": None,
            "disposition": "planned_output_not_a_precondition",
            "eligible_for_science": False,
        }
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": 7638,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": run_date,
        "worktree_root": str(root.resolve()),
        "honest_verdict": "complete_blocked_v666_scientific_producers_unavailable",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "gate_check_summary": blocked_summary(failures),
        "preconditions_checked": [deepcopy(dict(row)) for row in checks],
        "acceptance_gate_results": [
            _gate(
                "custody_validity",
                "validity",
                "present source bytes authenticate",
                True,
                True,
                True,
            ),
            _gate(
                "raw_branch_readiness",
                "readiness",
                "all six scientific producers are eligible",
                True,
                False,
                False,
            ),
            _gate(
                "probability_benefit",
                "probability_benefit",
                "registered Brier comparisons pass",
                True,
                None,
                False,
            ),
            _gate(
                "decision_utility",
                "utility",
                "cost, coverage, and false-accept gates pass",
                True,
                None,
                False,
            ),
            _gate(
                "retained_learning",
                "retention",
                "prequential and retention gates pass",
                True,
                None,
                False,
            ),
            _gate(
                "confirmatory_freshness",
                "freshness",
                "source roles remain unexposed",
                True,
                False,
                False,
            ),
        ],
        "rows": [],
        "sample_size_budget": [
            {
                "branch": branch,
                "independent_unit": "source_group",
                "intended": intended,
                "observed": 0,
                "excluded": 0,
                "censored": 0,
                "missing_external": intended,
                "arms_orders_views_seeds_multiply_samples": False,
            }
            for branch, intended in (
                ("fit_tune_policy", 120),
                ("online", 80),
                ("evaluation", 40),
                ("retention", 40),
            )
        ],
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "planned_inference_substrate_class": "aggregation",
        "MODEL_SPECS": [],
        "model_specs": [],
        "planned_MODEL_SPECS": [],
        "model_invoked": False,
        "historical_model_identity": HISTORICAL_MODEL_IDENTITY,
        "execution_venue": "host",
        "execution_venue_details": {
            "hostname": socket.gethostname(),
            "physical_device": "cpu",
            "gpu_uuid": None,
            "owned_pid": os.getpid(),
            "historical_gpu_evidence_used_as_current_execution": False,
        },
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "historical_invocation_counts": {
            "availability": "unavailable_missing_scientific_producers",
            "counts_inherited_as_current": False,
        },
        "duration_s": float(duration_s),
        "random_seed": {
            "static_bootstrap": 7638001,
            "learning_bootstrap": 7638002,
            "private_mutations": 7638003,
        },
        "source_artifact_hashes": source_rows,
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "terminal_reader_outcomes": {
            str(row.get("name")): bool(row.get("passed"))
            for row in validation_receipts
            if row.get("name") in TERMINAL_CHECK_NAMES
        },
        "verifier_is_oracle": False,
        "oracle_fixture_verdict_class": "circular_positive",
        "protocol_readiness_verdict_class": "null",
        "field_principles": field_principles(),
        "static_audit_eligible_score": 0,
        "learning_audit_eligible_score": 0,
        "audited_evidence_benefit_score": None,
        "audited_learning_benefit_score": None,
        "branch_dispositions": [
            {
                "branch": "static_probability_and_utility",
                "validity": "blocked_missing_raw_evaluation_rows_and_checkpoint",
                "readiness": "not_established",
                "probability_benefit": None,
                "utility_benefit": None,
                "unavailable_reason": "Experiments 7635 and 7636 did not produce eligible outputs.",
                "semantic_null": False,
                "historically_exposed": True,
                "scientific_hypothesis_retired": False,
            },
            {
                "branch": "online_learning_and_retention",
                "validity": "blocked_missing_online_events_and_retention_rows",
                "readiness": "not_established",
                "prequential_benefit": None,
                "retention_benefit": None,
                "joint_benefit": None,
                "unavailable_reason": "Experiment 7637 did not produce eligible event rows.",
                "semantic_null": False,
                "historically_exposed": True,
                "scientific_hypothesis_retired": False,
            },
        ],
        "mutation_rows": run_private_mutations(),
        "fresh_confirmatory_claim_allowed": False,
        "positive_claim": False,
        "external_publication_authorized": False,
        "submission_authorized": False,
        "purchase_authorized": False,
        "default_promotion_authorized": False,
        "generator_weights_immutable": True,
        "production_defaults_changed": False,
        "research_conductor_modified": False,
        "research_roadmap_modified": False,
        "applicable_numbered_e2e": [],
        "capability_e2e": "fresh_process_raw_to_claim_and_six_negative_mutations",
        "prior_failure_disposition": {
            "prior_experiment": "exp7624-evidence-audit",
            "prior_verdict": "complete_blocked_v665_scientific_producers_unavailable",
            "retirement_condition_met": True,
            "retired_scope": "repeat evidence audit before an eligible branch produces raw rows",
            "scientific_hypotheses_retired": False,
            "action": "close this unavailable branch without retrying unchanged external custody",
        },
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _receipts_pass(receipts: Sequence[Mapping[str, Any]]) -> bool:
    expected = {*AFFECTED_CHECK_NAMES, *TERMINAL_CHECK_NAMES}
    by_name = {str(row.get("name")): row for row in receipts}
    return set(by_name) == expected and all(
        row.get("exit_code") == 0
        and row.get("passed") is True
        and row.get("timed_out") is not True
        and str(row.get("log_sha256") or "").startswith("sha256:")
        for row in by_name.values()
    )


def validate_artifact(value: Mapping[str, Any], *, root: Path = REPO_ROOT) -> JsonDict:
    """Reject custody drift or unavailable evidence presented as benefit."""

    errors: list[str] = []
    if value.get("schema") != SCHEMA or value.get("experiment_id") != EXPERIMENT_ID:
        errors.append("identity_mismatch")
    if value.get("milestone") != MILESTONE or value.get("run_date") != "20260925":
        errors.append("run_identity_mismatch")
    if not str(value.get("honest_verdict") or "").startswith("complete_"):
        errors.append("terminal_prefix_missing")
    if value.get("verdict_class") != "blocked":
        errors.append("blocked_class_required")
    if value.get("flagged_adversarial") is not False:
        errors.append("terminal_adversarial_outcome_invalid")
    if (
        value.get("MODEL_SPECS") != []
        or value.get("model_specs") != []
        or value.get("planned_MODEL_SPECS") != []
    ):
        errors.append("model_specs_not_empty")
    if value.get("model_invoked") is not False:
        errors.append("current_model_invoked")
    if value.get("invocation_counts") != ZERO_INVOCATION_COUNTS:
        errors.append("current_model_counts_nonzero")
    if value.get("inference_substrate") != "aggregation_from_upstream_artifacts":
        errors.append("substrate_mismatch")
    if value.get("inference_substrate_class") != "aggregation":
        errors.append("substrate_class_mismatch")
    if set(REQUIRED_PRINCIPLE_FIELDS) - set(value.get("field_principles") or {}):
        errors.append("field_principles_incomplete")
    if value.get("static_audit_eligible_score") != 0:
        errors.append("blocked_static_eligibility_nonzero")
    if value.get("learning_audit_eligible_score") != 0:
        errors.append("blocked_learning_eligibility_nonzero")
    if value.get("audited_evidence_benefit_score") is not None:
        errors.append("blocked_benefit_must_be_null")
    if value.get("audited_learning_benefit_score") is not None:
        errors.append("blocked_benefit_must_be_null")
    first = (value.get("gate_check_summary") or {}).get("first_failure")
    if not isinstance(first, Mapping) or set(first) != GATE_OPERAND_FIELDS:
        errors.append("blocked_gate_summary_invalid")
    gates = value.get("acceptance_gate_results") or []
    categories = {row.get("category") for row in gates}
    expected_categories = {
        "validity",
        "readiness",
        "probability_benefit",
        "utility",
        "retention",
        "freshness",
    }
    if categories != expected_categories:
        errors.append("acceptance_gate_categories_incomplete")
    if any(not row.get("condition") or not row.get("principle") for row in gates):
        errors.append("acceptance_gate_explanations_incomplete")
    if value.get("rows") != []:
        errors.append("blocked_rows_must_not_be_fabricated")
    mutations = value.get("mutation_rows") or []
    if (
        {row.get("mutation") for row in mutations} != set(MUTATIONS)
        or not all(row.get("passed") is True for row in mutations)
        or not all(row.get("tolerances_weakened") is False for row in mutations)
    ):
        errors.append("mutation_rows_invalid")
    if not _receipts_pass(value.get("validation_receipts") or []):
        errors.append("validation_receipts_failed")
    sources = value.get("source_artifact_hashes") or []
    upstream_names = {spec.upstream for spec in SOURCE_SPECS}
    recorded = {row.get("upstream") for row in sources if row.get("upstream") in upstream_names}
    if recorded != upstream_names:
        errors.append("producer_custody_incomplete")
    for receipt in sources:
        if receipt.get("sha256") is not None:
            try:
                authenticate_source_receipt(receipt, root)
            except ValueError as exc:
                errors.append(str(exc))
    previous_end = 0.0
    for span in value.get("phase_spans") or []:
        start = float(span.get("start_offset_s", -1.0))
        end = float(span.get("end_offset_s", -1.0))
        if start < previous_end or end < start:
            errors.append("phase_spans_overlap")
            break
        previous_end = end
    if value.get("reproducibility_checksum") != reproducibility_checksum(value):
        errors.append("checksum_mismatch")
    if errors:
        raise ValueError(";".join(dict.fromkeys(errors)))
    return {"valid": True}


def build_test_artifact(root: Path) -> JsonDict:
    """Build the current blocked shape with private validation receipts."""

    checks, sources = collect_preconditions(root)
    return build_blocked_artifact(
        root,
        checks,
        sources,
        validation_receipts=provisional_validation_receipts(root),
        duration_s=0.1,
        phase_spans=[],
        run_date="20260925",
    )


def cold_replay(path: Path, *, root: Path = REPO_ROOT) -> JsonDict:
    """Reload exact bytes and rerun terminal schema and custody guards."""

    value = load_json(path)
    if not value:
        raise ValueError("artifact_not_object")
    return validate_artifact(value, root=root)


def independent_replay(path: Path, *, root: Path = REPO_ROOT) -> JsonDict:
    """Rebuild source dispositions and mutations without trusting the artifact."""

    value = load_json(path)
    validate_artifact(value, root=root)
    upstream_names = {spec.upstream for spec in SOURCE_SPECS}
    recorded = {
        str(row.get("upstream")): row
        for row in value.get("source_artifact_hashes") or []
        if row.get("upstream") in upstream_names
    }
    rebuilt = {spec.upstream: classify_source(root, spec) for spec in SOURCE_SPECS}
    for upstream, receipt in rebuilt.items():
        prior = recorded.get(upstream) or {}
        for field in ("sha256", "bytes", "disposition", "eligible_for_science"):
            if prior.get(field) != receipt.get(field):
                raise ValueError(f"independent_source_mismatch:{upstream}:{field}")
    mutation_rows = run_private_mutations()
    if canonical_hash(mutation_rows) != canonical_hash(value.get("mutation_rows") or []):
        raise ValueError("independent_mutation_replay_mismatch")
    return {
        "valid": True,
        "source_count": len(rebuilt),
        "row_count": len(value.get("rows") or []),
        "mutation_count": len(mutation_rows),
    }


def build_validation_commands(
    root: Path, private_root: Path
) -> list[validation_scope.CommandSpec]:  # pragma: no cover - exercised by task E2E.
    """Freeze serial tests, changed-module coverage, and scoped static checks."""

    private_root.mkdir(parents=True, exist_ok=True)
    return validation_scope.build_scoped_commands(
        root,
        (TEST_PATH.as_posix(),),
        (MODULE_PATH.as_posix(),),
        static_paths=(WRAPPER_PATH.as_posix(),),
        basetemp=private_root,
        coverage_file=private_root / ".coverage.exp7638",
    )


def terminal_commands(
    candidate: Path, root: Path
) -> list[validation_scope.CommandSpec]:  # pragma: no cover - exercised by task E2E.
    """Build bounded fresh-process readers for one exact private candidate."""

    python = str(root / ".venv/bin/python")
    common = ("--root", str(root.resolve()), "--date", "20260925")
    return [
        validation_scope.CommandSpec(
            "fresh_process_cold_replay",
            (python, "-u", WRAPPER_PATH.as_posix(), *common, "--cold-replay", str(candidate)),
            "exact_terminal_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            "independent_raw_reduction",
            (
                python,
                "-u",
                WRAPPER_PATH.as_posix(),
                *common,
                "--independent-reduce",
                str(candidate),
            ),
            "exact_terminal_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "exact_terminal_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "exact_terminal_candidate",
            300.0,
        ),
    ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--date", default="20260925")
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    arguments = parser.parse_args(argv)
    arguments.root = arguments.root.resolve()
    if arguments.date != "20260925":
        raise ValueError("run_date_must_equal_20260925")
    return arguments


def progress(  # pragma: no cover - user-visible task heartbeat.
    started: float, phase: str, event: str, **details: Any
) -> None:
    """Flush every phase boundary so long waits remain observable."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7638] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _span(  # pragma: no cover - measured only by the declared entrypoint.
    phase: str, phase_started: float, run_started: float, units: int
) -> JsonDict:
    ended = time.monotonic()
    return {
        "phase": phase,
        "start_offset_s": phase_started - run_started,
        "end_offset_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": units,
        "pending_operation": None,
        "checkpoint_position": units,
    }


def _write_manifest(root: Path) -> Path:  # pragma: no cover - task E2E output.
    path = root / RAW_DIR / "affected_validation_manifest.json"
    atomic_json(
        path,
        {
            "experiment_id": EXPERIMENT_ID,
            "test_paths": [TEST_PATH.as_posix()],
            "changed_modules": [MODULE_PATH.as_posix()],
            "static_paths": [WRAPPER_PATH.as_posix()],
            "spec_paths": [SPEC_PATH.as_posix()],
        },
    )
    return path


def _task_source_receipt(  # pragma: no cover - task E2E output.
    root: Path, relative: Path, upstream: str
) -> JsonDict:
    receipt = source_receipt(root / relative, root, upstream)
    receipt.update(disposition="task_owned_source", eligible_for_science=False)
    return receipt


def run_experiment(  # pragma: no cover - exercised by the declared raw-to-report E2E.
    root: Path, run_date: str, output: Path
) -> JsonDict:
    """Authenticate inputs, validate exact bytes, and publish atomically."""

    root = root.resolve()
    if run_date != "20260925":
        raise ValueError("run_date_must_equal_20260925")
    destination = output if output.is_absolute() else root / output
    started = time.monotonic()
    spans: list[JsonDict] = []

    progress(started, "preconditions", "start", root=root)
    phase_started = time.monotonic()
    checks, sources = collect_preconditions(root)
    failures = [row for row in checks if row.get("passed") is not True]
    if not failures:
        raise RuntimeError("blocked_audit_requires_complete_branch_extension")
    spans.append(_span("preconditions", phase_started, started, len(checks)))
    progress(
        started,
        "preconditions",
        "complete",
        completed_units=len(checks),
        failed=len(failures),
    )

    progress(started, "manifest_and_mutations", "start", planned_units=len(MUTATIONS))
    phase_started = time.monotonic()
    manifest_path = _write_manifest(root)
    for relative, upstream in (
        (MODULE_PATH, "exp7638.implementation"),
        (WRAPPER_PATH, "exp7638.entrypoint"),
        (TEST_PATH, "exp7638.tests"),
        (SPEC_PATH, "exp7638.spec"),
        (manifest_path.relative_to(root), "exp7638.manifest"),
    ):
        sources.append(_task_source_receipt(root, relative, upstream))
    mutations = run_private_mutations()
    if not all(row["passed"] is True for row in mutations):
        raise RuntimeError("private_mutation_panel_failed")
    spans.append(_span("manifest_and_mutations", phase_started, started, len(mutations)))
    progress(started, "manifest_and_mutations", "complete", completed_units=len(mutations))

    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7638-", dir="/tmp")).resolve()
    commands = build_validation_commands(root, private_root / "pytest")
    progress(started, "scoped_validation", "before_subprocesses", planned_units=len(commands))
    phase_started = time.monotonic()
    affected = validation_scope.run_commands(
        root,
        commands,
        log_dir=private_root / "validation_logs",
        heartbeat_s=60.0,
    )
    for receipt in affected:
        receipt["worktree"] = str(root)
    spans.append(_span("scoped_validation", phase_started, started, len(affected)))
    affected_passed = validation_scope.reduce_required_checks(affected)["required_checks_passed"]
    progress(
        started,
        "scoped_validation",
        "after_subprocesses",
        completed_units=len(affected),
        passed=affected_passed,
    )
    if not affected_passed:
        raise RuntimeError("required_scoped_validation_failed")

    candidate = private_root / "exact_terminal_candidate.json"
    pending_terminal = [
        row for row in provisional_validation_receipts(root) if row["name"] in TERMINAL_CHECK_NAMES
    ]
    provisional = build_blocked_artifact(
        root,
        checks,
        sources,
        validation_receipts=[*affected, *pending_terminal],
        duration_s=time.monotonic() - started,
        phase_spans=spans,
        run_date=run_date,
    )
    atomic_json(candidate, provisional)

    plan = terminal_commands(candidate, root)
    progress(started, "terminal_validation", "before_subprocesses", planned_units=len(plan))
    phase_started = time.monotonic()
    terminal = validation_scope.run_commands(
        root,
        plan,
        log_dir=private_root / "terminal_logs_provisional",
        heartbeat_s=60.0,
    )
    for receipt in terminal:
        receipt["worktree"] = str(root)
    spans.append(_span("terminal_validation", phase_started, started, len(terminal)))
    terminal_passed = all(row.get("passed") is True for row in terminal)
    progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        completed_units=len(terminal),
        passed=terminal_passed,
    )
    if not terminal_passed:
        raise RuntimeError("terminal_candidate_validation_failed")

    final = build_blocked_artifact(
        root,
        checks,
        sources,
        validation_receipts=[*affected, *terminal],
        duration_s=time.monotonic() - started,
        phase_spans=spans,
        run_date=run_date,
    )
    validate_artifact(final, root=root)
    atomic_json(candidate, final)
    exact_plan = terminal_commands(candidate, root)
    progress(started, "exact_candidate", "before_subprocesses", planned_units=len(exact_plan))
    exact = validation_scope.run_commands(
        root,
        exact_plan,
        log_dir=private_root / "terminal_logs_exact",
        heartbeat_s=60.0,
    )
    exact_passed = all(row.get("passed") is True for row in exact)
    atomic_json(
        root / RAW_DIR / "exact_terminal_reader_outcomes.json",
        {"candidate_sha256": sha256_file(candidate), "receipts": exact},
    )
    progress(
        started,
        "exact_candidate",
        "after_subprocesses",
        completed_units=len(exact),
        passed=exact_passed,
    )
    if not exact_passed:
        raise RuntimeError("exact_terminal_candidate_validation_failed")

    progress(started, "publish", "before_atomic_terminal")
    atomic_json(destination, final)
    if sha256_file(destination) != sha256_file(candidate):
        raise RuntimeError("published_bytes_differ_from_exact_candidate")
    progress(
        started,
        "publish",
        "complete_terminal",
        bytes=destination.stat().st_size,
        verdict=final["verdict_class"],
    )
    return final


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI boundary.
    arguments = parse_args(argv)
    if arguments.cold_replay is not None:
        result = cold_replay(arguments.cold_replay, root=arguments.root)
        print(json.dumps({"mode": "cold_replay", **result}, sort_keys=True), flush=True)
        return 0
    if arguments.independent_reduce is not None:
        result = independent_replay(arguments.independent_reduce, root=arguments.root)
        print(json.dumps({"mode": "independent_reduction", **result}, sort_keys=True), flush=True)
        return 0
    run_experiment(arguments.root, arguments.date, arguments.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
