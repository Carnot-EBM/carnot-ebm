"""Audit V663 evidence value and delayed updates without loading a model.

The audit keeps missing captures separate from measured scientific failures.
It never substitutes the older scalar-map experiment for new evidence inputs.

Spec refs: REQ-REPORT-7596 and SCENARIO-REPORT-7596-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "exp7596-v663-evidence-audit"
MILESTONE = "2026.09.663"
SCHEMA = "carnot.exp7596.v663.evidence_audit.v1"
RESULT_PATH = Path("results/experiment_7596_v663_evidence_audit.json")
RAW_DIR = Path("results/raw/experiment_7596_v663_evidence_audit")
MODULE_PATH = Path("python/carnot/experiment_7596_v663_evidence_audit.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7596_v663_evidence_audit.py")
TEST_PATH = Path("tests/python/test_experiment_7596_v663_evidence_audit.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
AFFECTED_CHECK_NAMES = validation_scope.REQUIRED_CHECK_NAMES
TERMINAL_CHECK_NAMES = (
    "fresh_process_cold_replay",
    "independent_raw_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
ZERO_INVOCATION_COUNTS = {
    "model_loads": 0,
    "forward_calls": 0,
    "generation_calls": 0,
    "input_tokens": 0,
    "output_tokens": 0,
}


@dataclass(frozen=True)
class SourceSpec:
    """Name a planned producer and the conductor diagnostic for its absence."""

    upstream: str
    producer_path: Path
    conductor_path: Path


SOURCE_SPECS = (
    SourceSpec(
        "exp7590-evidence-pilot",
        Path("results/experiment_7590_v663_evidence_pilot.json"),
        Path("results/experiment_7590_evidence_pilot.json"),
    ),
    SourceSpec(
        "exp7591-fit-evidence",
        Path("results/experiment_7591_v663_fit_evidence.json"),
        Path("results/experiment_7591_fit_evidence.json"),
    ),
    SourceSpec(
        "exp7592-test-online-evidence",
        Path("results/experiment_7592_v663_test_online_evidence.json"),
        Path("results/experiment_7592_test_online_evidence.json"),
    ),
    SourceSpec(
        "exp7593-evidence-energy",
        Path("results/experiment_7593_v663_evidence_energy.json"),
        Path("results/experiment_7593_evidence_energy.json"),
    ),
    SourceSpec(
        "exp7594-decision-evaluation",
        Path("results/experiment_7594_v663_decision_evaluation.json"),
        Path("results/experiment_7594_decision_evaluation.json"),
    ),
    SourceSpec(
        "exp7595-guarded-learning",
        Path("results/experiment_7595_v663_guarded_learning.json"),
        Path("results/experiment_7595_guarded_learning.json"),
    ),
)


def canonical_hash(value: Any) -> str:
    """Hash canonical JSON so any changed audit operand changes identity."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash exact bytes because a trusted path does not imply trusted content."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def load_json(path: Path) -> JsonDict:
    """Return a JSON object, or an empty object for unreadable producer bytes."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _path_label(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def source_receipt(path: Path, root: Path, upstream: str) -> JsonDict:
    """Bind one source to its exact bytes and declared upstream identity."""

    return {
        "upstream": upstream,
        "path": _path_label(path, root),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def authenticate_source_receipt(receipt: Mapping[str, Any], root: Path) -> Path:
    """Reject a source whose path, byte count, or byte hash changed."""

    path = Path(str(receipt.get("path") or ""))
    resolved = path if path.is_absolute() else root / path
    if not resolved.is_file():
        raise ValueError(f"source_missing:{path}")
    if resolved.stat().st_size != receipt.get("bytes"):
        raise ValueError(f"source_size_mismatch:{path}")
    if sha256_file(resolved) != receipt.get("sha256"):
        raise ValueError(f"source_hash_mismatch:{path}")
    return resolved


def classify_source(root: Path, spec: SourceSpec) -> JsonDict:
    """Keep producer bytes, pre-gate evidence, flags, and absence distinct."""

    producer_path = root / spec.producer_path
    conductor_path = root / spec.conductor_path
    if producer_path.is_file():
        producer = load_json(producer_path)
        receipt = source_receipt(producer_path, root, spec.upstream)
        receipt.update(
            {
                "producer_path": spec.producer_path.as_posix(),
                "conductor_path": spec.conductor_path.as_posix(),
                "honest_verdict": producer.get("honest_verdict"),
                "verdict_class": producer.get("verdict_class"),
                "flagged_adversarial": producer.get("flagged_adversarial"),
            }
        )
        terminal = str(producer.get("honest_verdict") or "").startswith("complete_")
        if not producer or not terminal:
            receipt.update(disposition="invalid_producer", eligible_for_science=False)
        elif producer.get("flagged_adversarial") is True:
            receipt.update(disposition="flagged_producer", eligible_for_science=False)
        else:
            receipt.update(disposition="authenticated_producer", eligible_for_science=True)
        return receipt
    if conductor_path.is_file():
        conductor = load_json(conductor_path)
        receipt = source_receipt(conductor_path, root, spec.upstream)
        receipt.update(
            {
                "producer_path": spec.producer_path.as_posix(),
                "conductor_path": spec.conductor_path.as_posix(),
                "honest_verdict": conductor.get("honest_verdict"),
                "verdict_class": "blocked",
                "flagged_adversarial": False,
                "disposition": "conductor_pre_gate",
                "eligible_for_science": False,
            }
        )
        return receipt
    return {
        "upstream": spec.upstream,
        "path": spec.producer_path.as_posix(),
        "producer_path": spec.producer_path.as_posix(),
        "conductor_path": spec.conductor_path.as_posix(),
        "sha256": None,
        "bytes": None,
        "honest_verdict": None,
        "verdict_class": None,
        "flagged_adversarial": None,
        "disposition": "missing_producer",
        "eligible_for_science": False,
    }


def check_row(
    check: str,
    upstream: str,
    path: str,
    field: str,
    expected: Any,
    observed: Any,
    op: str,
) -> JsonDict:
    """Retain both operands so a blocked result remains reproducible."""

    if op == "eq":
        passed = observed == expected
    elif op == "in":
        passed = observed in expected
    else:
        raise ValueError(f"unknown_check_op:{op}")
    return {
        "check": check,
        "upstream": upstream,
        "path": path,
        "field": field,
        "op": op,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "passed": bool(passed),
        "required": True,
    }


def blocked_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name every failure and preserve the complete first failed operand."""

    failed = [row for row in checks if row.get("passed") is not True]
    if not failed:
        raise ValueError("blocked_summary_requires_failed_check")
    keys = ("check", "upstream", "path", "field", "op", "expected", "observed")
    return {
        "passed": False,
        "failed_count": len(failed),
        "failed_checks": [row.get("check") for row in failed],
        "first_failure": {key: deepcopy(failed[0].get(key)) for key in keys},
    }


def collect_preconditions(root: Path) -> tuple[list[JsonDict], list[JsonDict]]:
    """Check owned instructions and inventory every expected V663 source."""

    root = root.resolve()
    checks: list[JsonDict] = []
    for relative in (Path("AGENTS.md"), Path("CODEX.md"), Path("CLAUDE.md"), SPEC_PATH):
        checks.append(
            check_row(
                "required_path",
                "worktree",
                relative.as_posix(),
                "exists",
                True,
                (root / relative).is_file(),
                "eq",
            )
        )
    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    checks.append(
        check_row(
            "matching_requirement",
            "openspec",
            SPEC_PATH.as_posix(),
            "REQ-REPORT-7596",
            True,
            "REQ-REPORT-7596" in spec_text,
            "eq",
        )
    )
    receipts = [classify_source(root, spec) for spec in SOURCE_SPECS]
    by_upstream = {row["upstream"]: row for row in receipts}
    for spec in SOURCE_SPECS[-2:]:
        receipt = by_upstream[spec.upstream]
        checks.append(
            check_row(
                "required_scientific_producer",
                spec.upstream,
                spec.producer_path.as_posix(),
                "exists_and_eligible",
                True,
                receipt["disposition"] == "authenticated_producer",
                "eq",
            )
        )
    return checks, receipts


MUTATIONS = (
    "dropped_rows",
    "inflated_independent_count",
    "inverted_brier_sign",
    "unchanged_controls",
    "future_origin_shuffle",
    "leaked_roles",
    "duplicated_accepted_update",
)
MUTATION_FAILURES = {
    "dropped_rows": "static_row_count_mismatch",
    "inflated_independent_count": "independent_sample_count_mismatch",
    "inverted_brier_sign": "brier_sign_mismatch",
    "unchanged_controls": "control_unchanged",
    "future_origin_shuffle": "future_origin_shuffle",
    "leaked_roles": "gradient_role_leak",
    "duplicated_accepted_update": "duplicate_accepted_update",
}


def private_fixture() -> JsonDict:
    """Build a compact valid fixture used only to prove fail-closed readers."""

    return {
        "static_group_ids": ["s0", "s1"],
        "declared_static_count": 2,
        "online_group_ids": ["o0", "o1"],
        "declared_online_count": 2,
        "raw_brier": 0.24,
        "evidence_brier": 0.20,
        "brier_improvement": 0.04,
        "control_predictions": [0.2, 0.8],
        "treatment_predictions": [0.3, 0.7],
        "shuffle_pairs": [{"source_release": 0, "target_release": 1}],
        "gradient_roles": ["fit"],
        "accepted_update_ids": ["u0", "u1"],
    }


def validate_private_fixture(value: Mapping[str, Any]) -> list[str]:
    """Return every corruption detected without repairing private inputs."""

    errors: list[str] = []
    static_ids = list(value.get("static_group_ids") or [])
    online_ids = list(value.get("online_group_ids") or [])
    if len(static_ids) != value.get("declared_static_count"):
        errors.append("static_row_count_mismatch")
    if len(set(online_ids)) != value.get("declared_online_count"):
        errors.append("independent_sample_count_mismatch")
    expected_improvement = float(value.get("raw_brier", 0)) - float(value.get("evidence_brier", 0))
    if abs(expected_improvement - float(value.get("brier_improvement", 0))) > 1e-10:
        errors.append("brier_sign_mismatch")
    if value.get("control_predictions") == value.get("treatment_predictions"):
        errors.append("control_unchanged")
    if any(row["source_release"] > row["target_release"] for row in value["shuffle_pairs"]):
        errors.append("future_origin_shuffle")
    if set(value.get("gradient_roles") or []) & {"anchor", "evaluator"}:
        errors.append("gradient_role_leak")
    update_ids = list(value.get("accepted_update_ids") or [])
    if len(update_ids) != len(set(update_ids)):
        errors.append("duplicate_accepted_update")
    return errors


def mutate_private_fixture(value: JsonDict, mutation: str) -> str:
    """Apply one named corruption and return the changed private field path."""

    if mutation == "dropped_rows":
        value["static_group_ids"].pop()
        return "static_group_ids[-1]"
    if mutation == "inflated_independent_count":
        value["declared_online_count"] += 1
        return "declared_online_count"
    if mutation == "inverted_brier_sign":
        value["brier_improvement"] *= -1
        return "brier_improvement"
    if mutation == "unchanged_controls":
        value["control_predictions"] = deepcopy(value["treatment_predictions"])
        return "control_predictions"
    if mutation == "future_origin_shuffle":
        value["shuffle_pairs"][0]["source_release"] = 2
        return "shuffle_pairs[0].source_release"
    if mutation == "leaked_roles":
        value["gradient_roles"].append("anchor")
        return "gradient_roles[-1]"
    if mutation == "duplicated_accepted_update":
        value["accepted_update_ids"].append(value["accepted_update_ids"][0])
        return "accepted_update_ids[-1]"
    raise ValueError(f"unknown_mutation:{mutation}")


def run_private_mutations() -> list[JsonDict]:
    """Record changed private bytes and the exact check that rejected each."""

    receipts: list[JsonDict] = []
    for mutation in MUTATIONS:
        value = private_fixture()
        before = canonical_hash(value)
        changed_path = mutate_private_fixture(value, mutation)
        after = canonical_hash(value)
        failures = validate_private_fixture(value)
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
            }
        )
    return receipts


REQUIRED_PRINCIPLE_FIELDS = (
    "honest_verdict",
    "verdict_class",
    "flagged_adversarial",
    "gate_check_summary",
    "acceptance_gate_results",
    "rows",
    "sample_size_budget",
    "inference_substrate",
    "inference_substrate_class",
    "MODEL_SPECS",
    "invocation_counts",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "source_artifact_hashes",
    "validation_receipts",
    "verifier_is_oracle",
    "field_principles",
    "static_audit_ready_score",
    "online_audit_ready_score",
    "branch_conclusions",
    "mutation_receipts",
)


def field_principles() -> dict[str, str]:
    """Explain why each required result field exists."""

    return {
        "honest_verdict": "A complete prefix reports terminal work; completion does not prove benefit.",
        "verdict_class": "A closed class keeps missing external work blocked rather than partial or null.",
        "flagged_adversarial": "The terminal reader outcome cannot open a gate when evidence is flagged.",
        "gate_check_summary": "A blocked result retains every operand needed to reproduce the failure.",
        "acceptance_gate_results": "Validity, readiness, benefit, retention, and freshness stay separate.",
        "rows": "Each measured unit and arm must retain absolute operands, direction, seed, missingness, and provenance.",
        "sample_size_budget": "Source groups own inference; seeds and windows do not multiply sample size.",
        "inference_substrate": "Current aggregation cannot inherit upstream GPU execution.",
        "inference_substrate_class": "Actual and planned execution classes remain explicit.",
        "MODEL_SPECS": "No current LLM call means the current model roster is empty.",
        "invocation_counts": "Loads, forwards, generations, and tokens are counted independently from history.",
        "duration_s": "Monotonic current work excludes inherited time and artificial padding.",
        "random_seed": "Every stochastic audit stage must have an explicit replay seed.",
        "reproducibility_checksum": "One digest binds immutable sources, configuration, and terminal reduction.",
        "source_artifact_hashes": "Producer, absence, pre-gate, and flagged custody remain distinct.",
        "validation_receipts": "Commands, exits, worktree, and log hashes bind terminal checks.",
        "verifier_is_oracle": "Exact labels and hand-built controls cannot support an oracle-distinct claim.",
        "field_principles": "Each required field carries its own omission guard.",
        "static_audit_ready_score": "Static validity remains separate from static benefit.",
        "online_audit_ready_score": "Chronology validity remains separate from retention success.",
        "branch_conclusions": "Positive, null, missing, and invalid branches stay individually visible.",
        "mutation_receipts": "Actual private changes and failed checks replace a list of intended attacks.",
    }


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    """Bind stable result content while excluding current wall-clock measurements."""

    excluded = {"reproducibility_checksum", "duration_s", "phase_spans"}
    return canonical_hash(
        {key: deepcopy(item) for key, item in value.items() if key not in excluded}
    )


def _acceptance_gate(check: str, category: str, observed: str, passed: bool) -> JsonDict:
    return {
        "check": check,
        "category": category,
        "op": "eq",
        "expected": "qualified",
        "observed": observed,
        "passed": passed,
        "principle": "Validity, readiness, benefit, retention, and freshness remain separate.",
    }


def _branch_conclusions() -> list[JsonDict]:
    branches = (
        ("incremental_evidence", "exp7594-decision-evaluation"),
        ("probability", "exp7594-decision-evaluation"),
        ("decision_cost", "exp7594-decision-evaluation"),
        ("causal_update_behavior", "exp7595-guarded-learning"),
        ("retention", "exp7595-guarded-learning"),
        ("exposure", "exp7594-and-exp7595"),
    )
    return [
        {
            "branch": branch,
            "upstream": upstream,
            "validity": "blocked_missing_capture",
            "readiness": "not_established",
            "benefit": "not_measured",
            "retention": "not_measured" if branch == "retention" else "not_applicable",
            "freshness": "not_assessed",
            "conclusion": "missing_external_evidence_not_scientific_failure",
        }
        for branch, upstream in branches
    ]


def provisional_validation_receipts() -> list[JsonDict]:
    """Build structurally valid receipts used only by a pre-terminal candidate."""

    return [
        {
            "name": name,
            "command": f"pending exact candidate {name}",
            "command_argv": ["pending", name],
            "scope": "exact_candidate" if name in TERMINAL_CHECK_NAMES else "changed_files",
            "worktree": str(REPO_ROOT),
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
    source_hashes: Sequence[Mapping[str, Any]],
    *,
    validation_receipts: Sequence[Mapping[str, Any]],
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Publish external absence without inventing a measured zero or null."""

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": "20260924",
        "worktree_root": str(root.resolve()),
        "honest_verdict": "complete_blocked_missing_v663_evidence_producers",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "gate_check_summary": blocked_summary(checks),
        "preconditions_checked": [deepcopy(dict(row)) for row in checks],
        "acceptance_gate_results": [
            _acceptance_gate("producer_validity", "validity", "missing_external", False),
            _acceptance_gate("static_readiness", "readiness", "not_started", False),
            _acceptance_gate("scientific_benefit", "benefit", "not_measured", False),
            _acceptance_gate("retained_usefulness", "retention", "not_measured", False),
            _acceptance_gate("fresh_confirmatory_data", "freshness", "not_assessed", False),
        ],
        "rows": [],
        "sample_size_budget": [
            {
                "branch": "static",
                "independent_unit": "source_group",
                "intended": 40,
                "observed": 0,
                "excluded": 0,
                "censored": 0,
                "missing_external": 40,
            },
            {
                "branch": "online",
                "independent_unit": "source_group",
                "intended": 80,
                "observed": 0,
                "excluded": 0,
                "censored": 0,
                "missing_external": 80,
                "registered_orders": 5,
                "orders_multiply_independent_units": False,
            },
        ],
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "planned_inference_substrate_class": "aggregation",
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "duration_s": float(duration_s),
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "random_seed": 7596001,
        "stochastic_stages_started": [],
        "source_artifact_hashes": [deepcopy(dict(row)) for row in source_hashes],
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "verifier_is_oracle": False,
        "oracle_distinct_claim_allowed": False,
        "field_principles": field_principles(),
        "static_audit_ready_score": 0,
        "online_audit_ready_score": 0,
        "branch_conclusions": _branch_conclusions(),
        "mutation_receipts": run_private_mutations(),
        "fresh_confirmatory_claim_allowed": False,
        "exact_new_information_source_earned_continuation": None,
        "production_activation_authorized": False,
        "hidden_score_claimed": False,
        "prior_verdict_disposition": {
            "literal_prior_verdict": "complete_null_independent_static_and_learning_audit",
            "repeated": False,
            "retire_if_same_verdict": True,
            "action": "not_triggered_by_missing_external_resources",
            "scientific_hypothesis_retired": False,
        },
        "applicable_numbered_e2e": [],
        "submitted_externally": False,
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
    """Reject identity, custody, blocked semantics, or terminal receipt drift."""

    errors: list[str] = []
    if value.get("schema") != SCHEMA or value.get("experiment_id") != EXPERIMENT_ID:
        errors.append("identity_mismatch")
    if value.get("milestone") != MILESTONE or value.get("run_date") != "20260924":
        errors.append("run_identity_mismatch")
    if not str(value.get("honest_verdict") or "").startswith("complete_"):
        errors.append("terminal_prefix_missing")
    if value.get("verdict_class") != "blocked":
        errors.append("blocked_class_required")
    if value.get("flagged_adversarial") is not False:
        errors.append("terminal_adversarial_outcome_invalid")
    if value.get("MODEL_SPECS") != [] or value.get("model_specs") != []:
        errors.append("model_specs_not_empty")
    if (
        value.get("model_invoked") is not False
        or value.get("invocation_counts") != ZERO_INVOCATION_COUNTS
    ):
        errors.append("current_model_calls_nonzero")
    if value.get("inference_substrate_class") != "aggregation":
        errors.append("substrate_class_mismatch")
    if value.get("inference_substrate") != "aggregation_from_upstream_artifacts":
        errors.append("substrate_mismatch")
    if set(REQUIRED_PRINCIPLE_FIELDS) - set(value.get("field_principles") or {}):
        errors.append("field_principles_incomplete")
    if value.get("static_audit_ready_score") != 0 or value.get("online_audit_ready_score") != 0:
        errors.append("blocked_readiness_nonzero")
    expected_branches = {
        "incremental_evidence",
        "probability",
        "decision_cost",
        "causal_update_behavior",
        "retention",
        "exposure",
    }
    if {row.get("branch") for row in value.get("branch_conclusions") or []} != expected_branches:
        errors.append("branch_conclusions_incomplete")
    first = (value.get("gate_check_summary") or {}).get("first_failure")
    gate_fields = {"check", "upstream", "path", "field", "op", "expected", "observed"}
    if not isinstance(first, Mapping) or set(first) != gate_fields:
        errors.append("blocked_gate_summary_invalid")
    mutations = value.get("mutation_receipts") or []
    if (
        {row.get("mutation") for row in mutations} != set(MUTATIONS)
        or not all(row.get("passed") is True for row in mutations)
        or not all(row.get("before_sha256") != row.get("after_sha256") for row in mutations)
    ):
        errors.append("mutation_receipts_invalid")
    if not _receipts_pass(value.get("validation_receipts") or []):
        errors.append("validation_receipts_failed")
    if value.get("rows") != []:
        errors.append("blocked_rows_must_be_empty")
    if value.get("reproducibility_checksum") != reproducibility_checksum(value):
        errors.append("checksum_mismatch")
    for receipt in value.get("source_artifact_hashes") or []:
        if receipt.get("sha256") is not None:
            try:
                authenticate_source_receipt(receipt, root)
            except ValueError as exc:
                errors.append(str(exc))
    if errors:
        raise ValueError(";".join(dict.fromkeys(errors)))
    return {"valid": True}


def cold_replay(path: Path, *, root: Path = REPO_ROOT) -> JsonDict:
    """Reload the exact artifact in a fresh process and run all schema guards."""

    value = load_json(path)
    if not value:
        raise ValueError("artifact_not_object")
    return validate_artifact(value, root=root)


def independent_replay(path: Path, *, root: Path = REPO_ROOT) -> JsonDict:
    """Independently confirm that external absence yielded no comparative rows."""

    value = load_json(path)
    validate_artifact(value, root=root)
    sources = value.get("source_artifact_hashes") or []
    dispositions = {row.get("upstream"): row.get("disposition") for row in sources}
    if dispositions.get("exp7594-decision-evaluation") != "missing_producer":
        raise ValueError("static_absence_disposition_mismatch")
    if dispositions.get("exp7595-guarded-learning") != "missing_producer":
        raise ValueError("online_absence_disposition_mismatch")
    return {
        "valid": True,
        "row_count": len(value.get("rows") or []),
        "source_count": len(sources),
        "static_disposition": dispositions["exp7594-decision-evaluation"],
        "online_disposition": dispositions["exp7595-guarded-learning"],
    }


def build_validation_commands(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Freeze focused tests, changed-module coverage, lint, type, and spec checks."""

    private_root.mkdir(parents=True, exist_ok=True)
    coverage_file = private_root / ".coverage.exp7596"
    return validation_scope.build_scoped_commands(
        root,
        (TEST_PATH.as_posix(),),
        (MODULE_PATH.as_posix(),),
        static_paths=(WRAPPER_PATH.as_posix(),),
        basetemp=private_root,
        coverage_file=coverage_file,
    )


def terminal_commands(candidate: Path, root: Path) -> list[validation_scope.CommandSpec]:
    """Build bounded fresh-process readers for one exact candidate path."""

    python = str(root / ".venv/bin/python")
    common = ("--root", str(root.resolve()), "--date", "20260924")
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
            (
                python,
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "exact_terminal_candidate",
            300.0,
        ),
    ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--date", default="20260924")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    arguments = parser.parse_args(argv)
    arguments.root = arguments.root.resolve()
    if arguments.date != "20260924":
        raise ValueError("run_date_must_equal_20260924")
    return arguments


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Flush every phase boundary so a bounded audit never appears stalled."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7596] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _span(  # pragma: no cover - orchestration receipt exercised by the declared entrypoint.
    phase: str, phase_started: float, run_started: float, units: int
) -> JsonDict:
    ended = time.monotonic()
    return {
        "phase": phase,
        "start_offset_s": phase_started - run_started,
        "end_offset_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": units,
    }


def _write_manifest(root: Path) -> tuple[Path, JsonDict]:  # pragma: no cover
    path = root / RAW_DIR / "affected_validation_manifest.json"
    value = {
        "experiment_id": EXPERIMENT_ID,
        "test_paths": [TEST_PATH.as_posix()],
        "changed_modules": [MODULE_PATH.as_posix()],
        "static_paths": [WRAPPER_PATH.as_posix()],
        "spec_paths": [SPEC_PATH.as_posix()],
    }
    atomic_json(path, value)
    return path, value


def run_experiment(  # pragma: no cover - exercised by the declared read-only E2E.
    root: Path, run_date: str
) -> JsonDict:
    """Inventory, validate, replay, and atomically publish the blocked audit."""

    root = root.resolve()
    if run_date != "20260924":
        raise ValueError("run_date_must_equal_20260924")
    started = time.monotonic()
    spans: list[JsonDict] = []
    destination = root / RESULT_PATH

    progress(started, "preconditions", "start", root=root)
    phase_started = time.monotonic()
    checks, custody = collect_preconditions(root)
    spans.append(_span("preconditions", phase_started, started, len(checks)))
    failures = [row for row in checks if row.get("passed") is not True]
    progress(
        started,
        "preconditions",
        "complete",
        completed_units=len(checks),
        failed=len(failures),
    )
    if not failures:
        raise RuntimeError("full_scientific_reducer_not_reached_by_current_custody")

    progress(started, "manifest", "start")
    phase_started = time.monotonic()
    manifest_path, _manifest = _write_manifest(root)
    durable_sources = [deepcopy(row) for row in custody]
    for relative, upstream in (
        (MODULE_PATH, "exp7596.implementation"),
        (WRAPPER_PATH, "exp7596.entrypoint"),
        (TEST_PATH, "exp7596.tests"),
        (SPEC_PATH, "exp7596.spec"),
    ):
        receipt = source_receipt(root / relative, root, upstream)
        receipt.update(
            disposition="task_owned_source",
            eligible_for_science=False,
            producer_path=relative.as_posix(),
            conductor_path=None,
            honest_verdict=None,
            verdict_class=None,
            flagged_adversarial=False,
        )
        durable_sources.append(receipt)
    manifest_receipt = source_receipt(manifest_path, root, "exp7596.validation_manifest")
    manifest_receipt.update(
        disposition="task_owned_source",
        eligible_for_science=False,
        producer_path=_path_label(manifest_path, root),
        conductor_path=None,
        honest_verdict=None,
        verdict_class=None,
        flagged_adversarial=False,
    )
    durable_sources.append(manifest_receipt)
    spans.append(_span("manifest", phase_started, started, len(durable_sources)))
    progress(started, "manifest", "complete", completed_units=len(durable_sources))

    progress(started, "private_mutations", "start", planned_units=len(MUTATIONS))
    phase_started = time.monotonic()
    mutations = run_private_mutations()
    if not all(row["passed"] is True for row in mutations):
        raise RuntimeError("private_mutation_panel_failed")
    spans.append(_span("private_mutations", phase_started, started, len(mutations)))
    progress(
        started,
        "private_mutations",
        "complete",
        completed_units=len(mutations),
        passed=len(mutations),
    )

    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7596-", dir="/tmp")).resolve()
    commands = build_validation_commands(root, private_root / "pytest")
    progress(started, "scoped_validation", "before_subprocesses", planned_units=len(commands))
    phase_started = time.monotonic()
    affected_receipts = validation_scope.run_commands(
        root,
        commands,
        log_dir=private_root / "validation_logs",
        heartbeat_s=60.0,
    )
    for receipt in affected_receipts:
        receipt["worktree"] = str(root)
    spans.append(_span("scoped_validation", phase_started, started, len(affected_receipts)))
    affected_passed = validation_scope.reduce_required_checks(affected_receipts)[
        "required_checks_passed"
    ]
    progress(
        started,
        "scoped_validation",
        "after_subprocesses",
        completed_units=len(affected_receipts),
        passed=affected_passed,
    )
    if not affected_passed:
        raise RuntimeError("required_scoped_validation_failed")

    candidate = private_root / "terminal_candidate.json"
    provisional = build_blocked_artifact(
        root,
        checks,
        durable_sources,
        validation_receipts=[
            *affected_receipts,
            *[
                row
                for row in provisional_validation_receipts()
                if row["name"] in TERMINAL_CHECK_NAMES
            ],
        ],
        duration_s=time.monotonic() - started,
        phase_spans=spans,
    )
    atomic_json(candidate, provisional)

    first_plan = terminal_commands(candidate, root)
    progress(started, "terminal_validation", "before_subprocesses", planned_units=len(first_plan))
    phase_started = time.monotonic()
    terminal_receipts = validation_scope.run_commands(
        root,
        first_plan,
        log_dir=private_root / "terminal_logs_provisional",
        heartbeat_s=60.0,
    )
    for receipt in terminal_receipts:
        receipt["worktree"] = str(root)
    spans.append(_span("terminal_validation", phase_started, started, len(terminal_receipts)))
    terminal_passed = all(row.get("passed") is True for row in terminal_receipts)
    progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        completed_units=len(terminal_receipts),
        passed=terminal_passed,
    )
    if not terminal_passed:
        raise RuntimeError("terminal_candidate_validation_failed")

    final = build_blocked_artifact(
        root,
        checks,
        durable_sources,
        validation_receipts=[*affected_receipts, *terminal_receipts],
        duration_s=time.monotonic() - started,
        phase_spans=spans,
    )
    validate_artifact(final, root=root)
    atomic_json(candidate, final)

    exact_plan = terminal_commands(candidate, root)
    progress(
        started,
        "exact_candidate_validation",
        "before_subprocesses",
        planned_units=len(exact_plan),
    )
    exact_receipts = validation_scope.run_commands(
        root,
        exact_plan,
        log_dir=private_root / "terminal_logs_exact",
        heartbeat_s=60.0,
    )
    exact_passed = all(row.get("passed") is True for row in exact_receipts)
    progress(
        started,
        "exact_candidate_validation",
        "after_subprocesses",
        completed_units=len(exact_receipts),
        passed=exact_passed,
    )
    if not exact_passed:
        raise RuntimeError("exact_terminal_candidate_validation_failed")
    atomic_json(root / RAW_DIR / "exact_terminal_receipts.json", {"receipts": exact_receipts})

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
    run_experiment(arguments.root, arguments.date)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
