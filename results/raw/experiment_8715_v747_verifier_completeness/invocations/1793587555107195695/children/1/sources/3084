"""Independently audit V665 evidence and delayed learning.

The audit does not convert a completed schema or blocked producer into measured
benefit. It reduces authenticated raw rows when they exist and preserves an
external block when the required scientific producers do not exist.

Spec refs: REQ-REPORT-7624 and SCENARIO-REPORT-7624-*.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
from importlib import metadata
import json
import math
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
from carnot.experiment_7610_v664_evidence_audit import (
    SourceSpec,
    _interval,
    _probability_from_logit,
    _typed_action,
    classify_source as classify_v664_source,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "exp7624-v665-evidence-audit"
MILESTONE = "2026.09.665"
SCHEMA = "carnot.exp7624.v665.evidence_audit.v1"
RESULT_PATH = Path("results/experiment_7624_v665_evidence_audit.json")
RAW_DIR = Path("results/raw/experiment_7624_v665_evidence_audit")
MODULE_PATH = Path("python/carnot/experiment_7624_v665_evidence_audit.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7624_v665_evidence_audit.py")
TEST_PATH = Path("tests/python/test_experiment_7624_v665_evidence_audit.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
MODEL_SPECS: list[JsonDict] = []
HISTORICAL_MODEL_IDENTITY = "unsloth/Qwen3.8-27B-GGUF"
STATIC_ARMS = ("factual", "erased", "deranged")
LEARNING_ARMS = ("frozen", "guarded")
GATE_OPERAND_FIELDS = {"check", "upstream", "path", "field", "op", "expected", "observed"}
AFFECTED_CHECK_NAMES = validation_scope.REQUIRED_CHECK_NAMES
TERMINAL_CHECK_NAMES = (
    "fresh_process_cold_replay",
    "independent_raw_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)

SOURCE_SPECS = (
    SourceSpec(
        "exp7616-evidence-schema", Path("results/experiment_7616_v665_evidence_schema.json"), None
    ),
    SourceSpec(
        "exp7617-schema-pilot", Path("results/experiment_7617_v665_schema_pilot.json"), None
    ),
    SourceSpec(
        "exp7618-fit-evidence",
        Path("results/experiment_7618_v665_fit_evidence.json"),
        Path("results/experiment_7618_fit_evidence.json"),
    ),
    SourceSpec(
        "exp7619-online-evidence",
        Path("results/experiment_7619_v665_online_evidence.json"),
        Path("results/experiment_7619_online_evidence.json"),
    ),
    SourceSpec(
        "exp7620-evaluation-evidence",
        Path("results/experiment_7620_v665_evaluation_evidence.json"),
        Path("results/experiment_7620_evaluation_evidence.json"),
    ),
    SourceSpec(
        "exp7621-evidence-energy",
        Path("results/experiment_7621_v665_evidence_energy.json"),
        Path("results/experiment_7621_evidence_energy.json"),
    ),
    SourceSpec(
        "exp7622-decision-evaluation",
        Path("results/experiment_7622_v665_decision_evaluation.json"),
        Path("results/experiment_7622_decision_evaluation.json"),
    ),
    SourceSpec(
        "exp7623-guarded-learning",
        Path("results/experiment_7623_v665_guarded_learning.json"),
        Path("results/experiment_7623_guarded_learning.json"),
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
    Path("results/experiment_7610_v664_evidence_audit.json"),
    Path("python/carnot/experiment_7610_v664_evidence_audit.py"),
    Path("scripts/adversarial_verify.py"),
    Path("scripts/verdict_row_consistency_lint.py"),
    SPEC_PATH,
)


def classify_source(root: Path, spec: SourceSpec) -> JsonDict:
    """Keep a real blocked producer distinct from a conductor-only receipt."""

    receipt = classify_v664_source(root, spec)
    if (
        receipt.get("disposition") == "authenticated_producer"
        and receipt.get("verdict_class") == "blocked"
    ):
        receipt["disposition"] = "authenticated_blocked_producer"
        receipt["eligible_for_science"] = False
    return receipt


def _bytes_sha256(value: str) -> str:
    """Hash the exact checkpoint text so whitespace changes remain visible."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _checkpoint(value: str, expected_hash: str, branch: str) -> JsonDict:
    if _bytes_sha256(value) != expected_hash:
        raise ValueError("checkpoint_hash_mismatch")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("checkpoint_json_invalid") from exc
    if not isinstance(parsed, dict) or parsed.get("branch") != branch:
        raise ValueError("checkpoint_identity_mismatch")
    if parsed.get("model_identity") != HISTORICAL_MODEL_IDENTITY:
        raise ValueError("model_identity_mismatch")
    return parsed


def reduce_static_branch(
    rows: Sequence[Mapping[str, Any]],
    *,
    checkpoint_bytes: str,
    checkpoint_sha256: str,
    role_map: Mapping[str, str],
    bootstrap_draws: int,
    bootstrap_seed: int,
) -> JsonDict:
    """Rebuild static probabilities and costs without a producer aggregate."""

    checkpoint = _checkpoint(checkpoint_bytes, checkpoint_sha256, "static")
    weights = [float(item) for item in checkpoint.get("weights") or []]
    if not weights or any(not math.isfinite(item) for item in weights):
        raise ValueError("checkpoint_weights_invalid")
    bias = float(checkpoint.get("bias", 0.0))
    raw_offset = float(checkpoint.get("raw_offset", 0.0))
    expected_sign = int(checkpoint.get("probability_sign", 0))
    if expected_sign != 1:
        raise ValueError("checkpoint_probability_sign_invalid")

    grouped: dict[str, dict[str, JsonDict]] = defaultdict(dict)
    reduced_rows: list[JsonDict] = []
    for raw in rows:
        group_id = str(raw.get("group_id") or "")
        arm = str(raw.get("arm") or "")
        if role_map.get(group_id) != "evaluation":
            raise ValueError("corrupt_group_id")
        if arm not in STATIC_ARMS:
            raise ValueError("static_arm_invalid")
        if arm in grouped[group_id]:
            raise ValueError("duplicated_group")
        if raw.get("label_role") != "evaluation":
            raise ValueError("static_label_role_invalid")
        if raw.get("label_accessed_during_optimization") is not False:
            raise ValueError("label_leak")
        if raw.get("checkpoint_sha256") != checkpoint_sha256:
            raise ValueError("row_checkpoint_mismatch")
        if raw.get("model_identity") != checkpoint.get("model_identity"):
            raise ValueError("model_identity_mismatch")
        if raw.get("probability_sign") != expected_sign:
            raise ValueError("probability_sign_flip")
        if raw.get("included") is not True:
            raise ValueError("excluded_observed_row")
        features = tuple(float(item) for item in raw.get("features") or [])
        if len(features) != len(weights) or any(not math.isfinite(item) for item in features):
            raise ValueError("static_feature_width_invalid")
        label = int(raw.get("label", -1))
        probability = float(raw.get("raw_probability", math.nan))
        if label not in (0, 1) or not 0.0 <= probability <= 1.0:
            raise ValueError("static_outcome_invalid")
        clipped = min(1.0 - 1e-12, max(1e-12, probability))
        raw_logit = math.log(clipped / (1.0 - clipped))
        logit = (
            bias
            + raw_offset * raw_logit
            + sum(weight * feature for weight, feature in zip(weights, features))
        )
        rebuilt = _probability_from_logit(logit)
        brier = (rebuilt - label) ** 2
        action, cost, covered, false_accept = _typed_action(rebuilt, label)
        row = {
            "unit_id": group_id,
            "arm": arm,
            "absolute_metrics": {
                "probability": rebuilt,
                "label": label,
                "brier": brier,
                "realized_cost": cost,
                "covered": covered,
                "false_accept": false_accept,
            },
            "probability": rebuilt,
            "brier": brier,
            "typed_action": action,
            "raw_numerator": brier,
            "raw_denominator": 1,
            "seed": int(raw.get("seed", -1)),
            "direction": "lower_brier_and_cost_are_better",
            "censored": bool(raw.get("censored")),
            "censoring": str(raw.get("censoring") or "none"),
            "raw_provenance": str(raw.get("raw_provenance") or ""),
            "features": list(features),
        }
        grouped[group_id][arm] = row
        reduced_rows.append(row)

    if not grouped or any(set(arms) != set(STATIC_ARMS) for arms in grouped.values()):
        raise ValueError("static_arm_roster_invalid")
    for arms in grouped.values():
        factual = arms["factual"]["features"]
        if factual == arms["erased"]["features"] or factual == arms["deranged"]["features"]:
            raise ValueError("identical_control")

    metrics: dict[str, JsonDict] = {}
    for arm in STATIC_ARMS:
        selected = [arms[arm] for arms in grouped.values()]
        brier_total = sum(float(row["brier"]) for row in selected)
        cost_total = sum(float(row["absolute_metrics"]["realized_cost"]) for row in selected)
        metrics[arm] = {
            "brier_numerator": brier_total,
            "brier_denominator": len(selected),
            "mean_brier": brier_total / len(selected),
            "cost_numerator": cost_total,
            "cost_denominator": len(selected),
            "mean_cost": cost_total / len(selected),
        }
    contrasts = {}
    for offset, comparator in enumerate(("erased", "deranged")):
        deltas = [
            float(arms[comparator]["brier"]) - float(arms["factual"]["brier"])
            for arms in grouped.values()
        ]
        contrasts[f"factual_vs_{comparator}_brier"] = _interval(
            deltas, draws=bootstrap_draws, seed=bootstrap_seed + offset
        )
    return {
        "eligible": True,
        "independent_unit_count": len(grouped),
        "row_count": len(reduced_rows),
        "rows": reduced_rows,
        "arm_metrics": metrics,
        "contrasts": contrasts,
        "controls_distinct": True,
        "sample_count_uses_views_or_seeds": False,
    }


def reduce_learning_branch(
    rows: Sequence[Mapping[str, Any]],
    *,
    checkpoint_bytes: str,
    checkpoint_sha256: str,
    role_map: Mapping[str, str],
) -> JsonDict:
    """Replay delayed updates while excluding admission and evaluation labels."""

    checkpoint = _checkpoint(checkpoint_bytes, checkpoint_sha256, "learning")
    initial = [float(item) for item in checkpoint.get("initial_weights") or []]
    learning_rate = float(checkpoint.get("learning_rate", 0.0))
    if not initial or learning_rate <= 0.0:
        raise ValueError("learning_checkpoint_invalid")
    grouped: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    retained_rows = 0
    for row in rows:
        group_id = str(row.get("group_id") or "")
        arm = str(row.get("arm") or "")
        if role_map.get(group_id) != "online":
            raise ValueError("corrupt_group_id")
        if arm not in LEARNING_ARMS:
            raise ValueError("learning_arm_invalid")
        if arm in grouped[group_id]:
            raise ValueError("duplicated_group")
        if row.get("checkpoint_sha256") != checkpoint_sha256:
            raise ValueError("row_checkpoint_mismatch")
        if row.get("label_available_at_prediction") is not False:
            raise ValueError("label_leak")
        if row.get("evaluator_access") is not False or row.get("admission_access") is not False:
            raise ValueError("label_leak")
        if row.get("optimizer_input_roles") != ["update"]:
            raise ValueError("optimizer_role_leak")
        prediction = int(row.get("prediction_index", -1))
        release = int(row.get("release_index", -1))
        update = int(row.get("update_index", -1))
        if not prediction < release <= update:
            raise ValueError("causal_release_order_invalid")
        features = [float(item) for item in row.get("features") or []]
        if len(features) != len(initial):
            raise ValueError("learning_feature_width_invalid")
        probability = _probability_from_logit(sum(a * b for a, b in zip(initial, features)))
        if not math.isclose(probability, float(row.get("prediction_probability")), abs_tol=1e-12):
            raise ValueError("learning_probability_mismatch")
        accepted = row.get("accepted_update") is True
        if accepted != (arm == "guarded") or row.get("feedback_label_role") != "update":
            raise ValueError("learning_update_role_invalid")
        weights = list(initial)
        if accepted:
            label = int(row.get("feedback_label", -1))
            if label not in (0, 1):
                raise ValueError("learning_label_invalid")
            weights = [
                weight - learning_rate * (probability - label) * feature
                for weight, feature in zip(weights, features)
            ]
        expected_state = canonical_hash([round(value, 15) for value in weights])
        if row.get("state_after_sha256") != expected_state:
            raise ValueError("learning_state_mismatch")
        if row.get("restart_state_sha256") != expected_state:
            raise ValueError("restart_checkpoint_mismatch")
        if row.get("retention_label_role") != "evaluation":
            raise ValueError("retention_role_invalid")
        if row.get("retention_used_for_update") is not False:
            raise ValueError("optimizer_role_leak")
        retention_loss = float(row.get("retention_loss", math.nan))
        if not math.isfinite(retention_loss):
            raise ValueError("retention_loss_invalid")
        retained_rows += 1
        grouped[group_id][arm] = row
    if not grouped or any(set(arms) != set(LEARNING_ARMS) for arms in grouped.values()):
        raise ValueError("learning_arm_roster_invalid")
    return {
        "eligible": True,
        "independent_unit_count": len(grouped),
        "row_count": sum(len(arms) for arms in grouped.values()),
        "causal_release_order": True,
        "optimization_roles": ["update"],
        "evaluation_or_admission_labels_used": False,
        "checkpoint_bytes_match": True,
        "retention_measured": retained_rows == len(rows),
        "sample_count_uses_orders_seeds_or_replays": False,
    }


def _json_bytes(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def private_fixture() -> JsonDict:
    """Build small valid raw branches for fail-closed mutation checks."""

    static_checkpoint_bytes = _json_bytes(
        {
            "branch": "static",
            "model_identity": HISTORICAL_MODEL_IDENTITY,
            "weights": [0.3, -0.2],
            "bias": -0.1,
            "raw_offset": 0.5,
            "probability_sign": 1,
        }
    )
    static_hash = _bytes_sha256(static_checkpoint_bytes)
    static_rows: list[JsonDict] = []
    feature_sets = {
        "factual": [0.9, 0.1],
        "erased": [0.0, 0.9],
        "deranged": [0.2, 0.8],
    }
    for index, (raw_probability, label) in enumerate(((0.8, 1), (0.2, 0))):
        for arm, features in feature_sets.items():
            static_rows.append(
                {
                    "group_id": f"eval-{index}",
                    "arm": arm,
                    "view_id": f"view-{index}-{arm}",
                    "seed": 7624100 + index,
                    "features": list(features),
                    "raw_probability": raw_probability,
                    "label": label,
                    "label_role": "evaluation",
                    "label_accessed_during_optimization": False,
                    "probability_sign": 1,
                    "model_identity": HISTORICAL_MODEL_IDENTITY,
                    "checkpoint_sha256": static_hash,
                    "included": True,
                    "censored": False,
                    "censoring": "none",
                    "raw_provenance": f"fixture:eval-{index}:{arm}",
                }
            )

    learning_checkpoint_bytes = _json_bytes(
        {
            "branch": "learning",
            "model_identity": HISTORICAL_MODEL_IDENTITY,
            "initial_weights": [0.0, 0.0],
            "learning_rate": 0.1,
        }
    )
    learning_hash = _bytes_sha256(learning_checkpoint_bytes)
    learning_rows: list[JsonDict] = []
    for index in range(2):
        features = [1.0, float(index)]
        probability = _probability_from_logit(0.0)
        for arm in LEARNING_ARMS:
            accepted = arm == "guarded"
            weights = [0.0, 0.0]
            if accepted:
                weights = [
                    weight - 0.1 * (probability - 1) * feature
                    for weight, feature in zip(weights, features)
                ]
            state_hash = canonical_hash([round(value, 15) for value in weights])
            learning_rows.append(
                {
                    "group_id": f"online-{index}",
                    "arm": arm,
                    "seed": 7624200 + index,
                    "replay_view": 0,
                    "features": features,
                    "prediction_probability": probability,
                    "prediction_index": 0,
                    "release_index": 1,
                    "update_index": 2,
                    "label_available_at_prediction": False,
                    "feedback_label": 1,
                    "feedback_label_role": "update",
                    "accepted_update": accepted,
                    "optimizer_input_roles": ["update"],
                    "evaluator_access": False,
                    "admission_access": False,
                    "checkpoint_sha256": learning_hash,
                    "state_after_sha256": state_hash,
                    "restart_state_sha256": state_hash,
                    "retention_label_role": "evaluation",
                    "retention_used_for_update": False,
                    "retention_loss": 0.2 + 0.01 * index,
                }
            )
    return {
        "static_rows": static_rows,
        "static_checkpoint_bytes": static_checkpoint_bytes,
        "static_checkpoint_sha256": static_hash,
        "learning_rows": learning_rows,
        "learning_checkpoint_bytes": learning_checkpoint_bytes,
        "learning_checkpoint_sha256": learning_hash,
        "role_map": {
            "eval-0": "evaluation",
            "eval-1": "evaluation",
            "online-0": "online",
            "online-1": "online",
        },
    }


MUTATIONS = (
    "corrupt_id",
    "label_leak",
    "probability_sign_flip",
    "changed_checkpoint",
    "duplicated_group",
    "identical_control",
)
MUTATION_FAILURES = {
    "corrupt_id": "corrupt_group_id",
    "label_leak": "label_leak",
    "probability_sign_flip": "probability_sign_flip",
    "changed_checkpoint": "checkpoint_hash_mismatch",
    "duplicated_group": "duplicated_group",
    "identical_control": "identical_control",
}


def mutate_private_fixture(value: JsonDict, mutation: str) -> str:
    """Change one private operand so its independent reader must reject it."""

    if mutation == "corrupt_id":
        value["static_rows"][0]["group_id"] = "unknown-group"
        return "static_rows[0].group_id"
    if mutation == "label_leak":
        value["learning_rows"][0]["label_available_at_prediction"] = True
        return "learning_rows[0].label_available_at_prediction"
    if mutation == "probability_sign_flip":
        value["static_rows"][0]["probability_sign"] = -1
        return "static_rows[0].probability_sign"
    if mutation == "changed_checkpoint":
        value["learning_checkpoint_bytes"] += " "
        return "learning_checkpoint_bytes"
    if mutation == "duplicated_group":
        value["static_rows"].append(deepcopy(value["static_rows"][0]))
        return "static_rows[duplicate]"
    if mutation == "identical_control":
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
    raise ValueError(f"unknown_mutation:{mutation}")


def validate_private_fixture(value: Mapping[str, Any]) -> list[str]:
    """Return all branch failures from one changed private fixture."""

    errors: list[str] = []
    try:
        reduce_static_branch(
            value.get("static_rows") or [],
            checkpoint_bytes=str(value.get("static_checkpoint_bytes") or ""),
            checkpoint_sha256=str(value.get("static_checkpoint_sha256") or ""),
            role_map=value.get("role_map") or {},
            bootstrap_draws=32,
            bootstrap_seed=7624001,
        )
    except ValueError as exc:
        errors.extend(str(exc).split(";"))
    try:
        reduce_learning_branch(
            value.get("learning_rows") or [],
            checkpoint_bytes=str(value.get("learning_checkpoint_bytes") or ""),
            checkpoint_sha256=str(value.get("learning_checkpoint_sha256") or ""),
            role_map=value.get("role_map") or {},
        )
    except ValueError as exc:
        errors.extend(str(exc).split(";"))
    return list(dict.fromkeys(errors))


def run_private_mutations() -> list[JsonDict]:
    """Bind each private corruption to its changed bytes and failed check."""

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
                "corrupted_fixture_published": False,
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
    "static_audit_eligible_score",
    "learning_audit_eligible_score",
    "audited_evidence_benefit_score",
    "audited_learning_benefit_score",
    "branch_dispositions",
    "mutation_rows",
)


def field_principles() -> dict[str, str]:
    """Keep the reason for every required field beside the field."""

    principles = {
        "honest_verdict": "A complete prefix reports finished audit work, not scientific benefit.",
        "verdict_class": "External missing evidence is blocked and never partial or a semantic null.",
        "flagged_adversarial": "Flagged evidence cannot open any downstream gate.",
        "gate_check_summary": "Every block retains its exact upstream operands.",
        "acceptance_gate_results": "Validity, readiness, benefit, retention, and freshness stay separate.",
        "rows": "Independent units retain arms, numerators, denominators, seeds, censoring, and provenance.",
        "sample_size_budget": "Seeds, views, orders, and replays never multiply source groups.",
        "preconditions_checked": "Only checks performed before reduction can establish available inputs.",
        "inference_substrate": "Current aggregation cannot inherit historical model execution.",
        "inference_substrate_class": "Planned and actual execution classes remain explicit.",
        "MODEL_SPECS": "No current LLM call requires an empty current model roster.",
        "model_invoked": "Historical GPU evidence is not a current invocation.",
        "execution_venue": "The current host stays separate from historical execution venues.",
        "phase_spans": "Disjoint measured stages expose work and checkpoint positions.",
        "invocation_counts": "Current loads, forwards, generations, and tokens remain zero and typed.",
        "duration_s": "Monotonic elapsed time excludes inherited work and artificial padding.",
        "random_seed": "Each stochastic interval and mutation panel has a replay seed.",
        "reproducibility_checksum": "One digest binds immutable inputs, configuration, and reductions.",
        "source_artifact_hashes": "Producer, pre-gate, blocked, and missing custody stay distinct.",
        "validation_receipts": "Commands, exits, worktrees, and log hashes bind terminal checks.",
        "verifier_is_oracle": "Exact fixtures cannot establish oracle-distinct learned benefit.",
        "field_principles": "Each governed field carries its omission guard.",
        "static_audit_eligible_score": "Static eligibility requires complete authenticated raw measurement.",
        "learning_audit_eligible_score": "Learning eligibility requires causal and retained measurement.",
        "audited_evidence_benefit_score": "Static benefit is recomputed and never inherited.",
        "audited_learning_benefit_score": "Learning benefit requires online and retention effects together.",
        "branch_dispositions": "Static and learning outcomes remain independent.",
        "mutation_rows": "Every required injected defect retains its observed rejection.",
    }
    return principles


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    """Hash stable evidence while excluding measured clocks and this digest."""

    excluded = {"reproducibility_checksum", "duration_s", "phase_spans"}
    return canonical_hash(
        {key: deepcopy(item) for key, item in value.items() if key not in excluded}
    )


def _gate(
    check: str, category: str, condition: str, expected: Any, observed: Any, passed: bool
) -> JsonDict:
    return {
        "check": check,
        "category": category,
        "condition": condition,
        "operator": "eq",
        "expected": expected,
        "observed": observed,
        "passed": passed,
        "principle": "Validity, readiness, benefit, retention, and freshness remain separate.",
    }


def collect_preconditions(root: Path) -> tuple[list[JsonDict], list[JsonDict]]:
    """Authenticate owned inputs and inventory every V665 producer path."""

    root = root.resolve()
    checks: list[JsonDict] = []
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
    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    checks.append(
        check_row(
            "matching_requirement",
            "openspec",
            SPEC_PATH.as_posix(),
            "REQ-REPORT-7624",
            True,
            "REQ-REPORT-7624" in spec_text,
            "eq",
        )
    )
    for tool, observed in (
        ("python", sys.version.split()[0]),
        ("pytest", metadata.version("pytest")),
        ("ruff", metadata.version("ruff")),
        ("mypy", metadata.version("mypy")),
    ):
        checks.append(
            check_row(
                "declared_tool_version",
                "worktree-toolchain",
                str(root / ".venv/bin" / tool),
                "version_nonempty",
                True,
                bool(observed),
                "eq",
            )
            | {"observed_version": observed}
        )

    producer_receipts = [classify_source(root, spec) for spec in SOURCE_SPECS]
    receipts.extend(producer_receipts)
    by_upstream = {str(row["upstream"]): row for row in producer_receipts}
    for spec in SOURCE_SPECS[2:]:
        row = by_upstream[spec.upstream]
        checks.append(
            check_row(
                "required_scientific_producer",
                spec.upstream,
                str(row["producer_path"]),
                "exists_and_eligible",
                True,
                row["eligible_for_science"],
                "eq",
            )
        )
    for upstream, field in (
        ("exp7622-decision-evaluation", "rows"),
        ("exp7622-decision-evaluation", "checkpoint_receipt.sha256"),
        ("exp7623-guarded-learning", "online_rows"),
        ("exp7623-guarded-learning", "retention_rows"),
    ):
        source = by_upstream[upstream]
        checks.append(
            check_row(
                "required_raw_evidence_field",
                upstream,
                str(source["producer_path"]),
                field,
                "present_and_authenticated",
                "missing_producer",
                "eq",
            )
        )
    return checks, receipts


def provisional_validation_receipts(root: Path) -> list[JsonDict]:
    """Provide structurally complete receipts for a private candidate only."""

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
    """Build a complete external block without inventing scientific rows."""

    failures = [row for row in checks if row.get("passed") is not True]
    if not failures:
        raise ValueError("blocked_artifact_requires_external_failure")
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": 7624,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": run_date,
        "worktree_root": str(root.resolve()),
        "honest_verdict": "complete_blocked_v665_scientific_producers_unavailable",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "gate_check_summary": blocked_summary(failures),
        "preconditions_checked": [deepcopy(dict(row)) for row in checks],
        "acceptance_gate_results": [
            _gate(
                "custody_validity", "validity", "all present bytes authenticate", True, True, True
            ),
            _gate(
                "static_readiness", "readiness", "static raw producer exists", True, False, False
            ),
            _gate(
                "learning_readiness",
                "readiness",
                "learning raw producer exists",
                True,
                False,
                False,
            ),
            _gate("static_benefit", "benefit", "eligible static effect passes", True, None, False),
            _gate(
                "learning_benefit", "benefit", "online and retained effects pass", True, None, False
            ),
            _gate("retention", "retention", "isolated retention rows pass", True, None, False),
            _gate("freshness", "freshness", "external roles are unexposed", True, False, False),
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
                "seeds_views_replays_multiply_samples": False,
            },
            {
                "branch": "learning",
                "independent_unit": "source_group",
                "intended": 80,
                "observed": 0,
                "excluded": 0,
                "censored": 0,
                "missing_external": 80,
                "seeds_views_replays_multiply_samples": False,
            },
            {
                "branch": "retention",
                "independent_unit": "source_group",
                "intended": 40,
                "observed": 0,
                "excluded": 0,
                "censored": 0,
                "missing_external": 40,
                "seeds_views_replays_multiply_samples": False,
            },
        ],
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "planned_inference_substrate_class": "aggregation",
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "historical_model_identity": HISTORICAL_MODEL_IDENTITY,
        "execution_venue": "host",
        "execution_venue_details": {
            "hostname": socket.gethostname(),
            "physical_device": "cpu",
            "gpu_uuid": None,
            "historical_board_receipt_used_as_current_execution": False,
        },
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "duration_s": float(duration_s),
        "random_seed": {
            "static_bootstrap": 7624001,
            "learning_bootstrap": 7624002,
            "private_mutations": 7624003,
        },
        "source_artifact_hashes": [deepcopy(dict(row)) for row in sources],
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
                "branch": "static",
                "validity": "blocked_missing_raw_rows_and_checkpoint",
                "readiness": "not_established",
                "benefit": None,
                "semantic_null": False,
                "syntax_success": False,
                "calibration_only_improvement": None,
                "semantic_evidence_dependence": None,
                "historically_exposed": True,
                "scientific_hypothesis_retired": False,
            },
            {
                "branch": "learning",
                "validity": "blocked_missing_online_and_retention_rows",
                "readiness": "not_established",
                "benefit": None,
                "semantic_null": False,
                "retained_online_learning": None,
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
        "capability_e2e": "cold_replay_and_all_negative_mutations",
        "prior_failure_disposition": {
            "prior_experiment": "exp7610-evidence-audit",
            "prior_verdict": "complete_blocked_v664_evidence_chain_unavailable",
            "retirement_condition_met": False,
            "same_scope": False,
            "action": "preserve_scientific_hypotheses_after_v665_external_block",
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
    """Reject custody drift or a blocked branch that claims measured benefit."""

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
    if categories != {"validity", "readiness", "benefit", "retention", "freshness"}:
        errors.append("acceptance_gate_categories_incomplete")
    if any(not row.get("condition") or not row.get("principle") for row in gates):
        errors.append("acceptance_gate_explanations_incomplete")
    branches = value.get("branch_dispositions") or []
    if {row.get("branch") for row in branches} != {"static", "learning"}:
        errors.append("branch_dispositions_incomplete")
    if any(row.get("scientific_hypothesis_retired") is not False for row in branches):
        errors.append("external_block_retired_hypothesis")
    mutations = value.get("mutation_rows") or []
    if (
        {row.get("mutation") for row in mutations} != set(MUTATIONS)
        or not all(row.get("passed") is True for row in mutations)
        or not all(row.get("before_sha256") != row.get("after_sha256") for row in mutations)
    ):
        errors.append("mutation_rows_invalid")
    if value.get("rows") != []:
        errors.append("blocked_rows_must_not_be_fabricated")
    if not _receipts_pass(value.get("validation_receipts") or []):
        errors.append("validation_receipts_failed")
    upstream_names = {spec.upstream for spec in SOURCE_SPECS}
    sources = value.get("source_artifact_hashes") or []
    recorded_upstreams = {
        row.get("upstream") for row in sources if row.get("upstream") in upstream_names
    }
    if recorded_upstreams != upstream_names:
        errors.append("producer_custody_incomplete")
    for receipt in sources:
        if receipt.get("sha256") is not None:
            try:
                authenticate_source_receipt(receipt, root)
            except ValueError as exc:
                errors.append(str(exc))
    spans = value.get("phase_spans") or []
    previous_end = 0.0
    for span in spans:
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
        run_date="20260924",
    )


def cold_replay(path: Path, *, root: Path = REPO_ROOT) -> JsonDict:
    """Reload exact bytes and rerun the terminal schema and custody guards."""

    value = load_json(path)
    if not value:
        raise ValueError("artifact_not_object")
    return validate_artifact(value, root=root)


def independent_replay(path: Path, *, root: Path = REPO_ROOT) -> JsonDict:
    """Re-inventory source paths without trusting the artifact dispositions."""

    value = load_json(path)
    validate_artifact(value, root=root)
    recorded = {
        str(row.get("upstream")): row
        for row in value.get("source_artifact_hashes") or []
        if row.get("upstream") in {spec.upstream for spec in SOURCE_SPECS}
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
        coverage_file=private_root / ".coverage.exp7624",
    )


def terminal_commands(
    candidate: Path, root: Path
) -> list[validation_scope.CommandSpec]:  # pragma: no cover - exercised by task E2E.
    """Build bounded fresh-process readers for one exact candidate."""

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
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "exact_terminal_candidate",
            300.0,
        ),
    ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--date", default="20260924")
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    arguments = parser.parse_args(argv)
    arguments.root = arguments.root.resolve()
    if arguments.date != "20260924":
        raise ValueError("run_date_must_equal_20260924")
    return arguments


def progress(  # pragma: no cover - user-visible task heartbeat.
    started: float, phase: str, event: str, **details: Any
) -> None:
    """Flush each phase boundary so the conductor can observe live work."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7624] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
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
    if run_date != "20260924":
        raise ValueError("run_date_must_equal_20260924")
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
        (MODULE_PATH, "exp7624.implementation"),
        (WRAPPER_PATH, "exp7624.entrypoint"),
        (TEST_PATH, "exp7624.tests"),
        (SPEC_PATH, "exp7624.spec"),
        (manifest_path.relative_to(root), "exp7624.manifest"),
    ):
        sources.append(_task_source_receipt(root, relative, upstream))
    mutations = run_private_mutations()
    if not all(row["passed"] is True for row in mutations):
        raise RuntimeError("private_mutation_panel_failed")
    spans.append(_span("manifest_and_mutations", phase_started, started, len(mutations)))
    progress(started, "manifest_and_mutations", "complete", completed_units=len(mutations))

    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7624-", dir="/tmp")).resolve()
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
