"""Tests for verifier-committed uncertainty routing.

Spec refs: REQ-VERIFY-7130, SCENARIO-VERIFY-7130-AUTHORITY,
SCENARIO-VERIFY-7130-LEAKAGE, SCENARIO-VERIFY-7130-UNCERTAINTY,
SCENARIO-VERIFY-7130-BUDGET, SCENARIO-VERIFY-7130-IDENTITY,
SCENARIO-VERIFY-7130-ROWS, SCENARIO-VERIFY-7130-RETRY, and
SCENARIO-VERIFY-7130-ARTIFACT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_7130_v626_verifier_committed_routing as mod


REPO = Path(__file__).resolve().parents[2]
RESULT = REPO / "results/experiment_7130_v626_verifier_committed_routing.json"


def _resolved_specs(tmp_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    paths = {}
    for model_id in mod.REQUIRED_MODEL_IDS:
        path = tmp_path / f"{model_id.rsplit('/', 1)[-1]}-Q4_K_M.gguf"
        path.write_bytes(model_id.encode("utf-8"))
        paths[model_id] = str(path)
    calls: list[dict[str, Any]] = []

    def pair(**kwargs: Any) -> list[dict[str, Any]]:
        calls.append(kwargs)
        ids = [mod.SOTA_REGISTRY_IDS[index] for index in kwargs["model_indices"]]
        return [
            {
                "name": model_id.rsplit("/", 1)[-1],
                "hf_id": model_id,
                "gpu": gpu,
                "model_path": paths[model_id],
            }
            for gpu, model_id in zip(kwargs["gpu_indices"], ids, strict=True)
        ]

    return mod.resolve_model_specs(cached_pair_func=pair), calls


@pytest.fixture(scope="module")
def artifact() -> dict[str, Any]:
    """REQ-VERIFY-7130 keeps the terminal deliverable independently readable."""

    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_req_verify_7130_spec_precedes_implementation() -> None:
    """REQ-VERIFY-7130 and every required scenario exist in the capability spec."""

    spec = (REPO / "openspec/capabilities/constraint-verification/spec.md").read_text(
        encoding="utf-8"
    )
    for anchor in (
        "REQ-VERIFY-7130",
        "SCENARIO-VERIFY-7130-AUTHORITY",
        "SCENARIO-VERIFY-7130-LEAKAGE",
        "SCENARIO-VERIFY-7130-UNCERTAINTY",
        "SCENARIO-VERIFY-7130-BUDGET",
        "SCENARIO-VERIFY-7130-IDENTITY",
        "SCENARIO-VERIFY-7130-ROWS",
        "SCENARIO-VERIFY-7130-RETRY",
        "SCENARIO-VERIFY-7130-ARTIFACT",
    ):
        assert anchor in spec


def test_req_verify_7130_resolves_three_models_through_cached_pair(tmp_path: Path) -> None:
    """REQ-VERIFY-7130 keeps the mandated order and embedded templates."""

    specs, calls = _resolved_specs(tmp_path)

    assert [row["hf_id"] for row in specs] == list(mod.REQUIRED_MODEL_IDS)
    assert calls == [
        {"gpu_indices": (0, 1), "preferred_quant": "Q4_K_M", "model_indices": (0, 2)},
        {"gpu_indices": (0, 1), "preferred_quant": "Q4_K_M", "model_indices": (1, 0)},
    ]
    assert all(row["resolution_method"] == "cached_sota_pair" for row in specs)
    assert all(row["chat_template_source"] == "embedded_gguf" for row in specs)
    assert mod.model_spec_errors(specs) == []


def test_exact_rejected_action_cannot_be_promoted() -> None:
    """REQ-VERIFY-7130 and SCENARIO-VERIFY-7130-AUTHORITY."""

    decision = mod.admit_action(
        unit_key="model|instance|uncertainty_router",
        exact_penalty=1,
        requested_action="accept",
        learned_priority_score=1.0,
    )

    assert decision["executed"] is False
    assert decision["promoted"] is False
    assert decision["final_action"] == "abstain"
    assert decision["rejection_reason"] == "exact_rejection_is_final"


def test_oracle_leakage_into_learned_score_fails_closed() -> None:
    """REQ-VERIFY-7130 and SCENARIO-VERIFY-7130-LEAKAGE."""

    safe = {
        "parsed_answer_hashes": ["sha256:" + "a" * 64, "sha256:" + "b" * 64],
        "parse_successes": [True, True],
        "completion_token_counts": [32, 48],
    }
    row = mod.estimate_uncertainty("model", "instance", safe)
    assert row["learner_feature_names"] == sorted(safe)
    assert 0.0 <= row["uncertainty"] <= 1.0

    for forbidden in (
        "exact_correct",
        "exact_penalty",
        "solver_receipt",
        "witness",
        "objective",
        "solution_set_hash",
        "answer_id",
    ):
        contaminated = deepcopy(safe)
        contaminated[forbidden] = 1
        with pytest.raises(ValueError, match="oracle_feature_forbidden"):
            mod.estimate_uncertainty("model", "instance", contaminated)


def test_uncertainty_collapse_is_measured_and_blocks_positive() -> None:
    """REQ-VERIFY-7130 and SCENARIO-VERIFY-7130-UNCERTAINTY."""

    collapsed = mod.uncertainty_diagnostics(
        [{"uncertainty": 0.0}, {"uncertainty": 0.0}, {"uncertainty": 0.0}]
    )
    varied = mod.uncertainty_diagnostics(
        [{"uncertainty": 0.0}, {"uncertainty": 0.5}, {"uncertainty": 1.0}]
    )

    assert collapsed == {"count": 3, "min": 0.0, "max": 0.0, "collapsed": True}
    assert varied == {"count": 3, "min": 0.0, "max": 1.0, "collapsed": False}
    assert mod.verdict_for_complete_run(promoted=0, accepted_errors=0, collapsed=True)[0] == (
        "null"
    )


def test_arm_budget_mismatch_is_rejected() -> None:
    """REQ-VERIFY-7130 and SCENARIO-VERIFY-7130-BUDGET."""

    assert mod.arm_budget_errors(mod.ARM_PLANS) == []
    assert {row["max_generation_tokens"] for row in mod.ARM_PLANS} == {192}

    changed = deepcopy(mod.ARM_PLANS)
    changed[0]["stages"][0]["max_tokens"] -= 1
    assert mod.arm_budget_errors(changed) == ["arm_budget_mismatch:single_shot:191:192"]


def test_model_and_family_metrics_cannot_be_pooled() -> None:
    """REQ-VERIFY-7130 and SCENARIO-VERIFY-7130-IDENTITY."""

    rows = [
        {
            "arm": arm,
            "model_id": model,
            "family": family,
            "parse_success": True,
            "exact_success": model.endswith("GGUF"),
            "exact_violation_count": 0,
            "abstained": False,
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "duration_s": 0.1,
        }
        for arm in mod.ARM_NAMES
        for model in mod.REQUIRED_MODEL_IDS
        for family in mod.CONSTRAINT_FAMILIES
    ]
    metrics = mod.aggregate_arm_rows(rows)

    assert len(metrics) == 36
    assert {
        (row["arm"], row["model_id"], row["family"]) for row in metrics
    } == set(mod.expected_metric_keys())
    assert mod.model_pooling_errors(metrics) == []
    assert mod.model_pooling_errors(metrics[:-1]) == ["model_family_metric_keys_mismatch"]


def test_missing_per_unit_rows_fail_completion() -> None:
    """REQ-VERIFY-7130 and SCENARIO-VERIFY-7130-ROWS."""

    bank = json.loads(mod.UPSTREAM_PATH.read_text(encoding="utf-8"))
    expected = mod.expected_unit_keys(bank)
    rows = [{"unit_key": key} for key in expected]

    assert len(expected) == 432
    assert mod.unit_row_errors(rows, expected) == []
    assert mod.unit_row_errors(rows[:-1], expected) == ["unit_row_keys_mismatch"]
    assert mod.unit_row_errors([*rows, rows[0]], expected) == ["duplicate_unit_key"]


def test_retry_receipt_is_bounded_non_oracle_and_same_model() -> None:
    """REQ-VERIFY-7130 and SCENARIO-VERIFY-7130-RETRY."""

    receipt = mod.bounded_verifier_receipt(
        model_id="unsloth/Qwen3.6-35B-A3B-GGUF",
        instance_id="sat-0:canonical",
        failed_constraint_classes=["parse", "clause"],
        exact_penalty=2,
    )

    assert receipt["same_model_required"] is True
    assert receipt["failed_constraint_count"] == 2
    assert receipt["failed_constraint_classes"] == ["clause", "parse"]
    assert not (
        set(receipt)
        & {"correct_answer", "witness", "objective", "solution_set_hash", "answer_id"}
    )
    with pytest.raises(ValueError, match="retry_model_family_mismatch"):
        mod.assert_retry_same_model(receipt, "unsloth/gemma-4-31B-it-GGUF")


def test_upstream_gate_and_hash_are_rechecked() -> None:
    """REQ-VERIFY-7130 requires the exact bare producer field and bank bytes."""

    bank = json.loads(mod.UPSTREAM_PATH.read_text(encoding="utf-8"))
    checks = mod.upstream_gate_checks(bank, mod.UPSTREAM_PATH)
    assert checks[0]["check"] == "sota_constraint_bank_ready_score"
    assert checks[0]["expected_value"] == 1
    assert checks[0]["observed_value"] == 1
    assert all(row["passed"] for row in checks)

    closed = deepcopy(bank)
    closed["sota_constraint_bank_ready_score"] = 0
    assert mod.upstream_gate_checks(closed, mod.UPSTREAM_PATH)[0]["passed"] is False


def test_terminal_artifact_replays_and_rejects_required_tampering(
    artifact: dict[str, Any],
) -> None:
    """REQ-VERIFY-7130 and SCENARIO-VERIFY-7130-ARTIFACT."""

    assert mod.validate_artifact(artifact) == []
    assert artifact["verifier_committed_routing_complete_score"] == 1
    assert artifact["exact_rejected_actions_promoted"] == 0
    assert artifact["accepted_error_rate"] == 0.0
    assert artifact["verifier_is_oracle"] is False

    mutations = []
    missing_row = deepcopy(artifact)
    missing_row["rows"].pop()
    mutations.append((missing_row, "unit_row_keys_mismatch"))

    pooled = deepcopy(artifact)
    pooled["arm_rows"].pop()
    mutations.append((pooled, "model_family_metric_keys_mismatch"))

    bad_budget = deepcopy(artifact)
    bad_budget["rows"][0]["max_generation_tokens"] = 191
    mutations.append((bad_budget, "row_arm_budget_mismatch"))

    leaked = deepcopy(artifact)
    leaked["uncertainty_rows"][0]["learner_feature_names"].append("exact_correct")
    mutations.append((leaked, "oracle_feature_forbidden"))

    promoted = deepcopy(artifact)
    rejected = next(row for row in promoted["rejected_promotion_rows"] if row["exact_penalty"] > 0)
    rejected["promoted"] = True
    mutations.append((promoted, "exact_rejected_action_promoted"))

    for changed, expected_error in mutations:
        changed["reproducibility_checksum"] = mod.artifact_checksum(changed)
        assert expected_error in mod.validate_artifact(changed)

