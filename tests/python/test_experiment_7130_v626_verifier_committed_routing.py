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
    assert {(row["arm"], row["model_id"], row["family"]) for row in metrics} == set(
        mod.expected_metric_keys()
    )
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
        set(receipt) & {"correct_answer", "witness", "objective", "solution_set_hash", "answer_id"}
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


def test_raw_shards_rebuild_the_complete_artifact(artifact: dict[str, Any]) -> None:
    """REQ-VERIFY-7130 replays persisted model bytes into every derived row."""

    bank = json.loads(mod.UPSTREAM_PATH.read_text(encoding="utf-8"))
    raw_rows = [
        row
        for manifest in artifact["raw_trace_manifest"]
        for row in mod.load_raw_rows(Path(manifest["path"]))
    ]
    rebuilt = mod.build_artifact(
        run_date=artifact["run_date"],
        duration_s=artifact["duration_s"],
        model_specs=artifact["MODEL_SPECS"],
        bank=bank,
        preconditions={"all_passed": True, "checks": []},
        raw_rows=raw_rows,
        raw_trace_manifest=artifact["raw_trace_manifest"],
        gpu_telemetry_rows=artifact["gpu_telemetry_rows"],
        source_artifact_hashes=artifact["source_artifact_hashes"],
    )

    assert len(raw_rows) == 541
    assert len(rebuilt["rows"]) == 432
    assert rebuilt["verifier_committed_routing_complete_score"] == 1
    assert mod.validate_artifact(rebuilt) == []

    retry_raw = next(row for row in raw_rows if row["stage"] == "retry")
    source = next(
        row
        for row in bank["rows"]
        if row["model_id"] == retry_raw["model_id"]
        and row["instance_id"] == retry_raw["instance_id"]
    )
    receipt = next(
        row for row in bank["solver_receipt_rows"] if row["instance_id"] == source["instance_id"]
    )
    unit_rows_without_retry = [
        row
        for row in raw_rows
        if row["model_id"] == retry_raw["model_id"]
        and row["instance_id"] == retry_raw["instance_id"]
        and row["stage"] != "retry"
    ]
    reduced_without_retry = mod._reduce_raw(
        {"rows": [source], "solver_receipt_rows": [receipt]},
        unit_rows_without_retry,
        [row for row in artifact["MODEL_SPECS"] if row["hf_id"] == source["model_id"]],
    )
    assert len(reduced_without_retry["rows"]) == 3
    assert (
        mod._reduce_raw({"rows": [source], "solver_receipt_rows": [receipt]}, [], [])["rows"] == []
    )


def test_blocked_partial_and_invalid_artifact_paths(artifact: dict[str, Any]) -> None:
    """SCENARIO-VERIFY-7130-ARTIFACT fails closed across every terminal class."""

    blocked = mod.build_artifact(
        run_date=artifact["run_date"],
        duration_s=0.0,
        model_specs=artifact["MODEL_SPECS"],
        bank=None,
        preconditions={
            "all_passed": False,
            "checks": [mod.gate_row("preflight", True, False, False)],
        },
    )
    assert blocked["verdict_class"] == "blocked"
    assert mod.validate_artifact(blocked) == []

    partial = mod.build_artifact(
        run_date=artifact["run_date"],
        duration_s=0.0,
        model_specs=artifact["MODEL_SPECS"],
        bank=None,
        preconditions={"all_passed": True, "checks": []},
    )
    assert partial["verdict_class"] == "partial"

    failed_raw = {
        "request_key": "missing|raw",
        "terminal_state": "failed",
    }
    partial_with_failed_raw = mod.build_artifact(
        run_date=artifact["run_date"],
        duration_s=0.0,
        model_specs=[],
        bank={"rows": [], "solver_receipt_rows": []},
        preconditions={"all_passed": True, "checks": []},
        raw_rows=[failed_raw],
    )
    assert partial_with_failed_raw["gate_check_summary"]["passed"] is False

    malformed_blocked = deepcopy(blocked)
    malformed_blocked.pop("rows")
    malformed_blocked["field_principles"]["rows"] = ""
    malformed_blocked["reproducibility_checksum"] = "corrupt"
    malformed_blocked["honest_verdict"] = "wrong"
    malformed_blocked["verifier_is_oracle"] = True
    malformed_blocked["gate_check_summary"] = {"passed": True, "failed_check": None}
    malformed_blocked["inference_substrate_class"] = "wrong"
    malformed_blocked["verifier_committed_routing_complete_score"] = 1
    blocked_errors = mod.validate_artifact(malformed_blocked)
    for expected in (
        "required_field_missing:rows",
        "field_principle_missing:rows",
        "reproducibility_checksum_mismatch",
        "honest_verdict_class_mismatch",
        "verifier_is_oracle_mismatch",
        "blocked_gate_summary_missing_failure",
        "blocked_substrate_class_mismatch",
        "blocked_complete_score_nonzero",
    ):
        assert expected in blocked_errors


def test_validator_rejects_all_derived_surface_tampering(
    artifact: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-VERIFY-7130-ARTIFACT recomputes every derived surface."""

    malformed = deepcopy(artifact)
    malformed["verdict_class"] = "unknown"
    malformed["honest_verdict"] = "wrong"
    malformed["verifier_is_oracle"] = True
    malformed.pop("duration_s")
    malformed["field_principles"]["duration_s"] = ""
    malformed["reproducibility_checksum"] = "corrupt"
    malformed_errors = mod.validate_artifact(malformed)
    for expected in (
        "required_field_missing:duration_s",
        "field_principle_missing:duration_s",
        "reproducibility_checksum_mismatch",
        "verdict_class_invalid",
        "honest_verdict_class_mismatch",
        "verifier_is_oracle_mismatch",
        "complete_score_mismatch",
    ):
        assert expected in malformed_errors

    changed = deepcopy(artifact)
    changed["inference_substrate"] = "wrong"
    changed["inference_substrate_class"] = "wrong"
    changed["execution_venue"] = "remote"
    changed["MODEL_SPECS"] = list(reversed(changed["MODEL_SPECS"]))
    changed["arm_plans"] = list(reversed(changed["arm_plans"]))
    changed["arm_rows"].pop()
    changed["uncertainty_diagnostics"] = {}
    changed["exact_rejected_actions_promoted"] = 99
    changed["accepted_error_rate"] = 0.5
    changed["retry_rows"][0]["retry_model_id"] = mod.REQUIRED_MODEL_IDS[0]
    changed["retry_rows"][0]["verifier_receipt"]["exact_penalty"] = 1
    changed["relabel_sensitivity_rows"].pop()
    changed["paraphrase_consistency_rows"].pop()
    changed["reproducibility_checksum"] = mod.artifact_checksum(changed)
    errors = mod.validate_artifact(changed)
    for expected in (
        "inference_substrate_mismatch",
        "inference_substrate_class_mismatch",
        "execution_venue_mismatch",
        "model_roster_mismatch",
        "arm_roster_mismatch",
        "model_family_metric_keys_mismatch",
        "arm_aggregate_mismatch",
        "uncertainty_summary_mismatch",
        "rejected_promotion_count_mismatch",
        "accepted_error_rate_mismatch",
        "retry_model_family_mismatch",
        "retry_receipt_oracle_leakage",
        "relabel_sensitivity_mismatch",
        "paraphrase_consistency_mismatch",
    ):
        assert expected in errors

    wrong_verdict = deepcopy(artifact)
    wrong_verdict["verdict_class"] = "null"
    wrong_verdict["honest_verdict"] = "null_complete_uncertainty_collapsed"
    wrong_verdict["reproducibility_checksum"] = mod.artifact_checksum(wrong_verdict)
    assert "complete_verdict_mismatch" in mod.validate_artifact(wrong_verdict)

    invalid_bank = tmp_path / "invalid-bank.json"
    invalid_bank.write_text("{", encoding="utf-8")
    monkeypatch.setattr(mod, "UPSTREAM_PATH", invalid_bank)
    assert "upstream_bank_unavailable" in mod.validate_artifact(artifact)


def test_helper_error_paths_are_explicit(artifact: dict[str, Any]) -> None:
    """REQ-VERIFY-7130 exposes roster, budget, uncertainty, and admission errors."""

    summary = mod.gate_summary([mod.gate_row("first", 1, 0, False)])
    assert summary["failed_check"] == "first"

    specs = deepcopy(artifact["MODEL_SPECS"])
    specs.reverse()
    specs[0].update(
        {
            "model_path": "",
            "quantization": "wrong",
            "chat_template_source": "wrong",
            "resolution_method": "wrong",
            "gpu_indices": [],
            "remote_allowed": True,
        }
    )
    specs[1]["model_path"] = "mmproj.gguf"
    spec_errors = mod.model_spec_errors(specs)
    assert "model_roster_mismatch" in spec_errors
    assert any(error.startswith("model_path_missing:") for error in spec_errors)
    assert any(error.startswith("model_path_not_language_gguf:") for error in spec_errors)
    assert any(error.startswith("model_quantization_mismatch:") for error in spec_errors)
    assert any(error.startswith("chat_template_source_mismatch:") for error in spec_errors)
    assert any(error.startswith("model_resolution_method_mismatch:") for error in spec_errors)
    assert any(error.startswith("model_execution_policy_mismatch:") for error in spec_errors)

    with pytest.raises(ValueError, match="uncertainty_requires_aligned_repeated_outputs"):
        mod.estimate_uncertainty(
            "model",
            "instance",
            {
                "parsed_answer_hashes": ["one"],
                "parse_successes": [True, False],
                "completion_token_counts": [1, 2],
            },
        )

    plans = deepcopy(mod.ARM_PLANS)
    plans.reverse()
    assert "arm_roster_mismatch" in mod.arm_budget_errors(plans)

    admitted = mod.admit_action(
        unit_key="model|instance|single_shot",
        exact_penalty=0,
        requested_action="accept",
        learned_priority_score=0.5,
    )
    assert admitted["executed"] is True
    assert admitted["final_action"] == "accept"
    assert mod.verdict_for_complete_run(promoted=1, accepted_errors=0, collapsed=False)[0] == (
        "disqualified"
    )

    assert mod._failed_constraint_classes("sat_logic", {}, False, {}) == ["parse"]
    assert mod._failed_constraint_classes(
        "sat_logic",
        {"status": "UNKNOWN"},
        True,
        {"constraint_violation_count": 1, "objective_matches": False},
    ) == ["clause", "objective", "status"]
    assert mod._failed_constraint_classes(
        "sat_logic",
        {"status": "SAT"},
        True,
        {"constraint_violation_count": 0, "objective_matches": True},
    ) == ["constraint"]

    incomplete_surface = [
        {
            "arm": mod.ARM_NAMES[0],
            "model_id": mod.REQUIRED_MODEL_IDS[0],
            "base_id": "base",
            "variant_kind": "canonical",
            "exact_success": True,
            "family": "sat_logic",
        }
    ]
    assert mod._surface_rows(incomplete_surface, "relabel") == []
    assert len(mod._model_confound_rows([], artifact["MODEL_SPECS"])) == 3

    assert "review" in mod._system_prompt("self_review", "reviewed_proposal")
    assert "commit" in mod._system_prompt("exact_commitment", "committed_proposal")
    assert "sample_a" in mod._system_prompt("uncertainty_router", "sample_a")
    assert "direct JSON" in mod._system_prompt("single_shot", "proposal")
    seed = mod._request_seed("model", "instance", "arm", "stage")
    assert seed == mod._request_seed("model", "instance", "arm", "stage")
    assert 0 <= seed <= 0x7FFFFFFF


def test_raw_loader_rejects_duplicates_and_hash_drift(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7130-ARTIFACT binds every persisted raw output."""

    missing = tmp_path / "missing.jsonl"
    assert mod.load_raw_rows(missing) == []

    row = {
        "request_key": "model|instance|arm|stage",
        "raw_text": "proposal",
        "raw_output_hash": mod.sha256_text("proposal"),
    }
    valid = tmp_path / "valid.jsonl"
    valid.write_text(mod.canonical_json(row) + "\n", encoding="utf-8")
    assert mod.load_raw_rows(valid) == [row]

    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(
        mod.canonical_json(row) + "\n" + mod.canonical_json(row) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate_request_key"):
        mod.load_raw_rows(duplicate)

    changed = {**row, "raw_output_hash": "sha256:" + "0" * 64}
    drifted = tmp_path / "drifted.jsonl"
    drifted.write_text(mod.canonical_json(changed) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="raw_output_hash_mismatch"):
        mod.load_raw_rows(drifted)
