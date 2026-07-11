"""Tests for Exp5568 cached verifier coevolution trigger audit.

Spec refs: REQ-VERIFY-5568, SCENARIO-VERIFY-5568.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5568_verifier_coevolution_trigger as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
TEST_PATH = Path("tests/python/test_experiment_5568_verifier_coevolution_trigger.py")


def _load_panel() -> dict[str, object]:
    return json.loads((REPO / mod.UPSTREAM_PANEL_RELATIVE_PATH).read_text(encoding="utf-8"))


def _load_corpus() -> dict[str, object]:
    return json.loads((REPO / mod.UPSTREAM_LABEL_RELATIVE_PATH).read_text(encoding="utf-8"))


def test_req_verify_5568_spec_declares_cached_trigger_contract() -> None:
    """REQ-VERIFY-5568: OpenSpec anchors cached-only residual trigger fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5568") : spec.index("### REQ-VERIFY-5567")]
    normalized = " ".join(section.split())

    assert "SCENARIO-VERIFY-5568" in section
    assert str(mod.RESULT_RELATIVE_PATH) in section
    assert str(mod.UPSTREAM_PANEL_RELATIVE_PATH) in section
    assert str(mod.UPSTREAM_LABEL_RELATIVE_PATH) in section
    assert "SHALL NOT invoke an LLM" in normalized
    assert "SHALL NOT retrain or tune a verifier" in normalized
    assert "SHALL NOT modify `scripts/research_conductor.py`" in normalized
    assert "`cached_only` SHALL be `true`" in normalized
    assert "`llm_invoked` SHALL be `false`" in normalized
    assert "`exact_validator_is_oracle` SHALL be `true`" in normalized
    assert "`verifier_is_oracle` SHALL be `false`" in normalized
    assert mod.INFERENCE_SUBSTRATE in section
    assert "false-accept rate exceeds 0.10" in normalized
    assert "family calibration gap exceeds 0.05" in normalized
    assert "perturbation changes a verdict on more than 0.05" in normalized
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_verify_5568_cached_artifact_emits_trigger(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5568: cached residuals emit the preregistered trigger."""

    artifact = mod.run(
        repo_root=tmp_path,
        panel_artifact=_load_panel(),
        corpus_artifact=_load_corpus(),
        tests_run=[{"command": str(TEST_PATH), "outcome": "passed"}],
    )
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert written == artifact
    assert artifact["upstream_panel_path"] == mod.UPSTREAM_PANEL_RELATIVE_PATH.as_posix()
    assert artifact["upstream_label_path"] == mod.UPSTREAM_LABEL_RELATIVE_PATH.as_posix()
    assert artifact["cached_only"] is True
    assert artifact["llm_invoked"] is False
    assert artifact["no_retraining_performed"] is True
    assert artifact["research_conductor_modified"] is False
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["exact_validator_is_oracle"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["n_independent_instances"] == 36
    assert artifact["scalability_metrics"]["residual_rows_reconstructed"] == 576
    assert artifact["scalability_metrics"]["raw_response_text_available"] is False
    assert artifact["faithfulness_metrics"]["overall"]["abstention_rate"] == pytest.approx(1.0)
    assert artifact["worst_family_false_accept_rate"] == pytest.approx(1.0)
    assert artifact["family_calibration_gap"] == pytest.approx(0.0)
    assert artifact["perturbation_flip_rate"] == pytest.approx(0.0)
    assert artifact["threshold_transferable"] is False
    assert artifact["verifier_coevolution_required"] is True
    assert artifact["bounded_next_action_recommendation"] == mod.BOUNDED_NEXT_ACTION
    assert artifact["honest_verdict"].startswith("complete:")
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)

    invalid_cells = [
        cell
        for cell in artifact["stratification_cells"]
        if cell["corruption_distance"] in (1, 2)
    ]
    assert invalid_cells
    assert {cell["generator_family"] for cell in invalid_cells} == {"gemma", "qwen"}
    assert {cell["verifier_arm"] for cell in invalid_cells} == set(mod.VERIFIER_ARMS)
    assert max(cell["false_accept_rate"] for cell in invalid_cells) == pytest.approx(1.0)
    assert all(cell["abstention_rate"] == pytest.approx(1.0) for cell in invalid_cells)

    mod.validate_artifact(artifact)


def test_req_verify_5568_reconstructs_aggregate_equivalent_rows_and_perturbations() -> None:
    """REQ-VERIFY-5568: residual rows preserve cached counts and exact labels."""

    panel = _load_panel()
    corpus = _load_corpus()
    residual_rows = mod.reconstruct_residual_rows(panel, corpus)

    assert len(residual_rows) == panel["n_candidate_labels"] == 576
    assert {row["generator_family"] for row in residual_rows} == {"gemma", "qwen"}
    assert {row["verifier_arm"] for row in residual_rows} == set(mod.VERIFIER_ARMS)
    assert {row["prediction_source"] for row in residual_rows} == {
        "aggregate_parser_failure_reconstruction"
    }
    assert all(row["cached_verdict"] == "abstain" for row in residual_rows)
    assert sum(row["false_accept"] for row in residual_rows) == 288
    assert sum(row["false_reject"] for row in residual_rows) == 288

    stratified = mod.compute_stratification_cells(residual_rows)
    assert len(stratified) == 96
    mixed = next(
        cell
        for cell in stratified
        if cell["generator_family"] == "qwen"
        and cell["constraint_family"] == "defaults_exceptions"
        and cell["corruption_distance"] == 0
        and cell["verifier_arm"] == "discrete_verdict"
    )
    assert mixed["n"] == 9
    assert mixed["false_reject_rate"] == pytest.approx(1.0)
    assert mixed["calibration_error"] == pytest.approx(1.0)

    robustness = mod.compute_robustness_metrics(residual_rows)
    assert robustness["candidate_order_reversal"]["verdict_flip_rate"] == pytest.approx(0.0)
    assert robustness["semantic_formatting"]["exact_label_flip_rate"] == pytest.approx(0.0)
    assert robustness["criteria_order_permutation"]["verdict_flip_rate"] == pytest.approx(0.0)
    assert robustness["max_verdict_flip_rate"] == pytest.approx(0.0)


def test_req_verify_5568_threshold_transfer_and_validation_fail_closed() -> None:
    """REQ-VERIFY-5568: thresholds are preregistered and blocked inputs stay blocked."""

    artifact = mod.build_artifact(panel_artifact=_load_panel(), corpus_artifact=_load_corpus())
    transfer = artifact["threshold_transfer"]

    assert transfer["paired_unit"] == "instance_id"
    assert transfer["n_independent_instances"] == 36
    assert transfer["threshold_transferable"] is False
    assert transfer["cross_family_intervals"]["discrete_verdict"]["false_accept_rate_diff"][
        "low"
    ] == pytest.approx(0.0)
    assert transfer["cross_family_intervals"]["granular_score"]["calibration_error_diff"][
        "high"
    ] == pytest.approx(0.0)

    no_trigger = mod.evaluate_trigger(
        worst_family_false_accept_rate=0.10,
        family_calibration_gap=0.05,
        perturbation_flip_rate=0.05,
    )
    assert no_trigger["verifier_coevolution_required"] is False
    assert no_trigger["triggered_by"] == []

    yes_trigger = mod.evaluate_trigger(
        worst_family_false_accept_rate=0.100001,
        family_calibration_gap=0.0,
        perturbation_flip_rate=0.0,
    )
    assert yes_trigger["verifier_coevolution_required"] is True
    assert yes_trigger["triggered_by"] == ["worst_family_false_accept_rate"]

    blocked = mod.build_artifact(panel_artifact={}, corpus_artifact={})
    assert blocked["cached_only"] is True
    assert blocked["llm_invoked"] is False
    assert blocked["verifier_coevolution_required"] is False
    assert blocked["honest_verdict"].startswith("blocked_")
    mod.validate_artifact(blocked)

    bad = deepcopy(artifact)
    bad["cached_only"] = False
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    with pytest.raises(ValueError, match="cached_only"):
        mod.validate_artifact(bad)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "bad"
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(bad_checksum)


def test_req_verify_5568_defensive_helpers_cover_cached_boundaries(tmp_path: Path) -> None:
    """REQ-VERIFY-5568: defensive branches stay deterministic and no-LLM."""

    panel_path = tmp_path / mod.UPSTREAM_PANEL_RELATIVE_PATH
    label_path = tmp_path / mod.UPSTREAM_LABEL_RELATIVE_PATH
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    panel_path.write_text(json.dumps(_load_panel()), encoding="utf-8")
    label_path.write_text(json.dumps(_load_corpus()), encoding="utf-8")
    artifact = mod.build_artifact(repo_root=tmp_path)
    assert artifact["cached_only"] is True
    assert artifact["n_independent_instances"] == 36

    candidates = [
        {"exact_label": "valid"},
        {"exact_label": "invalid"},
        {"exact_label": "valid"},
        {"exact_label": "invalid"},
        {"exact_label": "valid"},
        {"exact_label": "invalid"},
    ]
    predictions = mod._aggregate_predictions_for_candidates(
        candidates,
        {"parser_failures": 2, "tp": 1, "tn": 1, "fp": 2, "fn": 2},
    )
    assert predictions == ["abstain", "abstain", "valid", "invalid", "invalid", "valid"]

    one_family_rows = [
        {
            "generator_family": "qwen",
            "verifier_arm": "discrete_verdict",
            "instance_id": "i0",
            "exact_label": "invalid",
            "cached_verdict": "valid",
            "false_accept": True,
            "false_reject": False,
            "abstained": False,
        }
    ]
    interval = mod._paired_family_intervals(
        one_family_rows,
        arm="discrete_verdict",
        iterations=3,
        seed=mod.RANDOM_SEED,
    )
    assert interval["false_accept_rate_diff"]["n_bootstrap"] == 3
    assert mod._worst_false_accept_rate([]) == pytest.approx(0.0)
    assert mod._gap([]) == pytest.approx(0.0)
    assert mod._arms({}) == list(mod.VERIFIER_ARMS)
    assert mod._model_family({"hf_id": "local/Qwen"}) == "qwen"
    assert mod._model_family({"hf_id": "local/gemma"}) == "gemma"
    assert mod._model_family({"hf_id": "local/other"}) == "other"
    assert mod._honest_verdict(False).startswith("complete:")

    synthetic_transfer_rows = [
        {
            "generator_family": "qwen",
            "verifier_arm": "discrete_verdict",
            "instance_id": "i0",
            "exact_label": "valid",
            "cached_verdict": "invalid",
            "false_accept": False,
            "false_reject": True,
            "abstained": False,
        },
        {
            "generator_family": "gemma",
            "verifier_arm": "discrete_verdict",
            "instance_id": "i0",
            "exact_label": "valid",
            "cached_verdict": "valid",
            "false_accept": False,
            "false_reject": False,
            "abstained": False,
        },
    ]
    transfer = mod.compute_threshold_transfer(synthetic_transfer_rows)
    assert "family_calibration_gap_exceeds_threshold" in transfer["non_transfer_reasons"]

    trigger = mod.evaluate_trigger(
        worst_family_false_accept_rate=0.0,
        family_calibration_gap=0.050001,
        perturbation_flip_rate=0.050001,
    )
    assert trigger["triggered_by"] == ["family_calibration_gap", "perturbation_flip_rate"]

    assert mod._load_json(tmp_path / "missing.json")["load_error"] == "missing"
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    assert mod._load_json(malformed)["load_error"] == "json_decode"
    listed = tmp_path / "list.json"
    listed.write_text("[]", encoding="utf-8")
    assert mod._load_json(listed)["load_error"] == "json_not_object"
