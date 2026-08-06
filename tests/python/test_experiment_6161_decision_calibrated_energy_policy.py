"""Tests for Exp6161 decision-calibrated energy policy freeze.

Spec refs: REQ-VERIFY-6161, REQ-VERIFY-6161-1, REQ-VERIFY-6161-2,
REQ-VERIFY-6161-3, REQ-VERIFY-6161-4, REQ-VERIFY-6161-5,
REQ-VERIFY-6161-6, REQ-VERIFY-6161-7, REQ-VERIFY-6161-8,
REQ-VERIFY-6161-9, REQ-VERIFY-6161-10,
SCENARIO-VERIFY-6161-CALIBRATION-ONLY,
SCENARIO-VERIFY-6161-GROUPED-CV, SCENARIO-VERIFY-6161-CONTROLS,
SCENARIO-VERIFY-6161-FREEZE.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

import scripts.adversarial_verify as adversarial_verify
from carnot import experiment_6161_decision_calibrated_energy_policy as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/verifiable-reasoning/spec.md"


def _passing_exit_codes() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _run_artifact(tmp_path: Path, *, write: bool = False) -> dict[str, Any]:
    return mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        manifest_path=tmp_path / mod.MANIFEST_RELATIVE_PATH.name,
        test_exit_codes=_passing_exit_codes(),
        duration_s=1.5,
        write=write,
    )


def _write_artifact(tmp_path: Path, artifact: dict[str, Any]) -> Path:
    path = tmp_path / mod.RESULT_RELATIVE_PATH.name
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return path


def test_req_6161_spec_declares_calibration_only_policy_freeze() -> None:
    """REQ-VERIFY-6161: spec names the endpoint, fields, and scenarios."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-VERIFY-6161") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-6161-1",
        "REQ-VERIFY-6161-2",
        "REQ-VERIFY-6161-3",
        "REQ-VERIFY-6161-4",
        "REQ-VERIFY-6161-5",
        "REQ-VERIFY-6161-6",
        "REQ-VERIFY-6161-7",
        "REQ-VERIFY-6161-8",
        "REQ-VERIFY-6161-9",
        "REQ-VERIFY-6161-10",
        "SCENARIO-VERIFY-6161-CALIBRATION-ONLY",
        "SCENARIO-VERIFY-6161-GROUPED-CV",
        "SCENARIO-VERIFY-6161-CONTROLS",
        "SCENARIO-VERIFY-6161-FREEZE",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_6161_calibration_only_features_and_grouped_folds(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-6161-CALIBRATION-ONLY/GROUPED-CV: no held or oracle features."""

    artifact = _run_artifact(tmp_path)

    scan = artifact["precommitted_feature_allowlist_and_forbidden_scan"]
    assert scan["evaluated_partitions"] == ["calibration"]
    assert scan["calibration_label_read_count"] == 192
    assert scan["future_known_label_read_count"] == 0
    assert scan["shifted_family_held_label_read_count"] == 0
    assert scan["held_access_count"] == 0
    assert scan["forbidden_found_count"] == 0
    assert scan["ready_zero_if_forbidden"] is True
    assert set(mod.PRECOMMITTED_FEATURE_ALLOWLIST) == set(scan["allowlist"])
    for token in (
        "current_outcome",
        "exact_answer",
        "exact_labels",
        "chronological_index",
        "future_label",
        "held_label",
    ):
        assert token in scan["forbidden_tokens"]

    folds = artifact["calibration_group_and_fold_receipts"]
    assert folds["group_key"] == ["model_hf_id", "family"]
    assert folds["group_count"] == 8
    assert folds["fold_count"] == 4
    assert folds["calibration_row_count"] == 192
    assert folds["future_or_held_rows_used_for_fit_count"] == 0
    for fold in folds["folds"]:
        assert not set(fold["train_groups"]) & set(fold["validation_groups"])
        assert fold["train_row_count"] > 0
        assert fold["validation_row_count"] > 0


def test_req_6161_selects_policy_reports_metrics_and_freezes_manifest(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-6161-5/6/7/9/10: one zero-held policy manifest is frozen."""

    artifact = _run_artifact(tmp_path, write=True)

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete_ready"
    assert artifact["honest_verdict"].startswith("complete_ready:")
    assert "validly frozen" in artifact["honest_verdict"]
    assert artifact["decision_calibrated_policy_ready_score"] == 1.0
    assert artifact["held_access_count"] == 0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert mod.validate_artifact(artifact) is True
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text()) == artifact

    metrics = artifact[
        "per_model_calibration_cost_brier_ece_unsafe_safe_and_descriptive_ranking_metrics"
    ]
    selected = artifact["selected_policy_rationale_without_held_access"]
    assert selected["selected_arm"] == "decision_calibrated_task_energy"
    assert selected["selected_from_partitions"] == ["calibration"]
    assert selected["selection_uses_held_outcomes"] is False
    assert selected["objective_role"]["auroc_auprc"] == "descriptive_only"
    assert selected["control_outperformed_selected_count"] == 0

    for model_id, by_model in metrics["by_model"].items():
        assert model_id in mod.MANDATED_MODEL_IDS
        assert set(mod.CANDIDATE_ARMS) <= set(by_model["arms"])
        chosen = by_model["arms"]["decision_calibrated_task_energy"]
        assert chosen["row_count"] == 96
        assert chosen["unsafe_count"] == 36
        assert chosen["safe_count"] == 60
        assert chosen["unsafe_weighted_cost"] <= 0.0
        assert 0.0 <= chosen["brier"] <= 1.0
        assert 0.0 <= chosen["ece"] <= 1.0
        assert 0.0 <= chosen["auroc"] <= 1.0
        assert 0.0 <= chosen["auprc"] <= 1.0
        assert chosen["action_counts"]["abstain"] >= 0

    manifest = artifact["policy_manifest_path_hash_and_contents"]
    manifest_path = Path(manifest["path"])
    assert manifest_path.exists()
    assert manifest["sha256"] == mod.sha256_file(manifest_path)
    assert manifest["contents_hash"] == mod.policy_manifest_hash(manifest["contents"])
    contents = manifest["contents"]
    assert contents["selected_arm"] == "decision_calibrated_task_energy"
    assert contents["held_access_count_at_freeze"] == 0
    for field in (
        "score_code_hashes",
        "feature_schema",
        "task_statistics",
        "calibration_parameters",
        "threshold",
        "abstention_rule",
        "cost_table",
        "model_specific_policy_data",
        "bootstrap_evaluation_plan",
    ):
        assert field in contents


def test_scenario_6161_controls_threshold_freeze_and_drift(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6161-CONTROLS: controls and boundary cases cannot win."""

    artifact = _run_artifact(tmp_path)

    controls = artifact["shortcut_and_boundary_controls"]
    assert controls["all_required_controls_present"] is True
    assert controls["no_control_outperforms_selected"] is True
    for name in (
        "label_shuffle",
        "outcome_flip",
        "task_shuffle",
        "alias",
        "family_frequency",
        "model_identity",
        "constant_score",
        "threshold_boundary",
    ):
        assert controls[name]["passed"] is True

    freeze = artifact["score_threshold_abstention_and_cost_freeze_receipts"]
    assert freeze["frozen_before_held_access"] is True
    assert freeze["held_access_count_at_freeze"] == 0
    assert freeze["selected_arm"] == "decision_calibrated_task_energy"
    assert freeze["cost_table"]["table_id"] == "exp6159_unsafe_weighted_v1"
    assert freeze["threshold"] == artifact["policy_manifest_path_hash_and_contents"][
        "contents"
    ]["threshold"]

    drift = artifact["chronological_drift_diagnostics"]
    assert drift["chronological_index_used_as_score_feature"] is False
    assert drift["drift_windows"]
    assert all(window["row_count"] > 0 for window in drift["drift_windows"])


def test_req_6161_validation_fails_closed_on_leakage_manifest_and_held_access(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-6161-2/3/8/9/10: validation rejects unsafe policy artifacts."""

    artifact = _run_artifact(tmp_path, write=True)

    bad_forbidden = deepcopy(artifact)
    bad_forbidden["precommitted_feature_allowlist_and_forbidden_scan"][
        "forbidden_found_count"
    ] = 1
    bad_forbidden["decision_calibrated_policy_ready_score"] = mod.ready_score(
        bad_forbidden
    )
    bad_forbidden["status"] = mod.status(bad_forbidden)
    bad_forbidden["honest_verdict"] = mod.honest_verdict(bad_forbidden)
    bad_forbidden["reproducibility_checksum"] = mod.reproducibility_checksum(
        bad_forbidden
    )
    assert bad_forbidden["decision_calibrated_policy_ready_score"] == 0.0
    with pytest.raises(ValueError, match="forbidden"):
        mod.validate_artifact(bad_forbidden)

    bad_held = deepcopy(artifact)
    bad_held["held_access_count"] = 1
    bad_held["decision_calibrated_policy_ready_score"] = mod.ready_score(bad_held)
    bad_held["status"] = mod.status(bad_held)
    bad_held["honest_verdict"] = mod.honest_verdict(bad_held)
    bad_held["reproducibility_checksum"] = mod.reproducibility_checksum(bad_held)
    assert bad_held["decision_calibrated_policy_ready_score"] == 0.0
    with pytest.raises(ValueError, match="held_access_count"):
        mod.validate_artifact(bad_held)

    bad_manifest = deepcopy(artifact)
    bad_manifest["policy_manifest_path_hash_and_contents"]["contents"][
        "selected_arm"
    ] = "global_energy"
    bad_manifest["reproducibility_checksum"] = mod.reproducibility_checksum(
        bad_manifest
    )
    with pytest.raises(ValueError, match="policy_manifest"):
        mod.validate_artifact(bad_manifest)

    bad_score = deepcopy(artifact)
    bad_score["decision_calibrated_policy_ready_score"] = 0.0
    bad_score["reproducibility_checksum"] = mod.reproducibility_checksum(bad_score)
    with pytest.raises(ValueError, match="decision_calibrated_policy_ready_score"):
        mod.validate_artifact(bad_score)

    missing = dict(artifact)
    missing.pop("status")
    with pytest.raises(ValueError, match="missing required"):
        mod.validate_artifact(missing)


def test_req_6161_blocks_on_upstream_gate_and_adversarial_verify(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-6161-1/10: upstream blockers and cached substrate are explicit."""

    blocked = mod.run(
        result_path=tmp_path / "blocked.json",
        manifest_path=tmp_path / "blocked.manifest.json",
        exp6160_artifact={"sota_decision_corpus_ready_score": 0.0},
        test_exit_codes=_passing_exit_codes(),
        duration_s=0.25,
        write=False,
    )
    assert blocked["status"] == "blocked"
    assert blocked["decision_calibrated_policy_ready_score"] == 0.0
    assert "exp6160_ready" in blocked["honest_verdict"]
    assert blocked["structured_gate_receipt"]["calibration_permitted"] is False
    assert mod.validate_artifact(blocked) is True

    artifact = _run_artifact(tmp_path)
    report = adversarial_verify.verify_artifact(_write_artifact(tmp_path, artifact))
    kinds = {flag["kind"] for flag in report["flags"]}
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["preconditions_checked"]["llm_invocation_count"] == 0
    assert "DURATION_TOO_SHORT" not in kinds
    assert "METHODOLOGY_MISSING" not in kinds


def test_req_6161_helper_edges_and_validation_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-6161-3/10: edge helpers and validation branches fail closed."""

    assert mod._load_json(tmp_path / "missing.json") == {}
    assert mod._load_jsonl(tmp_path / "missing.jsonl") == []
    assert mod._std([1.0], default=2.0) == 2.0
    assert mod._brier([], []) == 0.0
    assert mod._ece([], []) == 0.0
    assert mod._select_threshold([], [], {}) == 0.0
    with pytest.raises(ValueError, match="unknown arm"):
        mod._score_entry({}, "unknown", {}, {})

    scan = mod._scan_features(
        [{"event_id": "fixture", "features": {"family": "exact_answer"}}],
        {"held_access_count": 0},
    )
    assert scan["forbidden_found_count"] > 0

    class NoShuffle:
        def __init__(self, _: str) -> None:
            pass

        def shuffle(self, __: list[str]) -> None:
            return None

    monkeypatch.setattr(mod.random, "Random", NoShuffle)
    assert mod._shuffled_task_map([{"family": "a"}, {"family": "b"}]) == {
        "a": "b",
        "b": "a",
    }

    blocked = mod.run(
        result_path=tmp_path / "blocked2.json",
        manifest_path=tmp_path / "blocked2.manifest.json",
        exp6147_artifact={"task_aware_energy_calibration_ready_score": 0.0},
        exp6159_artifact={"decision_calibrated_stream_ready_score": 0.0},
        exp6160_artifact={"sota_decision_corpus_ready_score": 0.0},
        test_exit_codes=_passing_exit_codes(),
        duration_s=0.1,
        write=False,
    )
    assert blocked["status"] == "blocked"
    assert "exp6147_ready" in blocked["honest_verdict"]
    assert "exp6159_ready" in blocked["honest_verdict"]

    artifact = _run_artifact(tmp_path)
    future_fit = deepcopy(artifact)
    future_fit["calibration_group_and_fold_receipts"][
        "future_or_held_rows_used_for_fit_count"
    ] = 1
    assert "future_or_held_rows_used_for_fit" in mod._blocked_reasons(future_fit)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = mod.sha256_text("wrong")
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)

    bad_provenance_type = deepcopy(artifact)
    bad_provenance_type["field_provenance"] = []
    bad_provenance_type["reproducibility_checksum"] = mod.reproducibility_checksum(
        bad_provenance_type
    )
    with pytest.raises(ValueError, match="field_provenance"):
        mod.validate_artifact(bad_provenance_type)

    bad_provenance = deepcopy(artifact)
    bad_provenance["field_provenance"]["status"]["principle"] = "wrong"
    bad_provenance["reproducibility_checksum"] = mod.reproducibility_checksum(
        bad_provenance
    )
    with pytest.raises(ValueError, match="field_provenance:status"):
        mod.validate_artifact(bad_provenance)

    bad_status = deepcopy(artifact)
    bad_status["status"] = "complete_null"
    bad_status["reproducibility_checksum"] = mod.reproducibility_checksum(bad_status)
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(bad_status)

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"] = "complete_ready: wrong"
    bad_verdict["reproducibility_checksum"] = mod.reproducibility_checksum(bad_verdict)
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(bad_verdict)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "cached_sota_event_energy_calibration"
    bad_substrate["decision_calibrated_policy_ready_score"] = mod.ready_score(
        bad_substrate
    )
    bad_substrate["status"] = mod.status(bad_substrate)
    bad_substrate["honest_verdict"] = mod.honest_verdict(bad_substrate)
    bad_substrate["reproducibility_checksum"] = mod.reproducibility_checksum(
        bad_substrate
    )
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(bad_substrate)

    bad_verifier = deepcopy(artifact)
    bad_verifier["verifier_is_oracle"] = True
    bad_verifier["decision_calibrated_policy_ready_score"] = mod.ready_score(
        bad_verifier
    )
    bad_verifier["status"] = mod.status(bad_verifier)
    bad_verifier["honest_verdict"] = mod.honest_verdict(bad_verifier)
    bad_verifier["reproducibility_checksum"] = mod.reproducibility_checksum(
        bad_verifier
    )
    with pytest.raises(ValueError, match="verifier_is_oracle"):
        mod.validate_artifact(bad_verifier)
