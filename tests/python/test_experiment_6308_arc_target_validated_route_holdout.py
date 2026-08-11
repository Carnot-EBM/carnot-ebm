"""Tests for Exp6308 ARC target-validated route holdout.

Spec refs: REQ-ARC-WMTE-6308,
SCENARIO-ARC-WMTE-6308-GATE-REPLAY,
SCENARIO-ARC-WMTE-6308-FOLD-ISOLATION,
SCENARIO-ARC-WMTE-6308-NO-REFIT-DEFAULT-OFF,
SCENARIO-ARC-WMTE-6308-PER-FOLD-GATE,
SCENARIO-ARC-WMTE-6308-ZERO-SOLVE-CREDIT.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_6308_arc_target_validated_route_holdout as exp6308


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"


def _artifact(tmp_path: Path) -> dict:
    return exp6308.run(
        date="20260811",
        result_path=tmp_path / exp6308.RESULT_RELATIVE_PATH.name,
        fold_manifest_path=tmp_path / exp6308.FOLD_MANIFEST_RELATIVE_PATH.name,
        raw_output_dir=tmp_path / "raw",
        duration_s=2.5,
        test_exit_codes={command: 0 for command in exp6308.DEFAULT_TEST_COMMANDS},
        model_resolver=exp6308.exp6307.deterministic_test_model_resolver,
        llm_runner=exp6308.deterministic_holdout_runner,
        write=True,
    )


def test_req_arc_wmte_6308_spec_declares_holdout_contract() -> None:
    """REQ-ARC-WMTE-6308: OpenSpec names the holdout and zero-credit contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-ARC-WMTE-6308") :]
    for marker in (
        "SCENARIO-ARC-WMTE-6308-GATE-REPLAY",
        "SCENARIO-ARC-WMTE-6308-FOLD-ISOLATION",
        "SCENARIO-ARC-WMTE-6308-NO-REFIT-DEFAULT-OFF",
        "SCENARIO-ARC-WMTE-6308-PER-FOLD-GATE",
        "SCENARIO-ARC-WMTE-6308-ZERO-SOLVE-CREDIT",
        "without refitting",
        "pooled mean SHALL NOT promote",
        exp6308.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert marker in section
    for arm in exp6308.ARMS:
        assert f"`{arm}`" in section
    for model_id in exp6308.MANDATED_MODEL_IDS:
        assert model_id in section
    for field in exp6308.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_arc_wmte_6308_gate_replay_and_registry_precheck() -> None:
    """SCENARIO-ARC-WMTE-6308-GATE-REPLAY: Exp6307 ready score gates execution."""

    upstream = exp6308.load_upstream_artifact()
    receipt = exp6308.structured_gate_receipt(upstream)
    registry = exp6308.registry_precheck()

    assert receipt["gate_passed"] is True
    assert receipt["actual_ready_score"] == 1.0
    assert receipt["required_ready_score"] == 1.0
    assert receipt["upstream_honest_verdict"].startswith("complete:")
    assert registry["registry_read_mode"] == "full_text"
    assert registry["target_present_in_registry"] is False
    assert registry["public_level_targeted"] is False
    assert registry["target_receipt"]["proposal_routing_only"] is True

    bad = dict(upstream)
    bad["arc_target_licensed_router_ready_score"] = 0.0
    blocked = exp6308.run(
        date="20260811",
        result_path=Path("/tmp/exp6308_blocked.json"),
        fold_manifest_path=Path("/tmp/exp6308_blocked_folds.json"),
        raw_output_dir=Path("/tmp/exp6308_blocked_raw"),
        upstream_artifact=bad,
        duration_s=0.1,
        model_resolver=exp6308.exp6307.deterministic_test_model_resolver,
        llm_runner=exp6308.deterministic_holdout_runner,
        write=False,
    )
    assert blocked["status"] == "blocked_exp6307_gate_not_ready"
    assert blocked["structured_gate_receipt"]["gate_passed"] is False
    assert blocked["arc_target_licensed_generalization_ready_score"] == 0.0
    exp6308.validate_artifact(blocked)


def test_scenario_arc_wmte_6308_fold_isolation_and_no_refit() -> None:
    """SCENARIO-ARC-WMTE-6308-FOLD-ISOLATION: held cells do not overlap 6307."""

    folds = exp6308.build_held_folds()
    payload = exp6308.held_fold_manifest_payload(folds)
    no_refit = exp6308.no_refit_receipts_by_fold(folds)

    assert payload["built_before_execution"] is True
    assert payload["no_overlap_with_exp6307_cells"] is True
    assert payload["overlap_counts"] == {
        "starting_history_hash": 0,
        "cell_key": 0,
        "fixture_id": 0,
    }
    assert {row["fold_id"] for row in payload["folds"]} == {fold.fold_id for fold in folds}
    assert any(row["adequately_powered"] is False for row in payload["folds"])
    assert any("navigation" in row["mechanics"] for row in payload["folds"])
    for receipt in no_refit.values():
        assert receipt["policy_refit_count"] == 0
        assert receipt["threshold_refit_count"] == 0
        assert receipt["per_game_threshold_count"] == 0
        assert receipt["adapter_refit_count"] == 0
        assert receipt["prompt_refit_count"] == 0


def test_scenario_arc_wmte_6308_default_off_and_matched_requests() -> None:
    """SCENARIO-ARC-WMTE-6308-NO-REFIT-DEFAULT-OFF: arms match and default is false."""

    models = exp6308.exp6307.deterministic_test_model_resolver(False)
    folds = exp6308.build_held_folds()
    requests = exp6308.build_holdout_requests(folds, models)
    exp6308.validate_holdout_requests(requests)
    default = exp6308.default_flag_receipt()

    assert default["flag_name"] == exp6308.DEFAULT_FLAG_NAME
    assert default["flag_value"] is False
    assert default["default_off_protected"] is True
    assert len(requests) == sum(len(f.windows) for f in folds) * len(models) * len(exp6308.ARMS)
    for group in exp6308.group_requests_by_cell(requests).values():
        by_arm = {row["arm"]: row for row in group}
        assert set(by_arm) == set(exp6308.ARMS)
        baseline = by_arm["router_off"]
        for arm in exp6308.ARMS:
            for key in (
                "fold_id",
                "held_game_id",
                "mechanic",
                "seed",
                "action_budget",
                "model_budget_tokens",
                "model_call_budget",
                "model_call_index",
                "model_id",
                "quantization",
                "starting_history_hash",
            ):
                assert by_arm[arm][key] == baseline[key]
        assert by_arm["router_off"]["route_active"] is False
        assert by_arm["retrieval_only_static_route"]["route_active"] is False
        assert by_arm["target_licensed_route"]["route_active"] is True


def test_scenario_arc_wmte_6308_artifact_schema_metrics_and_fold_gate(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6308-PER-FOLD-GATE: ready score is per-fold, not pooled."""

    artifact = _artifact(tmp_path)
    loaded = json.loads((tmp_path / exp6308.RESULT_RELATIVE_PATH.name).read_text(encoding="utf-8"))

    assert loaded == artifact
    assert set(exp6308.REQUIRED_ARTIFACT_FIELDS) == set(artifact["field_principles"])
    assert set(exp6308.REQUIRED_ARTIFACT_FIELDS) == set(artifact["field_provenance"])
    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["default_flag_name_value_and_receipt"]["flag_value"] is False
    assert set(artifact["models_used"]) == set(exp6308.MANDATED_MODEL_IDS)
    assert artifact["arc_target_licensed_generalization_ready_score"] == 1.0
    assert artifact["missing_underpowered_or_harmful_folds"]
    assert any(
        row["classification"] == "underpowered"
        for row in artifact["missing_underpowered_or_harmful_folds"]
    )
    assert all(
        row["fold_ready"] is True
        for row in artifact["paired_deltas_intervals_and_sample_sizes_by_fold"].values()
        if row["adequately_powered"]
    )
    assert all(
        row["baseline_harm_detected"] is False
        for row in artifact["baseline_harm_controls_by_fold"].values()
    )
    assert artifact["raw_output_paths_and_hashes"]
    assert artifact["reproducibility_checksum"] == exp6308.payload_checksum(artifact)
    exp6308.validate_artifact(artifact)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("hidden_game_source_access_count", 1, "hidden_game_source_access_count"),
        ("outer_loop_ground_truth_search_count", 1, "outer_loop_ground_truth_search_count"),
        ("arc_level_solve_claim_count", 1, "arc_level_solve_claim_count"),
        ("registry_update_count", 1, "registry_update_count"),
        ("source_model_weight_mutation_count", 1, "source_model_weight_mutation_count"),
        ("solve_provenance", "development_proxy", "solve_provenance"),
        ("verifier_is_oracle", True, "verifier_is_oracle"),
    ],
)
def test_scenario_arc_wmte_6308_zero_solve_credit_guards(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    """SCENARIO-ARC-WMTE-6308-ZERO-SOLVE-CREDIT: forbidden counters stay bare zero."""

    artifact = _artifact(tmp_path)
    bad = copy.deepcopy(artifact)
    bad[field] = value
    bad["reproducibility_checksum"] = exp6308.payload_checksum(bad)

    with pytest.raises(ValueError, match=message):
        exp6308.validate_artifact(bad)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda a: a["default_flag_name_value_and_receipt"].__setitem__(
                "flag_value", True
            ),
            "default_flag_name_value_and_receipt",
        ),
        (
            lambda a: a["no_refit_receipts_by_fold"][next(iter(a["no_refit_receipts_by_fold"]))].__setitem__(
                "threshold_refit_count", 1
            ),
            "no_refit_receipts_by_fold",
        ),
        (
            lambda a: a["paired_deltas_intervals_and_sample_sizes_by_fold"][
                "held_games_alpha"
            ].__setitem__("fold_ready", False),
            "paired_deltas_intervals_and_sample_sizes_by_fold",
        ),
        (
            lambda a: a["baseline_harm_controls_by_fold"]["held_games_alpha"].__setitem__(
                "baseline_harm_detected", True
            ),
            "baseline_harm_controls_by_fold",
        ),
        (lambda a: a.__setitem__("field_principles", {}), "field_principles"),
        (lambda a: a.__setitem__("field_provenance", {}), "field_provenance"),
        (lambda a: a.__setitem__("inference_substrate", "live_llm_inference"), "inference_substrate"),
        (lambda a: a.__setitem__("honest_verdict", "not_terminal"), "honest_verdict"),
        (lambda a: a.__setitem__("MODEL_SPECS", [dict(a["MODEL_SPECS"][0])]), "MODEL_SPECS"),
        (lambda a: a.__setitem__("models_used", [exp6308.MANDATED_MODEL_IDS[0]]), "models_used"),
        (
            lambda a: a["default_flag_name_value_and_receipt"].__setitem__(
                "default_off_protected", False
            ),
            "default_flag_name_value_and_receipt",
        ),
        (
            lambda a: a["registry_precheck_path_hash_and_target_receipt"].__setitem__(
                "target_present_in_registry", True
            ),
            "registry_precheck_path_hash_and_target_receipt",
        ),
        (
            lambda a: a["structured_gate_receipt"].__setitem__("gate_passed", False),
            "structured_gate_receipt",
        ),
        (
            lambda a: a["structured_gate_receipt"].__setitem__("actual_ready_score", 0.5),
            "structured_gate_receipt",
        ),
        (
            lambda a: a[
                "held_game_mechanic_model_seed_and_transition_fold_manifest_path_and_hash"
            ].__setitem__("sealed_before_execution", False),
            "held_game_mechanic_model_seed_and_transition_fold_manifest_path_and_hash",
        ),
        (
            lambda a: a.__setitem__("missing_underpowered_or_harmful_folds", []),
            "missing_underpowered_or_harmful_folds",
        ),
        (
            lambda a: a.__setitem__("arc_target_licensed_generalization_ready_score", 0.0),
            "arc_target_licensed_generalization_ready_score",
        ),
    ],
)
def test_scenario_arc_wmte_6308_artifact_validation_rejects_drift(
    tmp_path: Path, mutate, message: str
) -> None:
    """SCENARIO-ARC-WMTE-6308-ZERO-SOLVE-CREDIT: validation rejects protected drift."""

    artifact = _artifact(tmp_path)
    bad = copy.deepcopy(artifact)
    mutate(bad)
    bad["reproducibility_checksum"] = exp6308.payload_checksum(bad)

    with pytest.raises(ValueError, match=message):
        exp6308.validate_artifact(bad)


def test_scenario_arc_wmte_6308_request_and_checksum_edges(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-6308: helper guards catch arm mismatch and checksum drift."""

    models = exp6308.exp6307.deterministic_test_model_resolver(False)
    requests = exp6308.build_holdout_requests(exp6308.build_held_folds()[:1], models[:1])

    with pytest.raises(ValueError, match="three required arms"):
        exp6308.validate_holdout_requests(requests[:2])

    mismatched = [dict(row) for row in requests[:3]]
    mismatched[2]["model_budget_tokens"] = 999
    with pytest.raises(ValueError, match="matched cell model_budget_tokens"):
        exp6308.validate_holdout_requests(mismatched)

    bad_target = [dict(row) for row in requests[:3]]
    bad_target[2]["route_active"] = False
    with pytest.raises(ValueError, match="target_licensed_route license"):
        exp6308.validate_holdout_requests(bad_target)

    bad_router = [dict(row) for row in requests[:3]]
    bad_router[0]["route_active"] = True
    with pytest.raises(ValueError, match="router_off route_active"):
        exp6308.validate_holdout_requests(bad_router)

    bad_retrieval = [dict(row) for row in requests[:3]]
    bad_retrieval[1]["route_active"] = True
    with pytest.raises(ValueError, match="retrieval_only_static_route"):
        exp6308.validate_holdout_requests(bad_retrieval)

    artifact = _artifact(tmp_path)
    bad = dict(artifact)
    bad["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp6308.validate_artifact(bad)

    missing = dict(artifact)
    missing.pop("status")
    with pytest.raises(ValueError, match="missing fields"):
        exp6308.validate_artifact(missing)


def test_req_arc_wmte_6308_helper_edge_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-6308: helper edges preserve terminal and fold classifications."""

    assert exp6308._terminal_class({"flagged_adversarial": True}) == "flagged"
    assert exp6308._terminal_class({}) == "unknown"

    folds = exp6308.build_held_folds()
    partial = [
        {
            "request": {
                "cell_key": "partial",
                "arm": "router_off",
                "fold_id": folds[0].fold_id,
            },
            "text": "",
            "latency_s": 0.0,
        }
    ]
    paired = exp6308._paired_by_fold(partial, folds, {})
    assert paired[folds[0].fold_id]["sample_size_cells"] == 0

    preserved = exp6308._preserved_folds(
        {
            "missing": {"sample_size_cells": 0, "adequately_powered": True, "fold_ready": False},
            "harmful": {"sample_size_cells": 4, "adequately_powered": True, "fold_ready": True},
            "failed": {"sample_size_cells": 4, "adequately_powered": True, "fold_ready": False},
        },
        {"harmful": {"baseline_harm_detected": True}},
    )
    assert {row["classification"] for row in preserved} == {
        "missing",
        "harmful",
        "failed_adequate_fold",
    }

    receipt_path = tmp_path / "receipts.json"
    monkeypatch.setattr(exp6308, "EXTERNAL_TEST_RECEIPT_PATH", receipt_path)
    receipt_path.write_text(
        json.dumps({exp6308.FOCUSED_TEST_COMMAND: 3, "custom": None}), encoding="utf-8"
    )
    receipts = exp6308._read_external_test_receipts()
    assert receipts[exp6308.RUN_COMMAND] == 0
    assert receipts[exp6308.FOCUSED_TEST_COMMAND] == 3
    assert receipts["custom"] is None
