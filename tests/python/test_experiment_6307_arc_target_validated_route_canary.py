"""Tests for Exp6307 ARC target-validated route canary.

Spec refs: REQ-ARC-WMTE-6307,
SCENARIO-ARC-WMTE-6307-REGISTRY-PRECHECK,
SCENARIO-ARC-WMTE-6307-INACTIVE-HYPOTHESES,
SCENARIO-ARC-WMTE-6307-TARGET-LICENSE,
SCENARIO-ARC-WMTE-6307-MATCHED-THREE-ARM-CELLS,
SCENARIO-ARC-WMTE-6307-ARTIFACT-GUARDS.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_6307_arc_target_validated_route_canary as exp6307


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"


def _artifact(tmp_path: Path) -> dict:
    return exp6307.run(
        date="20260811",
        result_path=tmp_path / exp6307.RESULT_RELATIVE_PATH.name,
        fixture_manifest_path=tmp_path / exp6307.FIXTURE_MANIFEST_RELATIVE_PATH.name,
        live_window_manifest_path=tmp_path / exp6307.LIVE_WINDOW_MANIFEST_RELATIVE_PATH.name,
        raw_output_dir=tmp_path / "raw",
        duration_s=66.0,
        test_exit_codes={command: 0 for command in exp6307.DEFAULT_TEST_COMMANDS},
        model_resolver=exp6307.deterministic_test_model_resolver,
        llm_runner=exp6307.deterministic_test_llm_runner,
        write=True,
    )


def test_req_arc_wmte_6307_spec_declares_target_license_contract() -> None:
    """REQ-ARC-WMTE-6307: OpenSpec names the three-arm target-license contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-ARC-WMTE-6307") :]
    for marker in (
        "SCENARIO-ARC-WMTE-6307-REGISTRY-PRECHECK",
        "SCENARIO-ARC-WMTE-6307-INACTIVE-HYPOTHESES",
        "SCENARIO-ARC-WMTE-6307-TARGET-LICENSE",
        "SCENARIO-ARC-WMTE-6307-MATCHED-THREE-ARM-CELLS",
        "SCENARIO-ARC-WMTE-6307-ARTIFACT-GUARDS",
        "retrieval is not transfer",
        "exactly `1.0`",
        exp6307.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert marker in section
    for arm in exp6307.ARMS:
        assert f"`{arm}`" in section
    for model_id in exp6307.MANDATED_MODEL_IDS:
        assert model_id in section
    for field in exp6307.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_arc_wmte_6307_registry_precheck_and_fresh_seals(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6307-REGISTRY-PRECHECK: registry is checked before fresh seals."""

    registry = exp6307.registry_precheck()
    fixtures = exp6307.build_fresh_fixtures(seed=6307, per_mechanic=2)
    fixture_payload = exp6307.fixture_manifest_payload(fixtures, seed=6307)
    windows = exp6307.build_live_transition_windows(fixtures, seeds=(101, 202))
    window_payload = exp6307.live_window_manifest_payload(windows)

    fixture_receipt = exp6307.write_manifest(
        tmp_path / "fixtures.json", fixture_payload, write=True
    )
    window_receipt = exp6307.write_manifest(tmp_path / "windows.json", window_payload, write=True)

    assert registry["precheck_order"] == "registry_before_fixture_seal"
    assert registry["target_present_in_registry"] is False
    assert registry["public_level_targeted"] is False
    assert fixture_payload["freshness"] == "exp6307_seeded_synthetic_not_exp6294_or_registry_reuse"
    assert fixture_payload["prior_solve_duplication_count"] == 0
    assert fixture_payload["family_counts"] == {"push_block": 2, "toggle_move": 2}
    assert all(row["game_id"] is None for row in fixture_payload["fixtures"])
    assert all(row["hidden_source_used"] is False for row in window_payload["windows"])
    assert fixture_receipt["sha256"] == exp6307.sha256_json(fixture_payload)
    assert window_receipt["sha256"] == exp6307.sha256_json(window_payload)


def test_scenario_arc_wmte_6307_inactive_retrieval_and_target_license() -> None:
    """SCENARIO-ARC-WMTE-6307-TARGET-LICENSE: only target evidence activates the route."""

    policy = exp6307.TargetLicensePolicy()
    fixture = exp6307.build_fresh_fixtures(seed=6307, per_mechanic=1)[0]

    off = policy.evaluate(fixture.transitions, fixture.family, "router_off")
    retrieval = policy.evaluate(fixture.transitions, fixture.family, "retrieval_only_static_route")
    licensed = policy.evaluate(fixture.transitions, fixture.family, "target_licensed_route")
    false_license = policy.evaluate(fixture.transitions, "toggle_move", "target_licensed_route")

    assert off.route_active is False
    assert off.abstained is True
    assert retrieval.route_active is False
    assert retrieval.rejected is True
    assert retrieval.reason == "retrieval_is_not_transfer_without_target_license"
    assert licensed.route_active is True
    assert licensed.licensed is True
    assert licensed.mutation_receipt["mutation_control_rejected"] is True
    assert false_license.route_active is False
    assert false_license.rejected is True
    assert false_license.license_predicates["class_agrees_with_retrieval"] is False


def test_scenario_arc_wmte_6307_matched_three_arm_contract() -> None:
    """SCENARIO-ARC-WMTE-6307-MATCHED-THREE-ARM-CELLS: all arms match except route state."""

    fixtures = exp6307.build_fresh_fixtures(seed=6307, per_mechanic=1)
    windows = exp6307.build_live_transition_windows(fixtures, seeds=(303,))
    models = exp6307.deterministic_test_model_resolver(False)
    requests = exp6307.build_matched_arm_requests(windows, models)
    exp6307.validate_matched_requests(requests)

    assert len(requests) == len(windows) * len(models) * len(exp6307.ARMS)
    for group in exp6307.group_requests_by_cell(requests).values():
        by_arm = {row["arm"]: row for row in group}
        assert set(by_arm) == set(exp6307.ARMS)
        baseline = by_arm["router_off"]
        for arm in exp6307.ARMS:
            for key in (
                "fixture_id",
                "window_id",
                "mechanic",
                "seed",
                "action_budget",
                "model_budget_tokens",
                "model_id",
                "quantization",
                "starting_history_hash",
                "model_call_index",
            ):
                assert by_arm[arm][key] == baseline[key]
        assert by_arm["router_off"]["route_active"] is False
        assert by_arm["retrieval_only_static_route"]["route_active"] is False
        assert by_arm["target_licensed_route"]["route_active"] is True


def test_scenario_arc_wmte_6307_artifact_schema_metrics_and_receipts(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6307-ARTIFACT-GUARDS: complete artifact has exact ready score."""

    artifact = _artifact(tmp_path)
    loaded = json.loads((tmp_path / exp6307.RESULT_RELATIVE_PATH.name).read_text(encoding="utf-8"))

    assert loaded == artifact
    assert set(exp6307.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(exp6307.REQUIRED_ARTIFACT_FIELDS) == set(artifact["field_principles"])
    assert set(exp6307.REQUIRED_ARTIFACT_FIELDS) == set(artifact["field_provenance"])
    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert set(artifact["models_used"]) == set(exp6307.MANDATED_MODEL_IDS)
    assert artifact["duration_padding_count"] == 0
    assert artifact["arc_target_licensed_router_ready_score"] == 1.0
    assert artifact["actual_work_duration_receipt"]["measured_actual_work_s"] == 66.0
    assert artifact["random_seed"] == exp6307.RANDOM_SEEDS[0]
    assert artifact["paired_causal_deltas_intervals_and_sample_sizes"]["all_cells_ready"] is True
    assert artifact["baseline_harm_controls"]["baseline_harm_detected"] is False
    assert artifact["raw_output_paths_and_hashes"]
    assert artifact["reproducibility_checksum"] == exp6307.payload_checksum(artifact)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("hidden_game_source_access_count", 1, "hidden_game_source_access_count"),
        ("outer_loop_ground_truth_search_count", 1, "outer_loop_ground_truth_search_count"),
        ("arc_level_solve_claim_count", 1, "arc_level_solve_claim_count"),
        ("registry_update_count", 1, "registry_update_count"),
        ("source_model_weight_mutation_count", 1, "source_model_weight_mutation_count"),
        ("duration_padding_count", 1, "duration_padding_count"),
        ("solve_provenance", "development_proxy", "solve_provenance"),
        ("verifier_is_oracle", True, "verifier_is_oracle"),
    ],
)
def test_scenario_arc_wmte_6307_provenance_guard_rejects_forbidden_drift(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    """SCENARIO-ARC-WMTE-6307-ARTIFACT-GUARDS: forbidden counters stay bare zero."""

    artifact = _artifact(tmp_path)
    bad = dict(artifact)
    bad[field] = value
    bad["reproducibility_checksum"] = exp6307.payload_checksum(bad)

    with pytest.raises(ValueError, match=message):
        exp6307.validate_artifact(bad)


def test_scenario_arc_wmte_6307_validation_rejects_false_license_and_arm_drift(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-6307-ARTIFACT-GUARDS: false licenses and arm drift fail closed."""

    artifact = _artifact(tmp_path)

    bad_license = dict(artifact)
    bad_license["target_validation_predicates_and_mutation_receipts"] = {
        **artifact["target_validation_predicates_and_mutation_receipts"],
        "all_target_licensed_activations_mutation_proven": False,
    }
    bad_license["reproducibility_checksum"] = exp6307.payload_checksum(bad_license)
    with pytest.raises(ValueError, match="target_validation_predicates_and_mutation_receipts"):
        exp6307.validate_artifact(bad_license)

    bad_retrieval = dict(artifact)
    bad_retrieval["hypothesis_retrieval_activation_rejection_and_abstention_counts"] = {
        **artifact["hypothesis_retrieval_activation_rejection_and_abstention_counts"],
        "retrieval_only_activation_count": 1,
    }
    bad_retrieval["reproducibility_checksum"] = exp6307.payload_checksum(bad_retrieval)
    with pytest.raises(
        ValueError, match="hypothesis_retrieval_activation_rejection_and_abstention_counts"
    ):
        exp6307.validate_artifact(bad_retrieval)

    bad_arm = dict(artifact)
    bad_arm["router_off_retrieval_only_and_target_licensed_arm_contract"] = {
        **artifact["router_off_retrieval_only_and_target_licensed_arm_contract"],
        "all_cells_matched": False,
    }
    bad_arm["reproducibility_checksum"] = exp6307.payload_checksum(bad_arm)
    with pytest.raises(
        ValueError, match="router_off_retrieval_only_and_target_licensed_arm_contract"
    ):
        exp6307.validate_artifact(bad_arm)


def test_scenario_arc_wmte_6307_validation_rejects_model_substitution_and_harm(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-6307-ARTIFACT-GUARDS: models and harm controls are enforced."""

    artifact = _artifact(tmp_path)

    bad_model = dict(artifact)
    bad_model["MODEL_SPECS"] = [dict(bad_model["MODEL_SPECS"][0])]
    bad_model["models_used"] = [exp6307.MANDATED_MODEL_IDS[0]]
    bad_model["reproducibility_checksum"] = exp6307.payload_checksum(bad_model)
    with pytest.raises(ValueError, match="MODEL_SPECS"):
        exp6307.validate_artifact(bad_model)

    bad_harm = dict(artifact)
    bad_harm["baseline_harm_controls"] = {
        **artifact["baseline_harm_controls"],
        "baseline_harm_detected": True,
    }
    bad_harm["reproducibility_checksum"] = exp6307.payload_checksum(bad_harm)
    with pytest.raises(ValueError, match="baseline_harm_controls"):
        exp6307.validate_artifact(bad_harm)


def test_req_arc_wmte_6307_blocked_precondition_artifact_is_terminal(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-6307: missing mandated model writes zero-credit terminal artifact."""

    artifact = exp6307.run(
        date="20260811",
        result_path=tmp_path / "artifact.json",
        fixture_manifest_path=tmp_path / "fixtures.json",
        live_window_manifest_path=tmp_path / "windows.json",
        raw_output_dir=tmp_path / "raw",
        duration_s=0.5,
        model_resolver=exp6307.missing_gemma_test_model_resolver,
        llm_runner=exp6307.deterministic_test_llm_runner,
        write=True,
    )

    assert artifact["status"].startswith("blocked_")
    assert artifact["honest_verdict"].startswith("complete: blocked_")
    assert artifact["arc_target_licensed_router_ready_score"] == 0.0
    assert artifact["duration_padding_count"] == 0
    assert artifact["raw_output_paths_and_hashes"] == {}
    assert json.loads((tmp_path / "artifact.json").read_text(encoding="utf-8")) == artifact
    exp6307.validate_artifact(artifact)


def test_req_arc_wmte_6307_helper_edges_cover_fail_closed_paths() -> None:
    """REQ-ARC-WMTE-6307: helper edges preserve the fail-closed canary contract."""

    fixtures = exp6307.build_fresh_fixtures(seed=6307, per_mechanic=1)
    windows = exp6307.build_live_transition_windows(fixtures, seeds=(404,))
    models = exp6307.deterministic_test_model_resolver(False)
    requests = exp6307.build_matched_arm_requests(windows[:1], models[:1])

    missing_arm = [dict(row) for row in requests[:2]]
    with pytest.raises(ValueError, match="three required arms"):
        exp6307.validate_matched_requests(missing_arm)

    bad_router_off = [dict(row) for row in requests[:3]]
    bad_router_off[0]["route_active"] = True
    with pytest.raises(ValueError, match="router_off route_active"):
        exp6307.validate_matched_requests(bad_router_off)

    mismatched = [dict(row) for row in requests[:3]]
    mismatched[2]["seed"] = 999
    with pytest.raises(ValueError, match="matched cell seed"):
        exp6307.validate_matched_requests(mismatched)

    bad_retrieval = [dict(row) for row in requests[:3]]
    bad_retrieval[1]["route_active"] = True
    with pytest.raises(ValueError, match="retrieval_only_static_route"):
        exp6307.validate_matched_requests(bad_retrieval)

    bad_target = [dict(row) for row in requests[:3]]
    bad_target[2]["licensed"] = False
    with pytest.raises(ValueError, match="target_licensed_route license"):
        exp6307.validate_matched_requests(bad_target)

    bad_mutation = [dict(row) for row in requests[:3]]
    bad_mutation[2]["mutation_receipt"] = {
        **bad_mutation[2]["mutation_receipt"],
        "mutation_control_rejected": False,
    }
    with pytest.raises(ValueError, match="target_licensed_route mutation"):
        exp6307.validate_matched_requests(bad_mutation)

    with pytest.raises(ValueError, match="unknown arm"):
        exp6307.TargetLicensePolicy().evaluate(fixtures[0].transitions, fixtures[0].family, "bad")

    assert exp6307._extract_python("no code here") == "no code here"
    assert exp6307._executable_acceptance("```python\ndef engine(:\n```") is False
    assert exp6307._invalid_action_rate("no action markers") == 0.0
    assert exp6307._invalid_action_rate("ACTION4 ACTION9") == 0.5
    assert exp6307._mean_record([]) == {"mean": 0.0, "sample_size": 0}
    assert exp6307._paired_delta_receipt([])["sample_size_cells"] == 0
    assert (
        exp6307._paired_delta_receipt([{"request": {"cell_key": "partial", "arm": "router_off"}}])[
            "sample_size_cells"
        ]
        == 0
    )
    assert exp6307._model_revision("/cache/snapshots/rev123/model.gguf") == "rev123"
    assert exp6307._model_revision("") is None
    assert exp6307._model_revision("/cache/model.gguf") is None
    assert exp6307._quant_from_path("/cache/model-UD-Q4_K_M.gguf") == "UD-Q4_K_M"
    assert exp6307._quant_from_path("/cache/model.gguf") == exp6307.PREFERRED_QUANT
    assert exp6307._cuda_ready({}, [exp6307.MANDATED_MODEL_IDS[0]]) is False
    assert (
        exp6307._cuda_ready(
            {
                exp6307.MANDATED_MODEL_IDS[0]: {
                    "terminal": False,
                    "offload_observed": True,
                }
            },
            [exp6307.MANDATED_MODEL_IDS[0]],
        )
        is False
    )


def test_req_arc_wmte_6307_receipt_edges_are_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-6307: preflight and upstream receipts fail closed."""

    missing = tmp_path / "missing.json"
    assert exp6307._artifact_class(missing) == "missing"

    unloadable = tmp_path / "bad.json"
    unloadable.write_text("{not-json", encoding="utf-8")
    assert exp6307._artifact_class(unloadable) == "unloadable"

    flagged = tmp_path / "flagged.json"
    flagged.write_text(json.dumps({"flagged_adversarial": True}), encoding="utf-8")
    assert exp6307._artifact_class(flagged) == "flagged"

    unknown = tmp_path / "unknown.json"
    unknown.write_text(json.dumps({}), encoding="utf-8")
    assert exp6307._artifact_class(unknown) == "unknown"

    monkeypatch.setattr(exp6307, "REPO_ROOT", tmp_path)
    preflight = exp6307._exp6298_preflight_receipt()
    assert preflight["exists"] is False
    assert preflight["required_command"] == exp6307.EXP6298_PREFLIGHT_COMMAND

    receipt_path = tmp_path / "external_receipts.json"
    monkeypatch.setattr(exp6307, "EXTERNAL_TEST_RECEIPT_PATH", receipt_path)
    assert exp6307._read_external_test_receipts()[exp6307.RUN_COMMAND] == 0
    receipt_path.write_text("{bad", encoding="utf-8")
    assert exp6307._read_external_test_receipts()[exp6307.RUN_COMMAND] == 0
    receipt_path.write_text(
        json.dumps({exp6307.FOCUSED_TEST_COMMAND: 7, "custom": None}), encoding="utf-8"
    )
    receipts = exp6307._read_external_test_receipts()
    assert receipts[exp6307.RUN_COMMAND] == 0
    assert receipts[exp6307.FOCUSED_TEST_COMMAND] == 7
    assert receipts["custom"] is None


def test_req_arc_wmte_6307_baseline_harm_edges() -> None:
    """REQ-ARC-WMTE-6307: harm controls name missing arms, score, invalid, and latency drift."""

    def row(
        cell_key: str,
        arm: str,
        text: str,
        *,
        route_active: bool = False,
        latency_s: float = 0.0,
    ) -> dict:
        return {
            "request": {
                "cell_key": cell_key,
                "arm": arm,
                "mechanic": "push_block",
                "route_active": route_active,
            },
            "text": text,
            "latency_s": latency_s,
        }

    harm = exp6307._baseline_harm([row("missing", "router_off", "")])
    assert {"cell_key": "missing", "reason": "missing_arm"} in harm["harm_rows"]

    invalid = exp6307._baseline_harm(
        [
            row("invalid", "router_off", "ACTION4"),
            row("invalid", "retrieval_only_static_route", "ACTION4"),
            row("invalid", "target_licensed_route", "ACTION7 ACTION8 ACTION9", route_active=True),
        ]
    )
    assert any(
        item["reason"] == "invalid_rate_increase_vs_router_off" for item in invalid["harm_rows"]
    )

    score_drop = exp6307._baseline_harm(
        [
            row(
                "score",
                "router_off",
                "push_block push block toggle switch move contact object flip ACTION4\n"
                "def engine(grid, action, data):\n    return grid",
            ),
            row("score", "retrieval_only_static_route", "ACTION4"),
            row("score", "target_licensed_route", "ACTION4"),
        ]
    )
    assert any(item["reason"] == "proposal_score_drop" for item in score_drop["harm_rows"])

    latency = exp6307._baseline_harm(
        [
            row("latency", "router_off", "ACTION4", latency_s=1.0),
            row("latency", "retrieval_only_static_route", "ACTION4", latency_s=1.0),
            row("latency", "target_licensed_route", "ACTION4", route_active=True, latency_s=16.0),
        ]
    )
    assert any(item["reason"] == "latency_excess" for item in latency["harm_rows"])


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda a: a.__setitem__("field_principles", {}), "field_principles"),
        (lambda a: a.__setitem__("field_provenance", {}), "field_provenance"),
        (lambda a: a.__setitem__("inference_substrate", "offline_fixture"), "inference_substrate"),
        (lambda a: a.__setitem__("honest_verdict", "not_terminal"), "honest_verdict"),
        (lambda a: a.__setitem__("models_used", [exp6307.MANDATED_MODEL_IDS[0]]), "models_used"),
        (
            lambda a: a["registry_precheck_path_hash_and_target_receipt"].__setitem__(
                "target_present_in_registry", True
            ),
            "registry_precheck_path_hash_and_target_receipt",
        ),
        (
            lambda a: a["router_off_retrieval_only_and_target_licensed_arm_contract"].__setitem__(
                "retrieval_only_activation_count", 1
            ),
            "router_off_retrieval_only_and_target_licensed_arm_contract",
        ),
        (
            lambda a: a["matched_seed_history_action_budget_and_model_call_receipts"].__setitem__(
                "history_mismatch_count", 1
            ),
            "matched_seed_history_action_budget_and_model_call_receipts",
        ),
        (
            lambda a: a["target_validation_predicates_and_mutation_receipts"].__setitem__(
                "false_license_count", 1
            ),
            "target_validation_predicates_and_mutation_receipts",
        ),
        (lambda a: a.__setitem__("duration_s", 1.0), "actual_work_duration_receipt"),
        (
            lambda a: a["paired_causal_deltas_intervals_and_sample_sizes"].__setitem__(
                "all_cells_ready", False
            ),
            "paired_causal_deltas_intervals_and_sample_sizes",
        ),
        (
            lambda a: a.__setitem__("arc_target_licensed_router_ready_score", 0.0),
            "arc_target_licensed_router_ready_score",
        ),
        (
            lambda a: a["cuda_and_gpu_offload_receipts_by_model"][
                exp6307.MANDATED_MODEL_IDS[0]
            ].__setitem__("terminal", False),
            "cuda_and_gpu_offload_receipts_by_model",
        ),
    ],
)
def test_req_arc_wmte_6307_validate_artifact_edge_guards(
    tmp_path: Path, mutate, message: str
) -> None:
    """REQ-ARC-WMTE-6307: artifact validator rejects every protected readiness drift."""

    artifact = _artifact(tmp_path)
    bad = copy.deepcopy(artifact)
    mutate(bad)
    bad["reproducibility_checksum"] = exp6307.payload_checksum(bad)

    with pytest.raises(ValueError, match=message):
        exp6307.validate_artifact(bad)


def test_req_arc_wmte_6307_validation_rejects_checksum_drift(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-6307: reproducibility checksum catches artifact edits."""

    artifact = _artifact(tmp_path)
    bad = dict(artifact)
    bad["reproducibility_checksum"] = "sha256:bad"

    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp6307.validate_artifact(bad)


def test_req_arc_wmte_6307_validation_rejects_missing_required_field(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-6307: every required artifact field is mandatory."""

    artifact = _artifact(tmp_path)
    bad = dict(artifact)
    bad.pop("status")

    with pytest.raises(ValueError, match="missing fields"):
        exp6307.validate_artifact(bad)
