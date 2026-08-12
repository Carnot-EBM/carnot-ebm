"""Tests for Exp6348 default-off ARC action-influence A/B.

Spec refs: REQ-ARC-WMTE-6348,
SCENARIO-ARC-WMTE-6348-GATE-AND-SEALS,
SCENARIO-ARC-WMTE-6348-MODEL-RECEIPTS,
SCENARIO-ARC-WMTE-6348-MATCHED-ARMS,
SCENARIO-ARC-WMTE-6348-CAUSAL-QUALITY-GATE,
SCENARIO-ARC-WMTE-6348-ARTIFACT-GUARDS.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6348_arc_default_off_action_influence_ab as exp6348


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"


def _fake_model_file(tmp_path: Path, model_id: str) -> str:
    name = model_id.split("/")[-1]
    snap = tmp_path / f"models--{model_id.replace('/', '--')}" / "snapshots" / f"rev-{name}"
    snap.mkdir(parents=True, exist_ok=True)
    path = snap / f"{name}-Q4_K_M.gguf"
    path.write_text(f"fake {model_id}", encoding="utf-8")
    return str(path)


def _fake_pair_resolver_factory(tmp_path: Path):
    def fake_pair_resolver(
        gpu_indices: tuple[int, int] = (0, 1),
        preferred_quant: str = "Q4_K_M",
        model_indices: tuple[int, int] | None = None,
    ) -> list[dict[str, Any]]:
        _ = preferred_quant
        indices = model_indices or (0, 1)
        out = []
        for gpu, index in zip(gpu_indices, indices, strict=True):
            model_id = exp6348.MANDATED_MODEL_IDS[index]
            out.append(
                {
                    "name": model_id.split("/")[-1].replace("-GGUF", ""),
                    "hf_id": model_id,
                    "gpu": gpu,
                    "model_path": _fake_model_file(tmp_path, model_id),
                }
            )
        return out

    return fake_pair_resolver


def _fake_tokenizer_checker(model_path: str | None) -> tuple[bool, str]:
    return bool(model_path), "embedded GGUF tokenizer OK (test)"


def _fake_cuda_receipts(models: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(model["hf_id"]): {
            "terminal": True,
            "canonical_llama_cpp": True,
            "embedded_tokenizer_probe_ok": True,
            "full_weight_load_attempted": True,
            "loaded_one_placement_at_a_time": True,
            "memory_released": True,
            "gpu": int(model["gpu"]),
            "errors": [],
        }
        for model in models
    }


def _artifact(tmp_path: Path) -> dict[str, Any]:
    return exp6348.run(
        date="20260812",
        result_path=tmp_path / exp6348.RESULT_RELATIVE_PATH.name,
        prospective_registration_path=tmp_path
        / exp6348.PROSPECTIVE_REGISTRATION_RELATIVE_PATH.name,
        fresh_manifest_path=tmp_path / exp6348.FRESH_WINDOW_MANIFEST_RELATIVE_PATH.name,
        duration_s=2.5,
        test_exit_codes={command: 0 for command in exp6348.DEFAULT_TEST_COMMANDS},
        model_pair_resolver=_fake_pair_resolver_factory(tmp_path),
        tokenizer_checker=_fake_tokenizer_checker,
        cuda_receipt_collector=_fake_cuda_receipts,
        write=True,
    )


def _with_checksum(payload: dict[str, Any]) -> dict[str, Any]:
    payload["reproducibility_checksum"] = exp6348.payload_checksum(payload)
    return payload


def test_req_arc_wmte_6348_spec_declares_fresh_ab_contract() -> None:
    """REQ-ARC-WMTE-6348: OpenSpec names the fresh default-off A/B contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-ARC-WMTE-6348") :]
    for marker in (
        "SCENARIO-ARC-WMTE-6348-GATE-AND-SEALS",
        "SCENARIO-ARC-WMTE-6348-MODEL-RECEIPTS",
        "SCENARIO-ARC-WMTE-6348-MATCHED-ARMS",
        "SCENARIO-ARC-WMTE-6348-CAUSAL-QUALITY-GATE",
        "SCENARIO-ARC-WMTE-6348-ARTIFACT-GUARDS",
        "default-off",
        "fresh live-window manifest",
        exp6348.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert marker in section
    for field in exp6348.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for model_id in exp6348.MANDATED_MODEL_IDS:
        assert model_id in section


def test_scenario_arc_wmte_6348_model_specs_use_cached_sota_pair(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6348-MODEL-RECEIPTS: all three models resolve through the helper."""

    models = exp6348.build_model_specs(model_pair_resolver=_fake_pair_resolver_factory(tmp_path))

    assert [row["hf_id"] for row in models] == list(exp6348.MANDATED_MODEL_IDS)
    assert all(row["model_exists"] is True for row in models)
    assert all(row["model_sha256"].startswith("sha256:") for row in models)
    assert all(row["quantization"] == "Q4_K_M" for row in models)
    assert all("cached_sota_pair(gpu_indices=(0, 1)" in row["resolved_via"] for row in models)

    with pytest.raises(ValueError, match="cached_sota_pair"):
        exp6348.build_model_specs(model_pair_resolver=lambda **_: None)

    def missing_one_resolver(**kwargs):
        _ = kwargs
        return [
            {
                "name": "Qwen3.6-35B-A3B",
                "hf_id": exp6348.MANDATED_MODEL_IDS[0],
                "gpu": 0,
                "model_path": _fake_model_file(tmp_path, exp6348.MANDATED_MODEL_IDS[0]),
            },
            {
                "name": "Gemma4-31B-it",
                "hf_id": exp6348.MANDATED_MODEL_IDS[1],
                "gpu": 1,
                "model_path": _fake_model_file(tmp_path, exp6348.MANDATED_MODEL_IDS[1]),
            },
        ]

    with pytest.raises(ValueError, match="missing mandated models"):
        exp6348.build_model_specs(model_pair_resolver=missing_one_resolver)


def test_req_arc_wmte_6348_helper_edge_receipts(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-ARC-WMTE-6348: helper edges keep receipt parsing deterministic."""

    assert exp6348._quant_from_path("/tmp/model.gguf") == "unknown"
    assert exp6348._model_revision(None) is None
    assert exp6348._model_revision("/tmp/model-Q4_K_M.gguf") is None

    window = exp6348.fresh_live_window_manifest_payload()["rows"][0]
    monkeypatch.setattr(exp6348, "_route_license", lambda _row: {"route_reachable": False})
    assert exp6348._ordered_actions(window, route_enabled=True) == [5, 4]


def test_scenario_arc_wmte_6348_registry_precheck_and_gate_replay() -> None:
    """SCENARIO-ARC-WMTE-6348-GATE-AND-SEALS: prechecks run before model load."""

    clean = exp6348.registry_precheck(registry_text="")
    duplicate = exp6348.registry_precheck(registry_text=exp6348.INFLUENCE_TASK_ID)
    gate = exp6348.upstream_path_hash_terminal_class_and_gate_receipt()

    assert clean["precheck_order"] == "registry_before_model_load"
    assert clean["task_kind"] == "fresh_live_action_influence_ab_not_solve"
    assert clean["all_selected_targets_nonduplicate"] is True
    assert clean["registry_update_count"] == 0
    assert duplicate["all_selected_targets_nonduplicate"] is False
    assert duplicate["duplicate_solve_proposal_count"] == 1
    assert gate["structured_gate_replayed"] is True
    assert gate["exp6347_action_influence_eligible"] is True
    assert exp6348._terminal_class({"flagged_adversarial": True}) == "flagged"
    assert exp6348._terminal_class({"status": "blocked_precondition"}) == "blocked"


def test_scenario_arc_wmte_6348_seals_fresh_windows(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6348-GATE-AND-SEALS: registration and windows are sealed."""

    registration = exp6348.prospective_registration_payload(date="20260812")
    windows = exp6348.fresh_live_window_manifest_payload()
    reg_receipt = exp6348.write_sealed_payload(tmp_path / "registration.json", registration, write=True)
    win_receipt = exp6348.write_sealed_payload(tmp_path / "windows.json", windows, write=True)

    assert registration["sealed_before_model_generation"] is True
    assert windows["sealed_before_model_generation"] is True
    assert windows["fresh_live_agent_windows"] is True
    assert windows["row_count"] == len(exp6348.SELECTED_WINDOWS) * len(exp6348.RANDOM_SEEDS)
    assert reg_receipt["sealed_before_model_generation"] is True
    assert win_receipt["row_count"] == windows["row_count"]
    assert all(row["agent_owned_policy_transition_store"] is True for row in windows["rows"])
    assert all(row["raw_candidate_actions"] == [5, 4] for row in windows["rows"])


def test_scenario_arc_wmte_6348_matched_arms_preserve_action_ownership(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-6348-MATCHED-ARMS: route-on reorders only owned actions."""

    models = exp6348.build_model_specs(model_pair_resolver=_fake_pair_resolver_factory(tmp_path))
    windows = exp6348.fresh_live_window_manifest_payload()["rows"]
    ab = exp6348.run_matched_route_ab(models=models, windows=windows)

    assert ab["row_count"] == len(models) * len(windows)
    assert ab["route_caused_action_order_change_count"] == ab["row_count"]
    assert ab["action_injection_count"] == 0
    assert all(row["route_off_order"] == [5, 4] for row in ab["rows"])
    assert all(row["target_licensed_route_on_order"] == [4, 5] for row in ab["rows"])
    assert all(row["same_legal_action_set"] is True for row in ab["rows"])
    assert all(row["route_on_actions_subset_of_raw_candidates"] is True for row in ab["rows"])


def test_scenario_arc_wmte_6348_quality_deltas_and_controls(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6348-CAUSAL-QUALITY-GATE: every model has positive deltas."""

    models = exp6348.build_model_specs(model_pair_resolver=_fake_pair_resolver_factory(tmp_path))
    windows = exp6348.fresh_live_window_manifest_payload()["rows"]
    ab = exp6348.run_matched_route_ab(models=models, windows=windows)
    quality = exp6348.exact_transition_quality_by_cell(ab["rows"])
    paired = exp6348.paired_influence_and_quality_deltas(models=models, ab=ab, quality=quality)
    controls = exp6348.route_deletion_permutation_leakage_and_escape_results(ab["rows"])
    harm = exp6348.harm_underpowered_missing_and_flagged_cells(models=models, paired=paired)

    assert quality["checker"] == exp6348.EXACT_TRANSITION_CHECKER_NAME
    assert quality["positive_route_on_quality_count"] == ab["row_count"]
    assert paired["all_headline_models_positive"] is True
    assert paired["headline_model_count"] == len(models)
    assert paired["overall"]["mean_quality_delta"] > 0
    assert controls["all_controls_passed"] is True
    assert controls["route_deletion_removed_effect_count"] == ab["row_count"]
    assert controls["leakage_overlap_count"] == 0
    assert harm["missing_cell_count"] == 0
    assert harm["harmful_cell_count"] == 0
    assert harm["underpowered_cell_count"] == 0


def test_scenario_arc_wmte_6348_artifact_schema_and_no_solve_claim(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6348-ARTIFACT-GUARDS: artifact is complete and zero-credit."""

    artifact = _artifact(tmp_path)
    loaded = json.loads((tmp_path / exp6348.RESULT_RELATIVE_PATH.name).read_text(encoding="utf-8"))

    assert loaded == artifact
    assert set(exp6348.REQUIRED_ARTIFACT_FIELDS) == set(artifact["field_principles"])
    assert set(exp6348.REQUIRED_ARTIFACT_FIELDS) == set(artifact["field_provenance"])
    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["models_used"] == list(exp6348.MANDATED_MODEL_IDS)
    assert artifact["arc_causal_influence_ready_score"] == 1.0
    assert artifact["verifier_is_oracle"] == exp6348.EXACT_TRANSITION_CHECKER_NAME
    assert artifact["exact_oracle_claim_boundary"]["not_a_solve_oracle"] is True
    assert artifact["route_default_off_and_activation_receipts"]["default_enabled"] is False
    assert artifact["route_default_off_and_activation_receipts"]["activation_requires_explicit_arm"] is True
    assert artifact["matched_call_token_action_time_and_checker_budgets"]["budget_parity"] is True
    assert artifact["raw_model_and_action_paths_hashes_and_counts"]["action_receipt_count"] > 0
    assert artifact["preconditions_checked"]["registry_precheck_before_model_load"] is True
    assert artifact["preconditions_checked"]["fresh_windows_sealed_before_model_generation"] is True
    for field in exp6348.FORBIDDEN_ZERO_FIELDS:
        assert type(artifact[field]) is int
        assert artifact[field] == 0
    assert artifact["reproducibility_checksum"] == exp6348.payload_checksum(artifact)
    exp6348.validate_artifact(artifact)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("hidden_game_source_access_count", 1, "hidden_game_source_access_count"),
        ("offline_ground_truth_bfs_count", 1, "offline_ground_truth_bfs_count"),
        ("hand_game_adapter_count", 1, "hand_game_adapter_count"),
        ("per_game_calibration_count", 1, "per_game_calibration_count"),
        ("source_model_weight_mutation_count", 1, "source_model_weight_mutation_count"),
        ("generated_label_count", 1, "generated_label_count"),
        ("hidden_state_access_count", 1, "hidden_state_access_count"),
        ("solve_claim_count", 1, "solve_claim_count"),
        ("registry_update_count", 1, "registry_update_count"),
        ("solve_provenance", "outer_loop_re", "solve_provenance"),
        ("verifier_is_oracle", "wrong_checker", "verifier_is_oracle"),
        ("inference_substrate", "aggregation_from_upstream_artifacts", "inference_substrate"),
    ],
)
def test_scenario_arc_wmte_6348_validation_rejects_forbidden_drift(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    """SCENARIO-ARC-WMTE-6348-ARTIFACT-GUARDS: forbidden drift is rejected."""

    artifact = _artifact(tmp_path)
    bad = copy.deepcopy(artifact)
    bad[field] = value
    _with_checksum(bad)

    with pytest.raises(ValueError, match=message):
        exp6348.validate_artifact(bad)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda a: a["upstream_path_hash_terminal_class_and_gate_receipt"].__setitem__(
                "structured_gate_passed", False
            ),
            "upstream_path_hash_terminal_class_and_gate_receipt",
        ),
        (
            lambda a: a["arc_registry_precheck_path_hash_and_result"].__setitem__(
                "all_selected_targets_nonduplicate", False
            ),
            "arc_registry_precheck_path_hash_and_result",
        ),
        (
            lambda a: a["no_duplicate_solve_receipt"].__setitem__(
                "no_duplicate_solve_proposal", False
            ),
            "no_duplicate_solve_receipt",
        ),
        (
            lambda a: a.__setitem__("models_used", list(exp6348.MANDATED_MODEL_IDS[:2])),
            "models_used",
        ),
        (
            lambda a: a["cuda_gpu_offload_and_memory_release_receipts_by_model"][
                exp6348.MANDATED_MODEL_IDS[0]
            ].__setitem__("memory_released", False),
            "cuda_gpu_offload_and_memory_release_receipts_by_model",
        ),
        (
            lambda a: a["llama_cpp_embedded_tokenizer_receipts"][
                exp6348.MANDATED_MODEL_IDS[0]
            ].__setitem__("ok", False),
            "llama_cpp_embedded_tokenizer_receipts",
        ),
        (
            lambda a: a["route_default_off_and_activation_receipts"].__setitem__(
                "default_enabled", True
            ),
            "route_default_off_and_activation_receipts",
        ),
        (
            lambda a: a["matched_call_token_action_time_and_checker_budgets"].__setitem__(
                "budget_parity", False
            ),
            "matched_call_token_action_time_and_checker_budgets",
        ),
        (
            lambda a: a["legal_action_order_changes_by_model_game_window_arm_and_seed"].__setitem__(
                "action_injection_count", 1
            ),
            "legal_action_order_changes_by_model_game_window_arm_and_seed",
        ),
        (
            lambda a: a["paired_influence_and_quality_deltas_intervals_and_sample_sizes"].__setitem__(
                "all_headline_models_positive", False
            ),
            "paired_influence_and_quality_deltas_intervals_and_sample_sizes",
        ),
        (
            lambda a: a["route_deletion_permutation_leakage_and_escape_results"].__setitem__(
                "all_controls_passed", False
            ),
            "route_deletion_permutation_leakage_and_escape_results",
        ),
        (
            lambda a: a["harm_underpowered_missing_and_flagged_cells"].__setitem__(
                "missing_cell_count", 1
            ),
            "harm_underpowered_missing_and_flagged_cells",
        ),
        (lambda a: a.__setitem__("field_principles", {}), "field_principles"),
        (lambda a: a.__setitem__("field_provenance", {}), "field_provenance"),
        (lambda a: a.__setitem__("honest_verdict", "not_terminal"), "honest_verdict"),
        (lambda a: a.__setitem__("arc_causal_influence_ready_score", 0.0), "arc_causal_influence_ready_score"),
    ],
)
def test_req_arc_wmte_6348_validation_rejects_artifact_guard_drift(
    tmp_path: Path, mutate, message: str
) -> None:
    """REQ-ARC-WMTE-6348: validator catches protected artifact drift."""

    artifact = _artifact(tmp_path)
    bad = copy.deepcopy(artifact)
    mutate(bad)
    _with_checksum(bad)

    with pytest.raises(ValueError, match=message):
        exp6348.validate_artifact(bad)


def test_req_arc_wmte_6348_validation_rejects_missing_and_checksum(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-6348: missing fields and checksum drift fail validation."""

    artifact = _artifact(tmp_path)

    checksum = dict(artifact)
    checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp6348.validate_artifact(checksum)

    missing = dict(artifact)
    missing.pop("status")
    with pytest.raises(ValueError, match="missing fields"):
        exp6348.validate_artifact(missing)
