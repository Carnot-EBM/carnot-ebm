"""Tests for Exp6401 held active-goal causal ARC holdout.

Spec refs: REQ-ARC-ARM-6401,
SCENARIO-ARC-ARM-6401-GATE-AND-HOLDOUTS,
SCENARIO-ARC-ARM-6401-MATCHED-CAUSAL-ARMS,
SCENARIO-ARC-ARM-6401-FROZEN-ACTIONS,
SCENARIO-ARC-ARM-6401-PAIRED-METRICS,
SCENARIO-ARC-ARM-6401-ATTACKS-FAIL-CLOSED,
SCENARIO-ARC-ARM-6401-ARTIFACT-NO-SOLVE.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6401_arc_active_goal_causal_holdout as exp6401


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/arc-agi/spec.md"


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
        rows = {
            0: exp6401.MANDATED_MODEL_IDS[0],
            1: exp6401.MANDATED_MODEL_IDS[2],
            2: exp6401.MANDATED_MODEL_IDS[1],
        }
        indices = model_indices or (0, 1)
        return [
            {
                "name": rows[index].split("/")[-1].replace("-GGUF", ""),
                "hf_id": rows[index],
                "gpu": gpu,
                "model_path": _fake_model_file(tmp_path, rows[index]),
            }
            for gpu, index in zip(gpu_indices, indices, strict=True)
        ]

    return fake_pair_resolver


def _fake_tokenizer_checker(model_path: str | None) -> tuple[bool, str]:
    return bool(model_path), "embedded GGUF tokenizer OK (test)"


def _fake_cuda_receipts(models: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(model["hf_id"]): {
            "terminal": True,
            "gpu_offload_supported": True,
            "cuda_visible": True,
            "model_path": model["model_path"],
            "gpu": int(model["gpu"]),
            "errors": [],
        }
        for model in models
    }


def _artifact(tmp_path: Path) -> dict[str, Any]:
    return exp6401.run(
        date="20260813",
        result_path=tmp_path / exp6401.RESULT_RELATIVE_PATH.name,
        held_manifest_path=tmp_path / exp6401.HELD_WINDOW_MANIFEST_RELATIVE_PATH.name,
        duration_s=2.5,
        tests_run=tuple(exp6401.DEFAULT_TEST_COMMANDS),
        test_exit_codes={command: 0 for command in exp6401.DEFAULT_TEST_COMMANDS},
        model_pair_resolver=_fake_pair_resolver_factory(tmp_path),
        tokenizer_checker=_fake_tokenizer_checker,
        cuda_receipt_collector=_fake_cuda_receipts,
        write=True,
    )


def _with_checksum(payload: dict[str, Any]) -> dict[str, Any]:
    payload["reproducibility_checksum"] = exp6401.payload_checksum(payload)
    return payload


def test_req_arc_arm_6401_spec_declares_causal_holdout_contract() -> None:
    """REQ-ARC-ARM-6401: OpenSpec names the held causal route contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-ARC-ARM-6401") :]
    for marker in (
        "SCENARIO-ARC-ARM-6401-GATE-AND-HOLDOUTS",
        "SCENARIO-ARC-ARM-6401-MATCHED-CAUSAL-ARMS",
        "SCENARIO-ARC-ARM-6401-FROZEN-ACTIONS",
        "SCENARIO-ARC-ARM-6401-PAIRED-METRICS",
        "SCENARIO-ARC-ARM-6401-ATTACKS-FAIL-CLOSED",
        "SCENARIO-ARC-ARM-6401-ARTIFACT-NO-SOLVE",
        "eight fresh held live attempt windows",
        "48 visible transitions",
        exp6401.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert marker in section
    for field in exp6401.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for model_id in exp6401.MANDATED_MODEL_IDS:
        assert model_id in section


def test_scenario_arc_arm_6401_gate_receipts_replay_exp6400() -> None:
    """SCENARIO-ARC-ARM-6401-GATE-AND-HOLDOUTS: Exp6400 gates are replayed."""

    receipt = exp6401.exp6400_gate_receipts()

    assert receipt["all_gates_passed"] is True
    assert receipt["gates"][0]["artifact_field"] == "arc_active_goal_shadow_ready_score"
    assert receipt["gate_scalar_fields"]["arc_active_goal_shadow_ready_score"] == 1.0
    assert receipt["gate_scalar_fields"]["active_shadow_treatment_fired_count"] > 0
    assert receipt["gate_scalar_fields"]["delta_shadow_false_accept_count"] <= 0
    assert all(row["comparison_surface_finite_bare_number"] for row in receipt["gates"])
    assert receipt["route_disable_default_revalidated"] is True


def test_req_arc_arm_6401_model_specs_and_tokenizer_receipts(tmp_path: Path) -> None:
    """REQ-ARC-ARM-6401: all three GGUF models resolve through cached_sota_pair."""

    models, cached = exp6401.build_model_specs(
        model_pair_resolver=_fake_pair_resolver_factory(tmp_path)
    )
    tokenizers = exp6401.embedded_gguf_tokenizer_receipts(
        models,
        tokenizer_checker=_fake_tokenizer_checker,
    )
    files = exp6401.model_file_hashes_revisions_quantizations_and_tokenizers(
        models,
        tokenizers,
    )

    assert [row["hf_id"] for row in models] == list(exp6401.MANDATED_MODEL_IDS)
    assert cached["all_mandated_models_resolved"] is True
    assert all(row["model_exists"] is True for row in models)
    assert all(row["model_sha256"].startswith("sha256:") for row in models)
    assert all(row["quantization"] == "Q4_K_M" for row in models)
    assert all(row["ok"] is True for row in tokenizers.values())
    assert set(files) == set(exp6401.MANDATED_MODEL_IDS)

    direct = tmp_path / "direct.py"
    direct.write_text("AutoTokenizer\n", encoding="utf-8")
    attribute = tmp_path / "attribute.py"
    attribute.write_text("transformers.AutoTokenizer\n", encoding="utf-8")
    assert exp6401.autotokenizer_usage_count((direct, attribute)) == 2
    assert exp6401._quant_from_path("/tmp/model.gguf") == "unknown"
    assert exp6401._model_revision(None) is None
    assert exp6401._model_revision("/tmp/model-Q4_K_M.gguf") is None

    with pytest.raises(ValueError, match="cached_sota_pair"):
        exp6401.build_model_specs(model_pair_resolver=lambda **_: None)

    def missing_resolver(**kwargs):
        _ = kwargs
        return [
            {
                "name": "Qwen3.6-35B-A3B",
                "hf_id": exp6401.MANDATED_MODEL_IDS[0],
                "gpu": 0,
                "model_path": _fake_model_file(tmp_path, exp6401.MANDATED_MODEL_IDS[0]),
            },
            {
                "name": "Gemma4-31B-it",
                "hf_id": exp6401.MANDATED_MODEL_IDS[1],
                "gpu": 1,
                "model_path": _fake_model_file(tmp_path, exp6401.MANDATED_MODEL_IDS[1]),
            },
        ]

    with pytest.raises(ValueError, match="missing mandated models"):
        exp6401.build_model_specs(model_pair_resolver=missing_resolver)

    precheck = exp6401.arc_registry_and_claims_hashes(
        registry_text="",
        claims_text="experiment_6401_arc_active_goal_causal_holdout",
    )
    assert precheck["claims"]["solve_claim_count"] == 1


def test_scenario_arc_arm_6401_held_windows_are_fresh_and_disjoint(tmp_path: Path) -> None:
    """SCENARIO-ARC-ARM-6401-GATE-AND-HOLDOUTS: held windows exclude Exp6400."""

    manifest = exp6401.held_live_window_manifest_payload()
    receipt = exp6401.write_sealed_payload(tmp_path / "held.json", manifest, write=True)
    proof = manifest["exp6400_disjointness_proof"]

    assert manifest["sealed_before_evaluation"] is True
    assert manifest["held_live_attempt_windows"] is True
    assert manifest["window_count"] == 8
    assert manifest["visible_transition_count"] >= 48
    assert receipt["sha256"].startswith("sha256:")
    assert receipt["exp6400_disjointness"]["disjoint"] is True
    assert proof["overlap_window_ids"] == []
    assert proof["overlap_transition_hashes"] == []
    assert proof["proof_hash"].startswith("sha256:")
    assert all(not row["window_id"].startswith("exp6400_") for row in manifest["rows"])


def test_scenario_arc_arm_6401_matched_causal_arms_and_paired_metrics(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-ARM-6401-MATCHED-CAUSAL-ARMS: active probes add value."""

    models, _ = exp6401.build_model_specs(
        model_pair_resolver=_fake_pair_resolver_factory(tmp_path)
    )
    manifest = exp6401.held_live_window_manifest_payload()
    causal = exp6401.run_matched_causal_arms(models=models, windows=manifest["rows"])
    paired = causal["paired_tests_confidence_intervals_and_effective_sample_sizes"]

    assert causal["row_count"] == len(models) * manifest["window_count"] * 2
    assert causal["treatment_fired_counts"]["active_disagreement"] == len(models) * 8
    assert causal["treatment_fired_counts"]["passive_two_sided"] == 0
    assert causal["treatment_fired_counts"]["model_window_cells_where_treatment_did_not_fire"] == []
    assert causal["matched_work_and_legal_action_receipts"]["matched_work_passed"] is True
    assert causal["matched_work_and_legal_action_receipts"]["legal_action_sets_matched"] is True
    assert causal["delta_admission_precision"] == pytest.approx(0.75)
    assert causal["delta_false_accept_count"] == -9
    assert causal["delta_exact_progress_proxy"] > 0.0
    assert causal["route_promotion_eligible"] is True
    assert paired["effective_sample_size"] == len(models) * 8
    assert paired["progress_proxy"]["positive_count"] == paired["effective_sample_size"]
    assert paired["false_accept_count"]["negative_count"] == 9


def test_scenario_arc_arm_6401_frozen_actions_and_attack_matrix(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-ARM-6401-FROZEN-ACTIONS: attacks fail closed."""

    models, _ = exp6401.build_model_specs(
        model_pair_resolver=_fake_pair_resolver_factory(tmp_path)
    )
    causal = exp6401.run_matched_causal_arms(
        models=models,
        windows=exp6401.held_live_window_manifest_payload()["rows"],
    )
    rows = causal["pre_action_goal_probe_and_action_freeze_records"]
    attacks = exp6401.attack_matrix(rows=rows, model_ids=[row["hf_id"] for row in models])

    for row in rows:
        assert row["candidate_goals_frozen_before_outcome"] is True
        assert row["evidence_disposition_frozen_before_outcome"] is True
        assert row["probe_or_rank_frozen_before_outcome"] is True
        assert row["action_frozen_before_outcome"] is True
        assert row["environment_result_read_after_freeze"] is True
        assert row["post_action_transition_check"]["verifier_is_oracle"] is True
        if row["arm"] == "active_disagreement":
            assert row["selected_legal_probe"] == 4
            assert row["selected_action"] == 4
            assert row["treatment_fired"] is True
        else:
            assert row["passive_action_rank"][0] == 5
            assert row["selected_action"] == 5
    assert all(row["fail_closed"] for row in attacks)
    assert exp6401._expect_value_error("accepted", lambda: None)["fail_closed"] is False

    duplicate_row = copy.deepcopy(rows)
    duplicate_row[1]["arm"] = duplicate_row[0]["arm"]
    with pytest.raises(ValueError, match="duplicate model/window/arm"):
        exp6401.validate_causal_rows(duplicate_row, [row["hf_id"] for row in models])

    unequal_budget = copy.deepcopy(rows)
    unequal_budget[0]["action_budget"] += 1
    with pytest.raises(ValueError, match="unequal budgets"):
        exp6401.validate_causal_rows(unequal_budget, [row["hf_id"] for row in models])

    reused = copy.deepcopy(rows)
    reused[0]["window_id"] = "exp6400_live_shadow_push_a_l0_seed6400001"
    with pytest.raises(ValueError, match="window reuse"):
        exp6401.validate_causal_rows(reused, [row["hf_id"] for row in models])

    assert exp6401._mean_ci([]) == {"mean": 0.0, "ci_95": [0.0, 0.0], "n": 0}
    assert exp6401._mean_ci([2.0]) == {"mean": 2.0, "ci_95": [2.0, 2.0], "n": 1}
    assert exp6401._sign_test_two_sided([])["p_value"] == 1.0

    missing_arm = rows[:-1]
    paired = exp6401.paired_tests_confidence_intervals_and_effective_sample_sizes(missing_arm)
    assert paired["missing_paired_cell_count"] == 1
    with pytest.raises(ValueError, match="missing arm rows"):
        exp6401.validate_causal_rows(missing_arm, [row["hf_id"] for row in models])

    not_fired = copy.deepcopy(rows)
    active_index = next(index for index, row in enumerate(not_fired) if row["arm"] == exp6401.ACTIVE_ARM)
    not_fired[active_index]["treatment_fired"] = False
    fired = exp6401.treatment_fired_counts(not_fired)
    assert fired["model_window_cells_where_treatment_did_not_fire"]

    oracle_before = copy.deepcopy(rows)
    oracle_before[0]["oracle_before_action"] = True
    with pytest.raises(ValueError, match="oracle timing"):
        exp6401.validate_causal_rows(oracle_before, [row["hf_id"] for row in models])

    reordered = rows[16:] + rows[:16]
    with pytest.raises(ValueError, match="model row order in causal rows"):
        exp6401.validate_causal_rows(reordered, [row["hf_id"] for row in models])


def test_scenario_arc_arm_6401_artifact_schema_and_no_solve_claim(tmp_path: Path) -> None:
    """SCENARIO-ARC-ARM-6401-ARTIFACT-NO-SOLVE: artifact is complete."""

    artifact = _artifact(tmp_path)
    loaded = json.loads((tmp_path / exp6401.RESULT_RELATIVE_PATH.name).read_text(encoding="utf-8"))

    assert loaded == artifact
    assert set(exp6401.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert set(exp6401.REQUIRED_ARTIFACT_FIELDS).issubset(artifact["field_principles"])
    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["models_used"] == list(exp6401.MANDATED_MODEL_IDS)
    assert type(artifact["delta_false_accept_count"]) is int
    assert type(artifact["delta_exact_progress_proxy"]) is float
    assert artifact["delta_false_accept_count"] == -9
    assert artifact["delta_exact_progress_proxy"] > 0.0
    assert artifact["autotokenizer_usage_count"] == 0
    assert artifact["solve_claim_count"] == 0
    assert artifact["solve_registry_modified"] is False
    assert artifact["arc_active_goal_causal_ready_score"] == 1.0
    assert artifact["route_promotion_eligible"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["field_principles"]["exp6400.arc_active_goal_shadow_ready_score"]
    assert artifact["field_principles"]["delta_false_accept_count"]
    assert artifact["field_principles"]["delta_exact_progress_proxy"]
    assert artifact["reproducibility_checksum"] == exp6401.payload_checksum(artifact)
    exp6401.validate_artifact(artifact)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("hidden_source_access_count", 1, "hidden_source_access_count"),
        ("offline_ground_truth_search_count", 1, "offline_ground_truth_search_count"),
        ("per_game_adapter_count", 1, "per_game_adapter_count"),
        ("oracle_before_action_count", 1, "oracle_before_action_count"),
        ("solve_claim_count", 1, "solve_claim_count"),
        ("solve_registry_modified", True, "solve_registry_modified"),
        ("delta_false_accept_count", -9.0, "delta_false_accept_count"),
        ("delta_false_accept_count", 1, "delta_false_accept_count"),
        ("delta_exact_progress_proxy", "1.0", "delta_exact_progress_proxy"),
        ("arc_active_goal_causal_ready_score", 0.0, "arc_active_goal_causal_ready_score"),
        ("route_promotion_eligible", False, "route_promotion_eligible"),
        ("inference_substrate", "live_llm_inference", "inference_substrate"),
        ("verifier_is_oracle", True, "verifier_is_oracle"),
    ],
)
def test_scenario_arc_arm_6401_validation_rejects_forbidden_drift(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    """SCENARIO-ARC-ARM-6401-ATTACKS-FAIL-CLOSED: drift is rejected."""

    artifact = _artifact(tmp_path)
    bad = copy.deepcopy(artifact)
    bad[field] = value
    _with_checksum(bad)

    with pytest.raises(ValueError, match=message):
        exp6401.validate_artifact(bad)


def test_req_arc_arm_6401_validation_rejects_missing_and_nested_drift(
    tmp_path: Path,
) -> None:
    """REQ-ARC-ARM-6401: missing fields and nested drift fail validation."""

    artifact = _artifact(tmp_path)

    checksum = dict(artifact)
    checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp6401.validate_artifact(checksum)

    missing = dict(artifact)
    missing.pop("status")
    with pytest.raises(ValueError, match="missing fields"):
        exp6401.validate_artifact(missing)

    drift_cases = [
        (
            lambda a: a.__setitem__("models_used", list(exp6401.MANDATED_MODEL_IDS[:2])),
            "models_used",
        ),
        (
            lambda a: a["exp6400_gate_receipts"].__setitem__("all_gates_passed", False),
            "exp6400_gate_receipts",
        ),
        (
            lambda a: a["matched_work_and_legal_action_receipts"].__setitem__(
                "matched_work_passed", False
            ),
            "matched_work_and_legal_action_receipts",
        ),
        (
            lambda a: a[
                "held_live_window_manifest_path_hash_counts_and_exp6400_disjointness"
            ].__setitem__("window_count", 7),
            "held_live_window_manifest",
        ),
        (
            lambda a: a["oracle_timing_receipts"].__setitem__("all_actions_frozen_before_outcomes", False),
            "oracle_timing_receipts",
        ),
        (
            lambda a: a["oracle_timing_receipts"].__setitem__("all_environment_results_read_after_freeze", False),
            "oracle_timing_receipts",
        ),
        (
            lambda a: a["oracle_timing_receipts"].__setitem__("oracle_before_action_count", 1),
            "oracle_timing_receipts",
        ),
        (
            lambda a: a[
                "held_live_window_manifest_path_hash_counts_and_exp6400_disjointness"
            ]["exp6400_disjointness"].__setitem__("disjoint", False),
            "held_live_window_manifest",
        ),
        (
            lambda a: a[
                "window_action_oracle_model_state_legal_set_budget_duplicate_and_label_attack_matrix"
            ][0].__setitem__("fail_closed", False),
            "attack_matrix",
        ),
        (
            lambda a: a["protected_files_unchanged"]["ops/arc_solve_registry.yaml"].__setitem__(
                "unchanged", False
            ),
            "protected_files_unchanged",
        ),
        (
            lambda a: a["embedded_gguf_tokenizer_receipts"][
                exp6401.MANDATED_MODEL_IDS[0]
            ].__setitem__("ok", False),
            "embedded_gguf_tokenizer_receipts",
        ),
        (
            lambda a: a["cuda_offload_and_runtime_receipts_by_model"][
                exp6401.MANDATED_MODEL_IDS[0]
            ].__setitem__("terminal", False),
            "cuda_offload_and_runtime_receipts_by_model",
        ),
        (
            lambda a: a["field_principles"].pop(
                "exp6400.arc_active_goal_shadow_ready_score"
            ),
            "field_principles",
        ),
        (lambda a: a.__setitem__("field_principles", {}), "field_principles"),
        (lambda a: a.__setitem__("honest_verdict", "blocked"), "honest_verdict"),
    ]
    for mutate, message in drift_cases:
        bad = copy.deepcopy(artifact)
        mutate(bad)
        _with_checksum(bad)
        with pytest.raises(ValueError, match=message):
            exp6401.validate_artifact(bad)


def test_req_arc_arm_6401_build_artifact_uses_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-ARC-ARM-6401: build_artifact validates the runner output."""

    artifact = _artifact(tmp_path)

    def fake_run(**kwargs):
        assert kwargs["date"] == "20260813"
        assert kwargs["write"] is True
        return artifact

    monkeypatch.setattr(exp6401, "run", fake_run)

    built = exp6401.build_artifact(
        tmp_path,
        date="20260813",
        output_path=tmp_path / "out.json",
    )
    assert built["arc_active_goal_causal_ready_score"] == 1.0
