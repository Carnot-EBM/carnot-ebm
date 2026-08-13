"""Tests for Exp6400 default-off active-goal ARC shadow.

Spec refs: REQ-ARC-ARM-6400,
SCENARIO-ARC-ARM-6400-GATE-REPLAY,
SCENARIO-ARC-ARM-6400-MATCHED-SHADOW,
SCENARIO-ARC-ARM-6400-FROZEN-PROBES,
SCENARIO-ARC-ARM-6400-ATTACKS-FAIL-CLOSED,
SCENARIO-ARC-ARM-6400-ARTIFACT-NO-SOLVE.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6400_arc_default_off_active_goal_shadow as exp6400


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
            0: exp6400.MANDATED_MODEL_IDS[0],
            1: exp6400.MANDATED_MODEL_IDS[2],
            2: exp6400.MANDATED_MODEL_IDS[1],
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
    return exp6400.run(
        date="20260813",
        result_path=tmp_path / exp6400.RESULT_RELATIVE_PATH.name,
        fresh_manifest_path=tmp_path / exp6400.FRESH_WINDOW_MANIFEST_RELATIVE_PATH.name,
        duration_s=1.5,
        tests_run=tuple(exp6400.DEFAULT_TEST_COMMANDS),
        test_exit_codes={command: 0 for command in exp6400.DEFAULT_TEST_COMMANDS},
        model_pair_resolver=_fake_pair_resolver_factory(tmp_path),
        tokenizer_checker=_fake_tokenizer_checker,
        cuda_receipt_collector=_fake_cuda_receipts,
        write=True,
    )


def _with_checksum(payload: dict[str, Any]) -> dict[str, Any]:
    payload["reproducibility_checksum"] = exp6400.payload_checksum(payload)
    return payload


def test_req_arc_arm_6400_spec_declares_shadow_contract() -> None:
    """REQ-ARC-ARM-6400: OpenSpec names the deferred shadow replay."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-ARC-ARM-6400") :]
    for marker in (
        "SCENARIO-ARC-ARM-6400-GATE-REPLAY",
        "SCENARIO-ARC-ARM-6400-MATCHED-SHADOW",
        "SCENARIO-ARC-ARM-6400-FROZEN-PROBES",
        "SCENARIO-ARC-ARM-6400-ATTACKS-FAIL-CLOSED",
        "SCENARIO-ARC-ARM-6400-ARTIFACT-NO-SOLVE",
        "default-off shadow",
        "36 visible transitions",
        exp6400.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert marker in section
    for field in exp6400.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for model_id in exp6400.MANDATED_MODEL_IDS:
        assert model_id in section


def test_scenario_arc_arm_6400_gate_replay_uses_exp6393_scalars() -> None:
    """SCENARIO-ARC-ARM-6400-GATE-REPLAY: nested Exp6388 deltas are not trusted."""

    receipt = exp6400.exp6393_gate_receipts()

    assert receipt["all_gates_passed"] is True
    assert receipt["scalar_fields"]["delta_admission_precision_scalar"] == pytest.approx(0.75)
    assert receipt["scalar_fields"]["delta_false_accept_count_scalar"] == -9
    assert all(row["comparison_surface_finite_bare_number"] for row in receipt["gates"])
    assert receipt["deferred_exp6389_failure_repaired"] is True


def test_req_arc_arm_6400_model_specs_and_tokenizer_receipts(tmp_path: Path) -> None:
    """REQ-ARC-ARM-6400: all three GGUF models resolve through cached_sota_pair."""

    models, cached = exp6400.build_model_specs(
        model_pair_resolver=_fake_pair_resolver_factory(tmp_path)
    )
    tokenizers = exp6400.embedded_gguf_tokenizer_receipts(
        models,
        tokenizer_checker=_fake_tokenizer_checker,
    )
    files = exp6400.model_file_hashes_revisions_quantizations_and_tokenizers(
        models,
        tokenizers,
    )

    assert [row["hf_id"] for row in models] == list(exp6400.MANDATED_MODEL_IDS)
    assert cached["all_mandated_models_resolved"] is True
    assert all(row["model_exists"] is True for row in models)
    assert all(row["model_sha256"].startswith("sha256:") for row in models)
    assert all(row["quantization"] == "Q4_K_M" for row in models)
    assert all(row["ok"] is True for row in tokenizers.values())
    assert set(files) == set(exp6400.MANDATED_MODEL_IDS)

    direct = tmp_path / "direct.py"
    direct.write_text("AutoTokenizer\n", encoding="utf-8")
    attribute = tmp_path / "attribute.py"
    attribute.write_text("transformers.AutoTokenizer\n", encoding="utf-8")
    assert exp6400.autotokenizer_usage_count((direct, attribute)) == 2
    assert exp6400._quant_from_path("/tmp/model.gguf") == "unknown"
    assert exp6400._model_revision(None) is None
    assert exp6400._model_revision("/tmp/model-Q4_K_M.gguf") is None

    with pytest.raises(ValueError, match="cached_sota_pair"):
        exp6400.build_model_specs(model_pair_resolver=lambda **_: None)

    def missing_resolver(**kwargs):
        _ = kwargs
        return [
            {
                "name": "Qwen3.6-35B-A3B",
                "hf_id": exp6400.MANDATED_MODEL_IDS[0],
                "gpu": 0,
                "model_path": _fake_model_file(tmp_path, exp6400.MANDATED_MODEL_IDS[0]),
            },
            {
                "name": "Gemma4-31B-it",
                "hf_id": exp6400.MANDATED_MODEL_IDS[1],
                "gpu": 1,
                "model_path": _fake_model_file(tmp_path, exp6400.MANDATED_MODEL_IDS[1]),
            },
        ]

    with pytest.raises(ValueError, match="missing mandated models"):
        exp6400.build_model_specs(model_pair_resolver=missing_resolver)

    precheck = exp6400.arc_registry_and_claims_precheck_hashes(
        registry_text="",
        claims_text="experiment_6400_arc_default_off_active_goal_shadow",
    )
    assert precheck["claims"]["solve_claim_count"] == 1


def test_scenario_arc_arm_6400_fresh_windows_and_matched_shadow(tmp_path: Path) -> None:
    """SCENARIO-ARC-ARM-6400-MATCHED-SHADOW: six windows seal 36 transitions."""

    models, _ = exp6400.build_model_specs(
        model_pair_resolver=_fake_pair_resolver_factory(tmp_path)
    )
    manifest = exp6400.fresh_live_window_manifest_payload()
    receipt = exp6400.write_sealed_payload(tmp_path / "windows.json", manifest, write=True)
    shadow = exp6400.run_matched_shadow(models=models, windows=manifest["rows"])

    assert manifest["sealed_before_evaluation"] is True
    assert manifest["window_count"] == 6
    assert manifest["visible_transition_count"] >= 36
    assert receipt["sha256"].startswith("sha256:")
    assert all(row["agent_owned_policy_transition_store"] is True for row in manifest["rows"])
    assert shadow["row_count"] == len(models) * manifest["window_count"]
    assert shadow["active_shadow_treatment_fired_count"] == shadow["row_count"]
    assert shadow["executed_action_change_count"] == 0
    assert shadow["delta_shadow_false_accept_count"] == -9
    assert shadow["delta_shadow_admission_precision"] == pytest.approx(0.75)
    assert shadow["delta_shadow_exact_progress_proxy"] == pytest.approx(0.0)
    assert shadow["matched_work_receipts"]["matched_work_passed"] is True


def test_scenario_arc_arm_6400_frozen_probe_records_and_attack_matrix(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-ARM-6400-FROZEN-PROBES and attack controls fail closed."""

    models, _ = exp6400.build_model_specs(
        model_pair_resolver=_fake_pair_resolver_factory(tmp_path)
    )
    shadow = exp6400.run_matched_shadow(
        models=models,
        windows=exp6400.fresh_live_window_manifest_payload()["rows"],
    )
    attacks = exp6400.attack_matrix(
        rows=shadow["frozen_goal_probe_and_counterfactual_action_records"],
        model_ids=[row["hf_id"] for row in models],
    )

    for row in shadow["frozen_goal_probe_and_counterfactual_action_records"]:
        assert row["goal_probe_frozen_before_next_transition"] is True
        assert row["counterfactual_rank_frozen_before_next_transition"] is True
        assert row["shadow_executed_action"] == row["route_off_executed_action"]
        assert row["shadow_ranked_actions"][0] == row["legal_disagreement_probe"]
        assert row["post_action_transition_check"]["verifier_is_oracle"] is True
    assert all(row["fail_closed"] for row in attacks)
    assert exp6400._expect_value_error("accepted", lambda: None)["fail_closed"] is False

    counts = exp6400._empty_counts()
    exp6400._add_counts(counts, status="rejected", admissible_goal=True)
    assert counts["false_reject"] == 1

    rows = copy.deepcopy(shadow["frozen_goal_probe_and_counterfactual_action_records"])
    duplicate_row = copy.deepcopy(rows)
    duplicate_row[1]["window_id"] = duplicate_row[0]["window_id"]
    duplicate_row[1]["prefix_id"] = duplicate_row[0]["prefix_id"]
    with pytest.raises(ValueError, match="duplicate model/window/prefix"):
        exp6400.validate_shadow_rows(duplicate_row, [row["hf_id"] for row in models])

    truncated_prefix = copy.deepcopy(rows)
    truncated_prefix[0]["prefix_transition_count"] = 5
    with pytest.raises(ValueError, match="prefix truncation"):
        exp6400.validate_shadow_rows(truncated_prefix, [row["hf_id"] for row in models])

    leakage_flag = copy.deepcopy(rows)
    leakage_flag[0]["shadow_action_leaked_to_execution"] = True
    with pytest.raises(ValueError, match="shadow-to-action leakage"):
        exp6400.validate_shadow_rows(leakage_flag, [row["hf_id"] for row in models])

    reordered = rows[6:12] + rows[:6] + rows[12:]
    with pytest.raises(ValueError, match="model row order in shadow rows"):
        exp6400.validate_shadow_rows(reordered, [row["hf_id"] for row in models])


def test_scenario_arc_arm_6400_artifact_schema_and_no_solve_claim(tmp_path: Path) -> None:
    """SCENARIO-ARC-ARM-6400-ARTIFACT-NO-SOLVE: artifact is complete and zero-credit."""

    artifact = _artifact(tmp_path)
    loaded = json.loads((tmp_path / exp6400.RESULT_RELATIVE_PATH.name).read_text(encoding="utf-8"))

    assert loaded == artifact
    assert set(exp6400.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert set(exp6400.REQUIRED_ARTIFACT_FIELDS).issubset(artifact["field_principles"])
    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["models_used"] == list(exp6400.MANDATED_MODEL_IDS)
    assert type(artifact["active_shadow_treatment_fired_count"]) is int
    assert type(artifact["delta_shadow_false_accept_count"]) is int
    assert artifact["active_shadow_treatment_fired_count"] == 18
    assert artifact["delta_shadow_false_accept_count"] == -9
    assert artifact["autotokenizer_usage_count"] == 0
    assert artifact["executed_action_change_count"] == 0
    assert artifact["solve_claim_count"] == 0
    assert artifact["solve_registry_modified"] is False
    assert artifact["arc_active_goal_shadow_ready_score"] == 1.0
    assert artifact["verifier_is_oracle"] is False
    assert artifact["field_principles"]["delta_false_accept_count_scalar"]
    assert artifact["field_principles"]["active_shadow_treatment_fired_count"]
    assert artifact["reproducibility_checksum"] == exp6400.payload_checksum(artifact)
    exp6400.validate_artifact(artifact)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("hidden_source_access_count", 1, "hidden_source_access_count"),
        ("offline_ground_truth_search_count", 1, "offline_ground_truth_search_count"),
        ("per_game_adapter_count", 1, "per_game_adapter_count"),
        ("oracle_before_action_count", 1, "oracle_before_action_count"),
        ("executed_action_change_count", 1, "executed_action_change_count"),
        ("solve_claim_count", 1, "solve_claim_count"),
        ("solve_registry_modified", True, "solve_registry_modified"),
        ("active_shadow_treatment_fired_count", 0.5, "active_shadow_treatment_fired_count"),
        ("active_shadow_treatment_fired_count", 0, "active_shadow_treatment_fired_count"),
        ("delta_shadow_false_accept_count", -9.0, "delta_shadow_false_accept_count"),
        ("delta_shadow_false_accept_count", 1, "delta_shadow_false_accept_count"),
        ("arc_active_goal_shadow_ready_score", 0.0, "arc_active_goal_shadow_ready_score"),
        ("inference_substrate", "live_llm_inference", "inference_substrate"),
        ("verifier_is_oracle", True, "verifier_is_oracle"),
    ],
)
def test_scenario_arc_arm_6400_validation_rejects_forbidden_drift(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    """SCENARIO-ARC-ARM-6400-ATTACKS-FAIL-CLOSED: forbidden drift is rejected."""

    artifact = _artifact(tmp_path)
    bad = copy.deepcopy(artifact)
    bad[field] = value
    _with_checksum(bad)

    with pytest.raises(ValueError, match=message):
        exp6400.validate_artifact(bad)


def test_req_arc_arm_6400_validation_rejects_missing_and_checksum(tmp_path: Path) -> None:
    """REQ-ARC-ARM-6400: missing fields and checksum drift fail validation."""

    artifact = _artifact(tmp_path)

    checksum = dict(artifact)
    checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp6400.validate_artifact(checksum)

    missing = dict(artifact)
    missing.pop("status")
    with pytest.raises(ValueError, match="missing fields"):
        exp6400.validate_artifact(missing)

    bad_attack = copy.deepcopy(artifact)
    bad_attack["model_row_prefix_state_goal_duplicate_budget_and_action_leakage_attack_matrix"][0][
        "fail_closed"
    ] = False
    _with_checksum(bad_attack)
    with pytest.raises(ValueError, match="attack_matrix"):
        exp6400.validate_artifact(bad_attack)

    drift_cases = [
        (
            lambda a: a.__setitem__("models_used", list(exp6400.MANDATED_MODEL_IDS[:2])),
            "models_used",
        ),
        (
            lambda a: a["exp6393_gate_receipts"].__setitem__("all_gates_passed", False),
            "exp6393_gate_receipts",
        ),
        (
            lambda a: a["matched_work_receipts"].__setitem__("matched_work_passed", False),
            "matched_work_receipts",
        ),
        (
            lambda a: a["protected_files_unchanged"]["ops/arc_solve_registry.yaml"].__setitem__(
                "unchanged", False
            ),
            "protected_files_unchanged",
        ),
        (
            lambda a: a["embedded_gguf_tokenizer_receipts"][
                exp6400.MANDATED_MODEL_IDS[0]
            ].__setitem__("ok", False),
            "embedded_gguf_tokenizer_receipts",
        ),
        (
            lambda a: a["cuda_offload_and_runtime_receipts_by_model"][
                exp6400.MANDATED_MODEL_IDS[0]
            ].__setitem__("terminal", False),
            "cuda_offload_and_runtime_receipts_by_model",
        ),
        (lambda a: a.__setitem__("field_principles", {}), "field_principles"),
        (lambda a: a.__setitem__("honest_verdict", "blocked"), "honest_verdict"),
    ]
    for mutate, message in drift_cases:
        bad = copy.deepcopy(artifact)
        mutate(bad)
        _with_checksum(bad)
        with pytest.raises(ValueError, match=message):
            exp6400.validate_artifact(bad)


def test_req_arc_arm_6400_build_artifact_uses_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-ARC-ARM-6400: build_artifact validates the runner output."""

    artifact = _artifact(tmp_path)

    def fake_run(**kwargs):
        assert kwargs["date"] == "20260813"
        assert kwargs["write"] is True
        return artifact

    monkeypatch.setattr(exp6400, "run", fake_run)

    assert exp6400.build_artifact(tmp_path, date="20260813", output_path=tmp_path / "out.json")[
        "arc_active_goal_shadow_ready_score"
    ] == 1.0
