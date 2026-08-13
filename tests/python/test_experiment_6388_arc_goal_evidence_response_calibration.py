"""Tests for Exp6388 ARC goal-evidence response calibration.

Spec refs: REQ-ARC-ARM-6388,
SCENARIO-ARC-ARM-6388-MATCHED-PREFIXES,
SCENARIO-ARC-ARM-6388-FROZEN-PREDICTIONS,
SCENARIO-ARC-ARM-6388-METRICS-AND-CONTROLS,
SCENARIO-ARC-ARM-6388-ARTIFACT-NO-SOLVE.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from carnot import experiment_6388_arc_goal_evidence_response_calibration as exp6388


REPO = Path(__file__).resolve().parents[2]


def _token_receipts(model_specs: tuple[dict, ...]) -> dict[str, dict]:
    return {
        str(spec["hf_id"]): {
            "hf_id": spec["hf_id"],
            "model_path": spec["model_path"],
            "embedded_tokenizer_loadable": True,
            "tokenizer_source": "gguf_embedded_llama_cpp",
            "detail": "unit tokenizer receipt",
        }
        for spec in model_specs
    }


def test_req_arc_arm_6388_model_specs_include_three_cached_ggufs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-ARC-ARM-6388: all three mandated local model families are resolved."""

    calls: list[tuple[tuple[int, int], tuple[int, int] | None]] = []

    def fake_pair(*, gpu_indices=(0, 1), preferred_quant="Q4_K_M", model_indices=None):
        calls.append((gpu_indices, model_indices))
        rows = {
            0: {
                "name": "Qwen3.6-35B-A3B",
                "hf_id": exp6388.MANDATED_MODEL_IDS[0],
                "gpu": gpu_indices[0],
                "model_path": "/models/qwen.gguf",
            },
            1: {
                "name": "Gemma4-26B-A4B-it",
                "hf_id": exp6388.MANDATED_MODEL_IDS[2],
                "gpu": gpu_indices[1],
                "model_path": "/models/gemma26.gguf",
            },
            2: {
                "name": "Gemma4-31B-it",
                "hf_id": exp6388.MANDATED_MODEL_IDS[1],
                "gpu": gpu_indices[1],
                "model_path": "/models/gemma31.gguf",
            },
        }
        wanted = model_indices or (0, 1)
        return [dict(rows[index]) for index in wanted]

    monkeypatch.setattr(exp6388, "cached_sota_pair", fake_pair)

    specs, receipts = exp6388._model_specs_from_cached_sota_pair()

    assert tuple(spec["hf_id"] for spec in specs) == exp6388.MANDATED_MODEL_IDS
    assert {spec["model_path"] for spec in specs} == {
        "/models/qwen.gguf",
        "/models/gemma31.gguf",
        "/models/gemma26.gguf",
    }
    assert receipts["all_mandated_models_resolved"] is True
    assert calls == [((0, 1), (0, 1)), ((0, 1), (0, 2))]


def test_scenario_arc_arm_6388_matched_prefix_contract_and_freezing() -> None:
    """SCENARIO-ARC-ARM-6388-MATCHED-PREFIXES and frozen prediction receipts."""

    manifest = exp6388._sealed_visible_prefix_manifest()
    arm_contract = exp6388._preregistered_arm_contract(manifest)
    model_specs = exp6388._fixture_model_specs()
    receipts = exp6388._raw_prediction_receipts(model_specs, manifest)

    assert set(arm_contract["arms"]) == set(exp6388.ARMS)
    assert arm_contract["matched_work"]["model_calls_per_model_per_arm"] == len(manifest["prefixes"])
    assert arm_contract["matched_work"]["token_capacity"] == 256
    assert arm_contract["matched_work"]["trajectory_exposure"] == "matched_visible_prefix_only"

    expected_count = len(model_specs) * len(exp6388.ARMS) * len(manifest["prefixes"])
    assert len(receipts) == expected_count
    for row in receipts:
        assert row["frozen_before_evaluation"] is True
        assert row["evaluation_label_read_after_freeze"] is True
        assert row["next_legal_probe"] in row["legal_actions"]
        assert row["later_transition_used_for_calibration_only"] is True


def test_scenario_arc_arm_6388_metrics_controls_and_deltas() -> None:
    """SCENARIO-ARC-ARM-6388-METRICS-AND-CONTROLS."""

    manifest = exp6388._sealed_visible_prefix_manifest()
    model_specs = exp6388._fixture_model_specs()
    receipts = exp6388._raw_prediction_receipts(model_specs, manifest)
    counts, by_arm_model, curves, elimination = exp6388._calibration_tables(receipts)
    deltas = exp6388._active_vs_current_deltas(by_arm_model)
    controls = exp6388._control_receipts(receipts)

    assert counts["active_reward_machine_evidence"]["ALL"]["accepted"] > 0
    assert by_arm_model["active_reward_machine_evidence"][exp6388.MANDATED_MODEL_IDS[0]][
        "admission_precision"
    ] == pytest.approx(1.0)
    assert by_arm_model["current_gate"][exp6388.MANDATED_MODEL_IDS[0]][
        "admission_precision"
    ] < 1.0
    assert curves["active_reward_machine_evidence"][exp6388.MANDATED_MODEL_IDS[0]][
        "monotonic_non_decreasing"
    ] is True
    assert elimination["active_reward_machine_evidence"][exp6388.MANDATED_MODEL_IDS[0]][
        "treatment_fired"
    ] is True
    assert deltas["pooled_unrounded"] > 0.0
    assert deltas["by_model"][exp6388.MANDATED_MODEL_IDS[0]] > 0.0
    assert exp6388._active_false_accept_delta(counts) < 0
    assert all(row["passed"] for row in controls.values())


def test_req_arc_arm_6388_forbidden_tokenizer_counter_branches(tmp_path: Path) -> None:
    """REQ-ARC-ARM-6388: the static tokenizer guard counts forbidden imports."""

    direct = tmp_path / "direct.py"
    direct.write_text("AutoTokenizer\n", encoding="utf-8")
    attribute = tmp_path / "attribute.py"
    attribute.write_text("transformers.AutoTokenizer\n", encoding="utf-8")
    counts = exp6388._empty_counts()

    exp6388._accumulate(
        counts,
        {"status": "rejected", "admissible_goal": True},
    )

    assert counts["false_reject"] == 1
    assert exp6388._autotokenizer_usage_count((direct, attribute)) == 2


@pytest.mark.memory_watchdog_skip
def test_scenario_arc_arm_6388_artifact_no_solve_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-ARC-ARM-6388-ARTIFACT-NO-SOLVE."""

    output = tmp_path / "experiment_6388_arc_goal_evidence_response_calibration.json"
    model_specs = exp6388._fixture_model_specs()
    monkeypatch.setattr(
        exp6388,
        "_model_specs_from_cached_sota_pair",
        lambda: (model_specs, {"all_mandated_models_resolved": True, "unit": True}),
    )
    monkeypatch.setattr(exp6388, "_embedded_tokenizer_receipts", _token_receipts)
    monkeypatch.setattr(
        exp6388,
        "_cuda_runtime_receipts",
        lambda: {
            "cuda_device_count": 2,
            "both_gpus_visible": True,
            "llama_cpp_gpu_offload_supported": True,
            "disk_available_gb": 100.0,
        },
    )
    monkeypatch.setattr(
        exp6388,
        "_live_entrypoint_receipts",
        lambda: {
            "exp6387_live_reachable": True,
            "active_reward_machine_default_off": True,
            "two_sided_goal_contract_default_off": True,
        },
    )

    artifact = exp6388.build_artifact(
        REPO,
        date="20260813",
        output_path=output,
        tests_run=("focused-tests",),
        duration_s=0.5,
    )

    assert output.exists()
    assert set(exp6388.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert "solve_provenance" not in artifact
    assert artifact["status"] == "complete"
    assert artifact["MODEL_SPECS"] == [dict(row) for row in model_specs]
    assert artifact["models_used"] == list(exp6388.MANDATED_MODEL_IDS)
    assert artifact["autotokenizer_usage_count"] == 0
    assert artifact["arc_solve_claim"] is False
    assert artifact["verifier_is_oracle"] is False
    assert artifact["arc_evidence_calibration_ready_score"] == 1.0
    assert artifact["delta_admission_precision"]["pooled_unrounded"] > 0.0
    assert artifact["delta_false_accept_count"] < 0
    assert artifact["forbidden_access_and_registry_write_counts"]["registry_write_count"] == 0
    assert artifact["protected_files_unchanged"]["ops/arc_solve_registry.yaml"] is True
