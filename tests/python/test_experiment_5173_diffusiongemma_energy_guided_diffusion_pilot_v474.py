"""Tests for Exp 5173 DiffusionGemma energy-guided diffusion pilot.

Spec refs: REQ-VERIFY-5173, SCENARIO-VERIFY-5173-GATED,
SCENARIO-VERIFY-5173-PILOT.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5173_diffusiongemma_energy_guided_diffusion_pilot_v474 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"


def _gate(passed: bool = True) -> dict[str, Any]:
    return {
        "experiment": "experiment_5171_harden_set_encoder_cross_corpus_n30_v474",
        "gate_passed": passed,
        "held_out_task_n": 30,
        "cross_corpus_delta_n30": 0.5,
        "cross_corpus_delta_ci95_n30": [0.2667, 0.7],
        "honest_verdict": "success_arc_set_encoder_cross_corpus_gate_passed_n30",
    }


def _gpu(available: bool = True) -> mod.GpuAvailability:
    return mod.GpuAvailability(
        checked=True,
        gpu1_available=available,
        detail="gpu 1 idle" if available else "gpu 1 has python pid 1234",
    )


def _smoke(resolved: bool = True) -> mod.SmokeResult:
    return mod.SmokeResult(
        attempted=True,
        success=resolved,
        load_mode="4bit_nf4_devmap_auto_2gpu",
        model_class="DiffusionGemmaForBlockDiffusion",
        resolution=(
            "resolved by loading DiffusionGemmaForBlockDiffusion directly and "
            "passing decoder_input_ids on the first non-meta model device"
            if resolved
            else "blocked_diffusiongemma_meta_tensor_bug_unresolved: direct class load "
            "still raised Tensor.item() cannot be called on meta tensors"
        ),
        tried=[
            "AutoModelForCausalLM rejected DiffusionGemmaConfig",
            "DiffusionGemmaForBlockDiffusion direct load",
        ],
        error=None if resolved else "RuntimeError('Tensor.item() cannot be called on meta tensors')",
    )


def _arm_rows() -> list[dict[str, Any]]:
    return [
        {"task_id": "HumanEval/0", "unguided_passed": True, "guided_passed": True, "ar_passed": True},
        {"task_id": "HumanEval/1", "unguided_passed": False, "guided_passed": True, "ar_passed": True},
        {"task_id": "MBPP/2", "unguided_passed": False, "guided_passed": False, "ar_passed": True},
        {"task_id": "MBPP/3", "unguided_passed": False, "guided_passed": False, "ar_passed": False},
    ]


def test_req_verify_5173_spec_declares_pilot_contract() -> None:
    """REQ-VERIFY-5173: OpenSpec anchors the V474 DiffusionGemma pilot."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    start = spec.index("### REQ-VERIFY-5173")
    section = spec[start:]

    assert "SCENARIO-VERIFY-5173-GATED" in section
    assert "SCENARIO-VERIFY-5173-PILOT" in section
    assert mod.RESULT_RELATIVE_PATH in section
    assert "experiment_5171_harden_set_encoder_cross_corpus_n30_v474.json" in section
    assert "4-bit NF4" in section
    assert "device_map=\"auto\"" in section
    assert "unsloth/gemma-4-26B-A4B-it-GGUF" in section
    assert "verifier_is_oracle=true" in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_verify_5173_blocks_when_upstream_gate_failed() -> None:
    """SCENARIO-VERIFY-5173-GATED: a failed Exp5171 gate stops before GPU work."""

    artifact = mod.build_artifact(
        exp5171_gate=_gate(False),
        gpu_availability=None,
        smoke=None,
        arm_rows=[],
        compute_cost_per_arm={},
        tests_run=["focused"],
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_upstream_gate_not_passed"
    assert artifact["gpu1_availability_checked"]["value"] is False
    assert artifact["meta_tensor_bug_resolution"]["value"] == "not_attempted_upstream_gate_not_passed"
    assert artifact["pass_at_1_guided"]["value"] == 0.0


def test_scenario_verify_5173_blocks_when_gpu1_busy() -> None:
    """SCENARIO-VERIFY-5173-GATED: GPU-1 contention is reported, not ignored."""

    artifact = mod.build_artifact(
        exp5171_gate=_gate(True),
        gpu_availability=_gpu(False),
        smoke=None,
        arm_rows=[],
        compute_cost_per_arm={},
        tests_run=["focused"],
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_gpu1_busy"
    assert artifact["gpu1_availability_checked"]["value"] is True
    assert artifact["preconditions"]["gpu1"]["gpu1_available"] is False


def test_scenario_verify_5173_blocks_when_meta_tensor_bug_unresolved() -> None:
    """REQ-VERIFY-5173: unresolved DiffusionGemma meta tensors stop honestly."""

    artifact = mod.build_artifact(
        exp5171_gate=_gate(True),
        gpu_availability=_gpu(True),
        smoke=_smoke(False),
        arm_rows=[],
        compute_cost_per_arm={},
        tests_run=["focused"],
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_diffusiongemma_meta_tensor_bug_unresolved"
    assert "Tensor.item() cannot be called on meta tensors" in artifact[
        "meta_tensor_bug_resolution"
    ]["value"]
    assert "DiffusionGemmaForBlockDiffusion direct load" in artifact["preconditions"]["smoke"]["tried"]


def test_scenario_verify_5173_completed_artifact_reports_three_arms() -> None:
    """SCENARIO-VERIFY-5173-PILOT: three-arm pass@1, CIs, and compute cost are emitted."""

    artifact = mod.build_artifact(
        exp5171_gate=_gate(True),
        gpu_availability=_gpu(True),
        smoke=_smoke(True),
        arm_rows=_arm_rows(),
        compute_cost_per_arm={
            "unguided_diffusion": {"wall_clock_s": 40.0, "gpu_count": 2},
            "guided_diffusion": {"wall_clock_s": 46.0, "gpu_count": 2},
            "ar_best_of_n": {"wall_clock_s": 31.0, "gpu_count": 1},
        },
        tests_run=["focused"],
        bootstrap_resamples=256,
        random_seed=123,
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert "energy_guidance_helped_vs_unguided" in artifact["honest_verdict"]
    assert artifact["pass_at_1_unguided"]["value"] == pytest.approx(0.25)
    assert artifact["pass_at_1_guided"]["value"] == pytest.approx(0.5)
    assert artifact["pass_at_1_ar_baseline"]["value"] == pytest.approx(0.75)
    assert artifact["guided_vs_unguided_delta_ci95"]["value"][0] <= 0.25
    assert artifact["guided_vs_unguided_delta_ci95"]["value"][1] >= 0.25
    assert artifact["guided_vs_ar_delta_ci95"]["value"][0] <= -0.25
    assert artifact["guided_vs_ar_delta_ci95"]["value"][1] >= -0.25
    assert artifact["compute_cost_per_arm"]["value"]["guided_diffusion"]["gpu_count"] == 2
    assert artifact["verifier_is_oracle"]["value"] is True
    assert artifact["guidance_mechanism_design"]["value"] == mod.GUIDANCE_MECHANISM_DESIGN


def test_reweight_logits_applies_verifier_energy_penalty() -> None:
    """REQ-VERIFY-5173: the documented guidance rule has executable semantics."""

    adjusted = mod.reweight_logits_with_verifier_energy(
        logits=[3.0, 2.0, 1.0],
        verifier_energy=[0.0, 2.0, -1.0],
        lambda_energy=0.5,
    )

    assert adjusted == pytest.approx([3.0, 1.0, 1.5])


def test_utility_error_paths_and_checksum_inputs(tmp_path: Path) -> None:
    """REQ-VERIFY-5173: utility helpers fail closed on malformed inputs."""

    payload_path = tmp_path / "gate.json"
    payload_path.write_text(json.dumps(_gate(True)), encoding="utf-8")

    checksum = mod.stable_checksum({"path": payload_path, "gpu": _gpu(), "smoke": _smoke()})

    assert len(checksum) == 64
    assert mod.load_json(payload_path)["gate_passed"] is True
    assert mod.bootstrap_mean_ci([0.5]) == [0.5, 0.5]
    with pytest.raises(TypeError, match="not JSON serializable"):
        mod.stable_checksum({"bad": object()})
    with pytest.raises(ValueError, match="same length"):
        mod.reweight_logits_with_verifier_energy([1.0], [1.0, 2.0], lambda_energy=0.1)
    with pytest.raises(ValueError, match="non-negative"):
        mod.reweight_logits_with_verifier_energy([1.0], [1.0], lambda_energy=-0.1)
    with pytest.raises(ValueError, match="resamples"):
        mod.bootstrap_mean_ci([1.0, 0.0], resamples=0)


def test_completed_verdict_direction_variants() -> None:
    """SCENARIO-VERIFY-5173-PILOT: verdict text reflects helped/hurt/tied outcomes."""

    common = {
        "exp5171_gate": _gate(True),
        "gpu_availability": _gpu(True),
        "smoke": _smoke(True),
        "compute_cost_per_arm": {
            "unguided_diffusion": {"wall_clock_s": 1.0, "gpu_count": 2},
            "guided_diffusion": {"wall_clock_s": 1.0, "gpu_count": 2},
            "ar_best_of_n": {"wall_clock_s": 1.0, "gpu_count": 1},
        },
        "tests_run": ["focused"],
    }
    beat = mod.build_artifact(
        **common,
        arm_rows=[
            {"task_id": "t0", "unguided_passed": False, "guided_passed": True, "ar_passed": False}
        ],
    )
    hurt = mod.build_artifact(
        **common,
        arm_rows=[
            {"task_id": "t1", "unguided_passed": True, "guided_passed": False, "ar_passed": False}
        ],
    )
    tied = mod.build_artifact(
        **common,
        arm_rows=[
            {"task_id": "t2", "unguided_passed": False, "guided_passed": False, "ar_passed": False}
        ],
    )

    assert "energy_guidance_helped_vs_unguided_and_beat_ar" in beat["honest_verdict"]
    assert "energy_guidance_hurt_vs_unguided_and_tied_ar" in hurt["honest_verdict"]
    assert "energy_guidance_no_difference_vs_unguided_and_tied_ar" in tied["honest_verdict"]
    mod.validate_artifact(beat)
    mod.validate_artifact(hurt)
    mod.validate_artifact(tied)

    no_rows = mod.build_artifact(**common, arm_rows=[])
    assert no_rows["honest_verdict"] == "blocked_no_executable_code_rows"
    mod.validate_artifact(no_rows)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda artifact: {
                key: value for key, value in artifact.items() if key != "pass_at_1_guided"
            },
            "missing required fields",
        ),
        (lambda artifact: artifact | {"honest_verdict": "done"}, "honest_verdict"),
        (
            lambda artifact: artifact | {"verifier_is_oracle": {"value": False, "principle": mod.FIELD_PRINCIPLES["verifier_is_oracle"]}},
            "verifier_is_oracle",
        ),
        (
            lambda artifact: artifact | {"pass_at_1_guided": {"value": 1.5, "principle": mod.FIELD_PRINCIPLES["pass_at_1_guided"]}},
            "pass_at_1_guided",
        ),
        (
            lambda artifact: artifact | {"guided_vs_ar_delta_ci95": {"value": [0.1], "principle": mod.FIELD_PRINCIPLES["guided_vs_ar_delta_ci95"]}},
            "guided_vs_ar_delta_ci95",
        ),
        (
            lambda artifact: artifact | {"compute_cost_per_arm": {"value": {"guided_diffusion": {"wall_clock_s": -1.0, "gpu_count": 2}}, "principle": mod.FIELD_PRINCIPLES["compute_cost_per_arm"]}},
            "compute_cost_per_arm",
        ),
        (
            lambda artifact: artifact | {"meta_tensor_bug_resolution": {"value": "ok", "principle": "wrong"}},
            "declared principle",
        ),
        (lambda artifact: artifact | {"field_principles": {}}, "field_principles"),
        (
            lambda artifact: artifact | {"guided_vs_ar_delta_ci95": [0.0, 1.0]},
            "principle-wrapped",
        ),
        (lambda artifact: artifact | {"compute_cost_per_arm": {}}, "principle-wrapped"),
        (
            lambda artifact: artifact | {"compute_cost_per_arm": {"value": {}, "principle": mod.FIELD_PRINCIPLES["compute_cost_per_arm"]}},
            "compute_cost_per_arm",
        ),
        (
            lambda artifact: artifact | {"compute_cost_per_arm": {"value": "bad", "principle": mod.FIELD_PRINCIPLES["compute_cost_per_arm"]}},
            "compute_cost_per_arm",
        ),
        (
            lambda artifact: artifact | {"compute_cost_per_arm": {"value": {"unguided_diffusion": "bad", "guided_diffusion": {"wall_clock_s": 1.0, "gpu_count": 2}, "ar_best_of_n": {"wall_clock_s": 1.0, "gpu_count": 1}}, "principle": mod.FIELD_PRINCIPLES["compute_cost_per_arm"]}},
            "compute_cost_per_arm",
        ),
        (
            lambda artifact: artifact | {"compute_cost_per_arm": {"value": {"unguided_diffusion": {"wall_clock_s": 1.0, "gpu_count": 2}, "guided_diffusion": {"wall_clock_s": -1.0, "gpu_count": 2}, "ar_best_of_n": {"wall_clock_s": 1.0, "gpu_count": 1}}, "principle": mod.FIELD_PRINCIPLES["compute_cost_per_arm"]}},
            "compute_cost_per_arm",
        ),
        (
            lambda artifact: artifact | {"gpu1_availability_checked": {"value": "yes", "principle": mod.FIELD_PRINCIPLES["gpu1_availability_checked"]}},
            "gpu1_availability_checked",
        ),
        (
            lambda artifact: artifact | {"reproducibility_checksum": {"value": "short", "principle": mod.FIELD_PRINCIPLES["reproducibility_checksum"]}},
            "reproducibility_checksum",
        ),
        (lambda artifact: artifact | {"spec_refs": []}, "spec_refs"),
        (lambda artifact: artifact | {"tests_run": []}, "tests_run"),
    ],
)
def test_validate_artifact_rejects_invalid_schema(mutate: object, message: str) -> None:
    """REQ-VERIFY-5173: invalid pilot artifacts fail closed before writing."""

    artifact = mod.build_artifact(
        exp5171_gate=_gate(True),
        gpu_availability=_gpu(True),
        smoke=_smoke(True),
        arm_rows=_arm_rows(),
        compute_cost_per_arm={
            "unguided_diffusion": {"wall_clock_s": 40.0, "gpu_count": 2},
            "guided_diffusion": {"wall_clock_s": 46.0, "gpu_count": 2},
            "ar_best_of_n": {"wall_clock_s": 31.0, "gpu_count": 1},
        },
        tests_run=["focused"],
    )

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(mutate(artifact))


def test_write_result_is_stable_json(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5173-PILOT: writer emits the validated JSON artifact."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    artifact = mod.write_result(
        result_path=result_path,
        exp5171_gate=_gate(True),
        gpu_availability=_gpu(True),
        smoke=_smoke(True),
        arm_rows=_arm_rows(),
        compute_cost_per_arm={
            "unguided_diffusion": {"wall_clock_s": 40.0, "gpu_count": 2},
            "guided_diffusion": {"wall_clock_s": 46.0, "gpu_count": 2},
            "ar_best_of_n": {"wall_clock_s": 31.0, "gpu_count": 1},
        },
        tests_run=["focused"],
    )

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact


def test_main_writes_preflight_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """REQ-VERIFY-5173: CLI entrypoint writes a blocked preflight artifact."""

    gate_path = tmp_path / mod.EXP5171_RELATIVE_PATH
    gate_path.parent.mkdir(parents=True)
    gate_path.write_text(json.dumps(_gate(True)), encoding="utf-8")
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)

    mod.main()

    written = tmp_path / mod.RESULT_RELATIVE_PATH
    captured = capsys.readouterr().out
    assert written.exists()
    assert "blocked_gpu1_busy" in captured
