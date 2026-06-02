"""Tests for Exp 3736 real Thesis-A kill-gate part-(a) verdict.

Spec refs: REQ-EBT-3736, SCENARIO-EBT-3736-PASS,
SCENARIO-EBT-3736-UNTESTED, SCENARIO-EBT-3736-DIVERGED.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import experiment_3736_real_kill_gate_part_a_verdict as exp3736


SPEC_PATH = Path("openspec/capabilities/ebt-nrgpt/spec.md")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _exp3727() -> dict[str, object]:
    return {
        "honest_verdict": "complete: matched_compute_eval_harness_built",
        "flop_model_description": "FLOP model documented.",
        "matched_compute_report": {
            "ebt_total_flops": 10000,
            "ar_total_flops": 10000,
            "budget_match": {
                "ar_best_of_m": 5,
                "target_total_flops": 10000,
                "within_tolerance": True,
            },
        },
        "random_seed": 20260602,
        "reproducibility_checksum": "7" * 64,
        "duration_s": 1.9,
    }


def _exp3729() -> dict[str, object]:
    return {
        "honest_verdict": (
            "complete: kill_gate_part_a_FAIL_energy_as_generator_bounded_at_"
            "small_scale_honest_negative_stop"
        ),
        "ebt_trained_stably": False,
        "green_light_342": False,
        "kill_gate_conclusion": "BOUNDED: original infra false-negative.",
        "reproducibility_checksum": "9" * 64,
    }


def _chunk1(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "honest_verdict": (
            "complete: harness_fixed_ebt_train_chunk_2_steps_stable_so_far_"
            "loss_converging_no_nan_ar_baseline_co_trained_checkpointed"
        ),
        "inference_substrate": "live_llm_inference",
        "harness_fix_applied": True,
        "cumulative_steps_trained": 2,
        "ebt_loss_curve": [0.9902918338775635, 1.141614317893982],
        "ar_loss_curve": [5.67, 5.83],
        "nan_or_divergence_events": False,
        "stabilizers_applied": "replay_buffer, grad_clip",
        "peak_vram_mb": 100,
        "preconditions_checked": {
            "cuda": False,
            "ebt_vendored": True,
            "corpus_ok": True,
        },
        "random_seed": 3734,
        "reproducibility_checksum": "4" * 64,
        "duration_s": 1.64,
    }
    payload.update(overrides)
    return payload


def _chunk2(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "honest_verdict": "blocked_cuda",
        "inference_substrate": "live_llm_inference",
        "cumulative_steps_trained": 0,
        "ebt_loss_curve": [],
        "ar_loss_curve": [],
        "ebt_converged": False,
        "nan_or_divergence_events": False,
        "stabilizers_applied": "none",
        "peak_vram_mb": 0,
        "preconditions_checked": {
            "cuda": False,
            "ebt_vendored": True,
            "checkpoint_present": True,
        },
        "random_seed": 3734,
        "reproducibility_checksum": "",
        "duration_s": 0.05,
    }
    payload.update(overrides)
    return payload


def _seed_root(
    root: Path,
    *,
    exp3734: dict[str, object] | None = None,
    exp3735: dict[str, object] | None = None,
    include_exp3729: bool = True,
) -> None:
    _write_json(root / exp3736.EXP3727_REL_PATH, _exp3727())
    if include_exp3729:
        _write_json(root / exp3736.EXP3729_REL_PATH, _exp3729())
    if exp3734 is not None:
        _write_json(root / exp3736.EXP3734_REL_PATH, exp3734)
    if exp3735 is not None:
        _write_json(root / exp3736.EXP3735_REL_PATH, exp3735)


def test_req_ebt_3736_spec_anchor_exists() -> None:
    """REQ-EBT-3736: OpenSpec declares the real kill-gate contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-EBT-3736" in spec
    assert "SCENARIO-EBT-3736-PASS" in spec
    assert "SCENARIO-EBT-3736-UNTESTED" in spec
    assert "SCENARIO-EBT-3736-DIVERGED" in spec
    assert exp3736.OUTPUT_REL_PATH.as_posix() in spec


def test_scenario_ebt_3736_blocked_resume_is_untested_not_bounded(tmp_path: Path) -> None:
    """SCENARIO-EBT-3736-UNTESTED: blocked chunk2 leaves part-(a) untested."""

    _seed_root(tmp_path, exp3734=_chunk1(), exp3735=_chunk2())
    artifact = exp3736.build_artifact(
        tmp_path,
        started_s=10.0,
        now_s=10.5,
        adversarial_verify_report={"flags": []},
    )

    exp3736.validate_artifact(artifact)
    assert artifact["honest_verdict"] == exp3736.UNTESTED_VERDICT
    assert artifact["inference_substrate"] == exp3736.INFERENCE_SUBSTRATE
    assert artifact["ebt_trained_stably"] is False
    assert artifact["green_light_342"] is False
    assert artifact["training_actually_ran"] is True
    assert artifact["supersedes_exp3729"] is True
    assert artifact["real_run_diagnostics"]["cumulative_steps_trained"] == 2
    assert artifact["real_run_diagnostics"]["bounded_run_completed"] is False
    assert "UNTESTED" in artifact["kill_gate_conclusion"]
    assert "training did not complete" in artifact["kill_gate_conclusion"]
    assert "bounded at small scale" not in artifact["kill_gate_conclusion"]
    assert artifact["reproducibility_checksum"] == exp3736.payload_checksum(artifact)

    citations = {item["experiment_id"]: item for item in artifact["cited_upstream_artifacts"]}
    assert {3727, 3729, 3734, 3735} == set(citations)
    assert citations[3734]["sha256"] == exp3736.sha256_file(tmp_path / exp3736.EXP3734_REL_PATH)
    assert citations[3735]["sha256"] == exp3736.sha256_file(tmp_path / exp3736.EXP3735_REL_PATH)
    assert "ebt_loss_curve" in citations[3734]["fields_imported"]
    assert "ebt_converged" in citations[3735]["fields_imported"]


def test_scenario_ebt_3736_stable_real_run_green_lights_part_b(tmp_path: Path) -> None:
    """SCENARIO-EBT-3736-PASS: all four criteria must pass for green-light."""

    _seed_root(
        tmp_path,
        exp3734=_chunk1(ebt_loss_curve=[5.0, 4.6]),
        exp3735=_chunk2(
            honest_verdict=(
                "complete: ebt_train_resumed_total_100_steps_stable_converged_"
                "no_nan_ar_co_trained_ready_for_part_a_verdict"
            ),
            cumulative_steps_trained=100,
            ebt_loss_curve=[5.0, 4.2, 3.8, 3.72, 3.7],
            ar_loss_curve=[5.9, 5.8, 5.7],
            ebt_converged=True,
            gradient_norm_curve=[1.2, 1.0, 0.8],
            stabilizers_applied="replay_buffer, grad_clip",
            peak_vram_mb=4200,
            preconditions_checked={
                "cuda": True,
                "ebt_vendored": True,
                "checkpoint_present": True,
            },
            reproducibility_checksum="5" * 64,
            duration_s=1200.0,
        ),
    )
    (tmp_path / "results" / "experiment_3735_checkpoint.pt").touch()

    artifact = exp3736.build_artifact(
        tmp_path,
        started_s=1.0,
        now_s=1.25,
        adversarial_verify_report={"flags": []},
    )

    assert artifact["honest_verdict"] == exp3736.PASS_VERDICT
    assert artifact["ebt_trained_stably"] is True
    assert artifact["green_light_342"] is True
    assert artifact["training_actually_ran"] is True
    assert artifact["real_run_diagnostics"]["no_nan_inf_or_divergence"] is True
    assert artifact["real_run_diagnostics"]["gradient_norms_bounded"] is True
    assert artifact["real_run_diagnostics"]["non_runaway_convergence"] is True
    assert "results/experiment_3735_checkpoint.pt" in artifact["recommended_part_b_setup"]
    assert "10000" in artifact["recommended_part_b_setup"]
    assert "GREEN-LIGHT" in artifact["kill_gate_conclusion"]


def test_scenario_ebt_3736_genuine_divergence_is_bounded_negative(tmp_path: Path) -> None:
    """SCENARIO-EBT-3736-DIVERGED: real divergence is not downgraded to untested."""

    _seed_root(
        tmp_path,
        exp3734=_chunk1(ebt_loss_curve=[1.0, 0.5], gradient_norms_bounded=True),
        exp3735=_chunk2(
            honest_verdict="complete: ebt_train_resumed_diverged_at_step_12_genuine_part_a_signal_negative",
            cumulative_steps_trained=12,
            ebt_loss_curve=[1.0, 0.8, float("nan")],
            ebt_converged=False,
            nan_or_divergence_events=True,
            gradient_norms_bounded=False,
            preconditions_checked={
                "cuda": True,
                "ebt_vendored": True,
                "checkpoint_present": True,
            },
            reproducibility_checksum="6" * 64,
            duration_s=180.0,
        ),
    )

    artifact = exp3736.build_artifact(
        tmp_path,
        started_s=2.0,
        now_s=2.5,
        adversarial_verify_report={"flags": []},
    )

    assert artifact["honest_verdict"] == exp3736.DIVERGED_VERDICT
    assert artifact["ebt_trained_stably"] is False
    assert artifact["green_light_342"] is False
    assert artifact["training_actually_ran"] is True
    assert artifact["real_run_diagnostics"]["genuine_divergence"] is True
    assert "BOUNDED" in artifact["kill_gate_conclusion"]
    assert "energy-as-generator is bounded at small scale" in artifact["kill_gate_conclusion"]


def test_scenario_ebt_3736_runaway_collapse_is_genuine_divergence(tmp_path: Path) -> None:
    """SCENARIO-EBT-3736-DIVERGED: runaway negative energy is not convergence."""

    _seed_root(
        tmp_path,
        exp3734=_chunk1(ebt_loss_curve=[1.0, 0.5]),
        exp3735=_chunk2(
            honest_verdict="complete: ebt_train_resumed_total_100_steps_ready",
            cumulative_steps_trained=100,
            ebt_loss_curve=[1.0, -2_000_000.0],
            ebt_converged=True,
            gradient_norms_bounded=True,
            preconditions_checked={
                "cuda": True,
                "ebt_vendored": True,
                "checkpoint_present": True,
            },
            reproducibility_checksum="a" * 64,
            duration_s=200.0,
        ),
    )

    artifact = exp3736.build_artifact(
        tmp_path,
        started_s=2.0,
        now_s=2.5,
        adversarial_verify_report={"flags": []},
    )

    assert artifact["honest_verdict"] == exp3736.DIVERGED_VERDICT
    assert artifact["real_run_diagnostics"]["genuine_divergence"] is True
    assert artifact["real_run_diagnostics"]["non_runaway_convergence"] is False


def test_scenario_ebt_3736_invalid_loss_or_gradient_diagnostics_fail_closed(tmp_path: Path) -> None:
    """REQ-EBT-3736: invalid numeric diagnostics cannot become a pass."""

    _seed_root(
        tmp_path,
        exp3734=_chunk1(ebt_loss_curve=[1.0, 0.5]),
        exp3735=_chunk2(
            honest_verdict="complete: ebt_train_resumed_total_100_steps_ready",
            cumulative_steps_trained=100,
            ebt_loss_curve=[1.0, float("nan")],
            ebt_converged=True,
            gradient_norm_curve=[1.0, 200.0],
            preconditions_checked={
                "cuda": True,
                "ebt_vendored": True,
                "checkpoint_present": True,
            },
            reproducibility_checksum="b" * 64,
            duration_s=200.0,
        ),
    )

    artifact = exp3736.build_artifact(
        tmp_path,
        started_s=2.0,
        now_s=2.5,
        adversarial_verify_report={"flags": []},
    )

    assert artifact["honest_verdict"] == exp3736.DIVERGED_VERDICT
    assert artifact["real_run_diagnostics"]["no_nan_inf_or_divergence"] is False
    assert artifact["real_run_diagnostics"]["gradient_norms_bounded"] is False


def test_scenario_ebt_3736_unbounded_gradient_flag_is_genuine_divergence(tmp_path: Path) -> None:
    """SCENARIO-EBT-3736-DIVERGED: unbounded gradients fail the kill-gate."""

    _seed_root(
        tmp_path,
        exp3734=_chunk1(ebt_loss_curve=[1.0, 0.5]),
        exp3735=_chunk2(
            honest_verdict="complete: ebt_train_resumed_total_100_steps_ready",
            cumulative_steps_trained=100,
            ebt_loss_curve=[1.0, 0.8, 0.7],
            ebt_converged=True,
            gradient_norms_bounded=False,
            preconditions_checked={
                "cuda": True,
                "ebt_vendored": True,
                "checkpoint_present": True,
            },
            reproducibility_checksum="c" * 64,
            duration_s=200.0,
        ),
    )

    artifact = exp3736.build_artifact(
        tmp_path,
        started_s=2.0,
        now_s=2.5,
        adversarial_verify_report={"flags": []},
    )

    assert artifact["honest_verdict"] == exp3736.DIVERGED_VERDICT
    assert artifact["real_run_diagnostics"]["gradient_norms_bounded"] is False
    assert artifact["real_run_diagnostics"]["genuine_divergence"] is True


def test_req_ebt_3736_absent_artifact_fallback_does_not_crash(tmp_path: Path) -> None:
    """REQ-EBT-3736: missing real-run artifacts produce untested, not None-crash."""

    _seed_root(tmp_path, exp3734=None, exp3735=None, include_exp3729=False)

    artifact = exp3736.build_artifact(
        tmp_path,
        started_s=3.0,
        now_s=3.5,
        adversarial_verify_report={"flags": []},
    )

    assert artifact["honest_verdict"] == exp3736.UNTESTED_VERDICT
    assert artifact["training_actually_ran"] is False
    assert artifact["real_run_diagnostics"]["missing_or_blocked_artifacts"] == [3734, 3735]
    assert artifact["cited_upstream_artifacts"][0]["experiment_id"] == 3727


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.pop("honest_verdict"), "missing required"),
        (lambda p: p.update(honest_verdict="complete: wrong"), "terminal verdict"),
        (lambda p: p.update(inference_substrate="live_llm_inference"), "inference_substrate"),
        (lambda p: p.update(ebt_trained_stably="yes"), "ebt_trained_stably"),
        (lambda p: p.update(green_light_342="no"), "green_light_342"),
        (lambda p: p.update(ebt_trained_stably=True, green_light_342=False), "green_light_342"),
        (lambda p: p.update(training_actually_ran="yes"), "training_actually_ran"),
        (lambda p: p.update(supersedes_exp3729=False), "supersedes"),
        (lambda p: p.update(kill_gate_conclusion=""), "kill_gate_conclusion"),
        (lambda p: p.update(cited_upstream_artifacts=[]), "cite"),
        (lambda p: p.update(random_seed=3734), "random_seed"),
        (lambda p: p.update(reproducibility_checksum="bad"), "sha256"),
        (lambda p: p.update(duration_s=0.0), "duration_s"),
        (lambda p: p.update(field_principles=[]), "field_principles"),
        (lambda p: p["field_principles"].pop("green_light_342"), "field principles"),
        (lambda p: p.update(model_specs={}), "live-model markers"),
        (lambda p: p.update(adversarial_verify_report={"flags": [{"severity": "critical"}]}), "critical"),
        (lambda p: p.update(reproducibility_checksum="0" * 64), "does not match"),
    ],
)
def test_req_ebt_3736_validate_rejects_regressions(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    """REQ-EBT-3736: schema validation blocks dishonest verdict artifacts."""

    _seed_root(tmp_path, exp3734=_chunk1(), exp3735=_chunk2())
    artifact = exp3736.build_artifact(
        tmp_path,
        started_s=4.0,
        now_s=4.5,
        adversarial_verify_report={"flags": []},
    )
    broken = json.loads(json.dumps(artifact))
    mutate(broken)

    with pytest.raises(ValueError, match=message):
        exp3736.validate_artifact(broken)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.update(cited_upstream_artifacts="bad"), "cite"),
        (lambda p: p["cited_upstream_artifacts"].append(123), "object"),
        (lambda p: p["cited_upstream_artifacts"][0].update(experiment_id=3730), "Exp 3727"),
        (lambda p: p["cited_upstream_artifacts"][0].update(fields_imported=[]), "fields_imported"),
        (lambda p: p["cited_upstream_artifacts"][0].update(sha256="bad"), "sha256"),
    ],
)
def test_req_ebt_3736_validate_rejects_bad_citations(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    """REQ-EBT-3736: provenance citations must stay auditable."""

    _seed_root(tmp_path, exp3734=_chunk1(), exp3735=_chunk2())
    artifact = exp3736.build_artifact(
        tmp_path,
        started_s=5.0,
        now_s=5.5,
        adversarial_verify_report={"flags": []},
    )
    broken = json.loads(json.dumps(artifact))
    mutate(broken)

    with pytest.raises(ValueError, match=message):
        exp3736.validate_artifact(broken)


def test_scenario_ebt_3736_write_runs_adversarial_verify(tmp_path: Path) -> None:
    """REQ-EBT-3736: writing the artifact records verifier no-critical status."""

    _seed_root(tmp_path, exp3734=_chunk1(), exp3735=_chunk2())

    output = exp3736.write_artifact(tmp_path, started_s=1.0, now_s=1.5)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / exp3736.OUTPUT_REL_PATH
    assert payload["adversarial_verify_clean"] is True
    assert payload["adversarial_verify_report"]["critical_flag_count"] == 0
    assert payload["reproducibility_checksum"] == exp3736.payload_checksum(payload)
    exp3736.validate_artifact(payload)


def test_req_ebt_3736_main_writes_default_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-EBT-3736: CLI writes and prints the terminal verdict."""

    _seed_root(tmp_path, exp3734=_chunk1(), exp3735=_chunk2())
    monkeypatch.setattr(exp3736, "REPO_ROOT", tmp_path)

    assert exp3736.main([]) == 0

    payload = json.loads((tmp_path / exp3736.OUTPUT_REL_PATH).read_text(encoding="utf-8"))
    assert payload["honest_verdict"] == exp3736.UNTESTED_VERDICT
    assert exp3736.UNTESTED_VERDICT in capsys.readouterr().out


def test_req_ebt_3736_non_object_json_is_rejected(tmp_path: Path) -> None:
    """REQ-EBT-3736: malformed present artifacts fail explicitly."""

    _write_json(tmp_path / exp3736.EXP3727_REL_PATH, _exp3727())
    bad = tmp_path / exp3736.EXP3734_REL_PATH
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("[1, 2, 3]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a JSON object"):
        exp3736.build_artifact(
            tmp_path,
            started_s=6.0,
            now_s=6.5,
            adversarial_verify_report={"flags": []},
        )


def test_req_ebt_3736_missing_required_harness_fails_explicitly(tmp_path: Path) -> None:
    """REQ-EBT-3736: the matched-compute harness citation is required."""

    with pytest.raises(FileNotFoundError):
        exp3736.build_artifact(
            tmp_path,
            started_s=7.0,
            now_s=7.5,
            adversarial_verify_report={"flags": []},
        )


def test_req_ebt_3736_helper_edge_cases() -> None:
    """REQ-EBT-3736: helper edge cases stay deterministic."""

    assert exp3736._finite_curve("not-a-list") == []
    assert exp3736._runaway_collapse([]) is False
    assert exp3736._curve_explodes([1.0]) is False
    assert exp3736._is_blocked(None) is True
    assert exp3736._critical_flag_count(None) == 0
    assert exp3736._critical_flag_count({"flags": "bad"}) == 0
    assert exp3736._gradient_norms_exploded([{"gradient_norm_curve": [1.0, 200.0]}]) is True
    assert exp3736._safe_int(object()) == 0
    assert exp3736._is_sha256("z" * 64) is False
