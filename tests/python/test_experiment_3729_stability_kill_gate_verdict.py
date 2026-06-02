"""Tests for Exp 3729 stability kill-gate verdict.

Spec refs: REQ-EBT-3729, SCENARIO-EBT-3729, SCENARIO-EBT-3729-PASS.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from scripts import experiment_3729_stability_kill_gate_verdict as exp3729


def _write(path: Path, data: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _base_upstream(tmp_path: Path, exp3728: dict[str, object]) -> dict[str, Path]:
    results = tmp_path / "results"
    return {
        "3725": _write(
            results / "experiment_3725_ebt_fork_vendor_importable.json",
            {
                "honest_verdict": "complete: vendored",
                "importable": True,
                "smoke_energy_value": 0.5541654229164124,
                "random_seed": 42,
                "reproducibility_checksum": "2" * 64,
                "duration_s": 15,
            },
        ),
        "3726": _write(
            results / "experiment_3726_tiny_ebt_corpus_and_train_step_smoke.json",
            {
                "honest_verdict": "complete: train_step_smoke",
                "first_step_losses": [2.0, 1.0],
                "loss_finite": True,
                "loss_decreased": True,
                "ebt_param_count": 37_954_560,
                "peak_vram_mb": 1283,
                "random_seed": 3726,
                "reproducibility_checksum": "3" * 64,
                "duration_s": 7.84,
            },
        ),
        "3727": _write(
            results / "experiment_3727_matched_compute_eval_harness.json",
            {
                "honest_verdict": "complete: matched_compute_eval_harness",
                "matched_compute_report": {
                    "ebt_total_flops": 10000,
                    "ar_total_flops": 10000,
                    "budget_match": {"ar_best_of_m": 5, "within_tolerance": True},
                },
                "flop_model_description": "total inference FLOPs = parameter_count * sequence_tokens * forward_passes",
                "random_seed": 20260602,
                "reproducibility_checksum": "4" * 64,
                "duration_s": 1.9,
            },
        ),
        "3728": _write(
            results / "experiment_3728_bounded_checkpointed_train_ebt_and_ar.json",
            exp3728,
        ),
    }


def test_openspec_has_3729_requirement() -> None:
    """REQ-EBT-3729: the kill-gate implementation is OpenSpec anchored."""
    spec = Path("openspec/capabilities/ebt-nrgpt/spec.md").read_text(encoding="utf-8")

    assert "REQ-EBT-3729" in spec
    assert "SCENARIO-EBT-3729" in spec
    assert "SCENARIO-EBT-3729-PASS" in spec


def test_blocked_3728_writes_honest_negative_stop(tmp_path: Path) -> None:
    """SCENARIO-EBT-3729: blocked 3728 diagnostics stop the route honestly."""
    paths = _base_upstream(
        tmp_path,
        {
            "honest_verdict": "blocked_ebt",
            "cumulative_steps_trained": 0,
            "ebt_loss_curve": [],
            "ebt_converged": False,
            "nan_or_divergence_events": False,
            "stabilizers_applied": "none",
            "peak_vram_mb": 0,
            "preconditions_checked": {"ebt_vendored": False, "smoke_passed": False},
            "random_seed": 3728,
            "reproducibility_checksum": "0" * 64,
            "duration_s": 65.5,
        },
    )

    artifact = exp3729.build_artifact(paths, duration_s=0.25)

    assert artifact["honest_verdict"] == exp3729.FAIL_VERDICT
    assert artifact["inference_substrate"] == exp3729.INFERENCE_SUBSTRATE
    assert artifact["ebt_trained_stably"] is False
    assert artifact["green_light_342"] is False
    assert "bounded at small scale" in artifact["kill_gate_conclusion"]
    assert artifact["random_seed"] == 3729
    assert len(artifact["reproducibility_checksum"]) == 64
    assert exp3729.validate_artifact(artifact) == []
    exp3728_citation = next(
        item for item in artifact["cited_upstream_artifacts"] if item["experiment_id"] == 3728
    )
    assert exp3728_citation["sha256"] == exp3729.sha256_file(paths["3728"])
    assert "ebt_loss_curve" in exp3728_citation["fields_imported"]
    assert "nan_or_divergence_events" in exp3728_citation["fields_imported"]


def test_stable_3728_green_lights_matched_compute_setup(tmp_path: Path) -> None:
    """SCENARIO-EBT-3729-PASS: stable diagnostics green-light .342."""
    paths = _base_upstream(
        tmp_path,
        {
            "honest_verdict": "complete: stable_checkpointed_training",
            "cumulative_steps_trained": 100,
            "ebt_loss_curve": [5.0, 4.5, 4.0, 3.2],
            "ebt_converged": True,
            "nan_or_divergence_events": False,
            "gradient_norms_bounded": True,
            "stabilizers_applied": "lr_warmup, grad_clip",
            "peak_vram_mb": 4500,
            "checkpoint_path": "results/checkpoints/exp3728_step100.pt",
            "random_seed": 3728,
            "reproducibility_checksum": "8" * 64,
            "duration_s": 1500.0,
        },
    )

    artifact = exp3729.build_artifact(paths, duration_s=0.25)

    assert artifact["honest_verdict"] == exp3729.PASS_VERDICT
    assert artifact["ebt_trained_stably"] is True
    assert artifact["green_light_342"] is True
    assert "results/checkpoints/exp3728_step100.pt" in artifact["kill_gate_conclusion"]
    assert "10000" in artifact["kill_gate_conclusion"]
    assert "recommended_342_setup" in artifact
    exp3727_citation = next(
        item for item in artifact["cited_upstream_artifacts"] if item["experiment_id"] == 3727
    )
    assert "matched_compute_report.ebt_total_flops" in exp3727_citation["fields_imported"]
    assert exp3729.validate_artifact(artifact) == []


def test_green_light_uses_fallback_setup_when_3727_budget_missing(tmp_path: Path) -> None:
    """SCENARIO-EBT-3729-PASS: missing FLOP detail is named honestly."""
    paths = _base_upstream(
        tmp_path,
        {
            "honest_verdict": "complete: stable_checkpointed_training",
            "cumulative_steps_trained": 100,
            "ebt_loss_curve": [5.0, 3.2],
            "ebt_converged": True,
            "nan_or_divergence_events": False,
            "gradient_norms_bounded": True,
            "stabilizers_applied": "grad_clip",
            "peak_vram_mb": 4500,
            "random_seed": 3728,
            "reproducibility_checksum": "8" * 64,
            "duration_s": 1500.0,
        },
    )
    _write(
        paths["3727"],
        {
            "honest_verdict": "complete: matched_compute_eval_harness",
            "flop_model_description": "documented but budget absent",
            "reproducibility_checksum": "4" * 64,
        },
    )

    artifact = exp3729.build_artifact(paths, duration_s=0.25)

    assert "Exp 3727 matched-compute harness budget" in artifact["recommended_342_setup"]


@pytest.mark.parametrize(
    "exp3728_overrides",
    [
        {"ebt_loss_curve": [3.0, 3.2], "ebt_converged": True, "nan_or_divergence_events": False},
        {"ebt_loss_curve": [3.0, 2.0], "ebt_converged": True, "nan_or_divergence_events": True},
        {"ebt_loss_curve": [3.0, float("nan")], "ebt_converged": True, "nan_or_divergence_events": False},
    ],
)
def test_unstable_or_invalid_3728_diagnostics_stop(tmp_path: Path, exp3728_overrides: dict[str, object]) -> None:
    """REQ-EBT-3729: every stability condition must pass before green-lighting."""
    exp3728 = {
        "honest_verdict": "complete: attempted_training",
        "cumulative_steps_trained": 100,
        "stabilizers_applied": "grad_clip",
        "peak_vram_mb": 4500,
        "random_seed": 3728,
        "reproducibility_checksum": "8" * 64,
        "duration_s": 1500.0,
    }
    exp3728.update(exp3728_overrides)
    paths = _base_upstream(tmp_path, exp3728)

    artifact = exp3729.build_artifact(paths, duration_s=0.25)

    assert artifact["ebt_trained_stably"] is False
    assert artifact["green_light_342"] is False
    assert artifact["honest_verdict"] == exp3729.FAIL_VERDICT


def test_validate_artifact_reports_required_schema_failures() -> None:
    """REQ-EBT-3729: invalid verdict artifacts are rejected explicitly."""
    errors = exp3729.validate_artifact(
        {
            "honest_verdict": "complete: made_up",
            "inference_substrate": "live_llm_inference",
            "ebt_trained_stably": True,
            "green_light_342": False,
            "kill_gate_conclusion": "",
            "cited_upstream_artifacts": [],
            "random_seed": None,
            "reproducibility_checksum": "bad",
            "duration_s": 0,
        }
    )

    assert "honest_verdict must be one of the terminal kill-gate verdicts" in errors
    assert "inference_substrate must be aggregation_from_upstream_artifacts" in errors
    assert "green_light_342 must equal ebt_trained_stably" in errors
    assert "kill_gate_conclusion must be present" in errors
    assert "cited_upstream_artifacts must cite upstream artifacts" in errors
    assert "random_seed must equal 3729" in errors
    assert "reproducibility_checksum must be a sha256 hex string" in errors
    assert "duration_s must be positive" in errors


def test_validate_artifact_reports_missing_fields_and_malformed_citations() -> None:
    """REQ-EBT-3729: citation provenance must be well formed."""
    errors = exp3729.validate_artifact({})

    assert any(error.startswith("missing required fields:") for error in errors)
    assert "ebt_trained_stably must be boolean" in errors
    assert "green_light_342 must be boolean" in errors

    citation_errors = exp3729.validate_artifact(
        {
            "honest_verdict": exp3729.FAIL_VERDICT,
            "inference_substrate": exp3729.INFERENCE_SUBSTRATE,
            "ebt_trained_stably": False,
            "green_light_342": False,
            "kill_gate_conclusion": "BOUNDED: stop.",
            "cited_upstream_artifacts": [
                123,
                {"experiment_id": 3725, "fields_imported": [], "sha256": "bad"},
            ],
            "field_principles": dict(exp3729.FIELD_PRINCIPLES),
            "random_seed": 3729,
            "reproducibility_checksum": "a" * 64,
            "duration_s": 1.0,
        }
    )

    assert "cited_upstream_artifacts must include 3725, 3726, 3727, and 3728" in citation_errors
    assert "each citation must be an object" in citation_errors
    assert "each citation must include fields_imported" in citation_errors
    assert "each citation must include a sha256 hex string" in citation_errors


def test_loader_and_builder_error_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-EBT-3729: invalid source data and invalid built artifacts fail closed."""
    list_path = tmp_path / "not_object.json"
    list_path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a JSON object"):
        exp3729.load_json(list_path)

    assert exp3729._finite_loss_curve("not a list") == []

    paths = _base_upstream(
        tmp_path,
        {
            "honest_verdict": "blocked_ebt",
            "cumulative_steps_trained": 0,
            "ebt_loss_curve": [],
            "ebt_converged": False,
            "nan_or_divergence_events": False,
            "stabilizers_applied": "none",
            "peak_vram_mb": 0,
            "random_seed": 3728,
            "reproducibility_checksum": "0" * 64,
            "duration_s": 65.5,
        },
    )
    monkeypatch.setattr(exp3729, "validate_artifact", lambda artifact: ["forced validation error"])

    with pytest.raises(ValueError, match="forced validation error"):
        exp3729.build_artifact(paths, duration_s=0.25)


def test_main_writes_artifact_and_prints_terminal_verdict(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """REQ-EBT-3729: CLI writes the required result JSON."""
    paths = _base_upstream(
        tmp_path,
        {
            "honest_verdict": "blocked_ebt",
            "cumulative_steps_trained": 0,
            "ebt_loss_curve": [],
            "ebt_converged": False,
            "nan_or_divergence_events": False,
            "stabilizers_applied": "none",
            "peak_vram_mb": 0,
            "random_seed": 3728,
            "reproducibility_checksum": "0" * 64,
            "duration_s": 65.5,
        },
    )
    output_path = tmp_path / "results" / "experiment_3729_stability_kill_gate_verdict.json"

    rc = exp3729.main(
        [
            "--exp3725",
            str(paths["3725"]),
            "--exp3726",
            str(paths["3726"]),
            "--exp3727",
            str(paths["3727"]),
            "--exp3728",
            str(paths["3728"]),
            "--output",
            str(output_path),
        ]
    )

    assert rc == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["green_light_342"] is False
    assert exp3729.FAIL_VERDICT in capsys.readouterr().out


def test_main_revalidates_after_duration_update(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-EBT-3729: CLI fails closed if post-build validation fails."""
    paths = _base_upstream(
        tmp_path,
        {
            "honest_verdict": "blocked_ebt",
            "cumulative_steps_trained": 0,
            "ebt_loss_curve": [],
            "ebt_converged": False,
            "nan_or_divergence_events": False,
            "stabilizers_applied": "none",
            "peak_vram_mb": 0,
            "random_seed": 3728,
            "reproducibility_checksum": "0" * 64,
            "duration_s": 65.5,
        },
    )
    calls = {"n": 0}
    original_validate = exp3729.validate_artifact

    def validate_then_fail(artifact: dict[str, object]) -> list[str]:
        calls["n"] += 1
        if calls["n"] == 1:
            return original_validate(artifact)
        return ["post-build validation error"]

    monkeypatch.setattr(exp3729, "validate_artifact", validate_then_fail)

    with pytest.raises(ValueError, match="post-build validation error"):
        exp3729.main(
            [
                "--exp3725",
                str(paths["3725"]),
                "--exp3726",
                str(paths["3726"]),
                "--exp3727",
                str(paths["3727"]),
                "--exp3728",
                str(paths["3728"]),
                "--output",
                str(tmp_path / "results" / "bad.json"),
            ]
        )
