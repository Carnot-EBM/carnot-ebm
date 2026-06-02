"""Tests for Exp 3733 clean corrigendum of the Exp 3729 false-negative.

Spec: REQ-EBT-3733, SCENARIO-EBT-3733.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import experiment_3733_corrigendum_exp3729_false_negative as exp3733


SPEC_PATH = Path("openspec/capabilities/ebt-nrgpt/spec.md")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _seed_upstreams(root: Path) -> dict[str, dict[str, object]]:
    exp3726: dict[str, object] = {
        "schema": "carnot.experiment_3726_tiny_ebt_train_smoke.v1",
        "experiment": 3726,
        "honest_verdict": (
            "complete: tiny_ebt_38M_fits_3090_1283mb_single_train_step_loss_"
            "finite_and_decreasing_corpus_gsm8k_n2048"
        ),
        "first_step_losses": [-0.077116, -19.918041, -37.738224],
        "loss_finite": True,
        "loss_decreased": True,
        "ebt_param_count": 37954560,
        "peak_vram_mb": 1283,
        "n_train": 2048,
        "random_seed": 3726,
        "reproducibility_checksum": "8" * 64,
        "duration_s": 7.84,
    }
    exp3728: dict[str, object] = {
        "honest_verdict": "blocked_ebt",
        "cumulative_steps_trained": 0,
        "ebt_loss_curve": [],
        "ar_loss_curve": [],
        "ebt_converged": False,
        "nan_or_divergence_events": False,
        "stabilizers_applied": "none",
        "peak_vram_mb": 0,
        "preconditions_checked": {
            "ebt_vendored": False,
            "smoke_passed": False,
            "corpus_ok": False,
        },
        "random_seed": 3728,
        "reproducibility_checksum": "0" * 64,
        "duration_s": 65.5,
    }
    exp3729: dict[str, object] = {
        "schema": "carnot.experiment_3729_stability_kill_gate_verdict.v1",
        "experiment": 3729,
        "honest_verdict": (
            "complete: kill_gate_part_a_FAIL_energy_as_generator_bounded_at_"
            "small_scale_honest_negative_stop"
        ),
        "inference_substrate": (
            "aggregation_from_upstream_artifacts (principle: a verdict over "
            "upstream diagnostics, no live model)."
        ),
        "ebt_trained_stably": False,
        "green_light_342": False,
        "kill_gate_conclusion": (
            "BOUNDED: Exp 3728 does not show stable bounded convergence "
            "evidence (verdict=blocked_ebt, steps=0, loss_converged=False, "
            "nan_or_divergence_events=False). Energy-as-generator is bounded "
            "at small scale on this corpus and budget; stop the .342 "
            "matched-compute comparison unless a separately budgeted "
            "stabilization recipe is explicitly approved."
        ),
        "stability_diagnostics": {
            "source_honest_verdict": "blocked_ebt",
            "cumulative_steps_trained": 0,
            "ebt_loss_curve": [],
            "loss_converged": False,
            "ebt_converged": False,
            "nan_or_divergence_events": False,
            "gradient_norms_bounded": False,
            "stabilizers_applied": "none",
            "peak_vram_mb": 0,
            "checkpoint_path": None,
            "bounded_steps_present": False,
            "terminal_complete": False,
            "ebt_trained_stably": False,
        },
        "cited_upstream_artifacts": [],
        "field_principles": {"honest_verdict": "Terminal prefix."},
        "random_seed": 3729,
        "reproducibility_checksum": "2" * 64,
        "duration_s": 0.000369453,
    }
    _write_json(root / exp3733.EXP3726_REL_PATH, exp3726)
    _write_json(root / exp3733.EXP3728_REL_PATH, exp3728)
    _write_json(root / exp3733.EXP3729_REL_PATH, exp3729)
    return {"3726": exp3726, "3728": exp3728, "3729": exp3729}


def test_req_ebt_3733_spec_anchor_exists() -> None:
    """REQ-EBT-3733: OpenSpec declares the corrigendum contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-EBT-3733" in spec
    assert "SCENARIO-EBT-3733" in spec
    assert exp3733.OUTPUT_REL_PATH.as_posix() in spec


def test_scenario_ebt_3733_builds_clean_corrigendum_contract(tmp_path: Path) -> None:
    """SCENARIO-EBT-3733: the original FAIL is preserved and corrected."""

    upstreams = _seed_upstreams(tmp_path)
    artifact = exp3733.build_artifact(
        tmp_path,
        started_s=10.0,
        now_s=10.25,
        adversarial_verify_clean=True,
        adversarial_verify_report={"flags": []},
    )

    exp3733.validate_artifact(artifact)
    assert artifact["honest_verdict"] == exp3733.TERMINAL_VERDICT
    assert set(exp3733.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(exp3733.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["inference_substrate"] == exp3733.INFERENCE_SUBSTRATE
    assert artifact["original_exp3729_preserved"] is True
    assert artifact["original_exp3729"] == upstreams["3729"]
    assert artifact["false_negative_root_cause"] == exp3733.FALSE_NEGATIVE_ROOT_CAUSE
    assert artifact["positive_control_passed"] is True
    assert artifact["positive_control_evidence"] == {
        "experiment_id": 3726,
        "ebt_param_count": 37954560,
        "peak_vram_mb": 1283,
        "loss_finite": True,
        "loss_decreased": True,
        "first_step_loss": -0.077116,
        "last_step_loss": -37.738224,
    }
    assert artifact["part_a_status_corrected"] == exp3733.PART_A_STATUS
    assert artifact["energy_as_generator_not_retired"] is True
    assert artifact["corrected_status_label"] == "part_a_reopened_untested_not_bounded"
    assert isinstance(artifact["recommended_rerun_label"], str)
    assert "exp3734" in artifact["recommended_342_rerun"]
    assert "exp3735" in artifact["recommended_342_rerun"]
    assert artifact["adversarial_verify_clean"] is True
    assert artifact["adversarial_verify_report"] == {
        "flag_count": 0,
        "max_severity": -1,
        "flags": [],
    }
    assert artifact["reproducibility_checksum"] == exp3733.payload_checksum(artifact)
    assert artifact["duration_s"] == 0.25

    citations = {item["experiment_id"]: item for item in artifact["cited_upstream_artifacts"]}
    assert set(citations) == {3726, 3728, 3729}
    assert citations[3726]["sha256"] == exp3733.sha256_file(tmp_path / exp3733.EXP3726_REL_PATH)
    assert citations[3728]["sha256"] == exp3733.sha256_file(tmp_path / exp3733.EXP3728_REL_PATH)
    assert citations[3729]["sha256"] == exp3733.sha256_file(tmp_path / exp3733.EXP3729_REL_PATH)
    assert "loss_decreased" in citations[3726]["fields_imported"]
    assert "preconditions_checked.ebt_vendored" in citations[3728]["fields_imported"]
    assert "green_light_342" in citations[3729]["fields_imported"]

    encoded = json.dumps(artifact, sort_keys=True)
    assert "GGUF" not in encoded
    assert "CUDA" not in encoded
    assert "model_specs" not in artifact
    assert "target_model" not in artifact
    assert "candidate_value" not in artifact


def test_scenario_ebt_3733_write_runs_adversarial_verify(tmp_path: Path) -> None:
    """SCENARIO-EBT-3733: the written artifact confirms no critical flag."""

    _seed_upstreams(tmp_path)

    output = exp3733.write_artifact(tmp_path, started_s=1.0, now_s=1.5)
    payload = json.loads(output.read_text(encoding="utf-8"))

    exp3733.validate_artifact(payload)
    assert output == tmp_path / exp3733.OUTPUT_REL_PATH
    assert payload["adversarial_verify_clean"] is True
    assert payload["adversarial_verify_report"]["max_severity"] < 2
    assert payload["reproducibility_checksum"] == exp3733.payload_checksum(payload)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.pop("original_exp3729_preserved"), "missing required"),
        (lambda p: p.update(honest_verdict="complete: wrong"), "terminal verdict"),
        (lambda p: p.update(inference_substrate="aggregation_from_upstream_artifacts"), "inference_substrate"),
        (lambda p: p.update(original_exp3729_preserved=False), "original Exp 3729"),
        (lambda p: p["original_exp3729"].update(green_light_342=True), "original Exp 3729"),
        (lambda p: p.update(false_negative_root_cause="blocked_ebt"), "root cause"),
        (lambda p: p.update(positive_control_passed=False), "positive control"),
        (lambda p: p.update(part_a_status_corrected="bounded"), "UNTESTED"),
        (lambda p: p.update(energy_as_generator_not_retired=False), "not retired"),
        (lambda p: p.update(corrected_status_label=0.0), "status label"),
        (lambda p: p.update(cited_upstream_artifacts=[]), "cite"),
        (lambda p: p.update(random_seed=3729), "random_seed"),
        (lambda p: p.update(reproducibility_checksum="short"), "sha256"),
        (lambda p: p.update(duration_s=0.0), "duration_s"),
        (lambda p: p.update(model_specs={}), "model_specs"),
        (lambda p: p.update(target_model="forbidden"), "target_model"),
        (lambda p: p.update(recommended_rerun_label=0.0), "recommended rerun label"),
        (lambda p: p.update(correction_note="CUDA marker"), "GGUF/CUDA"),
        (lambda p: p.update(reproducibility_checksum="0" * 64), "does not match"),
        (lambda p: p.update(field_principles=[]), "field_principles"),
        (lambda p: p["field_principles"].pop("positive_control_passed"), "field principles"),
    ],
)
def test_req_ebt_3733_validate_rejects_regressions(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    """REQ-EBT-3733: schema validation blocks dishonest corrigenda."""

    _seed_upstreams(tmp_path)
    artifact = exp3733.build_artifact(
        tmp_path,
        started_s=1.0,
        now_s=1.25,
        adversarial_verify_clean=True,
        adversarial_verify_report={"flags": []},
    )

    broken = json.loads(json.dumps(artifact))
    mutate(broken)

    with pytest.raises(ValueError, match=message):
        exp3733.validate_artifact(broken)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.update(cited_upstream_artifacts="bad"), "cite"),
        (
            lambda p: p["cited_upstream_artifacts"].append(123),
            "object",
        ),
        (
            lambda p: p["cited_upstream_artifacts"][0].update(fields_imported=[]),
            "fields_imported",
        ),
        (
            lambda p: p["cited_upstream_artifacts"][0].update(sha256="bad"),
            "sha256",
        ),
    ],
)
def test_req_ebt_3733_validate_rejects_malformed_citations(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    """REQ-EBT-3733: citation provenance must be shaped for audit."""

    _seed_upstreams(tmp_path)
    artifact = exp3733.build_artifact(
        tmp_path,
        started_s=1.0,
        now_s=1.25,
        adversarial_verify_clean=True,
        adversarial_verify_report={"flags": []},
    )

    broken = json.loads(json.dumps(artifact))
    mutate(broken)
    broken["reproducibility_checksum"] = exp3733.payload_checksum(broken)

    with pytest.raises(ValueError, match=message):
        exp3733.validate_artifact(broken)


def test_req_ebt_3733_defensive_helpers_cover_bad_shapes() -> None:
    """REQ-EBT-3733: helper guards handle malformed data without fabricating."""

    with pytest.raises(ValueError, match="positive control losses"):
        exp3733._positive_control_evidence({"first_step_losses": []})

    assert exp3733._get_nested({"a": []}, "a.b") is None
    assert exp3733._adversarial_report_is_clean({"flags": "bad"}) is True
    assert exp3733._is_sha256("g" * 64) is False


def test_req_ebt_3733_build_fails_closed_on_missing_positive_control(
    tmp_path: Path,
) -> None:
    """REQ-EBT-3733: no positive control means no clean correction claim."""

    _seed_upstreams(tmp_path)
    _write_json(
        tmp_path / exp3733.EXP3726_REL_PATH,
        {
            "loss_finite": True,
            "loss_decreased": False,
            "ebt_param_count": 37954560,
            "peak_vram_mb": 1283,
            "first_step_losses": [-0.1, -0.2],
        },
    )

    with pytest.raises(ValueError, match="positive control"):
        exp3733.build_artifact(tmp_path)


@pytest.mark.parametrize(
    ("path_attr", "replacement", "message"),
    [
        (
            "EXP3728_REL_PATH",
            {
                "honest_verdict": "blocked_ebt",
                "cumulative_steps_trained": 1,
                "preconditions_checked": {
                    "ebt_vendored": False,
                    "smoke_passed": False,
                },
            },
            "zero-step",
        ),
        (
            "EXP3728_REL_PATH",
            {
                "honest_verdict": "blocked_ebt",
                "cumulative_steps_trained": 0,
                "preconditions_checked": {
                    "ebt_vendored": True,
                    "smoke_passed": False,
                },
            },
            "cwd/import",
        ),
        (
            "EXP3729_REL_PATH",
            {
                "honest_verdict": "complete: green_light",
                "green_light_342": True,
                "ebt_trained_stably": True,
                "kill_gate_conclusion": "STABLE",
            },
            "original Exp 3729 false-negative",
        ),
    ],
)
def test_req_ebt_3733_build_fails_closed_on_wrong_upstream_shape(
    tmp_path: Path,
    path_attr: str,
    replacement: dict[str, object],
    message: str,
) -> None:
    """REQ-EBT-3733: the correction requires the exact infra-false-negative shape."""

    _seed_upstreams(tmp_path)
    _write_json(tmp_path / getattr(exp3733, path_attr), replacement)

    with pytest.raises(ValueError, match=message):
        exp3733.build_artifact(tmp_path)


def test_req_ebt_3733_cli_writes_artifact(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """REQ-EBT-3733: the script entrypoint writes the required deliverable."""

    _seed_upstreams(tmp_path)

    assert exp3733.main(["--root", str(tmp_path)]) == 0
    payload = json.loads((tmp_path / exp3733.OUTPUT_REL_PATH).read_text(encoding="utf-8"))
    printed = capsys.readouterr().out

    assert payload["honest_verdict"] == exp3733.TERMINAL_VERDICT
    assert exp3733.TERMINAL_VERDICT in printed
