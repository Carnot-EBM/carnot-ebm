"""Tests for Exp 3742 v342 Thesis-A recovery capstone.

Spec refs: REQ-EBT-3742, SCENARIO-EBT-3742-UNTESTED,
SCENARIO-EBT-3742-FLAGGED.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import experiment_3742_capstone_v342 as exp3742


SPEC_PATH = Path("openspec/capabilities/ebt-nrgpt/spec.md")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact_3732() -> dict[str, object]:
    return {
        "schema": "carnot.archive_activation.v342",
        "experiment_id": 3732,
        "honest_verdict": (
            "complete: archived_v341_thesis_a_smoke_passed_but_killgate_was_"
            "infra_false_negative_part_a_reopened_untested_v342_active_"
            "paper_ready_true_frozen_headline_unchanged"
        ),
        "inference_substrate": (
            "aggregation_from_upstream_artifacts (principle: JSON-read + format)."
        ),
        "paper_ready_preserved": True,
        "paper_ready_evidence": {
            "g1": True,
            "g2": True,
            "g3": True,
            "g4": True,
            "paper_ready": True,
            "frozen_headline_unchanged": True,
            "frozen_headline_auroc": 0.9131,
        },
        "p01_status_preserved": "honest-negative-bounded",
        "v342_evidence": {"corrects_false_negative": True},
        "random_seed": 3732,
        "reproducibility_checksum": "2" * 64,
        "duration_s": 0.0001,
    }


def _artifact_3733() -> dict[str, object]:
    return {
        "experiment": 3733,
        "honest_verdict": (
            "complete: exp3729_killgate_corrected_infra_false_negative_"
            "part_a_reopened_untested_energy_as_generator_not_retired"
        ),
        "part_a_status_corrected": "UNTESTED_at_bounded_scale_not_bounded",
        "energy_as_generator_not_retired": True,
        "random_seed": 3733,
        "reproducibility_checksum": "3" * 64,
        "duration_s": 0.000635,
    }


def _artifact_3734() -> dict[str, object]:
    return {
        "honest_verdict": (
            "complete: harness_fixed_ebt_train_chunk_2_steps_stable_so_far_"
            "loss_converging_no_nan_ar_baseline_co_trained_checkpointed"
        ),
        "harness_fix_applied": True,
        "cumulative_steps_trained": 2,
        "ebt_loss_curve": [0.9902918338775635, 1.141614317893982],
        "ar_loss_curve": [5.6731038093566895, 5.8303961753845215],
        "nan_or_divergence_events": False,
        "stabilizers_applied": "grad_clip",
        "peak_vram_mb": 256,
        "random_seed": 3734,
        "reproducibility_checksum": "4" * 64,
        "duration_s": 1.64,
    }


def _artifact_3735() -> dict[str, object]:
    return {
        "honest_verdict": "blocked_cuda",
        "cumulative_steps_trained": 0,
        "ebt_loss_curve": [],
        "ar_loss_curve": [],
        "ebt_converged": False,
        "nan_or_divergence_events": False,
        "stabilizers_applied": "none",
        "peak_vram_mb": 0,
        "random_seed": 3735,
        "reproducibility_checksum": "",
        "duration_s": 0.05,
    }


def _artifact_3736(*, outcome: str = "untested") -> dict[str, object]:
    green = outcome == "green"
    bounded = outcome == "bounded"
    return {
        "schema": "carnot.experiment_3736_real_kill_gate_part_a_verdict.v1",
        "experiment": 3736,
        "honest_verdict": (
            exp3742.PART_A_GREEN_VERDICT
            if green
            else (
                "complete: real_kill_gate_part_a_genuinely_bounded_ebt_diverged_"
                "in_real_run_honest_negative"
                if bounded
                else "complete: real_kill_gate_part_a_untested_training_did_not_complete"
            )
        ),
        "green_light_342": green,
        "ebt_trained_stably": green,
        "training_actually_ran": True,
        "supersedes_exp3729": True,
        "kill_gate_conclusion": (
            "GREEN-LIGHT: trains stably enough to compare."
            if green
            else (
                "BOUNDED: genuine divergence; energy-as-generator is bounded at small scale."
                if bounded
                else "UNTESTED: training did not complete -- part-(a) remains untested."
            )
        ),
        "real_run_diagnostics": {
            "bounded_run_completed": green or bounded,
            "cumulative_steps_trained": 100 if green else 2,
            "genuine_divergence": bounded,
            "training_actually_ran": True,
        },
        "random_seed": 3736,
        "reproducibility_checksum": "6" * 64,
        "duration_s": 0.0001,
    }


def _artifact_3737(*, blocked: bool = True) -> dict[str, object]:
    return {
        "experiment": 3737,
        "honest_verdict": (
            "blocked_gate_check_failed"
            if blocked
            else "complete: ebt_generation_smoke_passed_bounded_heldout"
        ),
        "gate_check_summary": (
            "1 of 1 gate(s) failed; first failure: exp3736.green_light_342"
            if blocked
            else "gate passed"
        ),
        "random_seed": 3737,
        "reproducibility_checksum": "7" * 64,
        "duration_s": 0.0001,
    }


def _artifact_3738(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "experiment": 3738,
        "honest_verdict": "complete: matched_compute_comparison",
        "accuracy_delta": 0.04,
        "flops_matched_within_tolerance": True,
        "n_heldout": 128,
        "random_seed": 3738,
        "reproducibility_checksum": "8" * 64,
        "duration_s": 10.0,
    }
    payload.update(overrides)
    return payload


def _artifact_3739(*, outcome: str = "not-run") -> dict[str, object]:
    win = outcome == "win"
    bounded = outcome == "bounded"
    invalid = outcome == "invalid"
    return {
        "schema": "carnot.experiment_3739_kill_gate_part_b_verdict.v1",
        "experiment": 3739,
        "honest_verdict": (
            "complete: kill_gate_part_b_ebt_BEATS_ar_at_matched_compute_delta_0.04_n128"
            if win
            else (
                "complete: kill_gate_part_b_BOUNDED_ebt_does_not_beat_ar_at_"
                "equal_compute_honest_negative"
                if bounded
                else (
                    "complete: kill_gate_part_b_INVALID_flops_not_matched_rerun_exp3738_tighter_budget"
                    if invalid
                    else "complete: kill_gate_part_b_not_run_part_a_did_not_green_light"
                )
            )
        ),
        "thesis_a_outcome": (
            "ebt_beats_ar_at_matched_compute"
            if win
            else (
                "bounded_at_small_scale"
                if bounded
                else ("comparison_invalid" if invalid else "part_b_not_run")
            )
        ),
        "ebt_beats_ar_at_matched_compute": win,
        "accuracy_delta_cited": 0.04 if win else (-0.02 if bounded else None),
        "flops_matched_cited": True if (win or bounded) else None,
        "n_heldout_cited": 128 if (win or bounded) else None,
        "part_b_not_run_reason": None if (win or bounded or invalid) else "part-(a) did not green-light",
        "random_seed": 3739,
        "reproducibility_checksum": "9" * 64,
        "duration_s": 0.0001,
    }


def _artifact_3740() -> dict[str, object]:
    return {
        "honest_verdict": (
            "complete: fr11_v15_tier1_stabilizer_efficacy_tracker_recipe_"
            "recommended_state_persisted_preliminary_over_3_chunks"
        ),
        "tracker_state_persisted": True,
        "n_chunks_observed": 3,
        "recommended_recipe": {"stabilizers": ["grad_clip"], "is_preliminary_heuristic": True},
        "acceptance_gate": {"condition": "tracker_state_persisted == true", "passed": True},
        "random_seed": 3740,
        "reproducibility_checksum": "a" * 64,
        "duration_s": 0.852,
    }


def _artifact_3741() -> dict[str, object]:
    return {
        "experiment_id": 3741,
        "honest_verdict": (
            "complete: kv260_terminal_state_holds_ssh_reachable_"
            "accelerator_loadable_opportunistic_audit"
        ),
        "terminal_state_holds": True,
        "kv260_ssh_reachable": True,
        "kv260_overlay_loadable": True,
        "speedup_claim_made": False,
        "random_seed": 3741,
        "reproducibility_checksum": "b" * 64,
        "duration_s": 2.1224,
    }


def _seed_root(
    root: Path,
    *,
    part_a: str = "untested",
    part_b: str = "not-run",
    include_3738: bool = False,
    flagged: set[int] | None = None,
) -> None:
    flagged = flagged or set()
    payloads = {
        3732: _artifact_3732(),
        3733: _artifact_3733(),
        3734: _artifact_3734(),
        3735: _artifact_3735(),
        3736: _artifact_3736(outcome=part_a),
        3737: _artifact_3737(blocked=part_a != "green"),
        3739: _artifact_3739(outcome=part_b),
        3740: _artifact_3740(),
        3741: _artifact_3741(),
    }
    if include_3738:
        payloads[3738] = _artifact_3738(
            accuracy_delta=-0.02 if part_b == "bounded" else 0.04,
            flops_matched_within_tolerance=part_b != "invalid",
        )
    for experiment_id, payload in payloads.items():
        if experiment_id in flagged:
            payload["flagged_adversarial"] = True
        _write_json(root / exp3742.DEFAULT_UPSTREAM_PATHS[experiment_id], payload)


def test_req_ebt_3742_spec_anchor_exists() -> None:
    """REQ-EBT-3742: OpenSpec declares the v342 capstone contract."""
    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-EBT-3742" in spec
    assert "SCENARIO-EBT-3742-UNTESTED" in spec
    assert "SCENARIO-EBT-3742-FLAGGED" in spec
    assert exp3742.OUTPUT_REL_PATH.as_posix() in spec


def test_scenario_ebt_3742_observed_untested_not_run_capstone(tmp_path: Path) -> None:
    """SCENARIO-EBT-3742-UNTESTED: part-a untested keeps part-b not-run."""
    _seed_root(tmp_path)

    artifact = exp3742.build_artifact(
        tmp_path,
        started_s=10.0,
        now_s=10.25,
        adversarial_verify_report={"flags": []},
    )

    exp3742.validate_artifact(artifact)
    assert artifact["honest_verdict"] == (
        "complete: capstone_v342_thesis_a_false_negative_corrected_part_a_"
        "untested_part_b_not_run_paper_ready_true_frozen_headline_unchanged"
    )
    assert artifact["inference_substrate"] == exp3742.INFERENCE_SUBSTRATE
    assert artifact["false_negative_corrected"] is True
    assert artifact["thesis_a_part_a_outcome"] == "untested"
    assert artifact["thesis_a_part_b_outcome"] == "not-run"
    assert artifact["ebt_beats_ar_at_matched_compute"] is False
    assert artifact["paper_ready_preserved"] is True
    assert artifact["frozen_headline_unchanged"] is True
    assert artifact["frozen_fover_auroc"] == pytest.approx(0.9131)
    assert artifact["p01_energy_selection_status"] == "honest-negative-bounded"
    assert "GENERATION" in artifact["p01_energy_selection_boundary"]
    assert "energy-SELECTION" in artifact["p01_energy_selection_boundary"]
    assert artifact["missing_upstream_artifacts"] == [
        {
            "experiment_id": 3738,
            "path": str(tmp_path / exp3742.DEFAULT_UPSTREAM_PATHS[3738]),
            "reason": "artifact_missing",
        }
    ]
    assert artifact["flagged_artifacts_excluded"] == []
    assert {item["experiment_id"] for item in artifact["cited_upstream_artifacts"]} == {
        3732,
        3733,
        3734,
        3735,
        3736,
        3737,
        3739,
        3740,
        3741,
    }
    assert 3738 not in artifact["headline_aggregation_experiment_ids"]
    assert "INFRA FALSE-NEGATIVE" in artifact["milestone_summary"]
    assert "training did not complete" in artifact["part_a_summary"]
    assert "no live model" in artifact["inference_substrate"]
    assert "live_llm_inference" not in json.dumps(artifact, sort_keys=True)
    assert artifact["reproducibility_checksum"] == exp3742.payload_checksum(artifact)


def test_green_light_and_tiny_win_are_stated_narrowly(tmp_path: Path) -> None:
    """REQ-EBT-3742: green-light and win language must not become scale claims."""
    _seed_root(tmp_path, part_a="green", part_b="win", include_3738=True)

    artifact = exp3742.build_artifact(
        tmp_path,
        started_s=20.0,
        now_s=20.5,
        adversarial_verify_report={"flags": []},
    )

    assert artifact["honest_verdict"] == (
        "complete: capstone_v342_thesis_a_false_negative_corrected_part_a_"
        "green_light_part_b_ebt_beats_ar_paper_ready_true_frozen_headline_unchanged"
    )
    assert artifact["thesis_a_part_a_outcome"] == "stable-green-light"
    assert artifact["thesis_a_part_b_outcome"] == "ebt-beats-ar"
    assert artifact["ebt_beats_ar_at_matched_compute"] is True
    assert "trains stably enough to compare" in artifact["part_a_summary"]
    assert "beats AR at equal compute at this tiny scale" in artifact["part_b_summary"]
    assert "works at scale" not in artifact["milestone_summary"]
    assert {item["experiment_id"] for item in artifact["cited_upstream_artifacts"]} == set(range(3732, 3742))


def test_genuine_part_a_bounded_is_a_real_finding(tmp_path: Path) -> None:
    """REQ-EBT-3742: a genuine part-a negative bounds the route honestly."""
    _seed_root(tmp_path, part_a="bounded", part_b="not-run", include_3738=False)

    artifact = exp3742.build_artifact(
        tmp_path,
        started_s=1.0,
        now_s=1.25,
        adversarial_verify_report={"flags": []},
    )

    assert artifact["thesis_a_part_a_outcome"] == "genuinely-bounded"
    assert artifact["thesis_a_part_b_outcome"] == "not-run"
    assert artifact["honest_verdict"].endswith(
        "part_a_bounded_part_b_not_run_paper_ready_true_frozen_headline_unchanged"
    )
    assert "real finding" in artifact["part_a_summary"]


def test_part_b_bounded_and_invalid_outcomes_are_carried(tmp_path: Path) -> None:
    """REQ-EBT-3742: part-b bounded and invalid are distinct non-win states."""
    _seed_root(tmp_path / "bounded", part_a="green", part_b="bounded", include_3738=True)
    bounded = exp3742.build_artifact(
        tmp_path / "bounded",
        started_s=1.0,
        now_s=1.25,
        adversarial_verify_report={"flags": []},
    )
    assert bounded["thesis_a_part_b_outcome"] == "bounded"
    assert bounded["ebt_beats_ar_at_matched_compute"] is False
    assert "real finding" in bounded["part_b_summary"]

    _seed_root(tmp_path / "invalid", part_a="green", part_b="invalid", include_3738=True)
    invalid = exp3742.build_artifact(
        tmp_path / "invalid",
        started_s=2.0,
        now_s=2.25,
        adversarial_verify_report={"flags": []},
    )
    assert invalid["thesis_a_part_b_outcome"] == "invalid"
    assert invalid["ebt_beats_ar_at_matched_compute"] is False
    assert "compute-confounded" in invalid["part_b_summary"]


def test_scenario_ebt_3742_flagged_sources_are_excluded(tmp_path: Path) -> None:
    """SCENARIO-EBT-3742-FLAGGED: flagged upstream artifacts are quarantined."""
    _seed_root(tmp_path, flagged={3734})

    artifact = exp3742.build_artifact(
        tmp_path,
        started_s=2.0,
        now_s=2.25,
        adversarial_verify_report={"flags": []},
    )

    assert artifact["flagged_artifacts_excluded"] == [
        {
            "experiment_id": 3734,
            "path": str(tmp_path / exp3742.DEFAULT_UPSTREAM_PATHS[3734]),
            "reason": "flagged_adversarial=true",
        }
    ]
    assert 3734 not in {item["experiment_id"] for item in artifact["cited_upstream_artifacts"]}
    assert 3734 not in artifact["headline_aggregation_experiment_ids"]
    assert artifact["harness_training_summary"]["exp3734"] == "excluded_flagged_adversarial"


def test_validate_artifact_reports_schema_and_hygiene_errors() -> None:
    """REQ-EBT-3742: malformed capstones fail closed before publication."""
    errors = exp3742.validate_artifact({})
    assert any(error.startswith("missing required artifact fields:") for error in errors)
    assert "honest_verdict must be a terminal Exp 3742 verdict" in errors
    assert "inference_substrate must declare the v342 aggregation-only substrate" in errors
    assert "false_negative_corrected must be boolean" in errors
    assert "thesis_a_part_a_outcome must be a supported v342 outcome" in errors
    assert "thesis_a_part_b_outcome must be a supported v342 outcome" in errors
    assert "ebt_beats_ar_at_matched_compute must be a bare bool" in errors
    assert "paper_ready_preserved must be true" in errors
    assert "frozen_headline_unchanged must be true" in errors
    assert "flagged_artifacts_excluded must be a list" in errors
    assert "cited_upstream_artifacts must cite unflagged upstream artifacts" in errors
    assert "random_seed must equal 3742" in errors
    assert "reproducibility_checksum must be a sha256 hex string" in errors
    assert "duration_s must be numeric with the aggregation plausibility floor" in errors
    assert "field_principles must cover all required artifact fields" in errors

    valid = {
        "honest_verdict": exp3742.terminal_verdict("untested", "not-run"),
        "inference_substrate": exp3742.INFERENCE_SUBSTRATE,
        "false_negative_corrected": True,
        "thesis_a_part_a_outcome": "untested",
        "thesis_a_part_b_outcome": "not-run",
        "ebt_beats_ar_at_matched_compute": False,
        "paper_ready_preserved": True,
        "frozen_headline_unchanged": True,
        "flagged_artifacts_excluded": [],
        "cited_upstream_artifacts": [
            {
                "experiment_id": 3732,
                "path": "results/experiment_3732.json",
                "fields_imported": ["honest_verdict"],
                "sha256": "c" * 64,
            }
        ],
        "field_principles": dict(exp3742.FIELD_PRINCIPLES),
        "random_seed": 3742,
        "reproducibility_checksum": "d" * 64,
        "duration_s": 0.1,
    }
    valid["reproducibility_checksum"] = exp3742.payload_checksum(valid)
    assert exp3742.validate_artifact(valid) == []

    live_marker = dict(valid)
    live_marker["copied_substrate"] = "live_llm_inference"
    live_marker["reproducibility_checksum"] = exp3742.payload_checksum(live_marker)
    assert "artifact must not copy live-model substrate markers" in exp3742.validate_artifact(live_marker)

    bad_checksum = dict(valid)
    bad_checksum["reproducibility_checksum"] = "e" * 64
    assert "reproducibility_checksum does not match artifact content" in exp3742.validate_artifact(
        bad_checksum
    )

    bad_citation = dict(valid)
    bad_citation["cited_upstream_artifacts"] = [123, {"experiment_id": 3732}]
    bad_citation["reproducibility_checksum"] = exp3742.payload_checksum(bad_citation)
    citation_errors = exp3742.validate_artifact(bad_citation)
    assert "each citation must be an object" in citation_errors
    assert "each citation must include fields_imported" in citation_errors
    assert "each citation must include a sha256 hex string" in citation_errors

    impossible_win = dict(valid)
    impossible_win["thesis_a_part_b_outcome"] = "not-run"
    impossible_win["ebt_beats_ar_at_matched_compute"] = True
    impossible_win["reproducibility_checksum"] = exp3742.payload_checksum(impossible_win)
    assert "only a part-b win may set ebt_beats_ar_at_matched_compute=true" in exp3742.validate_artifact(
        impossible_win
    )


def test_main_writes_artifact_and_loader_rejects_arrays(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """REQ-EBT-3742: CLI writes stable JSON and invalid source JSON fails."""
    _seed_root(tmp_path)

    assert exp3742.main(["--root", str(tmp_path)]) == 0
    written = json.loads((tmp_path / exp3742.OUTPUT_REL_PATH).read_text(encoding="utf-8"))

    assert written["honest_verdict"] == exp3742.terminal_verdict("untested", "not-run")
    assert written["adversarial_verify_clean"] is True
    assert written["adversarial_verify_report"]["critical_flag_count"] == 0
    assert written["honest_verdict"] in capsys.readouterr().out

    list_path = tmp_path / "array.json"
    list_path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a JSON object"):
        exp3742.read_json_object(list_path)
