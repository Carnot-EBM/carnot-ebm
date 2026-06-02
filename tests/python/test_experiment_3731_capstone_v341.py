"""Tests for Exp 3731 Thesis-A EBT bring-up capstone.

Spec refs: REQ-EBT-3731, SCENARIO-EBT-3731, SCENARIO-EBT-3731-FLAGGED.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import experiment_3731_capstone_v341 as exp3731


def _write(path: Path, data: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _artifact_paths(
    tmp_path: Path,
    *,
    green_light_342: bool = False,
    flagged: set[int] | None = None,
) -> dict[int, Path]:
    flagged = flagged or set()
    results = tmp_path / "results"
    payloads: dict[int, dict[str, object]] = {
        3724: {
            "honest_verdict": (
                "complete: archived_v340_convergence_hardened_thesis_a_energy_generator_"
                "seeded_v341_active_paper_ready_true_frozen_headline_unchanged"
            ),
            "paper_ready_preserved": True,
            "paper_ready_evidence": {
                "g1": True,
                "g2": True,
                "g3": True,
                "g4": True,
                "paper_ready": True,
                "frozen_headline_unchanged": True,
            },
            "g_gates_preserved": {"g1": True, "g2": True, "g3": True, "g4": True},
            "frozen_headline_auroc_preserved": 0.9131,
            "p01_status_preserved": "honest-negative-bounded",
            "thesis_a_evidence": {"mechanism": "energy_as_generator_not_selector"},
            "random_seed": 3724,
            "reproducibility_checksum": "1" * 64,
            "duration_s": 0.0001,
        },
        3725: {
            "honest_verdict": "complete: ebt_vendored_energy_path_audited",
            "importable": True,
            "license_confirmed": True,
            "upstream_commit_sha": "19420cbeae655bbf11930219a675ade6897019e8",
            "smoke_energy_value": 0.5541654229164124,
            "energy_path_audit": "audited",
            "random_seed": 42,
            "reproducibility_checksum": "2" * 64,
            "duration_s": 15,
        },
        3726: {
            "honest_verdict": "complete: tiny_ebt_38M_fits_3090",
            "ebt_param_count": 37_954_560,
            "peak_vram_mb": 1283,
            "n_train": 2048,
            "loss_finite": True,
            "loss_decreased": True,
            "first_step_losses": [-0.1, -1.0],
            "random_seed": 3726,
            "reproducibility_checksum": "3" * 64,
            "duration_s": 7.84,
        },
        3727: {
            "honest_verdict": "complete: matched_compute_eval_harness_built",
            "unit_tests_added": "tests/python/test_matched_compute_eval_harness.py",
            "unit_tests_passed": "5_of_5_pass",
            "flop_model_description": "parameter_count * sequence_tokens * forward_passes",
            "matched_compute_report": {
                "ebt_total_flops": 10_000,
                "ar_total_flops": 10_000,
                "budget_match": {"ar_best_of_m": 5, "within_tolerance": True},
            },
            "random_seed": 20260602,
            "reproducibility_checksum": "4" * 64,
            "duration_s": 1.9,
        },
        3728: {
            "honest_verdict": "blocked_ebt" if not green_light_342 else "complete: stable",
            "cumulative_steps_trained": 0 if not green_light_342 else 100,
            "ebt_loss_curve": [] if not green_light_342 else [5.0, 3.0],
            "ar_loss_curve": [],
            "ebt_converged": green_light_342,
            "nan_or_divergence_events": False,
            "stabilizers_applied": "none" if not green_light_342 else "grad_clip",
            "peak_vram_mb": 0 if not green_light_342 else 4500,
            "random_seed": 3728,
            "reproducibility_checksum": "5" * 64,
            "duration_s": 65.5,
        },
        3729: {
            "honest_verdict": (
                "complete: kill_gate_part_a_PASS_ebt_trained_stably_green_light_342_matched_compute_comparison"
                if green_light_342
                else "complete: kill_gate_part_a_FAIL_energy_as_generator_bounded_at_small_scale_honest_negative_stop"
            ),
            "ebt_trained_stably": green_light_342,
            "green_light_342": green_light_342,
            "kill_gate_conclusion": "green-light .342 only" if green_light_342 else "bounded at small scale",
            "stability_diagnostics": {"ebt_trained_stably": green_light_342},
            "random_seed": 3729,
            "reproducibility_checksum": "6" * 64,
            "duration_s": 0.1,
        },
        3730: {
            "honest_verdict": "complete: kv260_terminal_state_holds",
            "terminal_state_holds": True,
            "kv260_ssh_reachable": True,
            "kv260_overlay_loadable": True,
            "speedup_claim_made": False,
            "random_seed": 3730,
            "reproducibility_checksum": "7" * 64,
            "duration_s": 6.4,
        },
    }
    paths: dict[int, Path] = {}
    for experiment_id, payload in payloads.items():
        if experiment_id in flagged:
            payload["flagged_adversarial"] = True
        paths[experiment_id] = _write(results / f"experiment_{experiment_id}.json", payload)
    return paths


def test_openspec_has_3731_requirement() -> None:
    """REQ-EBT-3731: the capstone is OpenSpec anchored before code."""
    spec = Path("openspec/capabilities/ebt-nrgpt/spec.md").read_text(encoding="utf-8")

    assert "REQ-EBT-3731" in spec
    assert "SCENARIO-EBT-3731" in spec
    assert "SCENARIO-EBT-3731-FLAGGED" in spec


def test_bounded_kill_gate_writes_honest_capstone(tmp_path: Path) -> None:
    """SCENARIO-EBT-3731: a bounded kill-gate cannot be promoted to thesis success."""
    artifact = exp3731.build_artifact(_artifact_paths(tmp_path), duration_s=0.25)

    assert artifact["honest_verdict"] == (
        "complete: capstone_v341_thesis_a_ebt_bringup_kill_gate_part_a_"
        "bounded_paper_ready_true_frozen_headline_unchanged"
    )
    assert artifact["inference_substrate"] == exp3731.INFERENCE_SUBSTRATE
    assert artifact["thesis_a_bringup_outcome"] == "bounded_at_small_scale_do_not_auto_propose_342"
    assert artifact["kill_gate_part_a_passed"] is False
    assert artifact["green_light_342"] is False
    assert artifact["paper_ready_preserved"] is True
    assert artifact["frozen_headline_unchanged"] is True
    assert artifact["frozen_fover_auroc"] == 0.9131
    assert artifact["p01_energy_selection_status"] == "honest-negative-bounded"
    assert artifact["flagged_artifacts_excluded"] == []
    assert {item["experiment_id"] for item in artifact["cited_upstream_artifacts"]} == set(range(3724, 3731))
    assert set(exp3731.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert "live_llm_inference" not in json.dumps(artifact)
    assert exp3731.validate_artifact(artifact) == []

    fallback_paths = _artifact_paths(tmp_path / "fallback")
    exp3724 = json.loads(fallback_paths[3724].read_text(encoding="utf-8"))
    exp3724.pop("g_gates_preserved")
    _write(fallback_paths[3724], exp3724)
    fallback_artifact = exp3731.build_artifact(fallback_paths, duration_s=0.25)
    assert fallback_artifact["g_gates_preserved"] == {"g1": True, "g2": True, "g3": True, "g4": True}


def test_green_light_is_narrow_not_energy_generator_success(tmp_path: Path) -> None:
    """REQ-EBT-3731: a pass only sanctions .342 matched-compute comparison."""
    artifact = exp3731.build_artifact(
        _artifact_paths(tmp_path, green_light_342=True),
        duration_s=0.25,
    )

    assert artifact["honest_verdict"].endswith("part_a_pass_paper_ready_true_frozen_headline_unchanged")
    assert artifact["kill_gate_part_a_passed"] is True
    assert artifact["green_light_342"] is True
    assert artifact["thesis_a_bringup_outcome"] == (
        "green_light_342_stable_enough_for_matched_compute_comparison_not_energy_as_generator_success"
    )
    assert "stable enough to run the matched-compute comparison" in artifact["thesis_a_bringup_summary"]
    assert "energy-as-generator works" not in artifact["thesis_a_bringup_summary"]


def test_flagged_upstream_is_excluded_from_headline_citations(tmp_path: Path) -> None:
    """SCENARIO-EBT-3731-FLAGGED: flagged sources are named and excluded."""
    paths = _artifact_paths(tmp_path, flagged={3726})

    artifact = exp3731.build_artifact(paths, duration_s=0.25)

    assert artifact["flagged_artifacts_excluded"] == [
        {
            "experiment_id": 3726,
            "path": str(paths[3726]),
            "reason": "flagged_adversarial=true",
        }
    ]
    assert 3726 not in {item["experiment_id"] for item in artifact["cited_upstream_artifacts"]}
    assert 3726 not in artifact["headline_aggregation_experiment_ids"]
    assert "tiny_ebt_3090_smoke" not in artifact["bringup_evidence"]


def test_validate_artifact_reports_schema_errors() -> None:
    """REQ-EBT-3731: malformed capstones fail schema validation explicitly."""
    missing_errors = exp3731.validate_artifact({})
    assert any(error.startswith("missing required fields:") for error in missing_errors)
    assert "green_light_342 must be boolean" in missing_errors
    assert "cited_upstream_artifacts must cite unflagged upstream artifacts" in missing_errors

    errors = exp3731.validate_artifact(
        {
            "honest_verdict": "complete: made_up",
            "inference_substrate": "hardware_smoke",
            "thesis_a_bringup_outcome": "",
            "kill_gate_part_a_passed": "yes",
            "green_light_342": True,
            "paper_ready_preserved": False,
            "frozen_headline_unchanged": False,
            "flagged_artifacts_excluded": "none",
            "cited_upstream_artifacts": [123, {"experiment_id": 3724, "fields_imported": [], "sha256": "bad"}],
            "field_principles": {},
            "random_seed": 0,
            "reproducibility_checksum": "bad",
            "duration_s": 0,
        }
    )

    assert "honest_verdict must be a terminal Exp 3731 verdict" in errors
    assert "inference_substrate must declare aggregation-only capstone provenance" in errors
    assert "thesis_a_bringup_outcome must be present" in errors
    assert "kill_gate_part_a_passed must be boolean" in errors
    assert "paper_ready_preserved must be true" in errors
    assert "frozen_headline_unchanged must be true" in errors
    assert "flagged_artifacts_excluded must be a list" in errors
    assert "each citation must be an object" in errors
    assert "each citation must include fields_imported" in errors
    assert "each citation must include a sha256 hex string" in errors
    assert "random_seed must equal 3731" in errors
    assert "reproducibility_checksum must be a sha256 hex string" in errors
    assert "duration_s must be positive" in errors
    assert "field_principles must cover all required artifact fields" in errors

    live_marker = {
        "honest_verdict": exp3731.BOUNDED_VERDICT,
        "inference_substrate": exp3731.INFERENCE_SUBSTRATE,
        "thesis_a_bringup_outcome": exp3731.BOUNDED_OUTCOME,
        "kill_gate_part_a_passed": False,
        "green_light_342": False,
        "paper_ready_preserved": True,
        "frozen_headline_unchanged": True,
        "flagged_artifacts_excluded": [],
        "cited_upstream_artifacts": [
            {"experiment_id": 3724, "fields_imported": ["honest_verdict"], "sha256": "a" * 64}
        ],
        "field_principles": dict(exp3731.FIELD_PRINCIPLES),
        "random_seed": 3731,
        "reproducibility_checksum": "b" * 64,
        "duration_s": 1.0,
        "copied_marker": "live_llm_inference",
    }
    assert "artifact must not copy live-model substrate markers" in exp3731.validate_artifact(live_marker)


def test_loader_builder_and_main_failure_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-EBT-3731: CLI writes valid JSON and invalid inputs fail closed."""
    list_path = tmp_path / "list.json"
    list_path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a JSON object"):
        exp3731.load_json(list_path)
    assert exp3731._get_nested({}, "missing.path") is None

    paths = _artifact_paths(tmp_path)
    output_path = tmp_path / "results" / "experiment_3731_capstone_v341.json"
    args = []
    for experiment_id in exp3731.UPSTREAM_IDS:
        args.extend([f"--exp{experiment_id}", str(paths[experiment_id])])
    args.extend(["--output", str(output_path)])

    assert exp3731.main(args) == 0
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["honest_verdict"] == exp3731.BOUNDED_VERDICT
    assert exp3731.BOUNDED_VERDICT in capsys.readouterr().out

    monkeypatch.setattr(exp3731, "validate_artifact", lambda artifact: ["forced validation error"])
    with pytest.raises(ValueError, match="forced validation error"):
        exp3731.build_artifact(paths, duration_s=0.25)
    with pytest.raises(ValueError, match="forced validation error"):
        exp3731.main(args)
