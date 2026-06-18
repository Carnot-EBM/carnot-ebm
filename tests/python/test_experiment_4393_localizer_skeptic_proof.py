"""Tests for Exp 4393 localizer skeptic-proof diagnostic.

Spec refs: REQ-VERIFY-4393, SCENARIO-VERIFY-4393.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4392_verifiable_process_data_localizer as exp4392
from carnot import experiment_4393_localizer_skeptic_proof as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _fover_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trace_idx in range(8):
        rows.extend(
            [
                {
                    "trace_id": f"trace-{trace_idx}",
                    "step_index": 0,
                    "partial_cot": "First claim is already wrong.",
                    "step_label": "wrong",
                    "cascade_score": 0.97,
                },
                {
                    "trace_id": f"trace-{trace_idx}",
                    "step_index": 1,
                    "partial_cot": "The suffix inherits the bad state.",
                    "step_label": "wrong",
                    "cascade_score": 0.62,
                },
            ]
        )
    return rows


def _a1_artifact(*, win: bool = True, n_synthetic: int = 32) -> dict[str, Any]:
    synthetic = exp4392.synthesize_verifiable_first_error_corpus(
        n_traces=n_synthetic,
        seed=4392,
    )
    localizer = exp4392.train_contrastive_localizer(synthetic)
    return {
        "experiment": "experiment_4392_verifiable_process_data_localizer",
        "honest_verdict": "success: fixture" if win else "complete: fixture_clean_null",
        "localizer_beats_ensemble_baseline": win,
        "synthesis_verification": exp4392.synthesis_verification_summary(synthetic),
        "model_specs": {
            "localizer": localizer.as_dict(),
            "synthesis_config": {"n": n_synthetic},
            "trm_training": "stood_down_not_invoked",
        },
        "random_seed": 4392,
        "reproducibility_checksum": "sha256:fixture",
    }


def test_req_verify_4393_spec_declares_skeptic_proof_contract() -> None:
    """REQ-VERIFY-4393: OpenSpec declares the three skeptic controls."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-4393",
        "SCENARIO-VERIFY-4393",
        "experiment_4393_localizer_skeptic_proof.json",
        "localizer_win_is_genuine",
        "beats_position_only_baseline",
        "template_ablation_drop",
        "held_out_real_localization_delta_ci95",
        "blocked_no_win_to_validate",
    ):
        assert marker in spec


def test_req_verify_4393_position_only_baseline_uses_only_empirical_step_distribution() -> None:
    """REQ-VERIFY-4393: position-only control can tie a step-0 localizer."""

    traces = exp4392.load_fover_real_traces_from_rows(_fover_rows())
    baseline = mod.PositionOnlyBaseline.fit(traces)
    successes = mod.successes_for_predictor(
        traces,
        baseline.predict_first_error_index,
    )

    assert baseline.position_counts == {0: 8}
    assert successes == [1] * 8
    assert mod.f1_from_successes(successes) == pytest.approx(1.0)


def test_req_verify_4393_defensive_controls_cover_empty_and_schema_paths(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-4393: blocked/control helpers stay terminal and explicit."""

    empty_trace = exp4392.ProcessTrace(
        trace_id="empty",
        source_domain="fixture",
        steps=(),
        first_error_index=None,
    )
    short_trace = exp4392.ProcessTrace(
        trace_id="short",
        source_domain="fixture",
        steps=(
            exp4392.ProcessStep(0, "a", False, {}),
            exp4392.ProcessStep(1, "b", True, {}),
        ),
        first_error_index=1,
    )
    clean_trace = exp4392.ProcessTrace(
        trace_id="clean",
        source_domain="fixture",
        steps=(exp4392.ProcessStep(0, "a", False, {}),),
        first_error_index=None,
    )

    assert mod.PositionOnlyBaseline({}).predict_first_error_index(short_trace) is None
    assert mod.PositionOnlyBaseline({3: 2}).predict_first_error_index(short_trace) == 1
    assert mod.successes_for_predictor([clean_trace, short_trace], lambda _trace: 1) == [1]
    assert mod.f1_from_successes([]) == 0.0
    assert mod._paired_delta_ci95([], [], seed=1, resamples=10) == [None, None]
    assert mod.scramble_synthetic_first_error_structure([empty_trace], seed=1) == [empty_trace]
    assert mod._synthetic_n_from_a1({"model_specs": {"synthesis_config": {"n": 7}}}) == 7
    assert mod._synthetic_n_from_a1({}) == exp4392.MIN_SYNTHETIC_TRACES
    assert mod._localizer_from_a1({"model_specs": {"localizer": {"weights": []}}}) is None

    template_gap = mod._missing_verifier_gap(
        {"beats_position_only_baseline": True},
        {"degrades_under_template_ablation": False},
    )
    heldout_gap = mod._missing_verifier_gap(
        {"beats_position_only_baseline": True},
        {"degrades_under_template_ablation": True},
    )
    assert template_gap["confounder"] == "template_ablation_no_material_drop"
    assert heldout_gap["confounder"] == "heldout_real_advantage_not_stable"

    gap_path = tmp_path / "ops" / "verifier_gaps.md"
    mod.append_missing_verifier_gaps(gap_path, [])
    assert not gap_path.exists()
    mod.append_missing_verifier_gaps(gap_path, [template_gap])
    first = gap_path.read_text(encoding="utf-8")
    mod.append_missing_verifier_gaps(gap_path, [template_gap])
    assert gap_path.read_text(encoding="utf-8") == first

    errors = mod.artifact_schema_errors(
        {
            "honest_verdict": "bad",
            "localizer_win_is_genuine": "false",
            "beats_position_only_baseline": "false",
            "template_ablation_drop": 0,
            "verifier_is_oracle": True,
        }
    )
    assert "missing preconditions_checked" in errors
    assert "honest_verdict must be terminal-prefixed" in errors
    assert "localizer_win_is_genuine must be bare bool" in errors
    assert "beats_position_only_baseline must be bare bool" in errors
    assert "template_ablation_drop must be bare float" in errors
    assert "verifier_is_oracle must be false" in errors


def test_req_verify_4393_template_ablation_reports_no_drop_when_scramble_still_transfers() -> None:
    """REQ-VERIFY-4393: no template-ablation drop quarantines the headline."""

    synthetic = exp4392.synthesize_verifiable_first_error_corpus(n_traces=48, seed=4392)
    a1 = exp4392.train_contrastive_localizer(synthetic)
    ablated = mod.scramble_synthetic_first_error_structure(synthetic, seed=4393)
    ablated_model = exp4392.train_contrastive_localizer(ablated)
    real = exp4392.load_fover_real_traces_from_rows(_fover_rows())
    report = mod.template_ablation_report(
        real,
        a1,
        ablated_model,
        seed=4393,
        bootstrap_resamples=120,
    )

    assert any(
        trace.first_error_index != synthetic[idx].first_error_index
        for idx, trace in enumerate(ablated)
    )
    assert report["a1_f1"] == pytest.approx(1.0)
    assert report["template_ablated_f1"] == pytest.approx(1.0)
    assert report["drop"] == pytest.approx(0.0)
    assert report["degrades_under_template_ablation"] is False


def test_scenario_verify_4393_run_experiment_quarantines_position_confounded_a1(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-4393: complete diagnostic writes terminal quarantine."""

    a1_path = tmp_path / "results" / "experiment_4392.json"
    fover_path = tmp_path / "data" / "steps.jsonl"
    artifact_path = tmp_path / "results" / "experiment_4393.json"
    gaps_path = tmp_path / "ops" / "verifier_gaps.md"
    _write_json(a1_path, _a1_artifact(win=True, n_synthetic=48))
    _write_jsonl(fover_path, _fover_rows())

    artifact = mod.run_experiment(
        mod.ExperimentConfig(
            repo_root=tmp_path,
            exp4392_artifact_path=a1_path,
            fover_step_corpus_path=fover_path,
            artifact_path=artifact_path,
            verifier_gaps_path=gaps_path,
            random_seed=4393,
            bootstrap_resamples=120,
            started_at=10.0,
            clock=lambda: 12.5,
        ),
        adversarial_verify_runner=lambda _path: {"returncode": 0, "stdout_tail": "clean"},
        write=True,
    )

    assert artifact_path.is_file()
    assert artifact["honest_verdict"] == "complete: a1_win_quarantined_as_artifact_confounded"
    assert artifact["localizer_win_is_genuine"] is False
    assert artifact["beats_position_only_baseline"] is False
    assert artifact["template_ablation_drop"] == pytest.approx(0.0)
    assert artifact["held_out_real_localization_delta_ci95"][0] > 0.0
    assert artifact["verifier_is_oracle"] is False
    assert artifact["a1_win_quarantined"] is True
    assert artifact["adversarial_verify"]["returncode"] == 0
    assert "GAP-4393-LOCALIZER-POSITION-OR-TEMPLATE-CONFOUND" in gaps_path.read_text(
        encoding="utf-8"
    )
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_verify_4393_blocks_without_a1_win(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4393: false A1 gate stops before re-testing."""

    a1_path = tmp_path / "results" / "experiment_4392.json"
    fover_path = tmp_path / "data" / "steps.jsonl"
    artifact_path = tmp_path / "results" / "experiment_4393.json"
    _write_json(a1_path, _a1_artifact(win=False))
    _write_jsonl(fover_path, _fover_rows())

    artifact = mod.run_experiment(
        mod.ExperimentConfig(
            repo_root=tmp_path,
            exp4392_artifact_path=a1_path,
            fover_step_corpus_path=fover_path,
            artifact_path=artifact_path,
            random_seed=4393,
            started_at=1.0,
            clock=lambda: 1.25,
        ),
        adversarial_verify_runner=lambda _path: {"returncode": 0},
        write=True,
    )

    assert artifact["honest_verdict"] == "blocked_no_win_to_validate"
    assert artifact["localizer_win_is_genuine"] is False
    assert artifact["beats_position_only_baseline"] is False
    assert artifact["template_ablation_drop"] == 0.0
    assert artifact["adversarial_verify"]["skipped"] == "blocked"
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_verify_4393_blocks_missing_a1_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4393: missing A1 artifact blocks as a resource failure."""

    artifact_path = tmp_path / "results" / "experiment_4393.json"

    artifact = mod.run_experiment(
        mod.ExperimentConfig(
            repo_root=tmp_path,
            exp4392_artifact_path=tmp_path / "missing_4392.json",
            fover_step_corpus_path=tmp_path / "missing_steps.jsonl",
            artifact_path=artifact_path,
            started_at=1.0,
            clock=lambda: 1.5,
        ),
        write=True,
    )

    assert artifact_path.is_file()
    assert artifact["honest_verdict"] == "blocked_a1_artifact"
    assert artifact["preconditions_checked"][0]["resource"] == "exp4392_a1_artifact"
    assert artifact["reproducibility_checksum"].startswith("sha256:")
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_verify_4393_blocks_missing_localizer_and_real_split(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-4393: malformed A1 or missing REAL split blocks."""

    fover_path = tmp_path / "data" / "steps.jsonl"
    _write_jsonl(fover_path, _fover_rows())

    no_localizer_path = tmp_path / "results" / "no_localizer_4392.json"
    no_localizer = _a1_artifact(win=True)
    no_localizer["model_specs"].pop("localizer")
    _write_json(no_localizer_path, no_localizer)
    blocked_localizer = mod.run_experiment(
        mod.ExperimentConfig(
            repo_root=tmp_path,
            exp4392_artifact_path=no_localizer_path,
            fover_step_corpus_path=fover_path,
            artifact_path=tmp_path / "results" / "blocked_localizer.json",
            started_at=2.0,
            clock=lambda: 2.5,
        ),
        write=True,
    )
    assert blocked_localizer["honest_verdict"] == "blocked_a1_localizer"

    no_real_path = tmp_path / "results" / "valid_4392.json"
    _write_json(no_real_path, _a1_artifact(win=True))
    blocked_real = mod.run_experiment(
        mod.ExperimentConfig(
            repo_root=tmp_path,
            exp4392_artifact_path=no_real_path,
            fover_step_corpus_path=tmp_path / "missing_steps.jsonl",
            artifact_path=tmp_path / "results" / "blocked_real.json",
            started_at=3.0,
            clock=lambda: 3.5,
        ),
        write=True,
    )
    assert blocked_real["honest_verdict"] == "blocked_held_out_real_split"


def test_scenario_verify_4393_write_false_and_adversarial_failure_paths(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-4393: write controls verifier execution and quarantine."""

    a1_path = tmp_path / "results" / "experiment_4392.json"
    fover_path = tmp_path / "data" / "steps.jsonl"
    _write_json(a1_path, _a1_artifact(win=True, n_synthetic=48))
    _write_jsonl(fover_path, _fover_rows())

    dry_run = mod.run_experiment(
        mod.ExperimentConfig(
            repo_root=tmp_path,
            exp4392_artifact_path=a1_path,
            fover_step_corpus_path=fover_path,
            artifact_path=tmp_path / "results" / "dry_run.json",
            random_seed=4393,
            bootstrap_resamples=120,
            started_at=4.0,
            clock=lambda: 4.5,
        ),
        write=False,
    )
    assert dry_run["adversarial_verify"] == {"returncode": None, "skipped": True}

    flagged = mod.run_experiment(
        mod.ExperimentConfig(
            repo_root=tmp_path,
            exp4392_artifact_path=a1_path,
            fover_step_corpus_path=fover_path,
            artifact_path=tmp_path / "results" / "flagged.json",
            verifier_gaps_path=tmp_path / "ops" / "flagged_gaps.md",
            random_seed=4393,
            bootstrap_resamples=120,
            started_at=5.0,
            clock=lambda: 5.5,
        ),
        adversarial_verify_runner=lambda _path: {"returncode": 1, "stdout_tail": "flag"},
        write=True,
    )
    assert flagged["flagged_adversarial"] is True
    assert flagged["a1_win_quarantined"] is True
