"""Tests for Exp 4416 hidden-state first-error falsification audit.

Spec refs: REQ-VERIFY-4416, SCENARIO-VERIFY-4416.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4416_hidden_state_localizer_falsification_audit as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _fover_rows(n_bad: int = 8, *, family: str = "fixture") -> list[dict[str, Any]]:
    rows = [
        {
            "question_id": f"{family}_correct",
            "source": family,
            "label": "correct",
            "confidence": 0.95,
            "step_text": f"{family} verified reference step",
        }
    ]
    for idx in range(n_bad):
        rows.append(
            {
                "question_id": f"{family}_bad_{idx}",
                "source": family,
                "label": "incorrect",
                "confidence": 0.15,
                "step_text": f"{family} failed step {idx}",
            }
        )
    return rows


class FakeExtractor:
    def __init__(self, *, available: bool = True) -> None:
        self.available = available

    def check(self) -> mod.HiddenStatePathStatus:
        return mod.HiddenStatePathStatus(
            available=self.available,
            detail="fake extractor ready" if self.available else "fake extractor unavailable",
            model_specs={
                "model_id": "fixture-hidden-model",
                "hidden_state_extraction_path": "injected_test_extractor",
                "gguf_tokenizer_rule": "not_gguf",
            },
        )

    def capture(self, traces: list[mod.HiddenStateTrace]) -> list[list[list[float]]]:
        captured: list[list[list[float]]] = []
        for trace in traces:
            per_trace: list[list[float]] = []
            for step_index, _text in enumerate(trace.steps):
                if step_index == trace.first_error_index:
                    per_trace.append([2.0, 0.0])
                else:
                    per_trace.append([0.1, 0.0])
            captured.append(per_trace)
        return captured


def _fixture_config(
    tmp_path: Path,
    *,
    artifact_name: str = "experiment_4416.json",
    n_bad: int = 8,
    min_capture_error_traces: int = 8,
    min_eval_traces: int = 3,
) -> mod.ExperimentConfig:
    fover_path = tmp_path / "data" / f"{artifact_name}.fover.jsonl"
    step_path = tmp_path / "data" / f"{artifact_name}.steps.jsonl"
    exp2850_path = tmp_path / "results" / f"{artifact_name}.2850.json"
    exp4403_path = tmp_path / "results" / f"{artifact_name}.4403.json"
    _write_jsonl(fover_path, _fover_rows(n_bad=n_bad))
    _write_jsonl(step_path, [{"question_id": "s", "step_label": "wrong", "partial_cot": "bad"}])
    _write_json(exp2850_path, {"n_examples": 1000, "honest_verdict": "complete: fixture"})
    _write_json(
        exp4403_path,
        {
            "localization_f1_by_domain": {"FoVer": {"real_intervention_localizer": 1.0}},
            "position_only_baseline_f1": 1.0,
            "honest_verdict": "complete: clean_powered_null_position_only_not_beaten",
        },
    )
    return mod.ExperimentConfig(
        repo_root=tmp_path,
        fover_row_corpus_path=fover_path,
        fover_step_corpus_path=step_path,
        exp2850_artifact_path=exp2850_path,
        exp4403_artifact_path=exp4403_path,
        verifier_gaps_path=tmp_path / "ops" / "verifier_gaps.md",
        artifact_path=tmp_path / "results" / artifact_name,
        min_capture_error_traces=min_capture_error_traces,
        min_eval_traces=min_eval_traces,
        heldout_fraction=0.75,
        bootstrap_resamples=40,
        started_at=10.0,
        clock=lambda: 12.0,
    )


def test_req_verify_4416_spec_declares_hidden_state_audit_contract() -> None:
    """REQ-VERIFY-4416: OpenSpec declares the hidden-state falsification audit."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-4416",
        "SCENARIO-VERIFY-4416",
        "experiment_4416_hidden_state_localizer_falsification_audit.json",
        "hidden_state_localizer_has_nonposition_signal",
        "localization_f1_comparison",
        "position_only_baseline_f1",
        "blocked_no_hidden_state_extraction_path",
        "GGUF tokenizer rule",
    ):
        assert marker in spec


def test_req_verify_4416_builds_powered_real_fover_error_traces() -> None:
    """REQ-VERIFY-4416: real FoVer failed rows become first-error traces."""

    traces = mod.build_real_fover_error_traces_from_rows(_fover_rows(n_bad=5))

    assert len(traces) == 5
    assert all(trace.first_error_index == 0 for trace in traces)
    assert all(trace.position_bin == "0" for trace in traces)
    assert mod.position_bin_counts(traces) == {"0": 5}


def test_req_verify_4416_hidden_margin_probe_requires_nonposition_delta() -> None:
    """REQ-VERIFY-4416: hidden signal is true only when CI beats position-only."""

    train = [
        mod.HiddenStateTrace("train-a", ("bad", "downstream"), 0, "0", "a"),
        mod.HiddenStateTrace("train-b", ("bad", "downstream"), 0, "0", "b"),
        mod.HiddenStateTrace("train-c", ("prefix", "bad"), 1, "1", "c"),
        mod.HiddenStateTrace("train-d", ("prefix", "bad"), 1, "1", "d"),
    ]
    heldout = [
        mod.HiddenStateTrace("held-a", ("prefix", "bad"), 1, "1", "c"),
        mod.HiddenStateTrace("held-b", ("prefix", "bad"), 1, "1", "d"),
    ]
    extractor = FakeExtractor()
    train_features = mod.transport_margin_features(train, extractor.capture(train))
    heldout_features = mod.transport_margin_features(heldout, extractor.capture(heldout))
    probe = mod.HiddenStateMarginProbe.fit(train_features)
    baseline = mod.PositionOnlyBaseline.fit(train)

    report = mod.evaluate_hidden_state_probe(
        heldout,
        heldout_features,
        probe,
        baseline,
        text_localizer_f1=1.0,
        seed=4416,
        bootstrap_resamples=30,
    )

    assert report["hidden_state_probe_f1"] == pytest.approx(1.0)
    assert report["position_only_baseline_f1"] == pytest.approx(0.0)
    assert report["delta_vs_position_only"] == pytest.approx(1.0)
    assert report["delta_ci95"] == [1.0, 1.0]
    assert mod.has_nonposition_signal(report) is True


def test_scenario_verify_4416_run_experiment_writes_clean_null(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4416: position-saturated hidden states close the localizer."""

    config = _fixture_config(tmp_path)
    artifact = mod.run_experiment(
        config,
        hidden_state_extractor=FakeExtractor(),
        adversarial_verify_runner=lambda _path: {"returncode": 0, "stdout_tail": "clean"},
        write=True,
    )

    assert config.artifact_path.is_file()
    assert artifact["honest_verdict"] == "complete: clean_powered_null_position_only_not_beaten"
    assert artifact["hidden_state_localizer_has_nonposition_signal"] is False
    assert artifact["localization_f1_comparison"]["position_only_baseline_f1"] == pytest.approx(1.0)
    assert artifact["localization_f1_comparison"]["hidden_state_probe_f1"] == pytest.approx(1.0)
    assert artifact["localization_f1_comparison"]["delta_vs_position_only"] == pytest.approx(0.0)
    assert artifact["position_only_baseline_f1"] == pytest.approx(1.0)
    assert artifact["n_traces"] == 6
    assert artifact["verifier_is_oracle"] is False
    assert artifact["hidden_state_capture_receipts"]["n_per_position_bin"] == {"0": 8}
    assert artifact["adversarial_verify"]["returncode"] == 0
    assert artifact["missing_verifier_gaps"][0]["gap_id"] == (
        "GAP-FOVER-HIDDEN-STATE-LOCALIZATION-POSITION-SATURATED"
    )
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_verify_4416_blocks_before_capture_on_missing_resources(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4416: blocked corpus and hidden-path verdicts are terminal."""

    missing = mod.run_experiment(
        _fixture_config(tmp_path, artifact_name="missing.json", n_bad=2, min_capture_error_traces=4),
        hidden_state_extractor=FakeExtractor(),
        write=True,
    )
    assert missing["honest_verdict"] == "blocked_cached_corpus_unavailable"
    assert missing["adversarial_verify"]["skipped"] == "blocked"
    assert missing["hidden_state_capture_receipts"]["n_captured_traces"] == 0
    assert mod.artifact_schema_errors(missing) == []

    hidden = mod.run_experiment(
        _fixture_config(tmp_path, artifact_name="hidden.json"),
        hidden_state_extractor=FakeExtractor(available=False),
        write=True,
    )
    assert hidden["honest_verdict"] == "blocked_no_hidden_state_extraction_path"
    assert hidden["preconditions_checked"][-2]["resource"] == "hidden_state_extraction_path"
    assert hidden["adversarial_verify"]["skipped"] == "blocked"
    assert mod.artifact_schema_errors(hidden) == []


def test_req_verify_4416_schema_and_repro_helpers_are_strict(tmp_path: Path) -> None:
    """REQ-VERIFY-4416: schema helpers reject non-bare gate fields."""

    assert mod.round_float(None) is None
    assert mod.round_float(math.nan) is None
    assert mod._paired_delta_ci95([], [], seed=1, resamples=10) == [None, None]
    assert mod._vector_mean([]) == []
    assert mod.text_localizer_f1_from_exp4403({}) is None
    assert mod.text_localizer_f1_from_exp4403({"position_only_baseline_f1": 1.0}) is None
    assert (
        mod.text_localizer_f1_from_exp4403(
            {"localization_f1_by_domain": {"FoVer": {"real_intervention_localizer": 0.75}}}
        )
        == pytest.approx(0.75)
    )

    path = tmp_path / "missing.json"
    checksum = mod.reproducibility_checksum(
        source_paths=[path],
        payload={"b": 2, "a": 1},
    )
    assert checksum.startswith("sha256:")
    assert mod._load_exp2850_check(path).available is False
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{bad", encoding="utf-8")
    assert mod._load_exp2850_check(bad_json).available is False
    assert mod._text_localizer_payload(path) == {}
    assert mod._text_localizer_payload(bad_json) == {}

    step_missing = mod._step_corpus_check(tmp_path / "missing_steps.jsonl")
    assert step_missing.available is False
    no_labels = tmp_path / "no_labels.jsonl"
    _write_jsonl(no_labels, [{"question_id": "x", "partial_cot": "no label"}])
    assert mod._step_corpus_check(no_labels).available is False
    bad_step = tmp_path / "bad_steps.jsonl"
    bad_step.write_text("{bad\n", encoding="utf-8")
    assert mod._step_corpus_check(bad_step).available is False

    trace = mod.HiddenStateTrace("short", ("only",), 0, "0", "row")
    assert mod.PositionOnlyBaseline({}).predict_first_error_index(trace) is None
    assert mod.PositionOnlyBaseline({5: 2}).predict_first_error_index(trace) == 0
    assert mod.HiddenStateMarginProbe.fit([]).predict_first_error_index("missing", []) is None
    assert mod.has_nonposition_signal({"delta_ci95": "bad"}) is False
    with pytest.raises(ValueError, match="hidden state count mismatch"):
        mod.transport_margin_features([trace], [[[0.0], [1.0]]])

    all_rows_path = tmp_path / "all_rows.jsonl"
    _write_jsonl(all_rows_path, _fover_rows(n_bad=1))
    assert len(mod.load_real_fover_error_traces(all_rows_path)) == 1

    missing_verify = mod.run_adversarial_verify(tmp_path / "artifact.json", repo_root=tmp_path)
    assert missing_verify["returncode"] is None
    script = tmp_path / "scripts" / "adversarial_verify.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("import sys\nprint('ok')\nsys.exit(0)\n", encoding="utf-8")
    verify = mod.run_adversarial_verify(tmp_path / "artifact.json", repo_root=tmp_path)
    assert verify["returncode"] == 0
    assert "ok" in verify["stdout_tail"]

    errors = mod.artifact_schema_errors(
        {
            "honest_verdict": "bad",
            "hidden_state_localizer_has_nonposition_signal": "false",
            "localization_f1_comparison": [],
            "position_only_baseline_f1": 1,
            "n_traces": "0",
            "verifier_is_oracle": True,
        }
    )
    assert "missing preconditions_checked" in errors
    assert "honest_verdict must be terminal-prefixed" in errors
    assert "hidden_state_localizer_has_nonposition_signal must be bare bool" in errors
    assert "localization_f1_comparison must be dict" in errors
    assert "position_only_baseline_f1 must be bare float" in errors
    assert "n_traces must be bare int" in errors
    assert "verifier_is_oracle must be false" in errors


def test_scenario_verify_4416_dry_run_flagged_and_bad_row_branches(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4416: dry-run, flagged, and unreadable row paths stay explicit."""

    dry = mod.run_experiment(
        _fixture_config(tmp_path, artifact_name="dry.json"),
        hidden_state_extractor=FakeExtractor(),
        write=False,
    )
    assert dry["adversarial_verify"] == {"returncode": None, "skipped": True}

    flagged = mod.run_experiment(
        _fixture_config(tmp_path, artifact_name="flagged.json"),
        hidden_state_extractor=FakeExtractor(),
        adversarial_verify_runner=lambda _path: {"returncode": 1, "stdout_tail": "flag"},
        write=True,
    )
    assert flagged["flagged_adversarial"] is True

    config = _fixture_config(tmp_path, artifact_name="bad_rows.json")
    config.fover_row_corpus_path.write_text("{bad\n", encoding="utf-8")
    blocked = mod.run_experiment(
        config,
        hidden_state_extractor=FakeExtractor(),
        write=True,
    )
    assert blocked["honest_verdict"] == "blocked_cached_corpus_unavailable"
    assert any(
        check["resource"] == "cached_real_fover_failed_traces" and "unreadable" in check["detail"]
        for check in blocked["preconditions_checked"]
    )

    missing_config = _fixture_config(tmp_path, artifact_name="missing_rows.json")
    missing_config.fover_row_corpus_path.unlink()
    missing = mod.run_experiment(
        missing_config,
        hidden_state_extractor=FakeExtractor(),
        write=True,
    )
    assert missing["honest_verdict"] == "blocked_cached_corpus_unavailable"
