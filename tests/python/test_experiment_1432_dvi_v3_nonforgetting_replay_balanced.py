"""Tests for Exp 1432 DVI v3 replay-heldout nonforgetting repair.

Spec: REQ-VERIFY-1432, SCENARIO-VERIFY-1432.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from carnot.reporting import dvi_v3_nonforgetting_replay_balanced as mod


def _write_checkpoint(path: Path, *, confidence: float = 0.5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez(
            handle,
            metric=np.zeros(128, dtype=np.float32),
            bias=np.asarray([0.0], dtype=np.float32),
            secl_bin_values=np.full(10, confidence, dtype=np.float32),
            secl_global_value=np.asarray([confidence], dtype=np.float32),
            secl_n_bins=np.asarray([10], dtype=np.int32),
            fresh_cases_used=np.asarray([59], dtype=np.int32),
        )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _exp1394(checkpoint_path: Path, *, dvi_v2_delta: float = 0.011458) -> dict[str, Any]:
    return {
        "status": "complete",
        "dvi_v2_deployed": True,
        "checkpoint_path": str(checkpoint_path),
        "dvi_v2_auroc_delta": dvi_v2_delta,
        "secl_ece_reduction_pct": 45.35096,
    }


def _exp1395(promoted_ids: list[str], demoted_ids: list[str]) -> dict[str, Any]:
    return {
        "status": "complete",
        "fresh_verified_sample_count": len(promoted_ids),
        "memory_updates": {
            "promoted": [f"dvi_v2:fover:{case_id}" for case_id in promoted_ids],
            "demoted": [f"dvi_v2:fover:{case_id}" for case_id in demoted_ids],
        },
    }


def _exp1415(
    *,
    dvi_v3_delta: float = 0.011842,
    dvi_v2_delta: float = 0.011458,
    nonforgetting_rate: float = 0.968604,
) -> dict[str, Any]:
    return {
        "status": "blocked",
        "dvi_v3_deployed": False,
        "dvi_v3_auroc_delta": dvi_v3_delta,
        "dvi_v2_auroc_delta_baseline": dvi_v2_delta,
        "nonforgetting_rate": nonforgetting_rate,
        "block_reasons": ["nonforgetting_below_gate"],
    }


def _fover_row(case_id: str, label: str) -> dict[str, str]:
    return {
        "question_id": case_id,
        "question": f"Question for {case_id}",
        "step_text": f"{label} reasoning trace for {case_id}",
        "label": label,
        "source": "unit_fover",
    }


def _training_result(*, auroc_delta: float, bias: float = 10.0) -> mod.base.DviV3TrainingResult:
    return mod.base.DviV3TrainingResult(
        baseline_auroc=0.4,
        trained_auroc=0.4 + auroc_delta,
        auroc_delta=auroc_delta,
        metric=np.zeros(128, dtype=np.float32),
        bias=bias,
        loss_history=[0.7, 0.69],
        source_checkpoint_path="/tmp/dvi_v2.pt",
    )


def test_req_verify_1432_writes_in_progress_artifact_first(tmp_path: Path) -> None:
    """REQ-VERIFY-1432: bootstrap output exists before source artifact loading."""

    out_path = tmp_path / mod.OUTPUT_FILE

    artifact = mod.write_in_progress_artifact(out_path, project_root="/repo")

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == artifact
    assert written["status"] == "in_progress"
    assert written["dvi_v3_deployed"] is False
    assert written["dvi_v3_auroc_delta"] is None
    assert written["dvi_v2_auroc_delta_baseline"] == mod.DVI_V2_AUROC_DELTA_BASELINE
    assert written["nonforgetting_rate"] is None
    assert written["replay_balance_applied"] is False
    assert written["threshold_calibration_applied"] is False
    assert written["fresh_cases_used"] == 0
    assert written["tests_run"] == []
    assert written["honest_verdict"] == "in_progress"


def test_req_verify_1432_diagnoses_exp1415_threshold_failure() -> None:
    """REQ-VERIFY-1432: positive AUROC plus replay-only block is thresholding."""

    diagnosis = mod.diagnose_exp1415_failure(_exp1415())

    assert diagnosis.failure_mode == "thresholding"
    assert diagnosis.auroc_improved_over_v2 is True
    assert diagnosis.nonforgetting_below_gate is True
    assert "nonforgetting_below_gate" in diagnosis.evidence


def test_req_verify_1432_diagnosis_and_split_edges() -> None:
    """REQ-VERIFY-1432: diagnosis and replay split branches stay deterministic."""

    assert (
        mod.diagnose_exp1415_failure(
            _exp1415(dvi_v3_delta=0.0, nonforgetting_rate=0.5)
        ).failure_mode
        == "model_update_drift"
    )
    sampling = _exp1415()
    sampling["block_reasons"] = []
    assert mod.diagnose_exp1415_failure(sampling).failure_mode == "sampling_imbalance"
    assert (
        mod.diagnose_exp1415_failure(_exp1415(nonforgetting_rate=1.0)).failure_mode
        == "no_failure_detected"
    )
    assert mod.split_replay_cases([]) == mod.ReplaySplit(calibration=[], holdout=[])

    one_case = [mod.dvi.DviCase(case_id="one", text="trace", label=1, source="test")]
    split = mod.split_replay_cases(one_case)
    assert split.calibration == one_case
    assert split.holdout == one_case


def test_scenario_verify_1432_calibrates_threshold_from_replay_holdout() -> None:
    """SCENARIO-VERIFY-1432: replay calibration can repair confidence-threshold drift."""

    head = mod.secl.HistogramECEConfidenceHead(
        bin_values=np.full(10, 0.5, dtype=np.float64),
        global_value=0.5,
    )
    replay_cases = [
        mod.dvi.DviCase(case_id=f"replay_{index}", text="bad trace", label=1, source="test")
        for index in range(4)
    ]

    calibration = mod.calibrate_threshold_for_nonforgetting(
        replay_cases=replay_cases[:2],
        metric=np.zeros(128, dtype=np.float32),
        bias=10.0,
        confidence_head=head,
        base_dvi_threshold=mod.fr11.DVI_INCORRECT_THRESHOLD,
        base_secl_threshold=mod.fr11.SECL_CONFIDENCE_THRESHOLD,
        candidate_secl_thresholds=(0.5, 0.500001),
        min_nonforgetting_rate=0.99,
    )
    holdout_rate = mod.measure_nonforgetting_rate(
        replay_cases=replay_cases[2:],
        metric=np.zeros(128, dtype=np.float32),
        bias=10.0,
        confidence_head=head,
        dvi_incorrect_threshold=calibration.dvi_incorrect_threshold,
        secl_confidence_threshold=calibration.secl_confidence_threshold,
    )

    assert calibration.threshold_calibration_applied is True
    assert calibration.secl_confidence_threshold == pytest.approx(0.500001)
    assert calibration.calibration_nonforgetting_rate == 1.0
    assert holdout_rate == 1.0


def test_req_verify_1432_calibration_returns_best_available_without_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-1432: calibration reports the best threshold when the gate stays closed."""

    rates = {0.5: 0.1, 0.6: 0.6, 0.7: 0.4}

    def fake_rate(**kwargs: Any) -> float:
        return rates[round(float(kwargs["secl_confidence_threshold"]), 1)]

    monkeypatch.setattr(mod, "measure_nonforgetting_rate", fake_rate)
    calibration = mod.calibrate_threshold_for_nonforgetting(
        replay_cases=[mod.dvi.DviCase(case_id="r", text="trace", label=1, source="test")],
        metric=np.zeros(128, dtype=np.float32),
        bias=0.0,
        confidence_head=mod.secl.HistogramECEConfidenceHead(
            bin_values=np.full(10, 0.5),
            global_value=0.5,
        ),
        candidate_secl_thresholds=(0.5, 0.6, 0.7),
        min_nonforgetting_rate=0.99,
    )

    assert calibration.secl_confidence_threshold == 0.6
    assert calibration.calibration_nonforgetting_rate == 0.6
    assert calibration.threshold_calibration_applied is True


def test_scenario_verify_1432_run_deploys_when_calibrated_gates_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-1432: deploy requires AUROC baseline and held-out replay pass."""

    results = tmp_path / "results"
    results.mkdir()
    v2_checkpoint = tmp_path / "verify" / "dvi_v2_secl_combined_checkpoint.pt"
    v3_checkpoint = tmp_path / "verify" / "dvi_v3_repaired_checkpoint.pt"
    exp1394_path = results / mod.EXP1394_FILE
    exp1395_path = results / mod.EXP1395_FILE
    exp1415_path = results / mod.EXP1415_FILE
    fover_path = tmp_path / "fover_corpus.jsonl"
    out_path = results / mod.OUTPUT_FILE
    fresh_ids = ["fresh_correct", "fresh_incorrect"]
    replay_ids = ["replay_bad_0", "replay_bad_1", "replay_bad_2", "replay_bad_3"]
    _write_checkpoint(v2_checkpoint, confidence=0.5)
    _write_json(exp1394_path, _exp1394(v2_checkpoint))
    _write_json(exp1395_path, _exp1395(fresh_ids, replay_ids))
    _write_json(exp1415_path, _exp1415())
    _write_jsonl(
        fover_path,
        [
            _fover_row("fresh_correct", "correct"),
            _fover_row("fresh_incorrect", "incorrect"),
            *[_fover_row(case_id, "incorrect") for case_id in replay_ids],
            *[
                _fover_row(f"holdout_{index}", "correct" if index % 2 == 0 else "incorrect")
                for index in range(8)
            ],
        ],
    )
    monkeypatch.setattr(mod, "run_dvi_v3_training", lambda **_: _training_result(auroc_delta=0.02))
    monkeypatch.setattr(
        mod,
        "measure_secl_preservation",
        lambda **_: mod.base.SECLPreservationResult(0.4, 0.2, 50.0, True),
    )

    artifact = mod.run(
        exp1394_path=exp1394_path,
        exp1395_path=exp1395_path,
        exp1415_path=exp1415_path,
        fover_path=fover_path,
        out_path=out_path,
        checkpoint_path=v3_checkpoint,
        project_root=tmp_path,
        expected_fresh_count=2,
        tests_run=["pytest targeted"],
    )

    assert json.loads(out_path.read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "complete"
    assert artifact["dvi_v3_deployed"] is True
    assert artifact["dvi_v3_auroc_delta"] == 0.02
    assert artifact["dvi_v2_auroc_delta_baseline"] == 0.011458
    assert artifact["nonforgetting_rate"] == 1.0
    assert artifact["replay_balance_applied"] is True
    assert artifact["threshold_calibration_applied"] is True
    assert artifact["fresh_cases_used"] == 2
    assert artifact["tests_run"] == ["pytest targeted"]
    assert artifact["honest_verdict"] == "dvi_v3_deployed_replay_heldout_threshold_calibrated"
    assert v3_checkpoint.exists()


def test_req_verify_1432_blocks_when_v2_checkpoint_inactive(tmp_path: Path) -> None:
    """REQ-VERIFY-1432: inactive DVI v2 source writes a blocked preflight artifact."""

    results = tmp_path / "results"
    results.mkdir()
    exp1394_path = results / mod.EXP1394_FILE
    exp1395_path = results / mod.EXP1395_FILE
    exp1415_path = results / mod.EXP1415_FILE
    out_path = results / mod.OUTPUT_FILE
    _write_json(
        exp1394_path,
        {"status": "complete", "dvi_v2_deployed": False, "checkpoint_path": "/missing.pt"},
    )
    _write_json(exp1395_path, _exp1395(["fresh"], []))
    _write_json(exp1415_path, _exp1415())

    artifact = mod.run(
        exp1394_path=exp1394_path,
        exp1395_path=exp1395_path,
        exp1415_path=exp1415_path,
        fover_path=tmp_path / "unused.jsonl",
        out_path=out_path,
        project_root=tmp_path,
        tests_run=["preflight"],
    )

    assert artifact["status"] == "blocked"
    assert artifact["fresh_cases_used"] == 0
    assert artifact["tests_run"] == ["preflight"]
    assert artifact["honest_verdict"] == "dvi_v3_blocked_exp1394_dvi_v2_not_deployed"


def test_req_verify_1432_records_checkpoint_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-1432: failed checkpoint persistence downgrades deployment."""

    results = tmp_path / "results"
    results.mkdir()
    v2_checkpoint = tmp_path / "verify" / "dvi_v2_secl_combined_checkpoint.pt"
    exp1394_path = results / mod.EXP1394_FILE
    exp1395_path = results / mod.EXP1395_FILE
    exp1415_path = results / mod.EXP1415_FILE
    fover_path = tmp_path / "fover_corpus.jsonl"
    out_path = results / mod.OUTPUT_FILE
    fresh_ids = ["fresh_correct", "fresh_incorrect"]
    replay_ids = ["replay_bad_0", "replay_bad_1"]
    _write_checkpoint(v2_checkpoint, confidence=0.5)
    _write_json(exp1394_path, _exp1394(v2_checkpoint))
    _write_json(exp1395_path, _exp1395(fresh_ids, replay_ids))
    _write_json(exp1415_path, _exp1415())
    _write_jsonl(
        fover_path,
        [
            _fover_row("fresh_correct", "correct"),
            _fover_row("fresh_incorrect", "incorrect"),
            *[_fover_row(case_id, "incorrect") for case_id in replay_ids],
        ],
    )
    monkeypatch.setattr(mod, "run_dvi_v3_training", lambda **_: _training_result(auroc_delta=0.02))
    monkeypatch.setattr(
        mod,
        "measure_secl_preservation",
        lambda **_: mod.base.SECLPreservationResult(0.4, 0.2, 50.0, True),
    )
    monkeypatch.setattr(mod, "save_repaired_checkpoint", lambda *args, **kwargs: False)

    artifact = mod.run(
        exp1394_path=exp1394_path,
        exp1395_path=exp1395_path,
        exp1415_path=exp1415_path,
        fover_path=fover_path,
        out_path=out_path,
        project_root=tmp_path,
        expected_fresh_count=2,
    )

    assert artifact["status"] == "blocked"
    assert artifact["block_reasons"] == ["dvi_v3_checkpoint_write_failed"]


def test_req_verify_1432_blocks_when_calibration_cannot_repair_nonforgetting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-1432: deployment stays blocked below the held-out replay gate."""

    results = tmp_path / "results"
    results.mkdir()
    v2_checkpoint = tmp_path / "verify" / "dvi_v2_secl_combined_checkpoint.pt"
    exp1394_path = results / mod.EXP1394_FILE
    exp1395_path = results / mod.EXP1395_FILE
    exp1415_path = results / mod.EXP1415_FILE
    fover_path = tmp_path / "fover_corpus.jsonl"
    out_path = results / mod.OUTPUT_FILE
    fresh_ids = ["fresh_correct", "fresh_incorrect"]
    replay_ids = ["replay_bad_0", "replay_bad_1"]
    _write_checkpoint(v2_checkpoint, confidence=0.5)
    _write_json(exp1394_path, _exp1394(v2_checkpoint))
    _write_json(exp1395_path, _exp1395(fresh_ids, replay_ids))
    _write_json(exp1415_path, _exp1415())
    _write_jsonl(
        fover_path,
        [
            _fover_row("fresh_correct", "correct"),
            _fover_row("fresh_incorrect", "incorrect"),
            *[_fover_row(case_id, "incorrect") for case_id in replay_ids],
        ],
    )
    monkeypatch.setattr(mod, "run_dvi_v3_training", lambda **_: _training_result(auroc_delta=0.02))
    monkeypatch.setattr(
        mod,
        "measure_secl_preservation",
        lambda **_: mod.base.SECLPreservationResult(0.4, 0.2, 50.0, True),
    )

    artifact = mod.run(
        exp1394_path=exp1394_path,
        exp1395_path=exp1395_path,
        exp1415_path=exp1415_path,
        fover_path=fover_path,
        out_path=out_path,
        project_root=tmp_path,
        expected_fresh_count=2,
        candidate_secl_thresholds=(0.5,),
        tests_run=["pytest targeted"],
    )

    assert artifact["status"] == "blocked"
    assert artifact["dvi_v3_deployed"] is False
    assert artifact["threshold_calibration_applied"] is False
    assert artifact["nonforgetting_rate"] == 0.0
    assert "nonforgetting_below_gate" in artifact["honest_verdict"]


def test_req_verify_1432_validates_required_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-1432: final artifacts must expose the required field contract."""

    valid = mod.write_in_progress_artifact(tmp_path / "progress.json", project_root="/repo")
    missing = dict(valid)
    missing.pop("dvi_v3_auroc_delta")

    with pytest.raises(AssertionError, match="missing required fields"):
        mod.validate_artifact(missing)

    bad_status = dict(valid, status="done")
    with pytest.raises(AssertionError, match="unsupported status"):
        mod.validate_artifact(bad_status)

    bad_tests = dict(valid, tests_run="pytest")
    with pytest.raises(AssertionError, match="tests_run must be a list"):
        mod.validate_artifact(bad_tests)

    terminal = dict(valid, status="complete", fresh_cases_used=2, dvi_v3_deployed=True)
    with pytest.raises(AssertionError, match="requires an existing checkpoint"):
        mod.validate_artifact(terminal)

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_text("placeholder", encoding="utf-8")
    terminal = dict(
        valid,
        status="complete",
        dvi_v3_deployed=True,
        dvi_v3_checkpoint_path=str(checkpoint),
        nonforgetting_rate=0.98,
        dvi_v3_auroc_delta=0.02,
        dvi_v2_auroc_delta_baseline=0.011458,
    )
    with pytest.raises(AssertionError, match="held-out nonforgetting gate"):
        mod.validate_artifact(terminal)

    terminal = dict(
        terminal,
        nonforgetting_rate=1.0,
        dvi_v3_auroc_delta=0.01,
        dvi_v2_auroc_delta_baseline=0.011458,
    )
    with pytest.raises(AssertionError, match="AUROC nonregression"):
        mod.validate_artifact(terminal)

    terminal = dict(
        valid,
        status="blocked",
        dvi_v3_auroc_delta=0.02,
        nonforgetting_rate=1.0,
        dvi_v3_deployed=False,
        dvi_v3_checkpoint_path=str(tmp_path / "not_deployed.pt"),
    )
    with pytest.raises(AssertionError, match="must not expose"):
        mod.validate_artifact(terminal)


def test_req_verify_1432_small_helpers() -> None:
    """REQ-VERIFY-1432: small helper branches are deterministic."""

    assert mod._deployment_block_reasons(
        dvi_v3_delta=0.01,
        dvi_v2_delta_baseline=0.02,
        nonforgetting_rate=0.98,
        min_nonforgetting_rate=0.99,
    ) == ["dvi_v3_delta_below_dvi_v2_baseline", "nonforgetting_below_gate"]
    assert mod._honest_verdict(False, []) == "dvi_v3_blocked_unknown_reason"
    assert mod._float("bad", 3.5) == 3.5
