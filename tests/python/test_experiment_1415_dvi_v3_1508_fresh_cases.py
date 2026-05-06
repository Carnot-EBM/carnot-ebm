"""Tests for Exp 1415 DVI v3 on the Exp 1395 fresh verified corpus.

Spec: REQ-VERIFY-1415, SCENARIO-VERIFY-1415.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from carnot.reporting import dvi_v3_1508_fresh_cases as mod


def _write_checkpoint(path: Path, *, bias: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez(
            handle,
            metric=np.zeros(128, dtype=np.float32),
            bias=np.asarray([bias], dtype=np.float32),
            secl_bin_values=np.ones(10, dtype=np.float32),
            secl_global_value=np.asarray([1.0], dtype=np.float32),
            secl_n_bins=np.asarray([10], dtype=np.int32),
            fresh_cases_used=np.asarray([59], dtype=np.int32),
        )


def _exp1394(checkpoint_path: Path, *, dvi_v2_delta: float = 0.011458) -> dict[str, Any]:
    return {
        "status": "complete",
        "dvi_v2_deployed": True,
        "checkpoint_path": str(checkpoint_path),
        "dvi_v2_auroc_delta": dvi_v2_delta,
        "secl_ece_reduction_pct": 45.35096,
        "honest_verdict": "dvi_v2_secl_combined_deployed_positive_auroc_delta_ece_reduced",
    }


def _exp1395(promoted_ids: list[str], demoted_ids: list[str] | None = None) -> dict[str, Any]:
    return {
        "status": "complete",
        "fresh_verified_sample_count": len(promoted_ids),
        "memory_updates": {
            "promoted": [f"dvi_v2:fover:{case_id}" for case_id in promoted_ids],
            "demoted": [f"dvi_v2:fover:{case_id}" for case_id in (demoted_ids or [])],
        },
        "honest_verdict": "fr11_self_learning_v5_dvi_v2_secl_headline_allowed",
    }


def _fover_row(case_id: str, label: str) -> dict[str, str]:
    return {
        "question_id": case_id,
        "question": f"Question for {case_id}",
        "step_text": f"{label} reasoning trace for {case_id}",
        "label": label,
        "source": "unit_fover",
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_req_verify_1415_writes_in_progress_artifact_first(tmp_path: Path) -> None:
    """REQ-VERIFY-1415: bootstrap output exists before source loading."""

    out_path = tmp_path / mod.OUTPUT_FILE

    artifact = mod.write_in_progress_artifact(out_path, project_root="/repo")

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == artifact
    assert written["status"] == "in_progress"
    assert written["fresh_verified_cases_used"] == 0
    assert written["dvi_v2_auroc_delta_baseline"] == mod.DVI_V2_AUROC_DELTA_BASELINE
    assert written["dvi_v3_deployed"] is False
    assert written["honest_verdict"] == "in_progress"


def test_req_verify_1415_loads_exact_1508_exp1395_fresh_ids() -> None:
    """REQ-VERIFY-1415: fresh cases are the 1508 Exp 1395 DVI v2 promotions."""

    promoted_ids = [f"fresh_{index}" for index in range(mod.FRESH_VERIFIED_CASE_COUNT)]
    exp1395 = _exp1395(promoted_ids)

    ids = mod.fresh_case_ids_from_exp1395(exp1395)

    assert len(ids) == mod.FRESH_VERIFIED_CASE_COUNT
    assert ids[0] == "fresh_0"
    assert ids[-1] == "fresh_1507"


def test_req_verify_1415_rejects_fresh_count_mismatch() -> None:
    """REQ-VERIFY-1415: a complete v3 run must not silently drop fresh IDs."""

    exp1395 = _exp1395(["only_one"])
    exp1395["fresh_verified_sample_count"] = mod.FRESH_VERIFIED_CASE_COUNT

    with pytest.raises(ValueError, match="Exp 1395 fresh verified count mismatch"):
        mod.fresh_case_ids_from_exp1395(exp1395)


def test_req_verify_1415_rejects_malformed_sources(tmp_path: Path) -> None:
    """REQ-VERIFY-1415: malformed artifacts fail before partial training starts."""

    non_object = tmp_path / "list.json"
    non_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact must be a JSON object"):
        mod.load_json(non_object)

    with pytest.raises(ValueError, match="memory_updates.promoted must be a list"):
        mod.fresh_case_ids_from_exp1395({"memory_updates": {"promoted": "bad"}})

    duplicate = _exp1395(["dup", "dup"])
    with pytest.raises(ValueError, match="contain duplicates"):
        mod.fresh_case_ids_from_exp1395(duplicate, expected_count=2)

    assert mod.replay_case_ids_from_exp1395({"memory_updates": {"demoted": "bad"}}) == []
    replay = _exp1395(["fresh"], demoted_ids=["a", "b", "a"])
    assert mod.replay_case_ids_from_exp1395(replay, max_replay_cases=2) == ["a", "b"]


def test_req_verify_1415_reconstructs_labeled_fresh_and_replay_cases() -> None:
    """REQ-VERIFY-1415: promoted and replay IDs join back to labeled FoVer rows."""

    exp1395 = _exp1395(
        ["fresh_correct", "fresh_incorrect"],
        demoted_ids=["replay_demoted"],
    )
    rows = [
        _fover_row("fresh_correct", "correct"),
        _fover_row("fresh_incorrect", "incorrect"),
        _fover_row("replay_demoted", "incorrect"),
    ]

    fresh_cases = mod.load_fresh_verified_cases(exp1395, rows, expected_count=2)
    replay_cases = mod.load_replay_cases(exp1395, rows)

    assert [case.case_id for case in fresh_cases] == ["fresh_correct", "fresh_incorrect"]
    assert [case.label for case in fresh_cases] == [0, 1]
    assert all(case.source == "exp1395_dvi_v2_secl_fresh_verified_fover" for case in fresh_cases)
    assert [case.case_id for case in replay_cases] == ["replay_demoted"]


def test_req_verify_1415_rejects_missing_fresh_fover_row() -> None:
    """REQ-VERIFY-1415: every promoted Exp 1395 ID must resolve to a FoVer row."""

    exp1395 = _exp1395(["missing"])

    with pytest.raises(ValueError, match="fresh case missing from FoVer corpus"):
        mod.load_fresh_verified_cases(exp1395, [], expected_count=1)


def test_req_verify_1415_rejects_single_class_training(tmp_path: Path) -> None:
    """REQ-VERIFY-1415: v3 training needs both correct and incorrect labels."""

    checkpoint = tmp_path / "checkpoint.pt"
    _write_checkpoint(checkpoint)
    cases = [mod.dvi.DviCase(case_id="only_correct", text="correct only", label=0, source="test")]

    with pytest.raises(ValueError, match="must contain both correct and incorrect labels"):
        mod.run_dvi_v3_training(
            fresh_cases=cases,
            holdout_rows=[_fover_row("holdout", "correct")],
            source_checkpoint_path=checkpoint,
        )


def test_req_verify_1415_nonforgetting_and_secl_empty_edges() -> None:
    """REQ-VERIFY-1415: empty replay and unmeasurable SECL cases stay explicit."""

    head = mod.secl.HistogramECEConfidenceHead(
        bin_values=np.ones(10, dtype=np.float64),
        global_value=1.0,
    )

    assert (
        mod.measure_nonforgetting_rate(
            replay_cases=[],
            metric=np.zeros(128, dtype=np.float32),
            bias=0.0,
            confidence_head=head,
        )
        == 1.0
    )

    result = mod.measure_secl_preservation(
        metric=np.zeros(128, dtype=np.float32),
        bias=0.0,
        confidence_head=head,
        holdout_rows=[{"question_id": "bad", "step_text": "", "label": "unknown"}],
        v2_ece_reduction_pct=10.0,
    )
    assert result == mod.SECLPreservationResult(
        ece_before=0.0,
        ece_after=0.0,
        ece_reduction_pct=0.0,
        preserved=False,
    )


def test_scenario_verify_1415_run_deploys_when_gates_are_met(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1415: runner records metrics and writes a checkpoint on deploy."""

    results = tmp_path / "results"
    results.mkdir()
    v2_checkpoint = tmp_path / "verify" / "dvi_v2_secl_combined_checkpoint.pt"
    v3_checkpoint = tmp_path / "verify" / "dvi_v3_1508_fresh_cases_checkpoint.pt"
    exp1394_path = results / mod.EXP1394_FILE
    exp1395_path = results / mod.EXP1395_FILE
    fover_path = tmp_path / "fover_corpus.jsonl"
    out_path = results / mod.OUTPUT_FILE
    fresh_ids = [f"fresh_{index}" for index in range(6)]
    replay_ids = [f"replay_{index}" for index in range(4)]
    rows = [
        _fover_row(case_id, "correct" if index % 2 == 0 else "incorrect")
        for index, case_id in enumerate([*fresh_ids, *replay_ids])
    ]
    rows.extend(
        _fover_row(f"holdout_{index}", "correct" if index % 2 == 0 else "incorrect")
        for index in range(20)
    )
    _write_checkpoint(v2_checkpoint)
    _write_json(exp1394_path, _exp1394(v2_checkpoint, dvi_v2_delta=-1.0))
    _write_json(exp1395_path, _exp1395(fresh_ids, replay_ids))
    _write_jsonl(fover_path, rows)

    artifact = mod.run(
        exp1394_path=exp1394_path,
        exp1395_path=exp1395_path,
        fover_path=fover_path,
        out_path=out_path,
        checkpoint_path=v3_checkpoint,
        project_root=tmp_path,
        expected_fresh_count=6,
        dvi_v2_auroc_delta_baseline=-1.0,
        min_nonforgetting_rate=0.0,
        require_secl_preserved=False,
        n_epochs=2,
        tests_run=["pytest targeted"],
    )

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == artifact
    mod.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["fresh_verified_cases_used"] == 6
    assert artifact["dvi_v2_auroc_delta_baseline"] == -1.0
    assert artifact["dvi_v3_deployed"] is True
    assert artifact["dvi_v3_checkpoint_path"] == str(v3_checkpoint)
    assert v3_checkpoint.exists()
    assert isinstance(artifact["dvi_v3_auroc_delta"], float)
    assert 0.0 <= artifact["nonforgetting_rate"] <= 1.0
    assert artifact["tests_run"] == ["pytest targeted"]


def test_scenario_verify_1415_run_blocks_when_v2_checkpoint_inactive(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1415: inactive DVI v2 emits an honest blocked artifact."""

    results = tmp_path / "results"
    results.mkdir()
    exp1394_path = results / mod.EXP1394_FILE
    exp1395_path = results / mod.EXP1395_FILE
    out_path = results / mod.OUTPUT_FILE
    _write_json(
        exp1394_path,
        {
            "status": "complete",
            "dvi_v2_deployed": False,
            "checkpoint_path": str(tmp_path / "missing.pt"),
        },
    )
    _write_json(exp1395_path, _exp1395(["fresh"]))

    artifact = mod.run(
        exp1394_path=exp1394_path,
        exp1395_path=exp1395_path,
        fover_path=tmp_path / "unused.jsonl",
        out_path=out_path,
        project_root=tmp_path,
        expected_fresh_count=1,
        tests_run=["blocked preflight"],
    )

    assert artifact["status"] == "blocked"
    assert artifact["fresh_verified_cases_used"] == 0
    assert artifact["dvi_v3_auroc_delta"] is None
    assert artifact["nonforgetting_rate"] is None
    assert artifact["tests_run"] == ["blocked preflight"]
    assert artifact["honest_verdict"] == "dvi_v3_blocked_exp1394_dvi_v2_not_deployed"


def test_scenario_verify_1415_blocks_without_deploy_checkpoint_when_delta_regresses(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-1415: blocked v3 artifacts explain why no checkpoint deployed."""

    training_result = mod.DviV3TrainingResult(
        baseline_auroc=0.5,
        trained_auroc=0.49,
        auroc_delta=-0.01,
        metric=np.zeros(128, dtype=np.float32),
        bias=0.0,
        loss_history=[0.7, 0.69],
        source_checkpoint_path="/tmp/dvi_v2.pt",
    )
    secl_result = mod.SECLPreservationResult(
        ece_before=0.4,
        ece_after=0.5,
        ece_reduction_pct=-25.0,
        preserved=False,
    )

    artifact = mod.build_artifact(
        fresh_verified_cases_used=mod.FRESH_VERIFIED_CASE_COUNT,
        replay_cases_used=10,
        dvi_v2_auroc_delta_baseline=0.011458,
        training_result=training_result,
        nonforgetting_rate=1.0,
        secl_preservation=secl_result,
        deployed=False,
        checkpoint_path=tmp_path / "blocked.pt",
        source_checkpoint_path="/tmp/dvi_v2.pt",
        started_at="2026-05-06T00:00:00+00:00",
        duration_s=0.0,
        train_cases_used=mod.FRESH_VERIFIED_CASE_COUNT,
        heldout_cases_used=20,
        tests_run=["pytest targeted"],
        block_reasons=["dvi_v3_delta_not_improved"],
        project_root="/repo",
    )

    mod.validate_artifact(artifact)
    assert artifact["status"] == "blocked"
    assert artifact["dvi_v3_deployed"] is False
    assert artifact["dvi_v3_checkpoint_path"] is None
    assert "dvi_v3_delta_not_improved" in artifact["honest_verdict"]


def test_req_verify_1415_records_checkpoint_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-1415: checkpoint write failure downgrades deployment to blocked."""

    results = tmp_path / "results"
    results.mkdir()
    v2_checkpoint = tmp_path / "verify" / "dvi_v2_secl_combined_checkpoint.pt"
    exp1394_path = results / mod.EXP1394_FILE
    exp1395_path = results / mod.EXP1395_FILE
    fover_path = tmp_path / "fover_corpus.jsonl"
    out_path = results / mod.OUTPUT_FILE
    fresh_ids = ["fresh_correct", "fresh_incorrect"]
    _write_checkpoint(v2_checkpoint)
    _write_json(exp1394_path, _exp1394(v2_checkpoint, dvi_v2_delta=-1.0))
    _write_json(exp1395_path, _exp1395(fresh_ids))
    _write_jsonl(
        fover_path,
        [
            _fover_row("fresh_correct", "correct"),
            _fover_row("fresh_incorrect", "incorrect"),
            *(
                _fover_row(f"holdout_{index}", "correct" if index % 2 == 0 else "incorrect")
                for index in range(8)
            ),
        ],
    )
    monkeypatch.setattr(mod, "save_v3_checkpoint", lambda *args, **kwargs: False)

    artifact = mod.run(
        exp1394_path=exp1394_path,
        exp1395_path=exp1395_path,
        fover_path=fover_path,
        out_path=out_path,
        project_root=tmp_path,
        expected_fresh_count=2,
        dvi_v2_auroc_delta_baseline=-1.0,
        min_nonforgetting_rate=0.0,
        require_secl_preserved=False,
        n_epochs=2,
    )

    assert artifact["status"] == "blocked"
    assert artifact["block_reasons"] == ["dvi_v3_checkpoint_write_failed"]


def test_req_verify_1415_validation_errors(tmp_path: Path) -> None:
    """REQ-VERIFY-1415: artifact validator catches schema and deploy drift."""

    valid = mod.write_in_progress_artifact(tmp_path / "progress.json", project_root="/repo")
    missing = dict(valid)
    missing.pop("tests_run")
    with pytest.raises(AssertionError, match="missing required fields"):
        mod.validate_artifact(missing)

    bad_status = dict(valid, status="done")
    with pytest.raises(AssertionError, match="unsupported status"):
        mod.validate_artifact(bad_status)

    bad_tests = dict(valid, tests_run="pytest")
    with pytest.raises(AssertionError, match="tests_run must be a list"):
        mod.validate_artifact(bad_tests)

    terminal = dict(valid, status="complete", tests_run=[])
    with pytest.raises(AssertionError, match="fresh_verified_cases_used"):
        mod.validate_artifact(terminal)

    terminal = dict(
        valid,
        status="complete",
        fresh_verified_cases_used=mod.FRESH_VERIFIED_CASE_COUNT,
        dvi_v3_auroc_delta=None,
        tests_run=[],
    )
    with pytest.raises(AssertionError, match="requires AUROC delta"):
        mod.validate_artifact(terminal)

    terminal = dict(
        valid,
        status="complete",
        fresh_verified_cases_used=mod.FRESH_VERIFIED_CASE_COUNT,
        dvi_v3_auroc_delta=0.1,
        nonforgetting_rate=None,
        tests_run=[],
    )
    with pytest.raises(AssertionError, match="requires nonforgetting_rate"):
        mod.validate_artifact(terminal)

    terminal = dict(
        valid,
        status="complete",
        fresh_verified_cases_used=mod.FRESH_VERIFIED_CASE_COUNT,
        dvi_v3_auroc_delta=0.1,
        nonforgetting_rate=1.0,
        dvi_v3_deployed=True,
        dvi_v3_checkpoint_path=str(tmp_path / "missing.pt"),
        tests_run=[],
    )
    with pytest.raises(AssertionError, match="existing checkpoint"):
        mod.validate_artifact(terminal)

    terminal = dict(
        valid,
        status="blocked",
        fresh_verified_cases_used=mod.FRESH_VERIFIED_CASE_COUNT,
        dvi_v3_auroc_delta=0.1,
        nonforgetting_rate=1.0,
        dvi_v3_deployed=False,
        dvi_v3_checkpoint_path=str(tmp_path / "not_deployed.pt"),
        tests_run=[],
    )
    with pytest.raises(AssertionError, match="must not expose"):
        mod.validate_artifact(terminal)


def test_req_verify_1415_small_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-VERIFY-1415: small gate helpers expose deterministic edge behavior."""

    assert mod._deployment_block_reasons(
        dvi_v3_delta=0.0,
        dvi_v2_delta_baseline=1.0,
        nonforgetting_rate=0.5,
        min_nonforgetting_rate=0.9,
        secl_preserved=False,
        require_secl_preserved=True,
    ) == [
        "dvi_v3_delta_not_improved",
        "nonforgetting_below_gate",
        "secl_ece_reduction_not_preserved",
    ]
    assert mod._ece_reduction_pct(0.0, 1.0) == 0.0
    assert mod._honest_verdict(False, []) == "dvi_v3_blocked_unknown_reason"
    assert mod._int("bad", 7) == 7
    assert mod._float("bad", 3.5) == 3.5

    monkeypatch.setattr(mod, "REQUIRED_ARTIFACT_FIELDS", (*mod.REQUIRED_ARTIFACT_FIELDS, "x"))
    with pytest.raises(AssertionError, match="missing required fields"):
        mod._blocked_without_training(
            project_root="/repo",
            run_date="20260506",
            started_at="2026-05-06T00:00:00+00:00",
            duration_s=0.0,
            dvi_v2_auroc_delta_baseline=0.011458,
            tests_run=[],
            expected_fresh_count=mod.FRESH_VERIFIED_CASE_COUNT,
            block_reason="forced",
        )
