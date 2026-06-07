"""Tests for Exp 3927 non-degenerate competent-judge cascade router.

Spec refs: REQ-VERIFY-3927, SCENARIO-VERIFY-3927,
SCENARIO-VERIFY-3927-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.eval import non_degenerate_cascade_router_3927 as exp3927


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "verification" / "spec.md"


def _row(index: int, label: int, energy: float, judge: float) -> exp3927.ScoreRow:
    return exp3927.ScoreRow(
        index=index,
        gold_error=label,
        energy_score=energy,
        judge_score=judge,
        corpus_source="fixture",
        source_index=index,
    )


def _calibration_rows() -> tuple[exp3927.ScoreRow, ...]:
    return (
        _row(0, 0, 0.10, 0.05),
        _row(1, 0, 0.20, 0.05),
        _row(2, 0, 0.30, 0.05),
        _row(3, 0, 0.40, 0.05),
        _row(4, 0, 0.51, 0.05),
        _row(5, 1, 0.49, 0.95),
        _row(6, 1, 0.60, 0.95),
        _row(7, 1, 0.70, 0.95),
        _row(8, 1, 0.80, 0.95),
        _row(9, 1, 0.90, 0.95),
    )


def _artifact_rows(n_repeats: int = 6) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for repeat in range(n_repeats):
        for source in _calibration_rows():
            index = len(rows)
            rows.append(
                {
                    "index": index,
                    "gold_error": source.gold_error,
                    "label": "incorrect" if source.gold_error else "correct",
                    "energy_score": source.energy_score,
                    "competent_judge_score": source.judge_score,
                    "corpus_source": "fixture",
                    "source_index": index,
                    "corpus_item_id": f"fixture-{repeat}-{source.index}",
                    "question_id": f"q-{repeat}-{source.index}",
                }
            )
    return rows


def _write_exp3926_artifact(
    repo_root: Path,
    *,
    rows: list[dict[str, object]] | None = None,
    energy_ms: float = 1.0,
    judge_ms: float = 100.0,
    positive_control: bool = True,
) -> Path:
    output = repo_root / "results" / "experiment_3926_valid_efficiency_head_to_head.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    score_rows = rows if rows is not None else _artifact_rows()
    payload = {
        "experiment": 3926,
        "honest_verdict": "complete: fixture",
        "judge_positive_control_passed": positive_control,
        "energy_per_item_ms": energy_ms,
        "llm_per_item_ms": judge_ms,
        "cost_ratio_walltime": judge_ms / energy_ms,
        "energy_cost": {
            "auroc": 0.96,
            "per_item_wall_ms": energy_ms,
            "total_wall_s": energy_ms * len(score_rows) / 1000.0,
            "est_tokens": 10,
            "est_flops": 100,
            "n_items": len(score_rows),
        },
        "llm_cost": {
            "auroc": 1.0,
            "per_item_wall_ms": judge_ms,
            "total_wall_s": judge_ms * len(score_rows) / 1000.0,
            "est_tokens": 100,
            "est_flops": 1000,
            "n_items": len(score_rows),
        },
        "llm_judge_auroc": 1.0,
        "per_item_results": score_rows,
        "model_specs": {"model_used": "fixture-competent-judge"},
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output


def test_req_verify_3927_spec_anchor_exists() -> None:
    """REQ-VERIFY-3927: the competent cascade router is OpenSpec anchored."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-VERIFY-3927" in spec
    assert "SCENARIO-VERIFY-3927" in spec
    assert "non_degenerate_cascade_router_3927.py" in spec
    assert "results/experiment_3927_non_degenerate_cascade_router.json" in spec


def test_req_verify_3927_applies_close_call_escalation_and_costs() -> None:
    """REQ-VERIFY-3927: only rows inside the energy margin band use judge scores."""

    rows = (
        _row(0, 0, 0.10, 0.20),
        _row(1, 1, 0.49, 0.90),
        _row(2, 0, 0.51, 0.10),
        _row(3, 1, 0.90, 0.80),
    )

    scores, escalated = exp3927.apply_cascade_scores(rows, threshold=0.50, band=0.02)
    metrics = exp3927.evaluate_cascade(
        rows,
        threshold=0.50,
        band=0.02,
        energy_per_item_ms=1.0,
        judge_per_item_ms=10.0,
    )

    assert scores == pytest.approx((0.10, 0.90, 0.10, 0.90))
    assert escalated == (False, True, True, False)
    assert metrics["cascade_auroc"] == 1.0
    assert metrics["pure_judge_auroc"] == 1.0
    assert metrics["escalation_fraction"] == 0.5
    assert metrics["cascade_degenerate"] is False
    assert metrics["cascade_cost_ratio"] == pytest.approx(40.0 / 24.0)
    assert rows[0].as_dict()["judge_score"] == 0.20

    with pytest.raises(ValueError, match="at least one item"):
        exp3927._cost_ratio(  # noqa: SLF001
            n_items=0,
            n_escalated=0,
            energy_per_item_ms=1.0,
            judge_per_item_ms=10.0,
        )
    with pytest.raises(ValueError, match="cascade cost"):
        exp3927._cost_ratio(  # noqa: SLF001
            n_items=1,
            n_escalated=0,
            energy_per_item_ms=0.0,
            judge_per_item_ms=10.0,
        )


def test_req_verify_3927_tunes_nonzero_band_on_calibration_split() -> None:
    """REQ-VERIFY-3927: calibration selects a non-degenerate close-call band."""

    result = exp3927.tune_band(
        _calibration_rows(),
        threshold=0.50,
        energy_per_item_ms=1.0,
        judge_per_item_ms=100.0,
    )

    assert result["band"] > 0.0
    assert result["cascade_auroc"] == 1.0
    assert result["pure_judge_auroc"] == 1.0
    assert result["auroc_gap"] == 0.0
    assert result["escalation_fraction"] == 0.2
    assert result["cascade_degenerate"] is False
    assert result["cascade_cost_ratio"] > 3.0
    assert exp3927._classify_verdict(  # noqa: SLF001
        auroc_gap=0.0,
        cascade_cost_ratio=4.0,
        escalation_fraction=0.2,
        cascade_degenerate=False,
    ).startswith("complete: cascade_router_WINS_escfrac")


def test_scenario_verify_3927_complete_artifact_uses_cached_scores(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3927: cached competent scores write bare heldout metrics."""

    upstream_path = _write_exp3926_artifact(tmp_path)
    output_path = tmp_path / "results" / "experiment_3927_non_degenerate_cascade_router.json"
    artifact = exp3927.run_experiment(
        exp3927.CascadeConfig(
            repo_root=tmp_path,
            output_path=output_path,
            started_at=10.0,
            clock=lambda: 11.25,
            random_seed=3927,
        ),
        write=True,
    )

    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    exp3927.validate_artifact(persisted)
    assert artifact == persisted
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["escalation_fraction"] > 0.0
    assert artifact["cascade_degenerate"] is False
    assert artifact["n_calibration"] + artifact["n_heldout"] == len(_artifact_rows())
    assert artifact["inference_substrate"] == exp3927.INFERENCE_SUBSTRATE
    assert artifact["source_artifacts"]["exp3926"]["path"] == upstream_path.relative_to(tmp_path).as_posix()
    assert artifact["no_new_inference"] is True
    assert artifact["frozen_fover_auroc_unchanged"] == 0.9131
    assert not isinstance(artifact["cascade_cost_ratio"], dict)


def test_scenario_verify_3927_missing_or_failed_upstream_blocks(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3927-BLOCKED: missing or failed Exp3926 writes no claims."""

    output_path = tmp_path / "results" / "experiment_3927_non_degenerate_cascade_router.json"
    artifact = exp3927.run_experiment(
        exp3927.CascadeConfig(
            repo_root=tmp_path,
            output_path=output_path,
            started_at=4.0,
            clock=lambda: 4.5,
        ),
        write=True,
    )

    exp3927.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_upstream_valid_efficiency_missing"
    assert artifact["cascade_auroc"] is None
    assert artifact["cascade_cost_ratio"] is None
    assert artifact["cascade_degenerate"] is True
    assert artifact["n_calibration"] == 0
    assert artifact["n_heldout"] == 0
    assert output_path.is_file()

    _write_exp3926_artifact(tmp_path, positive_control=False)
    checks, blocked_reason, evidence = exp3927.probe_preconditions(tmp_path)
    assert blocked_reason == "blocked_upstream_valid_efficiency_missing"
    assert evidence is None
    assert checks[0].available is False
    assert "positive control" in checks[0].detail


def test_req_verify_3927_evidence_validation_errors(tmp_path: Path) -> None:
    """REQ-VERIFY-3927: malformed cached evidence fails closed."""

    assert exp3927._require_judge_per_item_ms({"judge_per_item_ms": 3.0}) == 3.0  # noqa: SLF001
    with pytest.raises(ValueError, match="missing judge_per_item_ms"):
        exp3927._require_judge_per_item_ms({})  # noqa: SLF001

    with pytest.raises(ValueError, match="calibration_fraction"):
        exp3927.split_rows(_calibration_rows(), random_seed=1, calibration_fraction=1.0)
    with pytest.raises(ValueError, match="both labels"):
        exp3927.split_rows((_row(0, 0, 0.1, 0.1),), random_seed=1, calibration_fraction=0.5)

    exp3926_path = tmp_path / "results" / "experiment_3926_valid_efficiency_head_to_head.json"
    exp3926_path.parent.mkdir(parents=True, exist_ok=True)

    exp3926_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        exp3927.load_exp3926_evidence(tmp_path)

    exp3926_path.write_text(json.dumps({"judge_positive_control_passed": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="no per_item_results"):
        exp3927.load_exp3926_evidence(tmp_path)

    _write_exp3926_artifact(tmp_path, rows=[{"competent_judge_score": 0.1}])
    with pytest.raises(ValueError, match="missing energy_score, gold_error"):
        exp3927.load_exp3926_evidence(tmp_path)

    rows = _artifact_rows()
    rows[0]["gold_error"] = 2
    _write_exp3926_artifact(tmp_path, rows=rows)
    with pytest.raises(ValueError, match="non-binary"):
        exp3927.load_exp3926_evidence(tmp_path)

    one_class_rows = [dict(row, gold_error=0) for row in _artifact_rows()]
    _write_exp3926_artifact(tmp_path, rows=one_class_rows)
    with pytest.raises(ValueError, match="both gold classes"):
        exp3927.load_exp3926_evidence(tmp_path)

    _write_exp3926_artifact(tmp_path)
    artifact = json.loads(exp3926_path.read_text(encoding="utf-8"))
    del artifact["energy_per_item_ms"]
    exp3926_path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="missing energy_per_item_ms"):
        exp3927.load_exp3926_evidence(tmp_path)

    artifact["energy_per_item_ms"] = -1.0
    exp3926_path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="must be positive"):
        exp3927.load_exp3926_evidence(tmp_path)

    _write_exp3926_artifact(tmp_path)
    artifact = json.loads(exp3926_path.read_text(encoding="utf-8"))
    artifact["cost_ratio_walltime"] = -1.0
    exp3926_path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="cost_ratio_walltime"):
        exp3927.load_exp3926_evidence(tmp_path)

    _write_exp3926_artifact(tmp_path, rows=[{"gold_error": 1, "energy_score": 0.9}])
    checks, blocked_reason, evidence = exp3927.probe_preconditions(tmp_path)
    assert blocked_reason == "blocked_upstream_valid_efficiency_missing"
    assert evidence is None
    assert "judge score" in checks[0].detail


def test_req_verify_3927_validate_artifact_rejects_bad_shapes(tmp_path: Path) -> None:
    """REQ-VERIFY-3927: artifact validation enforces terminal and bare fields."""

    artifact = exp3927.build_blocked_artifact(
        config=exp3927.CascadeConfig(repo_root=tmp_path, started_at=0.0, clock=lambda: 1.0),
        reason="blocked_upstream_valid_efficiency_missing",
        preconditions_checked=[
            exp3927.PreconditionCheck("exp3926_valid_scores_ready", False, "missing")
        ],
        started_at=0.0,
    )

    exp3927.validate_artifact(artifact)
    with pytest.raises(ValueError, match="missing required"):
        exp3927.validate_artifact({key: value for key, value in artifact.items() if key != "duration_s"})
    with pytest.raises(ValueError, match="terminal prefix"):
        exp3927.validate_artifact(dict(artifact, honest_verdict="pending"))
    with pytest.raises(ValueError, match="must not be"):
        exp3927.validate_artifact(dict(artifact, cascade_auroc={"value": 1.0}))
    with pytest.raises(ValueError, match="split counts"):
        exp3927.validate_artifact(dict(artifact, n_calibration=1))
    with pytest.raises(ValueError, match="duration_s"):
        exp3927.validate_artifact(dict(artifact, duration_s="1"))
    with pytest.raises(ValueError, match="sha256"):
        exp3927.validate_artifact(dict(artifact, reproducibility_checksum="bad"))
    with pytest.raises(ValueError, match="bare bool"):
        exp3927.validate_artifact(dict(artifact, cascade_degenerate="true"))

    complete = dict(
        artifact,
        honest_verdict=(
            "complete: cascade_router_WINS_escfrac0.1000_gap0.0000_"
            "10.00x_cheaper_at_matched_accuracy_non_degenerate"
        ),
        status=(
            "complete: cascade_router_WINS_escfrac0.1000_gap0.0000_"
            "10.00x_cheaper_at_matched_accuracy_non_degenerate"
        ),
        cascade_auroc=1.0,
        pure_judge_auroc=1.0,
        escalation_fraction=0.1,
        cascade_degenerate=False,
        cascade_cost_ratio=10.0,
        auroc_gap=0.0,
        band_tuned_on_calibration=0.01,
        n_calibration=10,
        n_heldout=10,
    )
    exp3927.validate_artifact(complete)
    with pytest.raises(ValueError, match="must be a bare float"):
        exp3927.validate_artifact(dict(complete, cascade_cost_ratio="10"))
    with pytest.raises(ValueError, match="bare ints"):
        exp3927.validate_artifact(dict(complete, n_calibration=10.0))
    with pytest.raises(ValueError, match="positive"):
        exp3927.validate_artifact(dict(complete, n_heldout=0))
    with pytest.raises(ValueError, match="non-degenerate"):
        exp3927.validate_artifact(dict(complete, cascade_degenerate=True))
    assert "MARGINAL" in exp3927._classify_verdict(  # noqa: SLF001
        auroc_gap=0.0,
        cascade_cost_ratio=4.0,
        escalation_fraction=0.0,
        cascade_degenerate=True,
    )


def test_req_verify_3927_cli_and_script_report_terminal_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-3927: CLI status follows the terminal artifact verdict."""

    assert exp3927.cli_main(["--repo-root", str(tmp_path)]) == 1
    _write_exp3926_artifact(tmp_path)
    assert exp3927.cli_main(["--repo-root", str(tmp_path)]) == 0

    from scripts.experiments import experiment_3927_non_degenerate_cascade_router as script

    monkeypatch.setattr(script, "cli_main", lambda argv: 7)
    assert script.main() == 7
