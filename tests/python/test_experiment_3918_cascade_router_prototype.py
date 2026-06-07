"""Tests for the Exp 3918 classifier-first cascade router.

Spec refs: REQ-VERIFY-3918, SCENARIO-VERIFY-3918,
SCENARIO-VERIFY-3918-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.eval import cascade_router_prototype_3918 as exp3918


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "verification" / "spec.md"


def _row(index: int, label: int, energy: float, llm: float) -> exp3918.ScoreRow:
    return exp3918.ScoreRow(
        index=index,
        gold_error=label,
        energy_score=energy,
        llm_score=llm,
        corpus_source="fixture",
        source_index=index,
    )


def _calibration_rows() -> tuple[exp3918.ScoreRow, ...]:
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


def _artifact_rows(n_repeats: int = 3) -> list[dict[str, object]]:
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
                    "llm_judge_score": source.llm_score,
                    "corpus_source": "fixture",
                    "source_index": index,
                    "corpus_item_id": f"fixture-{repeat}-{source.index}",
                    "question_id": f"q-{repeat}-{source.index}",
                }
            )
    return rows


def _write_exp3917_artifact(
    repo_root: Path,
    *,
    rows: list[dict[str, object]] | None = None,
    energy_ms: float = 1.0,
    llm_ms: float = 100.0,
) -> Path:
    output = repo_root / "results" / "experiment_3917_efficiency_head_to_head.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    score_rows = rows if rows is not None else _artifact_rows()
    payload = {
        "experiment": 3917,
        "honest_verdict": "complete: fixture",
        "cost_ratio_walltime": llm_ms / energy_ms,
        "energy_per_item_ms": energy_ms,
        "llm_per_item_ms": llm_ms,
        "energy_cost": {
            "auroc": 0.96,
            "per_item_wall_ms": energy_ms,
            "total_wall_s": energy_ms * len(score_rows) / 1000.0,
            "est_tokens": 10,
            "est_flops": 100,
            "n_items": len(score_rows),
        },
        "llm_judge_cost": {
            "auroc": 1.0,
            "per_item_wall_ms": llm_ms,
            "total_wall_s": llm_ms * len(score_rows) / 1000.0,
            "est_tokens": 100,
            "est_flops": 1000,
            "n_items": len(score_rows),
        },
        "llm_judge_auroc": 1.0,
        "per_item_results": score_rows,
        "model_specs": {"model_used": "fixture-judge"},
        "score_digests": {"fixture": "cached"},
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output


def test_req_verify_3918_spec_anchor_exists() -> None:
    """REQ-VERIFY-3918: the cascade prototype is OpenSpec anchored."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-VERIFY-3918" in spec
    assert "SCENARIO-VERIFY-3918" in spec
    assert "python/carnot/eval/cascade_router_prototype_3918.py" in spec
    assert "results/experiment_3918_cascade_router_prototype.json" in spec


def test_req_verify_3918_applies_close_call_escalation_and_costs() -> None:
    """REQ-VERIFY-3918: only rows inside the energy margin band use LLM scores."""

    rows = (
        _row(0, 0, 0.10, 0.20),
        _row(1, 1, 0.49, 0.90),
        _row(2, 0, 0.51, 0.10),
        _row(3, 1, 0.90, 0.80),
    )

    scores, escalated = exp3918.apply_cascade_scores(rows, threshold=0.50, band=0.02)
    metrics = exp3918.evaluate_cascade(
        rows,
        threshold=0.50,
        band=0.02,
        energy_per_item_ms=1.0,
        llm_per_item_ms=10.0,
    )

    assert scores == pytest.approx((0.10, 0.90, 0.10, 0.90))
    assert escalated == (False, True, True, False)
    assert metrics["cascade_auroc"] == 1.0
    assert metrics["pure_llm_auroc"] == 1.0
    assert metrics["escalation_fraction"] == 0.5
    assert metrics["cascade_cost_ratio"] == pytest.approx(40.0 / 24.0)
    assert rows[0].as_dict()["energy_score"] == 0.10

    with pytest.raises(ValueError, match="at least one item"):
        exp3918._cost_ratio(  # noqa: SLF001
            n_items=0,
            n_escalated=0,
            energy_per_item_ms=1.0,
            llm_per_item_ms=10.0,
        )
    with pytest.raises(ValueError, match="cascade cost"):
        exp3918._cost_ratio(  # noqa: SLF001
            n_items=1,
            n_escalated=0,
            energy_per_item_ms=0.0,
            llm_per_item_ms=10.0,
        )


def test_req_verify_3918_tunes_band_on_calibration_split() -> None:
    """REQ-VERIFY-3918: calibration selects a band that fixes close energy misses."""

    result = exp3918.tune_band(
        _calibration_rows(),
        threshold=0.50,
        energy_per_item_ms=1.0,
        llm_per_item_ms=100.0,
    )

    assert result["band"] > 0.0
    assert result["cascade_auroc"] == 1.0
    assert result["pure_llm_auroc"] == 1.0
    assert result["auroc_gap"] == 0.0
    assert result["escalation_fraction"] == 0.2
    assert result["cascade_cost_ratio"] > 3.0
    assert exp3918._classify_verdict(  # noqa: SLF001
        auroc_gap=0.0,
        cascade_cost_ratio=4.0,
        escalation_fraction=0.2,
    ).startswith("complete: cascade_router_WINS")


def test_req_verify_3918_split_and_evidence_validation_errors(tmp_path: Path) -> None:
    """REQ-VERIFY-3918: malformed splits and cached evidence fail closed."""

    with pytest.raises(ValueError, match="calibration_fraction"):
        exp3918.split_rows(_calibration_rows(), random_seed=1, calibration_fraction=1.0)
    with pytest.raises(ValueError, match="both labels"):
        exp3918.split_rows((_row(0, 0, 0.1, 0.1),), random_seed=1, calibration_fraction=0.5)

    exp3917_path = tmp_path / "results" / "experiment_3917_efficiency_head_to_head.json"
    exp3917_path.parent.mkdir(parents=True, exist_ok=True)

    exp3917_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        exp3918.load_exp3917_evidence(tmp_path)

    exp3917_path.write_text(json.dumps({"per_item_results": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="no per_item_results"):
        exp3918.load_exp3917_evidence(tmp_path)

    rows = _artifact_rows()
    rows[0]["gold_error"] = 2
    _write_exp3917_artifact(tmp_path, rows=rows)
    with pytest.raises(ValueError, match="non-binary"):
        exp3918.load_exp3917_evidence(tmp_path)

    one_class_rows = [dict(row, gold_error=0) for row in _artifact_rows()]
    _write_exp3917_artifact(tmp_path, rows=one_class_rows)
    with pytest.raises(ValueError, match="both gold classes"):
        exp3918.load_exp3917_evidence(tmp_path)

    _write_exp3917_artifact(tmp_path)
    artifact = json.loads(exp3917_path.read_text(encoding="utf-8"))
    del artifact["energy_per_item_ms"]
    exp3917_path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="missing energy_per_item_ms"):
        exp3918.load_exp3917_evidence(tmp_path)

    artifact["energy_per_item_ms"] = -1.0
    exp3917_path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="must be positive"):
        exp3918.load_exp3917_evidence(tmp_path)


def test_scenario_verify_3918_complete_artifact_uses_cached_scores(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3918: runnable cached evidence writes bare heldout metrics."""

    upstream_path = _write_exp3917_artifact(tmp_path)
    output_path = tmp_path / "results" / "experiment_3918_cascade_router_prototype.json"
    artifact = exp3918.run_experiment(
        exp3918.CascadeConfig(
            repo_root=tmp_path,
            output_path=output_path,
            started_at=10.0,
            clock=lambda: 11.25,
            random_seed=3918,
        ),
        write=True,
    )

    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    exp3918.validate_artifact(persisted)
    assert artifact == persisted
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["n_calibration"] + artifact["n_heldout"] == len(_artifact_rows())
    assert artifact["inference_substrate"] == exp3918.INFERENCE_SUBSTRATE
    assert artifact["source_artifacts"]["exp3917"]["path"] == upstream_path.relative_to(tmp_path).as_posix()
    assert artifact["no_new_inference"] is True
    assert artifact["frozen_fover_auroc_unchanged"] == 0.9131
    assert not isinstance(artifact["cascade_cost_ratio"], dict)


def test_scenario_verify_3918_missing_upstream_blocks_without_claims(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3918-BLOCKED: missing Exp 3917 writes a blocked artifact."""

    output_path = tmp_path / "results" / "experiment_3918_cascade_router_prototype.json"
    artifact = exp3918.run_experiment(
        exp3918.CascadeConfig(
            repo_root=tmp_path,
            output_path=output_path,
            started_at=4.0,
            clock=lambda: 4.5,
        ),
        write=True,
    )

    exp3918.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_upstream_efficiency_missing"
    assert artifact["cascade_auroc"] is None
    assert artifact["cascade_cost_ratio"] is None
    assert artifact["n_calibration"] == 0
    assert artifact["n_heldout"] == 0
    assert output_path.is_file()


def test_req_verify_3918_precondition_rejects_malformed_cached_rows(tmp_path: Path) -> None:
    """REQ-VERIFY-3918: malformed cached per-item rows are not promoted."""

    _write_exp3917_artifact(tmp_path, rows=[{"gold_error": 1, "energy_score": 0.9}])

    checks, blocked_reason, evidence = exp3918.probe_preconditions(tmp_path)

    assert blocked_reason == "blocked_upstream_efficiency_missing"
    assert evidence is None
    assert checks[0].available is False
    assert "llm_judge_score" in checks[0].detail


def test_req_verify_3918_validate_artifact_rejects_bad_shapes(tmp_path: Path) -> None:
    """REQ-VERIFY-3918: artifact validation enforces terminal and bare fields."""

    artifact = exp3918.build_blocked_artifact(
        config=exp3918.CascadeConfig(repo_root=tmp_path, started_at=0.0, clock=lambda: 1.0),
        reason="blocked_upstream_efficiency_missing",
        preconditions_checked=[
            exp3918.PreconditionCheck("exp3917_cached_scores_ready", False, "missing")
        ],
        started_at=0.0,
    )

    exp3918.validate_artifact(artifact)
    with pytest.raises(ValueError, match="missing required"):
        exp3918.validate_artifact({key: value for key, value in artifact.items() if key != "duration_s"})
    with pytest.raises(ValueError, match="terminal prefix"):
        exp3918.validate_artifact(dict(artifact, honest_verdict="pending"))
    with pytest.raises(ValueError, match="must not be"):
        exp3918.validate_artifact(dict(artifact, cascade_auroc={"value": 1.0}))
    with pytest.raises(ValueError, match="split counts"):
        exp3918.validate_artifact(dict(artifact, n_calibration=1))
    with pytest.raises(ValueError, match="duration_s"):
        exp3918.validate_artifact(dict(artifact, duration_s="1"))
    with pytest.raises(ValueError, match="sha256"):
        exp3918.validate_artifact(dict(artifact, reproducibility_checksum="bad"))

    complete = dict(
        artifact,
        honest_verdict="complete: cascade_router_WINS_gap0.0000_10.00x_cheaper_at_matched_accuracy_escfrac0.1000",
        status="complete: cascade_router_WINS_gap0.0000_10.00x_cheaper_at_matched_accuracy_escfrac0.1000",
        cascade_auroc=1.0,
        pure_llm_auroc=1.0,
        escalation_fraction=0.1,
        cascade_cost_ratio=10.0,
        auroc_gap=0.0,
        band_tuned_on_calibration=0.01,
        n_calibration=10,
        n_heldout=10,
    )
    exp3918.validate_artifact(complete)
    with pytest.raises(ValueError, match="must be a bare float"):
        exp3918.validate_artifact(dict(complete, cascade_cost_ratio="10"))
    with pytest.raises(ValueError, match="bare ints"):
        exp3918.validate_artifact(dict(complete, n_calibration=10.0))
    with pytest.raises(ValueError, match="positive"):
        exp3918.validate_artifact(dict(complete, n_heldout=0))


def test_req_verify_3918_cli_and_script_report_terminal_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-3918: CLI status follows the terminal artifact verdict."""

    assert exp3918.cli_main(["--repo-root", str(tmp_path)]) == 1
    _write_exp3917_artifact(tmp_path)
    assert exp3918.cli_main(["--repo-root", str(tmp_path)]) == 0

    from scripts.experiments import experiment_3918_cascade_router_prototype as script

    monkeypatch.setattr(script, "cli_main", lambda argv: 7)
    assert script.main() == 7
