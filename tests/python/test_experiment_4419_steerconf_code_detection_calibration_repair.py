"""Tests for Exp 4419 SteerConf detector-calibration repair.

Spec refs: REQ-VERIFY-4419, SCENARIO-VERIFY-4419.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_4419_steerconf_code_detection_calibration_repair as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _rows(
    domain: str,
    n: int,
    *,
    high: float = 0.82,
    low: float = 0.18,
) -> list[mod.ScoredCandidate]:
    return [
        mod.ScoredCandidate(
            domain=domain,
            task_id=f"{domain}/task/{idx // 2}",
            candidate_id=f"{domain}:{idx}",
            is_correct=idx % 2 == 0,
            verifier_score=high if idx % 2 == 0 else low,
            valid_output=True,
            source=f"{domain}.fixture",
            semantic_key=f"answer-{idx}",
        )
        for idx in range(n)
    ]


def test_req_verify_4419_spec_declares_steerconf_contract() -> None:
    """REQ-VERIFY-4419: OpenSpec declares the SteerConf repair contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-4419",
        "SCENARIO-VERIFY-4419",
        "experiment_4419_steerconf_code_detection_calibration_repair.json",
        "blocked_cached_pools_unavailable",
        "blocked_no_steering_signal_path",
        "steered_confidence_added_auroc",
        "verifier_is_oracle=false",
    ):
        assert marker in spec


def test_req_verify_4419_steering_features_are_cached_and_label_independent() -> None:
    """REQ-VERIFY-4419: steering probes use cached scores, not correctness labels."""

    rows = [
        mod.ScoredCandidate("code_humaneval", "t0", "a", True, 0.8, semantic_key="a"),
        mod.ScoredCandidate("code_humaneval", "t0", "b", False, 0.2, semantic_key="b"),
        mod.ScoredCandidate("code_humaneval", "t1", "c", True, 0.55, semantic_key="c"),
        mod.ScoredCandidate("code_humaneval", "t1", "d", False, 0.45, semantic_key="d"),
    ]
    label_flipped = [
        mod.ScoredCandidate(
            row.domain,
            row.task_id,
            row.candidate_id,
            not row.is_correct,
            row.verifier_score,
            row.valid_output,
            row.source,
            row.semantic_key,
        )
        for row in rows
    ]

    steered = mod.derive_steered_confidence_features(rows)
    flipped = mod.derive_steered_confidence_features(label_flipped)

    assert steered.available is True
    assert steered.feature_names == mod.STEERED_FEATURE_NAMES
    assert [item.feature_vector for item in steered.rows] == [
        item.feature_vector for item in flipped.rows
    ]
    for item in steered.rows:
        assert item.conservative_confidence <= item.verifier_score <= item.optimistic_confidence
        assert item.confidence_consistency == pytest.approx(
            1.0 - item.steer_width,
            abs=1e-9,
        )


def test_scenario_verify_4419_complete_artifact_reports_required_domain_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-4419: powered pools emit SteerConf calibration fields."""

    source = tmp_path / "cached_pool.json"
    _write_json(source, {"ok": True})
    artifact_path = tmp_path / "results" / "experiment_4419.json"
    gaps_path = tmp_path / "ops" / "verifier_gaps.md"
    rows_by_domain = {
        "fover": _rows("fover", 8),
        "gap4_arc": _rows("gap4_arc", 8),
        "code_humaneval": _rows("code_humaneval", 8),
        "gsm8k": _rows("gsm8k", 8),
    }
    monkeypatch.setattr(
        mod,
        "load_raw_domain_rows",
        lambda _cfg: (
            rows_by_domain,
            [
                mod.pool_record(domain, [source], len(rows))
                for domain, rows in rows_by_domain.items()
            ],
            [],
            [source],
        ),
    )

    artifact = mod.run_experiment(
        mod.ExperimentConfig(
            artifact_path=artifact_path,
            verifier_gaps_path=gaps_path,
            min_powered_n=4,
            bootstrap_resamples=80,
            random_control_replicates=8,
            calibration_steps=80,
            started_at=10.0,
            clock=lambda: 12.0,
        ),
        adversarial_verify_runner=lambda _path: {"returncode": 0, "flags": []},
        write=True,
    )

    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["detection_calibrated_multi_domain"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["preconditions_checked"][0]["resource"].endswith("_proper_pool")
    assert artifact["preconditions_checked"][-2]["resource"] == "steering_signal_path"
    assert artifact["steering_feature_summary"]["feature_names"] == list(
        mod.STEERED_FEATURE_NAMES
    )
    for result in artifact["detection_by_domain"]:
        assert set(
            (
                "domain",
                "detection_auroc",
                "auroc_ci95",
                "ece_uncalibrated",
                "ece_lodo_calibrated",
                "risk_coverage",
                "random_score_control",
                "steered_confidence_added_auroc",
                "n",
            )
        ).issubset(result)
    assert artifact["adversarial_verify"]["returncode"] == 0
    assert mod.artifact_schema_errors(artifact) == []
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact


def test_scenario_verify_4419_blocks_when_cached_pools_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-4419: missing powered pools stop before scoring."""

    source = tmp_path / "cached_pool.json"
    _write_json(source, {"ok": True})
    monkeypatch.setattr(
        mod,
        "load_raw_domain_rows",
        lambda _cfg: (
            {"fover": _rows("fover", 4), "gap4_arc": _rows("gap4_arc", 4)},
            [
                mod.pool_record("fover", [source], 4),
                mod.pool_record("gap4_arc", [source], 4),
            ],
            [{"domain": "code_humaneval", "reason": "missing"}],
            [source],
        ),
    )

    artifact = mod.run_experiment(
        mod.ExperimentConfig(
            artifact_path=tmp_path / "results" / "experiment_4419.json",
            min_powered_n=5,
            started_at=1.0,
            clock=lambda: 2.0,
        ),
        write=True,
    )

    assert artifact["honest_verdict"] == "blocked_cached_pools_unavailable"
    assert artifact["detection_calibrated_multi_domain"] is False
    assert artifact["detection_by_domain"] == []
    assert artifact["verifier_is_oracle"] is False
    assert artifact["adversarial_verify"]["status"] == "not_run_blocked_preconditions"
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_verify_4419_blocks_when_no_steering_signal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-4419: degenerate cached scores are not called steering."""

    source = tmp_path / "cached_pool.json"
    _write_json(source, {"ok": True})
    rows_by_domain = {
        "fover": _rows("fover", 8, high=0.5, low=0.5),
        "gap4_arc": _rows("gap4_arc", 8, high=0.5, low=0.5),
        "code_humaneval": _rows("code_humaneval", 8, high=0.5, low=0.5),
        "gsm8k": _rows("gsm8k", 8, high=0.5, low=0.5),
    }
    monkeypatch.setattr(
        mod,
        "load_raw_domain_rows",
        lambda _cfg: (
            rows_by_domain,
            [
                mod.pool_record(domain, [source], len(rows))
                for domain, rows in rows_by_domain.items()
            ],
            [],
            [source],
        ),
    )

    artifact = mod.run_experiment(
        mod.ExperimentConfig(
            artifact_path=tmp_path / "results" / "experiment_4419.json",
            min_powered_n=4,
            started_at=1.0,
            clock=lambda: 2.0,
        ),
        write=True,
    )

    assert artifact["honest_verdict"] == "blocked_no_steering_signal_path"
    assert artifact["detection_by_domain"] == []
    assert artifact["steering_feature_summary"]["available"] is False
    assert mod.artifact_schema_errors(artifact) == []


def test_req_verify_4419_schema_errors_and_gap_entry() -> None:
    """REQ-VERIFY-4419: schema and missing-verifier gap guards fail closed."""

    bad = {
        "detection_calibrated_multi_domain": "true",
        "verifier_is_oracle": True,
        "detection_by_domain": {},
        "preconditions_checked": {},
        "inference_substrate": "wrong",
    }

    errors = mod.artifact_schema_errors(bad)

    assert "missing:honest_verdict" in errors
    assert "invalid:detection_calibrated_multi_domain" in errors
    assert "invalid:verifier_is_oracle" in errors
    assert "invalid:detection_by_domain" in errors
    gap = mod.missing_gap_entries(
        [{"domain": "code_humaneval", "auroc_ci95": [0.45, 0.55], "n": 539}]
    )
    assert gap[0]["gap_id"] == "GAP-4419-CODE-HUMANEVAL-STEERCONF-DETECTOR-CHANCE"
