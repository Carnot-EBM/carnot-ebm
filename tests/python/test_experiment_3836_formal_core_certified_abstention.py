"""Tests for Exp 3836 formal-core certified abstention.

Spec: REQ-SPOE-3836, SCENARIO-SPOE-3836.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from carnot.pipeline import formal_core_certified_abstention_3836 as exp
from carnot.pipeline.risk_coverage_abstention_3718 import AbstentionExample


UPSTREAM = {
    "exp3835": {
        "path": "results/experiment_3835_formal_core_5seed_ci.json",
        "sha256": "sha3835",
        "formal_only_auroc_mean": 0.894662,
    },
    "exp3771": {
        "path": "results/experiment_3771_certified_abstention_operating_point.json",
        "sha256": "sha3771",
        "selected_threshold": 0.733216,
        "coverage_at_operating_point": 0.998218,
    },
}


def _examples(kind: str, *, n: int = 2000) -> list[AbstentionExample]:
    rng = np.random.default_rng(3836)
    examples: list[AbstentionExample] = []
    for idx in range(n):
        label = 1 if idx < n * 0.02 else 0
        if kind == "formal_core_certified_threshold_shipped":
            score = rng.uniform(0.92, 1.0) if label else rng.uniform(0.0, 0.18)
        elif kind == "formal_core_certified_threshold_weak":
            score = 0.5
        else:
            raise ValueError(kind)
        examples.append(
            AbstentionExample(
                label=label,
                energy_score=float(score),
                baseline_score=0.0,
                example_id=f"row-{idx}",
            )
        )
    return examples


def test_formal_core_score_uses_tier0r_and_tier0u_only() -> None:
    """REQ-SPOE-3836: fr11_session_memory is not part of the formal-core scorer."""

    scores = exp.formal_core_scores_from_verifier_scores(
        {
            "tier0r_curry_howard": [1.0, 0.4],
            "tier0u_logical_consistency": [0.5, 0.0],
            "tier0s_arithmetic_gap": [1.0, 1.0],
        }
    )
    assert scores == [0.95, 0.36]

    with pytest.raises(ValueError, match="fr11_session_memory"):
        exp.formal_core_scores_from_verifier_scores(
            {
                "tier0r_curry_howard": [1.0],
                "tier0u_logical_consistency": [0.0],
                "fr11_session_memory": [1.0],
            }
        )
    with pytest.raises(ValueError, match="requires tier0r"):
        exp.formal_core_scores_from_verifier_scores({"tier0r_curry_howard": [1.0]})
    with pytest.raises(ValueError, match="lengths must match"):
        exp.formal_core_scores_from_verifier_scores(
            {
                "tier0r_curry_howard": [1.0, 0.0],
                "tier0u_logical_consistency": [1.0],
            }
        )


def test_build_artifact_formal_core_certified_threshold_shipped() -> None:
    """SCENARIO-SPOE-3836: a strong formal core emits the shipped verdict."""

    artifact = exp.build_artifact_from_examples(
        _examples("formal_core_certified_threshold_shipped"),
        started_s=100.0,
        now_s=104.0,
        tests_run=["pytest 3836"],
        min_examples=200,
        preconditions_checked=[{"resource": "synthetic", "available": True}],
        cited_upstream_artifacts=UPSTREAM,
    )

    exp.validate_artifact(artifact)
    assert artifact["formal_core_certified_threshold"] is not None
    assert artifact["formal_core_certified_coverage_at_risk_0_05"] > 0.90
    assert artifact["formal_core_certified_risk_bound"] <= exp.TARGET_RISK
    assert artifact["coverage_delta_vs_full_ensemble"] == pytest.approx(
        artifact["formal_core_certified_coverage_at_risk_0_05"] - 0.998218,
        abs=1e-6,
    )
    assert artifact["honest_verdict"].startswith(
        "complete: formal_core_certified_abstention_threshold"
    )
    assert artifact["honest_verdict"].endswith("_at_risk_0.05_contamination_free")
    assert "formal_core_certified_threshold" in artifact["field_provenance"]
    assert artifact["duration_s"] == 4.0


def test_build_artifact_formal_core_certified_threshold_weak() -> None:
    """SCENARIO-SPOE-3836: uncertifiable formal scores keep the full ensemble as product."""

    artifact = exp.build_artifact_from_examples(
        _examples("formal_core_certified_threshold_weak"),
        started_s=10.0,
        now_s=11.5,
        min_examples=200,
        preconditions_checked=[{"resource": "synthetic", "available": True}],
        cited_upstream_artifacts=UPSTREAM,
    )

    exp.validate_artifact(artifact)
    assert artifact["formal_core_certified_threshold"] is None
    assert artifact["formal_core_certified_coverage_at_risk_0_05"] == 0.0
    assert artifact["honest_verdict"].startswith(
        "complete: formal_core_certified_abstention_WEAK_coverage0.0"
    )


def test_blocked_precondition_artifact() -> None:
    """REQ-SPOE-3836: failed gates produce blocked artifacts without fabricated metrics."""

    artifact = exp.build_blocked_artifact(
        "blocked_exp3835_missing_or_weak",
        preconditions_checked=[
            {
                "resource": "exp3835_formal_only_auroc_gate",
                "available": False,
                "detail": "missing",
            }
        ],
        started_s=1.0,
        now_s=2.0,
    )

    exp.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_exp3835_missing_or_weak"
    assert artifact["formal_core_certified_threshold"] is None
    assert artifact["formal_core_certified_coverage_at_risk_0_05"] == 0.0
    assert artifact["risk_coverage_curve"] == []


def test_load_formal_core_examples_scores_cached_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-SPOE-3836: cached FoVer rows are scored with the formal-core scalar."""

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "fover_corpus_v4.json").write_text(
        json.dumps(
            [
                {"question_id": "a", "step_text": "ok", "label": "correct"},
                {"question_id": "b", "step_text": "bad", "label": "incorrect"},
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        exp,
        "_score_text_verifiers",
        lambda texts: {
            "tier0r_curry_howard": [0.1, 0.9],
            "tier0u_logical_consistency": [1.0, 0.0],
            "tier0s_arithmetic_gap": [1.0, 1.0],
        },
    )

    examples, status = exp.load_formal_core_examples(tmp_path)

    assert status["n_examples"] == 2
    assert [example.label for example in examples] == [0, 1]
    assert [example.energy_score for example in examples] == [0.19, 0.81]
    assert (
        status["formal_score_definition"]
        == "0.9*tier0r_curry_howard + 0.1*tier0u_logical_consistency"
    )


def test_build_artifact_blocks_when_preconditions_fail(tmp_path: Path) -> None:
    """SCENARIO-SPOE-3836: missing resources block before scoring."""

    artifact = exp.build_artifact(tmp_path, started_s=5.0, now_s=6.0)

    assert artifact["honest_verdict"].startswith("blocked_")
    assert artifact["duration_s"] == 1.0


def test_validate_artifact_rejects_missing_fields() -> None:
    """REQ-SPOE-3836: artifact contract validation rejects incomplete payloads."""

    with pytest.raises(ValueError, match="missing required artifact fields"):
        exp.validate_artifact({})


def test_reproducibility_checksum_ignores_duration() -> None:
    """REQ-SPOE-3836: wall-clock duration is not part of the deterministic checksum."""

    artifact = exp.build_artifact_from_examples(
        _examples("formal_core_certified_threshold_shipped"),
        started_s=100.0,
        now_s=104.0,
        min_examples=200,
        preconditions_checked=[{"resource": "synthetic", "available": True}],
        cited_upstream_artifacts=UPSTREAM,
    )
    changed = dict(artifact, duration_s=999.0)

    assert exp.reproducibility_checksum(artifact) == exp.reproducibility_checksum(changed)


def test_defensive_contract_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-SPOE-3836: defensive guards keep malformed inputs from passing silently."""

    monkeypatch.setattr(
        exp.importlib, "import_module", lambda name: (_ for _ in ()).throw(ImportError(name))
    )
    checks, verdict = exp.check_preconditions(tmp_path)
    assert verdict == "blocked_carnot_verify_import"
    assert checks[0]["available"] is False

    assert exp.load_formal_core_examples(tmp_path)[1]["status"] == "missing"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "fover_corpus_v4.json").write_text("{}", encoding="utf-8")
    assert exp.load_formal_core_examples(tmp_path)[1]["status"] == "blocked"
    (data_dir / "fover_corpus_v4.json").write_text(
        json.dumps([{"label": "correct"}, "skip-me"]),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        exp,
        "_score_text_verifiers",
        lambda texts: {
            "tier0r_curry_howard": [0.0],
            "tier0u_logical_consistency": [0.0],
        },
    )
    examples, _status = exp.load_formal_core_examples(tmp_path)
    assert len(examples) == 1
    (data_dir / "fover_corpus_v4.json").write_text(
        json.dumps(["skip-me", {"label": "correct"}]),
        encoding="utf-8",
    )
    examples, _status = exp.load_formal_core_examples(tmp_path)
    assert examples == []

    strong = exp.build_artifact_from_examples(
        _examples("formal_core_certified_threshold_shipped"),
        min_examples=200,
        preconditions_checked=[{"resource": "synthetic", "available": True}],
        cited_upstream_artifacts=UPSTREAM,
    )
    invalid_coverage = dict(strong, formal_core_certified_coverage_at_risk_0_05=0.9)
    invalid_coverage["honest_verdict"] = (
        "complete: formal_core_certified_abstention_threshold0.1_coverage0.9_at_risk_0.05_contamination_free"
    )
    invalid_payloads = [
        dict(strong, honest_verdict="bad"),
        dict(strong, duration_s=-1),
        dict(strong, field_provenance=[]),
        dict(strong, field_provenance={}),
        dict(strong, conformal_delta=0.1),
        dict(strong, formal_core_certified_coverage_at_risk_0_05=-1),
        dict(strong, n_calibration=1),
        dict(strong, n_test=1),
        dict(strong, formal_core_certified_threshold=None),
        invalid_coverage,
    ]
    weak = exp.build_artifact_from_examples(
        _examples("formal_core_certified_threshold_weak"),
        min_examples=200,
        preconditions_checked=[{"resource": "synthetic", "available": True}],
        cited_upstream_artifacts=UPSTREAM,
    )
    invalid_payloads.append(dict(weak, formal_core_certified_coverage_at_risk_0_05=0.91))
    for payload in invalid_payloads:
        with pytest.raises(ValueError):
            exp.validate_artifact(payload)

    assert exp.risk_coverage_curve([], []) == []
    assert exp.load_upstream_artifacts(tmp_path)["exp3835"]["sha256"] is None
    results_dir = tmp_path / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / "experiment_3835_formal_core_5seed_ci.json").write_text(
        json.dumps({"formal_only_auroc_mean": 0.9}),
        encoding="utf-8",
    )
    (results_dir / "experiment_3771_certified_abstention_operating_point.json").write_text(
        json.dumps({"coverage_at_operating_point": 0.9, "selected_threshold": 0.1}),
        encoding="utf-8",
    )
    assert exp.load_upstream_artifacts(tmp_path)["exp3835"]["sha256"] is not None
    assert exp._repo_path(tmp_path, Path("/abs/path")) == Path("/abs/path")


def test_build_artifact_and_write_success_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-SPOE-3836: top-level builder and writer persist a valid artifact."""

    monkeypatch.setattr(
        exp,
        "check_preconditions",
        lambda root: ([{"resource": "synthetic", "available": True}], None),
    )
    monkeypatch.setattr(
        exp,
        "load_formal_core_examples",
        lambda root: (
            _examples("formal_core_certified_threshold_shipped"),
            {"status": "loaded", "n_examples": 2000},
        ),
    )
    monkeypatch.setattr(exp, "load_upstream_artifacts", lambda root: dict(UPSTREAM))

    artifact = exp.build_artifact(tmp_path, started_s=20.0, now_s=23.0)
    exp.validate_artifact(artifact)
    assert artifact["duration_s"] == 3.0
    assert artifact["corpus_status"]["status"] == "loaded"

    output = exp.write_artifact(tmp_path, output_path=Path("results/out.json"))
    assert output.exists()
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["honest_verdict"].startswith(
        "complete: formal_core_certified_abstention_threshold"
    )


def test_check_preconditions_success(tmp_path: Path) -> None:
    """REQ-SPOE-3836: all declared resources passing returns no blocked verdict."""

    (tmp_path / "data").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "data" / "fover_test_v4.json").write_text("[]", encoding="utf-8")
    (tmp_path / "data" / "fover_corpus_v4.json").write_text("[]", encoding="utf-8")
    (tmp_path / "results" / "experiment_3835_formal_core_5seed_ci.json").write_text(
        json.dumps({"formal_only_auroc_mean": 0.9}),
        encoding="utf-8",
    )
    (tmp_path / "results" / "experiment_3771_certified_abstention_operating_point.json").write_text(
        json.dumps({"coverage_at_operating_point": 0.9}),
        encoding="utf-8",
    )

    checks, verdict = exp.check_preconditions(tmp_path)

    assert verdict is None
    assert all(check["available"] for check in checks)


def test_build_artifact_from_examples_blocks_on_too_few_examples() -> None:
    """REQ-SPOE-3836: insufficient candidate rows block instead of certifying."""

    artifact = exp.build_artifact_from_examples(
        _examples("formal_core_certified_threshold_shipped", n=20),
        min_examples=200,
        preconditions_checked=[{"resource": "synthetic", "available": True}],
        cited_upstream_artifacts=UPSTREAM,
        extra={"note": "covered"},
    )
    exp.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_formal_core_candidate_rows_unavailable"
    assert artifact["note"] == "covered"


def test_clean_examples_and_json_helpers_cover_edge_inputs(tmp_path: Path) -> None:
    """REQ-SPOE-3836: helper functions handle non-finite and non-object inputs."""

    examples = [
        AbstentionExample(label=1, energy_score=float("nan"), baseline_score=0.0),
        AbstentionExample(label=0, energy_score=0.1, baseline_score=0.0),
    ]
    assert len(exp._clean_examples(examples)) == 1

    payload = tmp_path / "payload.json"
    payload.write_text("[]", encoding="utf-8")
    assert exp._read_json_if_exists(payload) == {}
    assert exp._numeric_or_none("0.9") is None
    with pytest.raises(ValueError):
        exp.validate_artifact(
            dict(
                exp.build_blocked_artifact("blocked_x", preconditions_checked=[]), duration_s="bad"
            )
        )

    assert exp._full_ensemble_coverage(None) is None
    assert exp._full_ensemble_coverage({"exp3771": "bad"}) is None
    assert "unavailable" in exp._honest_comparison(0.0, None)
    assert "above" in exp._honest_comparison(1.0, 0.9)
    assert exp._sha256_file(tmp_path / "missing.json") is None
