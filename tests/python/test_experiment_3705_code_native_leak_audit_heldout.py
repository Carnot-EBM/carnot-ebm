"""Tests for Exp 3705 code-native leak audit and held-out replication.

Spec: REQ-CODE-3705, SCENARIO-CODE-3705.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts import experiment_3705_code_native_leak_audit_heldout as exp3705


def _metric(point: float | None, low: float | None, high: float | None, n: int = 120) -> dict[str, Any]:
    return {
        "point": point,
        "ci95": None if low is None or high is None else [low, high],
        "n": 0 if point is None else n,
        "n_positive_errors": 0 if point is None else n // 2,
        "n_negative_correct": 0 if point is None else n // 2,
        "bootstrap_seeds": [3705, 3706, 3707, 3708, 3709],
        "seed_mean_aurocs": [] if point is None else [point] * 5,
    }


def _findings(*, leak: bool, heldout_ge_099: bool = False) -> dict[str, Any]:
    return {
        "in_corpus_construction": {
            "separable_by_construction": leak,
            "label_correlated_metadata": leak,
            "score_gap_separable": leak,
        },
        "contamination_check": {
            "exact_candidate_overlap": 0,
            "task_id_overlap": 0,
            "kernel_mmd_rbf": 0.0,
        },
        "heldout_implausible_perfect_red_flag": heldout_ge_099,
        "verifier_authenticity": {
            "ast_parse_used": True,
            "runtime_execution_trace_used": True,
            "constant_score": False,
            "heuristic_gap_disclosed": True,
        },
    }


def _row(*, label: bool, mutation: str, task_id: str, code: str) -> dict[str, Any]:
    return {
        "candidate_code": code,
        "label": label,
        "source": f"fixture_{mutation}",
        "task_id": task_id,
        "candidate_sha256": f"{task_id}-{mutation}",
        "metadata": {
            "corpus": "FixtureCode",
            "entry_point": "add_one",
            "mutation": mutation,
            "stable_id": task_id,
        },
        "test_outcome": "candidate_passed" if label else "candidate_failed_tests",
    }


def _fixture_rows() -> list[dict[str, Any]]:
    correct = "def add_one(x: int) -> int:\n    return x + 1\n"
    wrong = "def add_one(x: int) -> int:\n    return None\n    return x + 1\n"
    rows: list[dict[str, Any]] = []
    for idx in range(4):
        rows.append(_row(label=True, mutation="canonical", task_id=f"task-{idx}", code=correct))
        rows.append(_row(label=False, mutation="return_none", task_id=f"task-{idx}", code=wrong))
    return rows


@pytest.mark.parametrize(
    (
        "case_name",
        "blocked",
        "heldout_metric",
        "findings",
        "expected_verdict",
        "expected_survives",
        "expected_leak",
    ),
    [
        pytest.param(
            "code_signal_survives_heldout",
            False,
            _metric(0.74, 0.61, 0.86),
            _findings(leak=False),
            exp3705.SURVIVES_VERDICT,
            True,
            False,
            id="code_signal_survives_heldout",
        ),
        pytest.param(
            "code_signal_was_a_leak",
            False,
            _metric(0.993, 0.982, 1.0),
            _findings(leak=True, heldout_ge_099=True),
            exp3705.LEAK_VERDICT,
            False,
            True,
            id="code_signal_was_a_leak",
        ),
        pytest.param(
            "blocked",
            True,
            _metric(None, None, None, n=0),
            {},
            exp3705.BLOCKED_VERDICT,
            False,
            False,
            id="blocked",
        ),
    ],
)
def test_scenario_code_3705_parametrized_honest_outcomes(
    case_name: str,
    blocked: bool,
    heldout_metric: dict[str, Any],
    findings: dict[str, Any],
    expected_verdict: str,
    expected_survives: bool,
    expected_leak: bool,
) -> None:
    """SCENARIO-CODE-3705: outcomes cover survives, leak, and blocked."""

    artifact = exp3705.build_artifact_from_measurements(
        blocked=blocked,
        in_corpus_metric={} if blocked else _metric(1.0, 1.0, 1.0, n=60),
        in_corpus_auroc_diagnosis="" if blocked else "fixture diagnosis",
        heldout_metric=heldout_metric,
        heldout_calibration_brier_ece={} if blocked else {"brier": 0.18, "ece": 0.08},
        heldout_recall_at_fixed_fpr={} if blocked else {"0.10": {"code_native_recall": 0.5}},
        leak_audit_findings=findings,
        n_examples_heldout=0 if blocked else int(heldout_metric["n"]),
        n_examples_in_corpus=0 if blocked else 60,
        adversarial_verify_clean=not blocked,
        started_s=1.0,
        now_s=3.5,
        tests_run=[f"SCENARIO-CODE-3705 {case_name}"],
    )

    exp3705.validate_artifact(artifact)
    assert artifact["honest_verdict"] == expected_verdict
    assert artifact["code_signal_survives_heldout"] is expected_survives
    assert artifact["leak_detected"] is expected_leak
    assert type(artifact["code_signal_survives_heldout"]) is bool
    assert type(artifact["leak_detected"]) is bool
    assert artifact["inference_substrate"] == exp3705.INFERENCE_SUBSTRATE
    assert "GGUF" not in json.dumps(artifact)
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["tests_run"] == [f"SCENARIO-CODE-3705 {case_name}"]
    assert set(exp3705.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)


def test_req_code_3705_survival_gate_rejects_implausible_perfect_and_chance_ci() -> None:
    """REQ-CODE-3705: held-out >=0.99 and chance-touching CIs fail closed."""

    assert exp3705.code_signal_survives_heldout(_metric(0.74, 0.61, 0.86)) is True
    assert exp3705.code_signal_survives_heldout(_metric(0.99, 0.90, 1.0)) is False
    assert exp3705.code_signal_survives_heldout(_metric(1.0, 1.0, 1.0)) is False
    assert exp3705.code_signal_survives_heldout(_metric(0.74, 0.50, 0.86)) is False
    assert exp3705.code_signal_survives_heldout(_metric(0.49, 0.31, 0.64)) is False
    assert exp3705.code_signal_survives_heldout({}) is False


def test_req_code_3705_audit_detects_exp3658_label_correlated_separability() -> None:
    """REQ-CODE-3705: audit detects return-none construction and score gap."""

    audit = exp3705.audit_exp3658_corpus(_fixture_rows(), random_seed=7)

    assert audit["in_corpus_construction"]["separable_by_construction"] is True
    assert audit["in_corpus_construction"]["label_correlated_metadata"] is True
    assert audit["in_corpus_construction"]["score_gap_separable"] is True
    assert audit["in_corpus_construction"]["metadata_label_correlations"]["mutation"]["purity"] == 1.0
    assert audit["contamination_check"]["exact_candidate_overlap"] == 0
    assert audit["verifier_authenticity"]["ast_parse_used"] is True
    assert audit["verifier_authenticity"]["runtime_execution_trace_used"] is True
    assert "return_none" in exp3705.diagnose_in_corpus_auroc(audit)


def test_req_code_3705_build_artifact_from_fixture_rows(tmp_path: Path) -> None:
    """REQ-CODE-3705: row scoring builds required held-out audit fields."""

    rows = _fixture_rows()
    artifact = exp3705.build_artifact(
        tmp_path,
        in_corpus_rows=rows,
        heldout_rows=rows,
        min_heldout_examples=8,
        started_s=0.0,
        now_s=2.0,
        seeds=[3705, 3706, 3707, 3708, 3709],
        n_bootstrap=4,
        tests_run=["REQ-CODE-3705 fixture build"],
    )

    exp3705.validate_artifact(artifact)
    assert artifact["n_examples_heldout"] == 8
    assert artifact["n_seeds"] == 5
    assert artifact["heldout_code_auroc"] == 1.0
    assert artifact["code_signal_survives_heldout"] is False
    assert artifact["leak_detected"] is True
    assert artifact["heldout_code_auroc_ci"] == [1.0, 1.0]
    assert artifact["heldout_calibration_brier_ece"]["brier"] >= 0.0
    assert artifact["acceptance_gate"]["passed"] is True


def test_req_code_3705_validation_and_write_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-CODE-3705: artifact writing validates bare bools and adversarial pass."""

    artifact = exp3705.build_artifact_from_measurements(
        blocked=False,
        in_corpus_metric=_metric(1.0, 1.0, 1.0, n=60),
        in_corpus_auroc_diagnosis="fixture diagnosis",
        heldout_metric=_metric(0.74, 0.61, 0.86),
        heldout_calibration_brier_ece={"brier": 0.18, "ece": 0.08},
        heldout_recall_at_fixed_fpr={"0.10": {"code_native_recall": 0.5}},
        leak_audit_findings=_findings(leak=False),
        n_examples_heldout=120,
        n_examples_in_corpus=60,
        adversarial_verify_clean=True,
        started_s=0.0,
        now_s=1.0,
    )

    output = exp3705.write_artifact_from_measurements(
        tmp_path,
        output_path="results/exp3705.json",
        artifact=artifact,
    )
    saved = json.loads(output.read_text(encoding="utf-8"))

    exp3705.validate_artifact(saved)
    assert saved["acceptance_gate"]["passed"] is True

    monkeypatch.setattr(exp3705, "build_artifact", lambda *args, **kwargs: dict(artifact))
    monkeypatch.setattr(exp3705, "run_adversarial_verify_report", lambda path: {"flags": []})
    output = exp3705.write_artifact(tmp_path, output_path="results/write-exp3705.json")
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["adversarial_verify_clean"] is True
    assert written["adversarial_verify_report"] == {"flag_count": 0, "flags": []}

    for field, value, message in [
        ("code_signal_survives_heldout", {"value": True}, "code_signal_survives_heldout"),
        ("leak_detected", 1, "leak_detected"),
        ("adversarial_verify_clean", "true", "adversarial_verify_clean"),
        ("honest_verdict", "complete: unexpected", "honest_verdict"),
        ("duration_s", -1.0, "duration_s"),
    ]:
        broken = dict(saved)
        broken[field] = value
        with pytest.raises(ValueError, match=message):
            exp3705.validate_artifact(broken)

    missing = dict(saved)
    missing.pop("heldout_code_auroc")
    with pytest.raises(ValueError, match="missing required"):
        exp3705.validate_artifact(missing)


def test_req_code_3705_blocked_when_no_heldout_corpus(tmp_path: Path) -> None:
    """REQ-CODE-3705: no distinct held-out corpus yields blocked verdict."""

    artifact = exp3705.build_artifact(
        tmp_path,
        in_corpus_rows=_fixture_rows(),
        heldout_rows=[],
        started_s=0.0,
        now_s=1.0,
    )

    assert artifact["honest_verdict"] == exp3705.BLOCKED_VERDICT
    assert artifact["code_signal_survives_heldout"] is False
    assert artifact["leak_detected"] is False
    assert artifact["n_examples_heldout"] == 0
    assert artifact["acceptance_gate"]["passed"] is False


def test_req_code_3705_helper_edges_and_real_adversarial_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CODE-3705: helper branches fail closed and verifier loader works."""

    artifact = exp3705.build_artifact_from_measurements(
        blocked=False,
        in_corpus_metric=_metric(1.0, 1.0, 1.0, n=60),
        in_corpus_auroc_diagnosis="fixture diagnosis",
        heldout_metric=_metric(0.74, 0.61, 0.86),
        heldout_calibration_brier_ece={"brier": 0.18, "ece": 0.08},
        heldout_recall_at_fixed_fpr={"0.10": {"code_native_recall": 0.5}},
        leak_audit_findings=_findings(leak=False),
        n_examples_heldout=120,
        n_examples_in_corpus=60,
        adversarial_verify_clean=True,
        started_s=0.0,
        now_s=1.0,
    )
    output = exp3705.write_artifact_from_measurements(
        tmp_path,
        output_path="results/real-adversarial-exp3705.json",
        artifact=artifact,
    )

    assert "flags" in exp3705.run_adversarial_verify_report(output)
    saved_loader = exp3705.importlib.util.spec_from_file_location
    monkeypatch.setattr(exp3705.importlib.util, "spec_from_file_location", lambda *args: None)
    with pytest.raises(ImportError, match="adversarial_verify"):
        exp3705.run_adversarial_verify_report(output)
    monkeypatch.setattr(exp3705.importlib.util, "spec_from_file_location", saved_loader)

    assert exp3705.diagnose_in_corpus_auroc({}) == "blocked_no_in_corpus_diagnosis"
    assert (
        exp3705.diagnose_in_corpus_auroc(
            {"in_corpus_construction": {"separable_by_construction": False}}
        )
        == "No deterministic in-corpus separation found by the Exp3705 audit."
    )
    assert exp3705.leak_detected_from_findings(_findings(leak=True), _metric(0.74, 0.61, 0.86))
    assert exp3705.score_gap_summary([1], [0.2])["score_gap_separable"] is False
    assert exp3705.mean_features_by_label([], []) == {"error": {}, "correct": {}}
    assert exp3705.feature_matrix([]).shape == (0, 0)
    assert exp3705.rbf_mmd(np.zeros((0, 0)), np.zeros((1, 0))) == 0.0
    assert exp3705.adversarial_report_is_clean({"flags": 3}) is False
    assert exp3705._round(float("inf")) == float("inf")

    bad_substrate = dict(artifact, inference_substrate="live_llm_inference")
    with pytest.raises(ValueError, match="inference_substrate"):
        exp3705.validate_artifact(bad_substrate)
    bad_n = dict(artifact, n_examples_heldout="120")
    with pytest.raises(ValueError, match="n_examples_heldout"):
        exp3705.validate_artifact(bad_n)

    missing_path = tmp_path / "missing.jsonl"
    assert exp3705.read_jsonl(missing_path) == []
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text(json.dumps(_fixture_rows()[0]) + "\n", encoding="utf-8")
    assert exp3705.read_jsonl(rows_path)[0]["task_id"] == "task-0"
    loaded, status = exp3705._rows_or_default(tmp_path, None, Path("rows.jsonl"))
    assert loaded[0]["task_id"] == "task-0"
    assert status["source"] == "cached_jsonl"
