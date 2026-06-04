"""Tests for the Exp 3810 HTTP/REST abstention repair runner.

Spec: REQ-SPOE-3810, SCENARIO-SPOE-3810.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from carnot.pipeline import certified_abstention_surface as abstention


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts/experiment_3810_abstention_http_rest_surface_v2.py"
VENV_PYTHON = ROOT / ".venv/bin/python"


def _load_exp3810() -> ModuleType:
    spec = importlib.util.spec_from_file_location("experiment_3810", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scenario_spoe_3810_runner_http_e2e_uses_real_endpoint() -> None:
    """SCENARIO-SPOE-3810: runner posts a real cached batch over HTTP."""

    exp3810 = _load_exp3810()
    e2e = exp3810.run_http_e2e(ROOT)

    assert e2e["default_off_ok"] is True
    assert e2e["batch_works"] is True
    assert e2e["abstention_ok"] is True
    rows = {row["candidate_id"]: row for row in e2e["enabled_response"]["scores"]}
    assert rows["exp3810_above_threshold"]["verdict"] == "confident"
    assert rows["exp3810_below_threshold"]["verdict"] == "abstain"


def test_scenario_spoe_3810_run_writes_repair_artifact_with_root_cause(
    tmp_path: Path,
) -> None:
    """SCENARIO-SPOE-3810: complete artifact includes repair audit trail."""

    exp3810 = _load_exp3810()
    output_path = tmp_path / "results/experiment_3810.json"
    proposal_path = tmp_path / "docs/research-notes/abstention-http-rest.md"
    config = abstention.load_certified_abstention_config()
    root_cause = (
        "Exp 3801 used a supposed below-threshold HTTP smoke candidate that scored "
        "above the certified threshold, so the E2E expected the wrong branch."
    )

    artifact = exp3810.run(
        ROOT,
        output_path=output_path,
        doc_proposal_path=proposal_path,
        executable=str(VENV_PYTHON),
        diagnosis_runner=lambda _root, _threshold_path: {
            "root_cause": root_cause,
            "old_below_threshold_score": 0.903847,
            "old_below_threshold_verdict": "confident",
        },
        http_runner=lambda _root, _threshold_path: {
            "default_off_ok": True,
            "batch_works": True,
            "abstention_ok": True,
            "default_response": {"scores": [{"candidate_id": "a"}]},
            "enabled_response": {
                "scores": [
                    {
                        "candidate_id": "a",
                        "verdict": "confident",
                        "score": 0.99,
                        "coverage": config.coverage,
                        "risk": config.certified_risk_bound,
                        "delta": config.delta,
                    },
                    {
                        "candidate_id": "b",
                        "verdict": "abstain",
                        "score": 0.5,
                        "coverage": config.coverage,
                        "risk": config.certified_risk_bound,
                        "delta": config.delta,
                    },
                ]
            },
        },
    )

    assert output_path.exists()
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert proposal_path.exists()
    assert "POST /v1/score-candidates" in proposal_path.read_text(encoding="utf-8")
    assert set(exp3810.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert artifact["honest_verdict"] == exp3810.COMPLETE_VERDICT
    assert artifact["e2e_failure_root_cause"] == root_cause
    assert artifact["http_rest_surface_added"] is True
    assert artifact["batch_post_works"] is True
    assert artifact["default_off_preserves_prior_behavior"] is True
    assert artifact["certified_threshold_used"] == pytest.approx(config.threshold)
    assert artifact["e2e_http_abstention_passed"] is True
    assert artifact["doc_proposal_emitted_not_curated_edit"] is True
    assert artifact["operator_curated_docs_edited"] is False
    assert artifact["tests_assert_real_behavior"] is True
    assert artifact["model_specs"]["verifiers"] == list(exp3810.SCORING_VERIFIERS)
    assert artifact["random_seed"] == exp3810.RANDOM_SEED
    assert len(artifact["reproducibility_checksum"]) == 16


def test_req_spoe_3810_missing_threshold_blocks_without_surface_claims(
    tmp_path: Path,
) -> None:
    """REQ-SPOE-3810: missing Exp 3771 threshold blocks honestly."""

    exp3810 = _load_exp3810()
    output_path = tmp_path / "results/experiment_3810.json"

    artifact = exp3810.run(
        ROOT,
        output_path=output_path,
        certified_threshold_path=tmp_path / "results/missing_exp3771.json",
        executable=str(VENV_PYTHON),
        diagnosis_runner=lambda _root, _threshold_path: {
            "root_cause": "not run because threshold was missing"
        },
        http_runner=lambda _root, _threshold_path: {"abstention_ok": True},
    )

    assert artifact["honest_verdict"] == "blocked_no_certified_threshold"
    assert artifact["e2e_failure_root_cause"] == "not diagnosed: preconditions blocked"
    assert artifact["http_rest_surface_added"] is False
    assert artifact["batch_post_works"] is False
    assert artifact["default_off_preserves_prior_behavior"] is False
    assert artifact["e2e_http_abstention_passed"] is False
    assert artifact["doc_proposal_emitted_not_curated_edit"] is False
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact


def test_req_spoe_3810_failed_http_smoke_keeps_blocked_verdict(
    tmp_path: Path,
) -> None:
    """REQ-SPOE-3810: failed repair E2E does not fabricate completion."""

    exp3810 = _load_exp3810()
    output_path = tmp_path / "results/experiment_3810.json"

    assert exp3810.first_blocker({"package_import": {"available": False}}) == (
        "blocked_package_import"
    )

    artifact = exp3810.run(
        ROOT,
        output_path=output_path,
        doc_proposal_path=tmp_path / "docs/research-notes/failing-http.md",
        executable=str(VENV_PYTHON),
        diagnosis_runner=lambda _root, _threshold_path: {
            "root_cause": "old fixture scored above threshold"
        },
        http_runner=lambda _root, _threshold_path: {
            "default_off_ok": True,
            "batch_works": False,
            "abstention_ok": True,
        },
    )

    assert artifact["honest_verdict"] == "blocked_http_abstention_e2e_failed"
    assert artifact["http_rest_surface_added"] is False
    assert artifact["batch_post_works"] is False
    assert artifact["e2e_http_abstention_passed"] is False
