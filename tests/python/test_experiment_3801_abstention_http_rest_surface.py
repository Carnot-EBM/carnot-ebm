"""Tests for the Exp 3801 abstention HTTP/REST surface runner.

Spec: REQ-SPOE-3801, SCENARIO-SPOE-3801.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from carnot.pipeline import certified_abstention_surface as abstention
from carnot.pipeline import second_pair_detector as spd


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts/experiment_3801_abstention_http_rest_surface.py"
VENV_PYTHON = ROOT / ".venv/bin/python"


def _load_exp3801() -> ModuleType:
    spec = importlib.util.spec_from_file_location("experiment_3801", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _domain_examples(domain: str = "math", *, n: int = 80) -> list[spd.LabeledDetectorExample]:
    examples: list[spd.LabeledDetectorExample] = []
    for idx in range(n):
        label = 1 if idx < n // 2 else 0
        ensemble = 0.95 - 0.004 * idx if label else 0.05 + 0.001 * (idx - n // 2)
        confidence_error = 0.82 - 0.003 * idx if label else 0.18 + 0.001 * (idx - n // 2)
        examples.append(
            spd.LabeledDetectorExample(
                domain=domain,
                label=label,
                ensemble_energy=ensemble,
                confidence_error=confidence_error,
                example_id=f"{domain}-3801-runner-{idx}",
            )
        )
    return examples


def test_scenario_spoe_3801_runner_http_e2e_uses_real_endpoint() -> None:
    """SCENARIO-SPOE-3801: runner E2E posts to the actual local HTTP endpoint."""

    exp3801 = _load_exp3801()
    e2e = exp3801.run_http_e2e(ROOT, examples=_domain_examples())

    assert e2e["default_off_ok"] is True
    assert e2e["batch_works"] is True
    assert e2e["abstention_ok"] is True
    rows = {row["candidate_id"]: row for row in e2e["enabled_response"]["scores"]}
    assert rows["exp3801_confident_error"]["verdict"] == "confident"
    assert rows["exp3801_uncertain_midpoint"]["verdict"] == "abstain"


def test_scenario_spoe_3801_run_writes_complete_artifact_with_doc_proposal(
    tmp_path: Path,
) -> None:
    """SCENARIO-SPOE-3801: runner emits the HTTP wiring artifact."""

    exp3801 = _load_exp3801()
    output_path = tmp_path / "results/experiment_3801.json"
    proposal_path = tmp_path / "docs/research-notes/abstention-http-rest.md"
    config = abstention.load_certified_abstention_config()

    artifact = exp3801.run(
        ROOT,
        output_path=output_path,
        doc_proposal_path=proposal_path,
        executable=str(VENV_PYTHON),
        http_runner=lambda _root: {
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
    assert set(exp3801.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert artifact["honest_verdict"] == exp3801.COMPLETE_VERDICT
    assert artifact["http_rest_surface_added"] is True
    assert artifact["batch_post_works"] is True
    assert artifact["default_off_preserves_prior_behavior"] is True
    assert artifact["certified_threshold_used"] == pytest.approx(config.threshold)
    assert artifact["e2e_http_abstention_passed"] is True
    assert artifact["no_heavy_new_dependency"] is True
    assert artifact["doc_proposal_emitted_not_curated_edit"] is True
    assert artifact["operator_curated_docs_edited"] is False
    assert artifact["tests_assert_real_behavior"] is True
    assert artifact["model_specs"]["verifiers"] == list(exp3801.SCORING_VERIFIERS)
    assert artifact["model_specs"]["certified_threshold_source"] == config.threshold_source
    assert artifact["random_seed"] == exp3801.RANDOM_SEED
    assert len(artifact["reproducibility_checksum"]) == 16


def test_req_spoe_3801_missing_threshold_blocks_without_surface_claims(
    tmp_path: Path,
) -> None:
    """REQ-SPOE-3801: absent Exp 3771 threshold blocks honestly."""

    exp3801 = _load_exp3801()
    output_path = tmp_path / "results/experiment_3801.json"
    missing_threshold = tmp_path / "results/missing_exp3771.json"

    artifact = exp3801.run(
        ROOT,
        output_path=output_path,
        certified_threshold_path=missing_threshold,
        executable=str(VENV_PYTHON),
        http_runner=lambda _root: {"abstention_ok": True},
    )

    assert artifact["honest_verdict"] == "blocked_no_certified_threshold"
    assert artifact["http_rest_surface_added"] is False
    assert artifact["batch_post_works"] is False
    assert artifact["default_off_preserves_prior_behavior"] is False
    assert artifact["e2e_http_abstention_passed"] is False
    assert artifact["doc_proposal_emitted_not_curated_edit"] is False
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact


def test_req_spoe_3801_artifact_guard_and_e2e_failure_paths(tmp_path: Path) -> None:
    """REQ-SPOE-3801: non-threshold blockers and failed HTTP smoke do not fabricate pass."""

    exp3801 = _load_exp3801()
    output_path = tmp_path / "results/experiment_3801.json"

    assert exp3801.first_blocker({"package_import": {"available": False}}) == (
        "blocked_package_import"
    )

    artifact = exp3801.run(
        ROOT,
        output_path=output_path,
        doc_proposal_path=tmp_path / "docs/research-notes/failing-http.md",
        executable=str(VENV_PYTHON),
        http_runner=lambda _root: {
            "default_off_ok": True,
            "batch_works": False,
            "abstention_ok": True,
        },
    )
    assert artifact["honest_verdict"] == "blocked_http_abstention_e2e_failed"
    assert artifact["http_rest_surface_added"] is False
    assert artifact["batch_post_works"] is False
    assert artifact["e2e_http_abstention_passed"] is False

    original_required = exp3801.REQUIRED_ARTIFACT_FIELDS
    exp3801.REQUIRED_ARTIFACT_FIELDS = (*original_required, "missing_field")
    try:
        with pytest.raises(ValueError, match="missing required artifact fields"):
            exp3801.build_artifact(
                verdict=exp3801.COMPLETE_VERDICT,
                duration_s=1.0,
                threshold={"selected_threshold": 0.733216},
                preconditions={},
                http_e2e={
                    "default_off_ok": True,
                    "batch_works": True,
                    "abstention_ok": True,
                },
                doc_proposal_path=output_path,
                threshold_path=ROOT
                / "results/experiment_3771_certified_abstention_operating_point.json",
                output_path=output_path,
            )
    finally:
        exp3801.REQUIRED_ARTIFACT_FIELDS = original_required
