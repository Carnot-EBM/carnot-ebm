"""Tests for Exp 3779 abstention operating point product wiring.

Spec: REQ-SPOE-3779, SCENARIO-SPOE-3779.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting import abstention_operating_point_product_wiring_3779 as exp3779


ROOT = Path(__file__).resolve().parents[2]
VENV_PYTHON = ROOT / ".venv/bin/python"


def test_scenario_spoe_3779_run_writes_complete_artifact_with_doc_proposal(
    tmp_path: Path,
) -> None:
    """SCENARIO-SPOE-3779: runner emits the product-wiring artifact."""

    output_path = tmp_path / "results/experiment_3779.json"
    proposal_path = tmp_path / "docs/research-notes/abstention-mode-doc-proposal.md"

    artifact = exp3779.run(
        ROOT,
        output_path=output_path,
        doc_proposal_path=proposal_path,
        executable=str(VENV_PYTHON),
        mcp_runner=lambda _root, _exe, _candidates: exp3779.SurfaceCheck(
            name="mcp_protocol",
            passed=True,
            detail="injected protocol pass",
            data={"protocol": "mcp_stdio_json_rpc", "tool_name": "score_candidates"},
        ),
    )

    assert output_path.exists()
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert proposal_path.exists()
    assert "abstention_mode" in proposal_path.read_text(encoding="utf-8")
    assert set(exp3779.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert artifact["honest_verdict"] == exp3779.COMPLETE_VERDICT
    assert artifact["inference_substrate"] == exp3779.INFERENCE_SUBSTRATE
    assert artifact["abstention_mode_wired"] is True
    assert artifact["default_off_preserves_prior_behavior"] is True
    assert artifact["e2e_abstention_passed"] is True
    assert artifact["mcp_surface_confirmed"] is True
    assert artifact["doc_proposal_emitted_not_curated_edit"] is True
    assert artifact["tests_assert_real_behavior"] is True
    assert artifact["model_specs"]["verifiers"] == list(exp3779.SCORING_VERIFIERS)
    assert artifact["model_specs"]["certified_threshold_source"].endswith(
        "results/experiment_3771_certified_abstention_operating_point.json"
    )
    assert artifact["random_seed"] == exp3779.RANDOM_SEED
    assert len(artifact["reproducibility_checksum"]) == 16


def test_req_spoe_3779_missing_threshold_blocks_without_surface_claims(
    tmp_path: Path,
) -> None:
    """REQ-SPOE-3779: absent Exp 3771 threshold blocks honestly."""

    output_path = tmp_path / "results/experiment_3779.json"
    missing_threshold = tmp_path / "results/missing_exp3771.json"
    artifact = exp3779.run(
        ROOT,
        output_path=output_path,
        certified_threshold_path=missing_threshold,
        executable=str(VENV_PYTHON),
        mcp_runner=lambda _root, _exe, _candidates: exp3779.SurfaceCheck(
            name="mcp_protocol",
            passed=True,
            detail="should not be claimed when threshold is absent",
        ),
    )

    assert artifact["honest_verdict"] == "blocked_no_certified_threshold"
    assert artifact["abstention_mode_wired"] is False
    assert artifact["default_off_preserves_prior_behavior"] is False
    assert artifact["e2e_abstention_passed"] is False
    assert artifact["mcp_surface_confirmed"] is False
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
