"""Tests for Exp 3324 capstone .307 path-de-risking readout.

Spec refs: REQ-REPORT-3324, SCENARIO-REPORT-3324.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from carnot.reporting import capstone_v307_3324 as mod


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec/capabilities/research-reporting/spec.md"


def _write_json(root: Path, rel_path: Path, payload: Mapping[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _publication_gate(*, paper_ready: bool = False) -> dict[str, Any]:
    return {
        "paper_ready": paper_ready,
        "unmet_gates": [] if paper_ready else ["G2"],
        "gates": {
            "G1": {"pass": True, "detail": "fixture headline"},
            "G2": {"pass": paper_ready, "detail": "fixture reproducer"},
            "G3": {"pass": True, "detail": "fixture prose"},
            "G4": {"pass": True, "detail": "fixture trace"},
        },
    }


def _exp3322(verdict: str, **extra: Any) -> dict[str, Any]:
    payload = {
        "experiment_id": "exp3322",
        "task_id": "exp3322-energy-descent-vs-autoregressive-premise-v1",
        "energy_descent_vs_autoregressive_premise_v1_ready": True,
        "task_name": "gsm8k_fixture_split",
        "n_problems": 240,
        "ar_baseline_accuracy": 0.61,
        "energy_descent_accuracy": 0.68,
        "accuracy_delta": 0.07,
        "paired_significance": {"p_value": 0.03, "ci95": [0.01, 0.13]},
        "honest_verdict": verdict,
        "random_seed": 3322,
        "reproducibility_checksum": "exp3322-checksum",
    }
    payload.update(extra)
    return payload


def _exp3323(verdict: str, **extra: Any) -> dict[str, Any]:
    payload = {
        "experiment_id": "exp3323",
        "task_id": "exp3323-verifier-ensemble-lambda-min-diversity-audit-v1",
        "verifier_ensemble_lambda_min_diversity_audit_v1_ready": True,
        "k_verifiers": 5,
        "lambda_min_sigma": 0.18,
        "effective_k_participation_ratio": 3.4,
        "pairwise_max_correlation": 0.62,
        "honest_verdict": verdict,
        "random_seed": 3323,
        "reproducibility_checksum": "exp3323-checksum",
    }
    payload.update(extra)
    return payload


def test_req_report_3324_spec_anchor_declares_capstone_schema() -> None:
    """REQ-REPORT-3324: OpenSpec declares the capstone before implementation."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-3324" in spec
    assert "SCENARIO-REPORT-3324" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_report_3324_validated_and_grounded_scales_next_gap(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3324: validated Kona premise plus grounding hold selects scale."""

    _write_json(
        tmp_path,
        mod.KONA_PREMISE_REL_PATH,
        _exp3322("complete: energy_descent_beats_ar_premise_validated"),
    )
    _write_json(
        tmp_path,
        mod.GROUNDING_AUDIT_REL_PATH,
        _exp3323("complete: verifier_ensemble_diversity_sufficient_grounding_holds"),
    )

    artifact = mod.build_artifact(
        tmp_path,
        publication_gate_result=_publication_gate(paper_ready=True),
        started_s=1.0,
        now_s=4.25,
    )
    second = mod.build_artifact(
        tmp_path,
        publication_gate_result=_publication_gate(paper_ready=True),
        started_s=8.0,
        now_s=9.0,
    )
    output = mod.write_artifact(
        tmp_path,
        publication_gate_result=_publication_gate(paper_ready=True),
        started_s=2.0,
        now_s=3.5,
    )
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert mod.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["schema_version"] == mod.SCHEMA_VERSION
    assert artifact["experiment_id"] == "exp3324"
    assert artifact["task_id"] == mod.TASK_ID
    assert artifact["run_date"] == "20260529"
    assert artifact["milestone"] == "2026.05.307"
    assert artifact["capstone_v307_ready"] is True
    assert artifact["kona_premise_outcome"] == "validated"
    assert artifact["grounding_keystone_outcome"] == "holds"
    assert artifact["next_top_gap"] == "scale_substrate_intermediate (exp_NEXT_E 100-300M)"
    assert artifact["paper_ready"] is True
    assert artifact["publication_gate_unmet"] == []
    assert artifact["publication_gate_source"] == "scripts/publication_gate.py --json"
    assert artifact["duration_s"] == pytest.approx(3.25)
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert artifact["reproducibility_checksum"] == second["reproducibility_checksum"]
    assert artifact["source_checksums"][mod.KONA_PREMISE_REL_PATH.as_posix()] == _sha256(
        tmp_path / mod.KONA_PREMISE_REL_PATH
    )
    assert artifact["source_checksums"][mod.GROUNDING_AUDIT_REL_PATH.as_posix()] == _sha256(
        tmp_path / mod.GROUNDING_AUDIT_REL_PATH
    )
    assert saved["duration_s"] == pytest.approx(1.5)
    assert saved["honest_verdict"].startswith("complete:")
    assert "next_top_gap=scale_substrate_intermediate" in saved["honest_verdict"]
    assert artifact["no_new_model_execution"] is True
    assert artifact["no_new_verifier_run"] is True
    assert artifact["ops_status_modified_by_this_task"] is False
    assert artifact["ops_changelog_modified_by_this_task"] is False
    assert artifact["traceability_modified_by_this_task"] is False
    mod.validate_artifact(artifact)


def test_req_report_3324_unsupported_or_at_risk_next_gap_precedence(tmp_path: Path) -> None:
    """REQ-REPORT-3324: unsupported premise and at-risk grounding remain decisive."""

    _write_json(
        tmp_path,
        mod.KONA_PREMISE_REL_PATH,
        _exp3322("complete: energy_descent_below_ar_premise_unsupported_at_scale"),
    )
    _write_json(
        tmp_path,
        mod.GROUNDING_AUDIT_REL_PATH,
        _exp3323("complete: verifier_ensemble_null_space_collapse_confirmed_grounding_at_risk"),
    )

    artifact = mod.build_artifact(
        tmp_path,
        publication_gate_result=_publication_gate(),
        started_s=5.0,
        now_s=6.0,
    )

    assert artifact["capstone_v307_ready"] is True
    assert artifact["kona_premise_outcome"] == "unsupported"
    assert artifact["grounding_keystone_outcome"] == "at_risk"
    assert artifact["next_top_gap"] == "reconsider_foundation_model_endgame"
    assert artifact["paper_ready"] is False
    assert artifact["publication_gate_unmet"] == ["G2"]
    assert "kona=unsupported" in artifact["honest_verdict"]
    mod.validate_artifact(artifact)


def test_req_report_3324_missing_inputs_block_without_fabricating(tmp_path: Path) -> None:
    """REQ-REPORT-3324: absent upstream artifacts become blocked_* outcomes."""

    artifact = mod.build_artifact(
        tmp_path,
        publication_gate_result=_publication_gate(),
        started_s=9.0,
        now_s=8.0,
    )

    assert artifact["capstone_v307_ready"] is False
    assert artifact["kona_premise_outcome"] == "blocked_exp3322_missing"
    assert artifact["grounding_keystone_outcome"] == "blocked_exp3323_missing"
    assert artifact["next_top_gap"] == "complete_phase3_path_derisking_upstreams"
    assert artifact["duration_s"] == 0.0
    assert artifact["source_checksums"] == {}
    assert all(not row["present"] for row in artifact["source_artifacts"])
    assert artifact["honest_verdict"].startswith("complete:")
    mod.validate_artifact(artifact)


def test_req_report_3324_helpers_and_validation_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-3324: helper branches stay explicit and schema validation is strict."""

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")

    assert mod.read_json_object(bad_json) == {}
    assert mod.read_json_object(list_json) == {}
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod.sha256_file(tmp_path / "missing.json") is None
    assert mod._as_mapping([]) == {}
    assert mod._float_value(True) == 0.0
    assert mod._float_value("bad") == 0.0
    assert mod._terminal_prefix_ok("passed: done") is True
    assert mod._terminal_prefix_ok("blocked") is False
    assert mod._normalise_publication_gate({"paper_ready": True, "unmet_gates": ["G4"]})[
        "publication_gate_unmet"
    ] == ["G4"]
    assert mod._normalise_publication_gate({"paper_ready": False, "unmet": ["G1"]})[
        "publication_gate_unmet"
    ] == ["G1"]
    assert mod._normalise_publication_gate({})["publication_gate_unmet"] == [
        "G1",
        "G2",
        "G3",
        "G4",
    ]
    assert (
        mod._kona_premise_outcome(
            _exp3322(
                "complete: fixture",
                energy_descent_accuracy=0.66,
                ar_baseline_accuracy=0.61,
                paired_significance={"p_value": 0.20, "ci95": [-0.01, 0.11]},
            )
        )
        == "viable_not_superior"
    )
    assert (
        mod._kona_premise_outcome(
            _exp3322(
                "complete: fixture",
                energy_descent_accuracy=0.50,
                ar_baseline_accuracy=0.61,
                paired_significance={},
            )
        )
        == "unsupported"
    )
    assert (
        mod._kona_premise_outcome(_exp3322("complete: energy_descent_viable_not_superior_at_scale"))
        == "viable_not_superior"
    )
    assert (
        mod._kona_premise_outcome({"energy_descent_vs_autoregressive_premise_v1_ready": False})
        == "blocked_exp3322_not_ready"
    )
    assert mod._kona_premise_outcome({"honest_verdict": "blocked_cuda_unavailable"}) == (
        "blocked_cuda_unavailable"
    )
    assert mod._kona_premise_outcome({"honest_verdict": "complete: unknown"}) == (
        "blocked_exp3322_indeterminate"
    )
    assert (
        mod._grounding_keystone_outcome(
            _exp3323(
                "complete: fixture", lambda_min_sigma=0.12, effective_k_participation_ratio=3.0
            )
        )
        == "holds"
    )
    assert (
        mod._grounding_keystone_outcome(
            _exp3323(
                "complete: fixture", lambda_min_sigma=0.02, effective_k_participation_ratio=1.4
            )
        )
        == "at_risk"
    )
    assert (
        mod._grounding_keystone_outcome(
            {"verifier_ensemble_lambda_min_diversity_audit_v1_ready": False}
        )
        == "blocked_exp3323_not_ready"
    )
    assert mod._grounding_keystone_outcome({"honest_verdict": "blocked_corpus_missing"}) == (
        "blocked_corpus_missing"
    )
    assert mod._grounding_keystone_outcome({"honest_verdict": "complete: unknown"}) == (
        "blocked_exp3323_indeterminate"
    )
    assert (
        mod._next_top_gap("viable_not_superior", "holds")
        == "prove_energy_descent_superiority_before_scale"
    )
    assert mod._next_top_gap("validated", "at_risk") == "redesign_verifier_grounding_source"
    assert mod._next_top_gap("blocked_x", "holds") == "complete_phase3_path_derisking_upstreams"
    assert mod._next_top_gap("unexpected", "holds") == "complete_phase3_path_derisking_upstreams"

    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"paper_ready": False, "unmet_gates": ["G2", "G4"]}),
            stderr="",
        ),
    )
    assert mod.run_publication_gate(tmp_path)["unmet_gates"] == ["G2", "G4"]

    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="boom"),
    )
    assert mod.run_publication_gate(tmp_path)["unmet_gates"] == ["G1", "G2", "G3", "G4"]

    good = mod.build_artifact(
        tmp_path,
        publication_gate_result=_publication_gate(),
        started_s=1.0,
        now_s=1.0,
    )
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({})
    with pytest.raises(ValueError, match="experiment_id"):
        mod.validate_artifact(good | {"experiment_id": "bad"})
    with pytest.raises(ValueError, match="task_id"):
        mod.validate_artifact(good | {"task_id": "bad"})
    with pytest.raises(ValueError, match="milestone"):
        mod.validate_artifact(good | {"milestone": "bad"})
    with pytest.raises(ValueError, match="random_seed"):
        mod.validate_artifact(good | {"random_seed": 0})
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(good | {"inference_substrate": "live"})
    with pytest.raises(ValueError, match="kona_premise_outcome"):
        mod.validate_artifact(good | {"kona_premise_outcome": "bad"})
    with pytest.raises(ValueError, match="grounding_keystone_outcome"):
        mod.validate_artifact(good | {"grounding_keystone_outcome": "bad"})
    with pytest.raises(ValueError, match="capstone_v307_ready"):
        mod.validate_artifact(good | {"capstone_v307_ready": "false"})
    with pytest.raises(ValueError, match="paper_ready"):
        mod.validate_artifact(good | {"paper_ready": "false"})
    with pytest.raises(ValueError, match="publication_gate_unmet"):
        mod.validate_artifact(good | {"publication_gate_unmet": "G2"})
    with pytest.raises(ValueError, match="next_top_gap"):
        mod.validate_artifact(good | {"next_top_gap": ""})
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(good | {"reproducibility_checksum": ""})
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(good | {"honest_verdict": "blocked"})
    with pytest.raises(ValueError, match="no_push"):
        mod.validate_artifact(good | {"no_push": False})
