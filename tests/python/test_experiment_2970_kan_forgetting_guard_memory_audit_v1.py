"""Tests for Exp 2970 KAN constraint-memory forgetting guard audit.

Spec: REQ-LEARN-2970,
      SCENARIO-LEARN-2970,
      SCENARIO-LEARN-2970-BLOCKED.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import fr11_kan_forgetting_guard_memory_audit_v1 as exp


REQUIRED_FIELDS = {
    "honest_verdict",
    "kan_forgetting_guard_ready",
    "source_artifacts",
    "policies_compared",
    "current_domain_utility",
    "old_domain_utility",
    "forgetting_delta_by_policy",
    "selected_policy",
    "high_dimensional_claim_allowed",
    "hardware_cost_fields",
    "no_synthesis_claim",
    "no_analog_claim",
    "files_changed",
    "inference_substrate",
    "duration_s",
}


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_ready_inputs(root: Path) -> None:
    _write_json(
        root,
        exp.EXP2969_REL_PATH,
        {
            "honest_verdict": "complete: non_tautological_self_learning_ready",
            "continuous_self_learning_task": True,
            "non_tautological_self_learning_ready": True,
            "forgetting_guard_passed": True,
            "inference_substrate": "aggregation_from_upstream_artifacts",
        },
    )
    _write_json(
        root,
        exp.EXP2933_REL_PATH,
        {
            "honest_verdict": "complete: kan_rbf_importance_self_learning_passed",
            "kan_cl_self_learning_ready": True,
            "non_forgetting_passed": True,
            "inference_substrate": "local_training_simulation",
        },
    )
    _write_json(
        root,
        exp.EXP2893_REL_PATH,
        {
            "honest_verdict": "complete: tiny KAN PWA/MILP complexity accounting ready",
            "complexity_metrics": {
                "rm_count": 2,
                "bop_count": 96,
                "nabs_count": 4,
                "memory_table_entries": 8,
                "pwa_regions": 4,
                "milp_constraints": 27,
            },
            "hardware_execution_claim_made": False,
            "analog_kan_claim_made": False,
        },
    )


def _config(tmp_path: Path) -> exp.ExperimentConfig:
    return exp.ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp.OUTPUT_FILENAME,
        started_at=10.0,
        clock=lambda: 14.5,
    )


def test_req_learn_2970_spec_anchor_exists() -> None:
    """REQ-LEARN-2970: OpenSpec declares the KAN forgetting-guard audit."""

    spec = (exp.REPO_ROOT / "openspec/capabilities/self-learning/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-LEARN-2970" in spec
    assert "SCENARIO-LEARN-2970" in spec
    assert "SCENARIO-LEARN-2970-BLOCKED" in spec
    assert exp.OUTPUT_FILENAME in spec
    assert 'inference_substrate="deterministic_wiring"' in spec


def test_scenario_learn_2970_policy_comparison_exposes_forgetting() -> None:
    """SCENARIO-LEARN-2970: eager updating learns current constraints but forgets."""

    comparison = exp.build_policy_comparison()
    policies = {row["policy_name"]: row for row in comparison}

    assert set(policies) == {
        "frozen",
        "eager_update",
        "per_knot_importance_update",
        "adapter_style_update",
    }
    assert policies["frozen"]["current_domain_utility"] == pytest.approx(0.0)
    assert policies["frozen"]["old_domain_utility"] == pytest.approx(1.0)
    assert policies["frozen"]["forgetting_delta"] == pytest.approx(0.0)

    assert policies["eager_update"]["current_domain_utility"] == pytest.approx(1.0)
    assert policies["eager_update"]["old_domain_utility"] == pytest.approx(0.25)
    assert policies["eager_update"]["forgetting_delta"] == pytest.approx(0.75)
    assert policies["eager_update"]["forgetting_guard_passed"] is False

    assert policies["per_knot_importance_update"]["current_domain_utility"] == pytest.approx(1.0)
    assert policies["per_knot_importance_update"]["old_domain_utility"] == pytest.approx(1.0)
    assert policies["per_knot_importance_update"]["forgetting_guard_passed"] is True
    assert policies["adapter_style_update"]["old_domain_utility"] == pytest.approx(1.0)

    assert exp.select_policy(comparison) == "per_knot_importance_update"
    assert exp.select_policy(()) == "none"


def test_scenario_learn_2970_writes_ready_artifact_with_no_hardware_claims(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-2970: guarded per-knot memory produces the required artifact."""

    _write_ready_inputs(tmp_path)

    artifact = exp.write_artifact(_config(tmp_path))
    saved = json.loads((tmp_path / "results" / exp.OUTPUT_FILENAME).read_text(encoding="utf-8"))

    assert saved == artifact
    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["honest_verdict"] == "complete: kan_forgetting_guard_ready"
    assert artifact["kan_forgetting_guard_ready"] is True
    assert artifact["selected_policy"] == "per_knot_importance_update"
    assert artifact["high_dimensional_claim_allowed"] is False
    assert artifact["no_synthesis_claim"] is True
    assert artifact["no_analog_claim"] is True
    assert artifact["inference_substrate"] == "deterministic_wiring"
    assert artifact["duration_s"] == pytest.approx(4.5)
    assert (
        "results/experiment_2970_kan_forgetting_guard_memory_audit_v1.json"
        in artifact["files_changed"]
    )

    assert artifact["current_domain_utility"]["per_knot_importance_update"] == pytest.approx(1.0)
    assert artifact["old_domain_utility"]["eager_update"] == pytest.approx(0.25)
    assert artifact["forgetting_delta_by_policy"]["eager_update"] == pytest.approx(0.75)
    assert artifact["forgetting_delta_by_policy"]["per_knot_importance_update"] == pytest.approx(
        0.0
    )

    hardware = artifact["hardware_cost_fields"]
    assert hardware["derivable"] is True
    assert hardware["rm_count"] == 2
    assert hardware["bop_count"] == 96
    assert hardware["nabs_count"] == 4
    assert hardware["no_synthesis_claim"] is True
    assert hardware["no_analog_claim"] is True

    source_by_id = {source["experiment_id"]: source for source in artifact["source_artifacts"]}
    assert source_by_id["exp2969"]["present"] is True
    assert source_by_id["exp2969"]["required"] is True
    assert source_by_id["exp2969"]["sha256"] == _sha256(tmp_path / exp.EXP2969_REL_PATH)
    assert source_by_id["exp2933"]["present"] is True
    assert source_by_id["exp2893"]["present"] is True


def test_scenario_learn_2970_blocked_artifacts_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-LEARN-2970-BLOCKED: missing readiness or imports writes blocked JSON."""

    missing = exp.build_artifact(_config(tmp_path))
    assert missing["honest_verdict"] == "blocked_missing_exp2969_ready_artifact"
    assert missing["kan_forgetting_guard_ready"] is False
    assert missing["selected_policy"] == "none"
    assert missing["policies_compared"] == []
    assert missing["hardware_cost_fields"]["derivable"] is False
    assert REQUIRED_FIELDS <= set(missing)

    _write_json(
        tmp_path,
        exp.EXP2969_REL_PATH,
        {"non_tautological_self_learning_ready": False},
    )
    not_ready = exp.build_artifact(_config(tmp_path))
    assert not_ready["honest_verdict"] == "blocked_exp2969_not_ready"
    assert not_ready["high_dimensional_claim_allowed"] is False

    _write_json(
        tmp_path,
        exp.EXP2969_REL_PATH,
        {"non_tautological_self_learning_ready": True},
    )

    def _raise_import_error() -> Any:
        raise ImportError("fixture missing KAN helpers")

    monkeypatch.setattr(exp, "KAN_HELPERS_IMPORTER", _raise_import_error)
    missing_import = exp.build_artifact(_config(tmp_path))
    assert missing_import["honest_verdict"].startswith("blocked_missing_kan_import")
    assert "fixture missing KAN helpers" in missing_import["blockers"]

    malformed = tmp_path / exp.EXP2969_REL_PATH
    malformed.write_text("{", encoding="utf-8")
    assert exp.read_json_object(malformed) == {}
    malformed.write_text("[]", encoding="utf-8")
    assert exp.read_json_object(malformed) == {}


def test_req_learn_2970_validation_and_cost_defense(tmp_path: Path) -> None:
    """REQ-LEARN-2970-5: schema and hardware-claim drift are rejected."""

    _write_ready_inputs(tmp_path)
    artifact = exp.build_artifact(_config(tmp_path))
    assert exp.validate_artifact(artifact) == artifact

    incomplete = dict(artifact)
    incomplete.pop("old_domain_utility")
    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact(incomplete)

    bad_claim = dict(artifact, no_synthesis_claim=False)
    with pytest.raises(ValueError, match="claim boundary"):
        exp.validate_artifact(bad_claim)

    high_dim = dict(artifact, high_dimensional_claim_allowed=True)
    with pytest.raises(ValueError, match="high-dimensional"):
        exp.validate_artifact(high_dim)

    bad_cost_root = tmp_path / "bad-cost"
    _write_json(
        bad_cost_root,
        exp.EXP2893_REL_PATH,
        {"complexity_metrics": {"rm_count": 2, "bop_count": 96}},
    )
    cost = exp.hardware_cost_fields(bad_cost_root)
    assert cost["derivable"] is False
    assert cost["rm_count"] is None
    assert cost["no_synthesis_claim"] is True

    minimal_cost_root = tmp_path / "minimal-cost"
    _write_json(
        minimal_cost_root,
        exp.EXP2893_REL_PATH,
        {"complexity_metrics": {"rm_count": 2, "bop_count": 96, "nabs_count": 4}},
    )
    minimal_cost = exp.hardware_cost_fields(minimal_cost_root)
    assert minimal_cost["derivable"] is True
    assert minimal_cost["memory_table_entries"] is None
