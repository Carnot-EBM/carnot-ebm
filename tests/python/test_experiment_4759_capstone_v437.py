"""Tests for REQ-CAPSTONE-4759 / SCENARIO-CAPSTONE-4759."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_4759_capstone_v437 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _paper_gate(ready: bool = True) -> dict[str, Any]:
    return {
        "paper_ready": ready,
        "gates": {
            "G1": {"pass": ready, "detail": "headline measured"},
            "G2": {"pass": ready, "detail": "independent reproducer"},
            "G3": {"pass": ready, "detail": "narrowing clean"},
            "G4": {"pass": ready, "detail": "traceable artifact"},
        },
        "unmet_gates": [] if ready else ["G2"],
    }


def _a1(*, flagged: bool = True, structured_accuracy: float = 0.5) -> dict[str, Any]:
    return {
        "experiment": "experiment_4749_structured_engine_vs_freeform",
        "experiment_id": 4749,
        "flagged_adversarial": flagged,
        "honest_verdict": "complete_structured_engine_no_improvement_null",
        "structured_heldout_accuracy": structured_accuracy,
        "freeform_heldout_accuracy": 0.12,
        "l2_proposer_failed": True,
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "verifier_is_oracle": False,
    }


def _a2(
    *,
    satisfiable: bool = False,
    plan: bool = False,
    reproduced: bool = False,
) -> dict[str, Any]:
    return {
        "experiment": "experiment_4750_structural_alignment_detector_fix",
        "experiment_id": 4750,
        "honest_verdict": "complete_detector_fixed_but_no_bank_no_reachable_plan",
        "goal_predicate_satisfiable": satisfiable,
        "l2_plan_reaches_goal": plan,
        "offline_reproduced": reproduced,
        "reproduced_levels": 2 if reproduced else 1,
        "verifier_is_oracle": False,
    }


def _a3(*, banked: bool = True) -> dict[str, Any]:
    return {
        "experiment": "experiment_4751_levelup_selfplay",
        "experiment_id": 4751,
        "honest_verdict": "success: sk48_L2_offline_reproduced"
        if banked
        else "complete: sk48_delta_identified_no_bank",
        "target_game": "sk48",
        "reached_level": 2 if banked else 1,
        "new_levels_banked": 1 if banked else 0,
        "offline_reproduced": banked,
        "reproduced_levels": 2 if banked else 1,
        "reproducible_total_levels_before": 64,
        "reproducible_total_levels": 65 if banked else 64,
        "verifier_is_oracle": False,
    }


def _a4(*, flagged: bool = True, ready: bool = False) -> dict[str, Any]:
    return {
        "experiment": "experiment_4752_held_out_first_win_readiness",
        "experiment_id": 4752,
        "flagged_adversarial": flagged,
        "honest_verdict": "complete: held_out_first_win_flat_no_leaderboard_change",
        "first_win_rate_integrated": 0.04,
        "submission_package_ready": ready,
        "ready_for_operator_submit": ready,
        "verifier_is_oracle": False,
    }


def _artifacts(
    *,
    a1: dict[str, Any] | None = None,
    a2: dict[str, Any] | None = None,
    a3: dict[str, Any] | None = None,
    a4: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    return {
        "A1": a1 or _a1(),
        "A2": a2 or _a2(),
        "A3": a3 or _a3(),
        "A4": a4 or _a4(),
    }


def _hashes() -> dict[str, str]:
    return {name: f"sha256:{name.lower()}" for name in mod.UPSTREAM_SOURCES}


def test_req_capstone_4759_spec_declares_required_contract() -> None:
    """REQ-CAPSTONE-4759: OpenSpec declares the .437 capstone contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for ref in mod.SPEC_REFS:
        assert ref in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_4759_skips_flagged_artifacts_before_importing_numbers() -> None:
    """SCENARIO-CAPSTONE-4759: flagged A1/A4 artifacts cannot source scorecard numbers."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 65},
        registry_sha256="sha256:registry",
        publication_gate=_paper_gate(),
        duration_s=0.001,
    )

    assert artifact["honest_verdict"] == "success: real_bank_landed_sk48_L2_capstone_complete"
    assert artifact["bridge_crossed_for_solve"] is False
    assert artifact["reproducible_total_levels"] == 65
    assert artifact["paper_ready"] is True
    assert artifact["submission_package_ready"] is False
    assert artifact["verifier_is_oracle"] is False
    assert artifact["submitted_to_leaderboard"] is False

    skipped = {row["source"]: row for row in artifact["skipped_artifacts"]}
    assert skipped["A1"]["reason"] == "flagged_adversarial"
    assert skipped["A4"]["reason"] == "flagged_adversarial"

    cited = {row["experiment_id"]: row for row in artifact["cited_upstream_artifacts"]}
    assert cited[4749]["fields_imported"] == ["flagged_adversarial"]
    assert "structured_heldout_accuracy" not in cited[4749]["fields_imported"]
    assert cited[4752]["fields_imported"] == ["flagged_adversarial"]
    assert "first_win_rate_integrated" not in cited[4752]["fields_imported"]

    decision = artifact["induction_quality_decision"]
    assert decision["a1"]["decision"] == "skipped_flagged_adversarial"
    assert decision["a1"]["beat_0_12_freeform_baseline"] is None
    assert decision["a2"]["goal_predicate_satisfiable"] is False
    assert decision["a2"]["banked_l2"] is False
    assert decision["cleared_induction_quality_wall"] is False
    assert artifact["scorecard"]["A3"]["real_bank_landed"] is True
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_4759_clean_induction_levers_can_clear_wall() -> None:
    """SCENARIO-CAPSTONE-4759: clean A1/A2 evidence decides the .437 headline."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(
            a1=_a1(flagged=False, structured_accuracy=0.13),
            a2=_a2(satisfiable=True, plan=True, reproduced=True),
            a3=_a3(banked=False),
            a4=_a4(flagged=False, ready=True),
        ),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 64},
        registry_sha256="sha256:registry",
        publication_gate=_paper_gate(),
        duration_s=0.002,
    )

    assert artifact["honest_verdict"] == "success: induction_quality_wall_cleared_capstone_complete"
    assert artifact["bridge_crossed_for_solve"] is True
    assert artifact["submission_package_ready"] is True
    assert artifact["scorecard"]["A1"]["structured_heldout_accuracy"] == 0.13
    assert artifact["scorecard"]["A1"]["beat_0_12_freeform_baseline"] is True
    assert artifact["scorecard"]["A2"]["goal_predicate_satisfiable"] is True
    assert artifact["scorecard"]["A2"]["banked_l2"] is True
    cited = {row["experiment_id"]: row for row in artifact["cited_upstream_artifacts"]}
    assert "structured_heldout_accuracy" in cited[4749]["fields_imported"]
    assert "submission_package_ready" in cited[4752]["fields_imported"]
    assert mod.validate_artifact(artifact) == []


def test_run_capstone_writes_artifact_and_records_missing_gaps(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4759: missing upstreams are gaps, not fabricated failures."""

    for name in ("results", "ops", "scripts", "openspec/capabilities/capstone"):
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
    (tmp_path / "ops" / "arc_solve_registry.yaml").write_text(
        yaml.safe_dump({"reproducible_total_levels": 65}),
        encoding="utf-8",
    )
    (tmp_path / "openspec" / "capabilities" / "capstone" / "spec.md").write_text(
        "REQ-CAPSTONE-4759\n",
        encoding="utf-8",
    )
    sources = {"A1": _a1(), "A2": _a2(), "A3": _a3()}
    for source, payload in sources.items():
        (tmp_path / mod.UPSTREAM_SOURCES[source]).write_text(
            json.dumps(payload, sort_keys=True),
            encoding="utf-8",
        )

    artifact = mod.run_capstone(root=tmp_path, publication_gate=_paper_gate())

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written == artifact
    assert artifact["preconditions_checked"]["upstream_artifacts"]["A4"]["present"] is False
    assert artifact["missing_artifacts"] == ["A4"]
    assert artifact["reproducibility_checksum"].startswith("sha256:")
    assert mod.validate_artifact(artifact) == []


def test_validate_artifact_rejects_schema_violations() -> None:
    """SCENARIO-CAPSTONE-4759: malformed capstone artifacts fail closed."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(),
        artifact_sha256=_hashes(),
        registry={"reproducible_total_levels": 65},
        registry_sha256="sha256:registry",
        publication_gate=_paper_gate(),
        duration_s=0.001,
    )

    assert "missing_field:honest_verdict" in mod.validate_artifact(
        {key: value for key, value in artifact.items() if key != "honest_verdict"}
    )
    assert "honest_verdict_missing_terminal_prefix" in mod.validate_artifact(
        {**artifact, "honest_verdict": "not-terminal"}
    )
    assert "verifier_is_oracle_must_be_false" in mod.validate_artifact(
        {**artifact, "verifier_is_oracle": True}
    )
    assert "missing_principle:honest_verdict" in mod.validate_artifact(
        {**artifact, "field_principles": {}}
    )
    assert "invalid_cited_upstream_artifacts" in mod.validate_artifact(
        {**artifact, "cited_upstream_artifacts": [{"experiment_id": 1}]}
    )
    assert "invalid_reproducibility_checksum" in mod.validate_artifact(
        {**artifact, "reproducibility_checksum": ""}
    )
    assert "submitted_to_leaderboard_must_be_false" in mod.validate_artifact(
        {**artifact, "submitted_to_leaderboard": True}
    )


def test_defensive_branches_remain_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CAPSTONE-4759: defensive parser and missing-source branches stay honest."""

    assert mod._as_float(True) is None
    assert mod._as_float("not-a-number") is None
    assert mod._as_int(False) == 0
    assert mod._as_int("not-a-number") == 0
    assert mod._experiment_id("A1", {}) == 4749
    assert mod._gate_paper_ready({}) is False
    assert mod._a2_scorecard({"flagged_adversarial": True}, skipped=True)["decision"] == (
        "skipped_flagged_adversarial"
    )
    assert mod._a3_scorecard({"flagged_adversarial": True}, skipped=True)["decision"] == (
        "skipped_flagged_adversarial"
    )

    empty = mod.build_artifact(
        artifacts={},
        artifact_sha256={},
        registry={},
        registry_sha256=None,
        publication_gate={},
        duration_s=0.001,
    )
    assert empty["honest_verdict"] == "complete: capstone_aggregation_no_real_bank"
    assert empty["missing_artifacts"] == ["A1", "A2", "A3", "A4"]
    assert empty["paper_ready"] is False

    for name in ("results", "ops", "openspec/capabilities/capstone"):
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
    (tmp_path / "ops" / "arc_solve_registry.yaml").write_text(": bad: yaml", encoding="utf-8")
    (tmp_path / "openspec" / "capabilities" / "capstone" / "spec.md").write_text(
        "REQ-CAPSTONE-4759\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        mod,
        "_publication_gate_result",
        lambda _root: (_paper_gate(), {"available": True, "stubbed": True}),
    )

    artifact = mod.run_capstone(root=tmp_path)

    assert artifact["preconditions_checked"]["registry"]["yaml_loadable"] is False
    assert artifact["preconditions_checked"]["publication_gate"]["stubbed"] is True
