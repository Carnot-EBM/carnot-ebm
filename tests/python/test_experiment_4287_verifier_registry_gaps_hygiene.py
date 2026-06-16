"""Tests for Exp 4287 verifier registry/gaps hygiene.

Spec refs: REQ-VERIFY-4287, SCENARIO-VERIFY-4287.
"""

from __future__ import annotations

import json
import runpy
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_4287_verifier_registry_gaps_hygiene as exp4287_wrapper
from carnot.reporting import verifier_registry_gaps_hygiene_4287 as exp4287


REPO_ROOT = Path(__file__).parents[2]
SPEC_PATH = Path("openspec/capabilities/verification/spec.md")
RESULTS_WRAPPER_PATH = Path("results/experiment_4287_verifier_registry_gaps_hygiene.py")


def _minimal_registry() -> dict[str, Any]:
    return {
        "verifiers": [
            {
                "verifier_id": exp4287.GAP4_VERIFIER_ID,
                "domain": "arc_agi2_grid",
                "version": 1,
                "kind": "process_verifier",
                "eval": {"metric": "pass_at_1"},
                "registry_roles": [],
            }
        ]
    }


def _minimal_manifest() -> dict[str, Any]:
    return {"retired": [], "retired_experiments": [], "retired_extras": []}


def _write_minimal_repo(tmp_path: Path) -> None:
    (tmp_path / "ops").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "ops" / "verifier_registry.yaml").write_text(
        yaml.safe_dump(_minimal_registry(), sort_keys=False),
        encoding="utf-8",
    )
    (tmp_path / "ops" / "verifier_gaps.md").write_text(
        "# Verifier Gaps\n\nHistorical note remains.\n",
        encoding="utf-8",
    )
    (tmp_path / "ops" / "exclusion_manifest.yaml").write_text(
        yaml.safe_dump(_minimal_manifest(), sort_keys=False),
        encoding="utf-8",
    )
    for path in exp4287.REQUIRED_COPY_PATHS:
        source = REPO_ROOT / path
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def test_req_4287_spec_declared() -> None:
    """REQ-VERIFY-4287: OpenSpec declares the .396 hygiene artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-4287",
        "SCENARIO-VERIFY-4287",
        "python/carnot/experiment_4287_verifier_registry_gaps_hygiene.py",
        RESULTS_WRAPPER_PATH.as_posix(),
        exp4287.EXP4287_ARTIFACT_PATH,
        "blocked_v396_artifacts_missing",
        "diffusiongemma_guidance_moat=false",
        "arcgen_cross_family_holds=true",
        "cross_family_delta=1.0",
        "online_adaptation_helps=false",
        "static_cross_family_delta=0.5",
        "efficiency_parity_at_lower_cost=true",
        "accuracy_delta=0.4423076923",
        "cost_ratio=0.0000000195",
        "total_levels_solved=21",
        "new_levels_solved_this_task=1",
        exp4287.GAP_DIFFUSIONGEMMA_PARTIAL_STATE,
    ):
        assert marker in spec
    assert exp4287.FIELD_PRINCIPLES["honest_verdict"] in spec
    assert exp4287.FIELD_PRINCIPLES["regression_guard_passed"] in spec
    assert exp4287.FIELD_PRINCIPLES["gaps_logged"] in spec
    assert exp4287.FIELD_PRINCIPLES["reproducibility_checksum"] in spec
    assert exp4287_wrapper.main is exp4287.main


def test_scenario_4287_preconditions_outcomes_and_gap4_guard_are_stable() -> None:
    """SCENARIO-VERIFY-4287: .396 artifacts exist and GAP-4 does not regress."""

    preflight = exp4287.check_preconditions(REPO_ROOT)
    assert preflight["ok"] is True
    assert preflight["blocked_resource"] is None

    guard = exp4287.run_gap4_regression_guard(REPO_ROOT)
    assert guard["regression_guard_passed"] is True
    assert guard["prior_artifact_path"] == exp4287.EXP4277_PATH
    assert guard["recorded_arc1_rule_exec"] == {
        "n": 31,
        "vote_pass2": pytest.approx(0.4516),
        "gated_pass2": pytest.approx(0.5806),
        "headroom_recovered": 4,
        "vote_wins_lost": 0,
    }
    assert guard["replayed_arc1_rule_exec"] == guard["recorded_arc1_rule_exec"]

    outcomes = exp4287.load_v396_outcomes(REPO_ROOT)
    assert outcomes["diffusiongemma"]["diffusiongemma_guidance_moat"] is False
    assert outcomes["diffusiongemma"]["carnot_minus_unguided_delta"] == pytest.approx(0.0)
    assert outcomes["diffusiongemma"]["partial_state_support"] is False
    assert outcomes["arcgen_cross_family"]["arcgen_cross_family_holds"] is True
    assert outcomes["arcgen_cross_family"]["cross_family_delta"] == pytest.approx(1.0)
    assert outcomes["arcgen_cross_family"]["cross_family_ci95"] == [1.0, 1.0]
    assert outcomes["arcgen_cross_family"]["held_out_task_n"] == 50
    assert outcomes["self_learning"]["online_adaptation_helps"] is False
    assert outcomes["self_learning"]["static_cross_family_delta"] == pytest.approx(0.5)
    assert outcomes["self_learning"]["online_cross_family_delta"] == pytest.approx(0.5806451613)
    assert outcomes["self_learning"]["held_out_task_n"] == 102
    assert outcomes["efficiency"]["efficiency_parity_at_lower_cost"] is True
    assert outcomes["efficiency"]["accuracy_delta"] == pytest.approx(0.4423076923)
    assert outcomes["efficiency"]["cost_ratio"] == pytest.approx(1.95e-8)
    assert outcomes["arc_progress"]["total_levels_solved"] == 21
    assert outcomes["arc_progress"]["new_levels_solved_this_task"] == 1
    assert outcomes["arc_progress"]["game_advanced"] == "ls20-9607627b"


def test_req_4287_ledgers_record_v396_truth_and_new_gap() -> None:
    """REQ-VERIFY-4287: registry and gaps carry the .396 truth."""

    guard = exp4287.run_gap4_regression_guard(REPO_ROOT)
    outcomes = exp4287.load_v396_outcomes(REPO_ROOT)
    gaps_logged = exp4287.build_gap_entries(outcomes)

    registry, gaps_text, manifest, summary = exp4287.ensure_ledgers_record_v396(
        _minimal_registry(),
        "# Verifier Gaps\n\nHistorical note remains.\n",
        _minimal_manifest(),
        guard,
        outcomes,
        gaps_logged,
    )

    assert [entry["gap_id"] for entry in gaps_logged] == [
        exp4287.GAP_DIFFUSIONGEMMA_PARTIAL_STATE
    ]
    for entry in gaps_logged:
        assert set(entry) >= {
            "gap_id",
            "failure_mode",
            "missing_discriminator",
            "candidate_design",
            "priority",
        }
    assert summary == {
        "registry_reconciled": True,
        "manifest_reconciled": True,
        "gaps_logged_ids": [exp4287.GAP_DIFFUSIONGEMMA_PARTIAL_STATE],
    }

    gap4 = registry["verifiers"][0]
    assert gap4["eval"]["eval_exp_4287"] == exp4287.EXP4287_ARTIFACT_PATH
    assert gap4["eval"]["exp4287_regression_guard_passed"] is True
    assert gap4["eval"]["exp4287_diffusiongemma_guidance_moat"] is False
    assert gap4["eval"]["exp4287_diffusiongemma_partial_state_support"] is False
    assert gap4["eval"]["exp4287_arcgen_cross_family_holds"] is True
    assert gap4["eval"]["exp4287_arcgen_cross_family_delta"] == pytest.approx(1.0)
    assert gap4["eval"]["exp4287_online_adaptation_helps"] is False
    assert gap4["eval"]["exp4287_efficiency_parity_at_lower_cost"] is True
    assert gap4["eval"]["exp4287_arc_total_levels_solved"] == 21
    assert gap4["eval"]["exp4287_generalization_state"] == exp4287.V396_GENERALIZATION_STATE
    assert gap4["eval"]["exp4287_guidance_state"] == exp4287.V396_GUIDANCE_STATE
    assert gap4["eval"]["exp4287_gaps_logged"] == [
        exp4287.GAP_DIFFUSIONGEMMA_PARTIAL_STATE
    ]
    role = next(role for role in gap4["registry_roles"] if role["role_id"] == exp4287.V396_ROLE_ID)
    assert role["cross_family_status"] == exp4287.V396_GENERALIZATION_STATE
    assert role["diffusiongemma_guidance_state"] == exp4287.V396_GUIDANCE_STATE
    assert role["efficiency_parity_at_lower_cost"] is True
    assert role["arc_total_levels_solved"] == 21
    assert exp4287.registry_contains_v396(registry) is True
    assert exp4287.registry_contains_v396({}) is False

    missing_registry: dict[str, Any] = {}
    exp4287._ensure_gap4_eval(missing_registry, guard, outcomes, gaps_logged)
    assert missing_registry["verifiers"][0]["verifier_id"] == exp4287.GAP4_VERIFIER_ID
    empty_registry: dict[str, Any] = {}
    exp4287._ensure_v396_role(empty_registry, outcomes, gaps_logged)
    assert empty_registry == {}

    assert "Historical note remains." in gaps_text
    assert exp4287.GAP_DIFFUSIONGEMMA_PARTIAL_STATE in gaps_text
    assert "masked/partial diffusion token states" in gaps_text
    assert "learned partial-state diffusion scorer" in gaps_text
    assert manifest == _minimal_manifest()


def test_req_4287_build_artifact_validates_required_fields() -> None:
    """REQ-VERIFY-4287: terminal artifact exposes the required schema fields."""

    guard = exp4287.run_gap4_regression_guard(REPO_ROOT)
    outcomes = exp4287.load_v396_outcomes(REPO_ROOT)
    gaps_logged = exp4287.build_gap_entries(outcomes)
    artifact = exp4287.build_artifact(
        regression_guard=guard,
        v396_outcomes=outcomes,
        gaps_logged=gaps_logged,
        registry_reconciled=True,
        manifest_reconciled=True,
        reproducibility_checksum="sha256:" + "0" * 64,
        duration_s=0.25,
    )

    exp4287.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["regression_guard_passed"] is True
    assert artifact["registry_reconciled"] is True
    assert artifact["manifest_reconciled"] is True
    assert artifact["gaps_logged"] == gaps_logged
    assert artifact["field_principles"] == exp4287.FIELD_PRINCIPLES
    assert artifact["model_specs"]["method"] == "cached_v396_ledger_reconciliation"
    assert artifact["inference_substrate"] == exp4287.INFERENCE_SUBSTRATE

    for field in exp4287.REQUIRED_ARTIFACT_FIELDS:
        malformed = dict(artifact)
        malformed.pop(field)
        with pytest.raises(ValueError, match=field):
            exp4287.validate_artifact(malformed)
    with pytest.raises(ValueError, match="terminal prefix"):
        exp4287.validate_artifact({**artifact, "honest_verdict": "not terminal"})
    with pytest.raises(ValueError, match="BARE bool"):
        exp4287.validate_artifact({**artifact, "regression_guard_passed": "yes"})
    with pytest.raises(ValueError, match="registry_reconciled"):
        exp4287.validate_artifact({**artifact, "registry_reconciled": "yes"})
    with pytest.raises(ValueError, match="manifest_reconciled"):
        exp4287.validate_artifact({**artifact, "manifest_reconciled": "yes"})
    with pytest.raises(ValueError, match="gaps_logged"):
        exp4287.validate_artifact({**artifact, "gaps_logged": "gap"})
    with pytest.raises(ValueError, match="gap entry"):
        exp4287.validate_artifact({**artifact, "gaps_logged": [{"gap_id": "GAP"}]})
    with pytest.raises(ValueError, match="random_seed"):
        exp4287.validate_artifact({**artifact, "random_seed": True})
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp4287.validate_artifact({**artifact, "reproducibility_checksum": ""})
    with pytest.raises(ValueError, match="model_specs"):
        exp4287.validate_artifact({**artifact, "model_specs": {}})
    with pytest.raises(ValueError, match="field_principles"):
        exp4287.validate_artifact({**artifact, "field_principles": {}})
    with pytest.raises(ValueError, match="spec_refs"):
        exp4287.validate_artifact({**artifact, "spec_refs": ["REQ-VERIFY-4287"]})
    with pytest.raises(ValueError, match="inference_substrate"):
        exp4287.validate_artifact({**artifact, "inference_substrate": "live"})


def test_req_4287_defensive_helpers_cover_blocked_shapes(tmp_path: Path) -> None:
    """REQ-VERIFY-4287: malformed ledger/resource shapes block honestly."""

    list_manifest_path = tmp_path / "manifest.yaml"
    list_manifest_path.write_text("- not-a-mapping\n", encoding="utf-8")
    assert exp4287._load_manifest(list_manifest_path) == _minimal_manifest()

    bad_registry = tmp_path / "registry.yaml"
    bad_registry.write_text("- not-a-mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="registry"):
        exp4287._load_registry_for_check(bad_registry)

    blank_gaps = tmp_path / "gaps.md"
    blank_gaps.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="gaps ledger"):
        exp4287._load_gaps_for_check(blank_gaps)

    list_json = tmp_path / "list.json"
    list_json.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact"):
        exp4287._load_json_for_check(list_json)

    outcomes = exp4287.load_v396_outcomes(REPO_ROOT)
    supported = {**outcomes, "diffusiongemma": {**outcomes["diffusiongemma"]}}
    supported["diffusiongemma"]["partial_state_support"] = True
    assert exp4287.build_gap_entries(supported) == []


def test_scenario_4287_run_hygiene_writes_artifact_ledgers_and_manifest(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4287: run writes the deliverable JSON and reconciled ledgers."""

    _write_minimal_repo(tmp_path)
    artifact = exp4287.run_hygiene(tmp_path)
    exp4287.validate_artifact(artifact)

    assert artifact["regression_guard_passed"] is True
    assert artifact["registry_reconciled"] is True
    assert artifact["manifest_reconciled"] is True
    assert [entry["gap_id"] for entry in artifact["gaps_logged"]] == [
        exp4287.GAP_DIFFUSIONGEMMA_PARTIAL_STATE
    ]
    written = json.loads((tmp_path / exp4287.EXP4287_ARTIFACT_PATH).read_text(encoding="utf-8"))
    assert written == artifact
    registry = yaml.safe_load((tmp_path / exp4287.REGISTRY_PATH).read_text(encoding="utf-8"))
    assert exp4287.registry_contains_v396(registry) is True
    gaps = (tmp_path / exp4287.GAPS_PATH).read_text(encoding="utf-8")
    assert exp4287.GAP_DIFFUSIONGEMMA_PARTIAL_STATE in gaps
    manifest = yaml.safe_load((tmp_path / exp4287.EXCLUSION_MANIFEST_PATH).read_text(encoding="utf-8"))
    assert manifest == _minimal_manifest()


def test_req_4287_blocked_precondition_stops_without_fabricating(tmp_path: Path) -> None:
    """REQ-VERIFY-4287: missing artifacts write blocked_v396_artifacts_missing."""

    (tmp_path / "ops").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "ops" / "verifier_registry.yaml").write_text(
        yaml.safe_dump(_minimal_registry(), sort_keys=False),
        encoding="utf-8",
    )
    (tmp_path / "ops" / "verifier_gaps.md").write_text("# Verifier Gaps\n", encoding="utf-8")
    (tmp_path / "ops" / "exclusion_manifest.yaml").write_text(
        yaml.safe_dump(_minimal_manifest(), sort_keys=False),
        encoding="utf-8",
    )

    artifact = exp4287.run_hygiene(tmp_path)
    exp4287.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_v396_artifacts_missing"
    assert artifact["regression_guard_passed"] is False
    assert artifact["registry_reconciled"] is False
    assert artifact["manifest_reconciled"] is False
    assert artifact["gaps_logged"] == []
    assert artifact["reproducibility_checksum"].startswith("blocked:")
    assert exp4287.GAP_DIFFUSIONGEMMA_PARTIAL_STATE not in (
        tmp_path / "ops" / "verifier_gaps.md"
    ).read_text(encoding="utf-8")


def test_scenario_4287_results_entrypoint_runs_with_monkeypatched_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-4287: required results entrypoint delegates to Exp 4287."""

    called: list[bool] = []
    monkeypatch.setattr(exp4287, "main", lambda: called.append(True))

    runpy.run_path(RESULTS_WRAPPER_PATH.as_posix(), run_name="__main__")

    assert called == [True]
