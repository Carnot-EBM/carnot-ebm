"""Tests for Exp 4277 verifier registry/gaps hygiene.

Spec refs: REQ-VERIFY-4277, SCENARIO-VERIFY-4277.
"""

from __future__ import annotations

import json
import runpy
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_4277_verifier_registry_gaps_hygiene as exp4277_wrapper
from carnot.reporting import verifier_registry_gaps_hygiene_4277 as exp4277


REPO_ROOT = Path(__file__).parents[2]
SPEC_PATH = Path("openspec/capabilities/verification/spec.md")
RESULTS_WRAPPER_PATH = Path("results/experiment_4277_verifier_registry_gaps_hygiene.py")


def _minimal_registry() -> dict[str, Any]:
    return {
        "verifiers": [
            {
                "verifier_id": exp4277.GAP4_VERIFIER_ID,
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
    for path in exp4277.REQUIRED_COPY_PATHS:
        source = REPO_ROOT / path
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def test_req_4277_spec_declared() -> None:
    """REQ-VERIFY-4277: OpenSpec declares the .395 hygiene artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-4277",
        "SCENARIO-VERIFY-4277",
        "python/carnot/experiment_4277_verifier_registry_gaps_hygiene.py",
        RESULTS_WRAPPER_PATH.as_posix(),
        exp4277.EXP4277_ARTIFACT_PATH,
        "blocked_v395_artifacts_missing",
        "cross_family_win_holds=true",
        "cross_family_delta=0.4038461538",
        "online_adaptation_helps=false",
        "online_minus_static_delta=0.0961538462",
        "loader_repaired=true",
        "preflight_go=true",
        "guidance_selection_change_count=12",
        "total_levels_solved=20",
        "new_levels_solved_this_task=1",
        exp4277.CODE_RETIREMENT_ID,
        exp4277.REWARD_RETIREMENT_ID,
        exp4277.GAP_ONLINE_ADAPTATION_CALIBRATION,
    ):
        assert marker in spec
    assert exp4277.FIELD_PRINCIPLES["honest_verdict"] in spec
    assert exp4277.FIELD_PRINCIPLES["regression_guard_passed"] in spec
    assert exp4277.FIELD_PRINCIPLES["retirements_recorded"] in spec
    assert exp4277.FIELD_PRINCIPLES["gaps_logged"] in spec
    assert exp4277.FIELD_PRINCIPLES["reproducibility_checksum"] in spec
    assert exp4277_wrapper.main is exp4277.main


def test_scenario_4277_preconditions_outcomes_and_gap4_guard_are_stable() -> None:
    """SCENARIO-VERIFY-4277: .395 artifacts exist and GAP-4 does not regress."""

    preflight = exp4277.check_preconditions(REPO_ROOT)
    assert preflight["ok"] is True
    assert preflight["blocked_resource"] is None

    guard = exp4277.run_gap4_regression_guard(REPO_ROOT)
    assert guard["regression_guard_passed"] is True
    assert guard["prior_artifact_path"] == exp4277.EXP4266_PATH
    assert guard["recorded_arc1_rule_exec"] == {
        "n": 31,
        "vote_pass2": pytest.approx(0.4516),
        "gated_pass2": pytest.approx(0.5806),
        "headroom_recovered": 4,
        "vote_wins_lost": 0,
    }
    assert guard["replayed_arc1_rule_exec"] == guard["recorded_arc1_rule_exec"]

    outcomes = exp4277.load_v395_outcomes(REPO_ROOT)
    assert outcomes["cross_family"]["artifact_path"] == exp4277.EXP4271_PATH
    assert outcomes["cross_family"]["cross_family_win_holds"] is True
    assert outcomes["cross_family"]["cross_family_delta"] == pytest.approx(0.4038461538)
    assert outcomes["cross_family"]["cross_family_ci95"] == [0.25, 0.5576923077]
    assert outcomes["cross_family"]["held_out_task_n"] == 52
    assert outcomes["cross_family"]["generalization_state"] == "generalizes_held_out_family"
    assert outcomes["online_adaptation"]["online_adaptation_helps"] is False
    assert outcomes["online_adaptation"]["online_minus_static_delta"] == pytest.approx(
        0.0961538462
    )
    assert outcomes["online_adaptation"]["online_minus_static_ci95"] == [0.0, 0.1923076923]
    assert outcomes["diffusiongemma"]["loader_repaired"] is True
    assert outcomes["diffusiongemma"]["preflight_go"] is True
    assert outcomes["diffusiongemma"]["guidance_changes_selection"] is True
    assert outcomes["diffusiongemma"]["guidance_selection_change_count"] == 12
    assert outcomes["arc_progress"]["total_levels_solved"] == 20
    assert outcomes["arc_progress"]["new_levels_solved_this_task"] == 1
    assert outcomes["arc_progress"]["game_advanced"] == "wa30-ee6fef47"


def test_req_4277_ledgers_record_v395_truth_retirements_and_new_gap() -> None:
    """REQ-VERIFY-4277: registry, gaps, and manifest carry the .395 truth."""

    guard = exp4277.run_gap4_regression_guard(REPO_ROOT)
    outcomes = exp4277.load_v395_outcomes(REPO_ROOT)
    retirements = exp4277.build_retirement_entries(outcomes)
    gaps_logged = exp4277.build_gap_entries(outcomes)

    registry, gaps_text, manifest, summary = exp4277.ensure_ledgers_record_v395(
        _minimal_registry(),
        "# Verifier Gaps\n\nHistorical note remains.\n",
        _minimal_manifest(),
        guard,
        outcomes,
        retirements,
        gaps_logged,
    )

    assert [entry["id"] for entry in retirements] == [
        exp4277.CODE_RETIREMENT_ID,
        exp4277.REWARD_RETIREMENT_ID,
    ]
    assert [entry["gap_id"] for entry in gaps_logged] == [
        exp4277.GAP_ONLINE_ADAPTATION_CALIBRATION
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
        "retirements_recorded_ids": [
            exp4277.CODE_RETIREMENT_ID,
            exp4277.REWARD_RETIREMENT_ID,
        ],
        "gaps_logged_ids": [exp4277.GAP_ONLINE_ADAPTATION_CALIBRATION],
        "filled_gap_ids": [
            exp4277.GAP_CROSS_FAMILY_PROVENANCE_4266,
            exp4277.GAP_DIFFUSIONGEMMA_PREFLIGHT_4266,
        ],
    }

    gap4 = registry["verifiers"][0]
    assert gap4["eval"]["eval_exp_4277"] == exp4277.EXP4277_ARTIFACT_PATH
    assert gap4["eval"]["exp4277_regression_guard_passed"] is True
    assert gap4["eval"]["exp4277_cross_family_win_holds"] is True
    assert gap4["eval"]["exp4277_cross_family_delta"] == pytest.approx(0.4038461538)
    assert gap4["eval"]["exp4277_generalization_state"] == "generalizes_held_out_family"
    assert gap4["eval"]["exp4277_online_adaptation_helps"] is False
    assert gap4["eval"]["exp4277_diffusiongemma_preflight_go"] is True
    assert gap4["eval"]["exp4277_arc_total_levels_solved"] == 20
    assert gap4["eval"]["exp4277_retirements_recorded"] == [
        exp4277.CODE_RETIREMENT_ID,
        exp4277.REWARD_RETIREMENT_ID,
    ]
    role = next(role for role in gap4["registry_roles"] if role["role_id"] == exp4277.V395_ROLE_ID)
    assert role["cross_family_status"] == "generalizes_held_out_family"
    assert role["code_retirement_id"] == exp4277.CODE_RETIREMENT_ID
    assert role["reward_retirement_id"] == exp4277.REWARD_RETIREMENT_ID
    assert exp4277.registry_contains_v395(registry) is True
    assert exp4277.registry_contains_v395({}) is False

    missing_registry: dict[str, Any] = {}
    exp4277._ensure_gap4_eval(missing_registry, guard, outcomes, retirements, gaps_logged)
    assert missing_registry["verifiers"][0]["verifier_id"] == exp4277.GAP4_VERIFIER_ID
    empty_registry: dict[str, Any] = {}
    exp4277._ensure_v395_role(empty_registry, outcomes, retirements, gaps_logged)
    assert empty_registry == {}

    assert "Historical note remains." in gaps_text
    assert exp4277.GAP_CROSS_FAMILY_PROVENANCE_4266 in gaps_text
    assert "filled (arc_family_provenance_recovery_4270_cross_family_4271)" in gaps_text
    assert exp4277.GAP_DIFFUSIONGEMMA_PREFLIGHT_4266 in gaps_text
    assert "filled (diffusiongemma_loader_fix_preflight_4274)" in gaps_text
    assert exp4277.GAP_ONLINE_ADAPTATION_CALIBRATION in gaps_text
    assert "online_minus_static_delta=0.0961538462" in gaps_text

    assert exp4277._find_manifest_entry(manifest, exp4277.CODE_RETIREMENT_ID) is not None
    assert exp4277._find_manifest_entry(manifest, exp4277.REWARD_RETIREMENT_ID) is not None
    for entry_id in (exp4277.CODE_RETIREMENT_ID, exp4277.REWARD_RETIREMENT_ID):
        entry = exp4277._find_manifest_entry(manifest, entry_id)
        assert entry is not None
        assert entry["retire_if_same_verdict"] is True
        assert entry["recorded_by_artifact"] == exp4277.EXP4277_ARTIFACT_PATH


def test_req_4277_manifest_helpers_are_idempotent(tmp_path: Path) -> None:
    """REQ-VERIFY-4277: retirement manifest updates are normalized and idempotent."""

    list_manifest_path = tmp_path / "manifest.yaml"
    list_manifest_path.write_text("- not-a-mapping\n", encoding="utf-8")
    assert exp4277._load_manifest(list_manifest_path) == _minimal_manifest()

    assert exp4277._find_manifest_entry({"retired_extras": "bad-shape"}, "missing") is None
    manifest = _minimal_manifest()
    retirements = exp4277.build_retirement_entries(exp4277.load_v395_outcomes(REPO_ROOT))
    assert exp4277._ensure_manifest_retirements(manifest, retirements) is True
    assert exp4277._ensure_manifest_retirements(manifest, retirements) is False
    assert exp4277._find_manifest_entry(manifest, exp4277.CODE_RETIREMENT_ID) is not None
    assert exp4277._find_manifest_entry(manifest, exp4277.REWARD_RETIREMENT_ID) is not None


def test_req_4277_build_artifact_validates_required_fields() -> None:
    """REQ-VERIFY-4277: terminal artifact exposes the required schema fields."""

    guard = exp4277.run_gap4_regression_guard(REPO_ROOT)
    outcomes = exp4277.load_v395_outcomes(REPO_ROOT)
    retirements = exp4277.build_retirement_entries(outcomes)
    gaps_logged = exp4277.build_gap_entries(outcomes)
    artifact = exp4277.build_artifact(
        regression_guard=guard,
        v395_outcomes=outcomes,
        retirements_recorded=retirements,
        gaps_logged=gaps_logged,
        registry_reconciled=True,
        manifest_reconciled=True,
        reproducibility_checksum="sha256:" + "0" * 64,
        duration_s=0.25,
    )

    exp4277.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["regression_guard_passed"] is True
    assert artifact["registry_reconciled"] is True
    assert artifact["manifest_reconciled"] is True
    assert artifact["retirements_recorded"] == retirements
    assert artifact["gaps_logged"] == gaps_logged
    assert artifact["field_principles"] == exp4277.FIELD_PRINCIPLES
    assert artifact["model_specs"]["method"] == "cached_v395_ledger_reconciliation"
    assert artifact["inference_substrate"] == exp4277.INFERENCE_SUBSTRATE

    for field in exp4277.REQUIRED_ARTIFACT_FIELDS:
        malformed = dict(artifact)
        malformed.pop(field)
        with pytest.raises(ValueError, match=field):
            exp4277.validate_artifact(malformed)
    with pytest.raises(ValueError, match="terminal prefix"):
        exp4277.validate_artifact({**artifact, "honest_verdict": "not terminal"})
    with pytest.raises(ValueError, match="BARE bool"):
        exp4277.validate_artifact({**artifact, "regression_guard_passed": "yes"})
    with pytest.raises(ValueError, match="registry_reconciled"):
        exp4277.validate_artifact({**artifact, "registry_reconciled": "yes"})
    with pytest.raises(ValueError, match="manifest_reconciled"):
        exp4277.validate_artifact({**artifact, "manifest_reconciled": "yes"})
    with pytest.raises(ValueError, match="retirements_recorded"):
        exp4277.validate_artifact({**artifact, "retirements_recorded": "retired"})
    with pytest.raises(ValueError, match="retirement entry"):
        exp4277.validate_artifact({**artifact, "retirements_recorded": [{"id": "only"}]})
    with pytest.raises(ValueError, match="gaps_logged"):
        exp4277.validate_artifact({**artifact, "gaps_logged": "gap"})
    with pytest.raises(ValueError, match="gap entry"):
        exp4277.validate_artifact({**artifact, "gaps_logged": [{"gap_id": "GAP"}]})


def test_scenario_4277_run_hygiene_writes_artifact_ledgers_and_manifest(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4277: run writes the deliverable JSON and reconciled ledgers."""

    _write_minimal_repo(tmp_path)
    artifact = exp4277.run_hygiene(tmp_path)
    exp4277.validate_artifact(artifact)

    assert artifact["regression_guard_passed"] is True
    assert artifact["registry_reconciled"] is True
    assert artifact["manifest_reconciled"] is True
    assert [entry["id"] for entry in artifact["retirements_recorded"]] == [
        exp4277.CODE_RETIREMENT_ID,
        exp4277.REWARD_RETIREMENT_ID,
    ]
    assert [entry["gap_id"] for entry in artifact["gaps_logged"]] == [
        exp4277.GAP_ONLINE_ADAPTATION_CALIBRATION
    ]
    written = json.loads((tmp_path / exp4277.EXP4277_ARTIFACT_PATH).read_text(encoding="utf-8"))
    assert written == artifact
    registry = yaml.safe_load((tmp_path / exp4277.REGISTRY_PATH).read_text(encoding="utf-8"))
    assert exp4277.registry_contains_v395(registry) is True
    gaps = (tmp_path / exp4277.GAPS_PATH).read_text(encoding="utf-8")
    assert exp4277.GAP_ONLINE_ADAPTATION_CALIBRATION in gaps
    manifest = yaml.safe_load((tmp_path / exp4277.EXCLUSION_MANIFEST_PATH).read_text(encoding="utf-8"))
    assert exp4277._find_manifest_entry(manifest, exp4277.CODE_RETIREMENT_ID) is not None
    assert exp4277._find_manifest_entry(manifest, exp4277.REWARD_RETIREMENT_ID) is not None


def test_req_4277_blocked_precondition_stops_without_fabricating(tmp_path: Path) -> None:
    """REQ-VERIFY-4277: missing artifacts write blocked_v395_artifacts_missing."""

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

    artifact = exp4277.run_hygiene(tmp_path)
    exp4277.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_v395_artifacts_missing"
    assert artifact["regression_guard_passed"] is False
    assert artifact["registry_reconciled"] is False
    assert artifact["manifest_reconciled"] is False
    assert artifact["retirements_recorded"] == []
    assert artifact["gaps_logged"] == []
    assert artifact["reproducibility_checksum"].startswith("blocked:")
    assert exp4277.GAP_ONLINE_ADAPTATION_CALIBRATION not in (
        tmp_path / "ops" / "verifier_gaps.md"
    ).read_text(encoding="utf-8")


def test_scenario_4277_results_entrypoint_runs_with_monkeypatched_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-4277: required results entrypoint delegates to Exp 4277."""

    called: list[bool] = []
    monkeypatch.setattr(exp4277, "main", lambda: called.append(True))

    runpy.run_path(RESULTS_WRAPPER_PATH.as_posix(), run_name="__main__")

    assert called == [True]
