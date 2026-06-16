"""Tests for Exp 4299 verifier registry/gaps hygiene.

Spec refs: REQ-VERIFY-4299, SCENARIO-VERIFY-4299.
"""

from __future__ import annotations

import json
import runpy
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_4299_verifier_registry_gaps_hygiene as exp4299_wrapper
from carnot.reporting import verifier_registry_gaps_hygiene_4299 as exp4299


REPO_ROOT = Path(__file__).parents[2]
SPEC_PATH = Path("openspec/capabilities/verification/spec.md")
RESULTS_WRAPPER_PATH = Path("results/experiment_4299_verifier_registry_gaps_hygiene.py")


def _minimal_registry() -> dict[str, Any]:
    return {
        "verifiers": [
            {
                "verifier_id": exp4299.GAP4_VERIFIER_ID,
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _synthetic_4294() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: hardened_efficiency_pareto_holds_fixture",
        "efficiency_pareto_holds": True,
        "accuracy_energy_verifier": 0.75,
        "accuracy_best_judge": 0.5,
        "accuracy_delta_ci95": [0.05, 0.45],
        "cost_ratio": 0.01,
        "verifier_is_oracle": False,
        "judge_metrics": {
            "best": {
                "judge_id": "fixture-strong-judge",
                "accuracy": 0.5,
                "cost_per_1k": 0.25,
            }
        },
        "model_specs": {"fixture": True},
        "random_seed": 4294,
        "reproducibility_checksum": "sha256:" + "4" * 64,
    }


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
    for path in exp4299.REQUIRED_COPY_PATHS:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if path == exp4299.EXP4294_PATH:
            _write_json(target, _synthetic_4294())
        else:
            shutil.copy2(REPO_ROOT / path, target)
    for path in exp4299.OPTIONAL_COPY_PATHS:
        source = REPO_ROOT / path
        if source.exists():
            target = tmp_path / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def test_req_4299_spec_declared() -> None:
    """REQ-VERIFY-4299: OpenSpec declares the .397 hygiene artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-4299",
        "SCENARIO-VERIFY-4299",
        "python/carnot/experiment_4299_verifier_registry_gaps_hygiene.py",
        RESULTS_WRAPPER_PATH.as_posix(),
        exp4299.EXP4299_ARTIFACT_PATH,
        "blocked_v397_artifacts_missing",
        "cross_generator_holds=true",
        "cross_generator_delta=0.5",
        "partial_state_scorer_built=true",
        "partial_state_leak_free=true",
        "efficiency_pareto_holds",
        "online_adaptation_helps=true",
        "total_levels=22",
        exp4299.GAP_CROSS_GENERATOR_SELECTION,
        exp4299.GAP_LEAK_FREE_PARTIAL_STATE,
    ):
        assert marker in spec
    assert exp4299.FIELD_PRINCIPLES["honest_verdict"] in spec
    assert exp4299.FIELD_PRINCIPLES["regression_guard_passed"] in spec
    assert exp4299.FIELD_PRINCIPLES["gaps_logged"] in spec
    assert exp4299.FIELD_PRINCIPLES["reproducibility_checksum"] in spec
    assert exp4299_wrapper.main is exp4299.main


def test_scenario_4299_preconditions_outcomes_and_gap4_guard_are_stable(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-4299: .397 artifacts exist and GAP-4 does not regress."""

    _write_minimal_repo(tmp_path)

    preflight = exp4299.check_preconditions(tmp_path)
    assert preflight["ok"] is True
    assert preflight["blocked_resource"] is None

    guard = exp4299.run_gap4_regression_guard(tmp_path)
    assert guard["regression_guard_passed"] is True
    assert guard["prior_artifact_path"] == exp4299.EXP4287_PATH
    assert guard["recorded_arc1_rule_exec"] == {
        "n": 31,
        "vote_pass2": pytest.approx(0.4516),
        "gated_pass2": pytest.approx(0.5806),
        "headroom_recovered": 4,
        "vote_wins_lost": 0,
    }
    assert guard["replayed_arc1_rule_exec"] == guard["recorded_arc1_rule_exec"]

    outcomes = exp4299.load_v397_outcomes(tmp_path)
    assert outcomes["cross_generator"]["cross_generator_holds"] is True
    assert outcomes["cross_generator"]["cross_generator_delta"] == pytest.approx(0.5)
    assert outcomes["cross_generator"]["held_out_task_n"] == 24
    assert outcomes["partial_state"]["partial_state_scorer_built"] is True
    assert outcomes["partial_state"]["partial_state_leak_free"] is True
    assert outcomes["partial_state"]["partial_state_auroc"] == pytest.approx(0.966143)
    assert outcomes["in_generation"]["diffusiongemma_guidance_moat"] is True
    assert outcomes["in_generation"]["flagged_adversarial"] is True
    assert outcomes["efficiency"]["efficiency_pareto_holds"] is True
    assert outcomes["efficiency"]["cost_ratio"] == pytest.approx(0.01)
    assert outcomes["self_learning"]["online_adaptation_helps"] is True
    assert outcomes["self_learning"]["online_cross_family_delta"] == pytest.approx(0.4833333333)
    assert outcomes["arc_progress"]["total_levels"] == 22
    assert outcomes["arc_progress"]["new_levels_solved_this_task"] == 1


def test_req_4299_ledgers_record_v397_truth_and_conditional_gaps(tmp_path: Path) -> None:
    """REQ-VERIFY-4299: registry and gaps carry the .397 truth."""

    _write_minimal_repo(tmp_path)
    guard = exp4299.run_gap4_regression_guard(tmp_path)
    outcomes = exp4299.load_v397_outcomes(tmp_path)
    gaps_logged = exp4299.build_gap_entries(outcomes)

    registry, gaps_text, manifest, summary = exp4299.ensure_ledgers_record_v397(
        _minimal_registry(),
        "# Verifier Gaps\n\nHistorical note remains.\n",
        _minimal_manifest(),
        guard,
        outcomes,
        gaps_logged,
    )

    assert gaps_logged == []
    assert summary == {
        "registry_reconciled": True,
        "manifest_reconciled": True,
        "gaps_logged_ids": [],
    }

    gap4 = registry["verifiers"][0]
    assert gap4["eval"]["eval_exp_4299"] == exp4299.EXP4299_ARTIFACT_PATH
    assert gap4["eval"]["exp4299_regression_guard_passed"] is True
    assert gap4["eval"]["exp4299_cross_generator_holds"] is True
    assert gap4["eval"]["exp4299_partial_state_scorer_built"] is True
    assert gap4["eval"]["exp4299_partial_state_leak_free"] is True
    assert gap4["eval"]["exp4299_diffusiongemma_guidance_moat"] is True
    assert gap4["eval"]["exp4299_diffusiongemma_flagged_adversarial"] is True
    assert gap4["eval"]["exp4299_efficiency_pareto_holds"] is True
    assert gap4["eval"]["exp4299_online_adaptation_helps"] is True
    assert gap4["eval"]["exp4299_arc_total_levels"] == 22
    assert gap4["eval"]["exp4299_v397_hardened_state"] == exp4299.V397_HARDENED_STATE
    role = next(role for role in gap4["registry_roles"] if role["role_id"] == exp4299.V397_ROLE_ID)
    assert role["v397_hardened_state"] == exp4299.V397_HARDENED_STATE
    assert role["efficiency_pareto_holds"] is True
    assert role["arc_total_levels"] == 22
    assert exp4299.registry_contains_v397(registry) is True
    assert exp4299.registry_contains_v397({}) is False
    assert gaps_text == "# Verifier Gaps\n\nHistorical note remains.\n"
    assert manifest == _minimal_manifest()

    missing_registry: dict[str, Any] = {}
    exp4299._ensure_gap4_eval(missing_registry, guard, outcomes, gaps_logged)
    assert missing_registry["verifiers"][0]["verifier_id"] == exp4299.GAP4_VERIFIER_ID
    empty_registry: dict[str, Any] = {}
    exp4299._ensure_v397_role(empty_registry, outcomes, gaps_logged)
    assert empty_registry == {}

    failed = {
        **outcomes,
        "cross_generator": {**outcomes["cross_generator"], "cross_generator_holds": False},
        "partial_state": {**outcomes["partial_state"], "partial_state_leak_free": False},
    }
    gap_entries = exp4299.build_gap_entries(failed)
    assert [entry["gap_id"] for entry in gap_entries] == [
        exp4299.GAP_CROSS_GENERATOR_SELECTION,
        exp4299.GAP_LEAK_FREE_PARTIAL_STATE,
    ]
    for entry in gap_entries:
        assert set(entry) >= {
            "gap_id",
            "failure_mode",
            "missing_discriminator",
            "candidate_design",
            "priority",
        }
    _, failed_gaps_text, _, failed_summary = exp4299.ensure_ledgers_record_v397(
        _minimal_registry(),
        "# Verifier Gaps\n",
        _minimal_manifest(),
        guard,
        failed,
        gap_entries,
    )
    assert failed_summary["gaps_logged_ids"] == [
        exp4299.GAP_CROSS_GENERATOR_SELECTION,
        exp4299.GAP_LEAK_FREE_PARTIAL_STATE,
    ]
    assert exp4299.GAP_CROSS_GENERATOR_SELECTION in failed_gaps_text
    assert exp4299.GAP_LEAK_FREE_PARTIAL_STATE in failed_gaps_text


def test_req_4299_build_artifact_validates_required_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-4299: terminal artifact exposes the required schema fields."""

    _write_minimal_repo(tmp_path)
    guard = exp4299.run_gap4_regression_guard(tmp_path)
    outcomes = exp4299.load_v397_outcomes(tmp_path)
    artifact = exp4299.build_artifact(
        regression_guard=guard,
        v397_outcomes=outcomes,
        gaps_logged=[],
        registry_reconciled=True,
        manifest_reconciled=True,
        reproducibility_checksum="sha256:" + "0" * 64,
        duration_s=0.25,
    )

    exp4299.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["regression_guard_passed"] is True
    assert artifact["registry_reconciled"] is True
    assert artifact["manifest_reconciled"] is True
    assert artifact["gaps_logged"] == []
    assert artifact["field_principles"] == exp4299.FIELD_PRINCIPLES
    assert artifact["model_specs"]["method"] == "cached_v397_ledger_reconciliation"
    assert artifact["inference_substrate"] == exp4299.INFERENCE_SUBSTRATE

    for field in exp4299.REQUIRED_ARTIFACT_FIELDS:
        malformed = dict(artifact)
        malformed.pop(field)
        with pytest.raises(ValueError, match=field):
            exp4299.validate_artifact(malformed)
    with pytest.raises(ValueError, match="terminal prefix"):
        exp4299.validate_artifact({**artifact, "honest_verdict": "not terminal"})
    with pytest.raises(ValueError, match="BARE bool"):
        exp4299.validate_artifact({**artifact, "regression_guard_passed": "yes"})
    with pytest.raises(ValueError, match="registry_reconciled"):
        exp4299.validate_artifact({**artifact, "registry_reconciled": "yes"})
    with pytest.raises(ValueError, match="manifest_reconciled"):
        exp4299.validate_artifact({**artifact, "manifest_reconciled": "yes"})
    with pytest.raises(ValueError, match="gaps_logged"):
        exp4299.validate_artifact({**artifact, "gaps_logged": "gap"})
    with pytest.raises(ValueError, match="gap entry"):
        exp4299.validate_artifact({**artifact, "gaps_logged": [{"gap_id": "GAP"}]})
    with pytest.raises(ValueError, match="random_seed"):
        exp4299.validate_artifact({**artifact, "random_seed": True})
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp4299.validate_artifact({**artifact, "reproducibility_checksum": ""})
    with pytest.raises(ValueError, match="model_specs"):
        exp4299.validate_artifact({**artifact, "model_specs": {}})
    with pytest.raises(ValueError, match="field_principles"):
        exp4299.validate_artifact({**artifact, "field_principles": {}})
    with pytest.raises(ValueError, match="spec_refs"):
        exp4299.validate_artifact({**artifact, "spec_refs": ["REQ-VERIFY-4299"]})
    with pytest.raises(ValueError, match="inference_substrate"):
        exp4299.validate_artifact({**artifact, "inference_substrate": "live"})


def test_req_4299_defensive_helpers_cover_blocked_shapes(tmp_path: Path) -> None:
    """REQ-VERIFY-4299: malformed ledger/resource shapes block honestly."""

    list_manifest_path = tmp_path / "manifest.yaml"
    list_manifest_path.write_text("- not-a-mapping\n", encoding="utf-8")
    assert exp4299._load_manifest(list_manifest_path) == _minimal_manifest()

    bad_registry = tmp_path / "registry.yaml"
    bad_registry.write_text("- not-a-mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="registry"):
        exp4299._load_registry_for_check(bad_registry)

    blank_gaps = tmp_path / "gaps.md"
    blank_gaps.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="gaps ledger"):
        exp4299._load_gaps_for_check(blank_gaps)

    list_json = tmp_path / "list.json"
    list_json.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact"):
        exp4299._load_json_for_check(list_json)


def test_scenario_4299_run_hygiene_writes_artifact_ledgers_and_manifest(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-4299: run writes the deliverable JSON and reconciled ledgers."""

    _write_minimal_repo(tmp_path)
    artifact = exp4299.run_hygiene(tmp_path)
    exp4299.validate_artifact(artifact)

    assert artifact["regression_guard_passed"] is True
    assert artifact["registry_reconciled"] is True
    assert artifact["manifest_reconciled"] is True
    assert artifact["gaps_logged"] == []
    written = json.loads((tmp_path / exp4299.EXP4299_ARTIFACT_PATH).read_text(encoding="utf-8"))
    assert written == artifact
    registry = yaml.safe_load((tmp_path / exp4299.REGISTRY_PATH).read_text(encoding="utf-8"))
    assert exp4299.registry_contains_v397(registry) is True
    gaps = (tmp_path / exp4299.GAPS_PATH).read_text(encoding="utf-8")
    assert exp4299.GAP_CROSS_GENERATOR_SELECTION not in gaps
    manifest = yaml.safe_load((tmp_path / exp4299.EXCLUSION_MANIFEST_PATH).read_text(encoding="utf-8"))
    assert manifest == _minimal_manifest()


def test_req_4299_blocked_precondition_stops_without_fabricating(tmp_path: Path) -> None:
    """REQ-VERIFY-4299: missing artifacts write blocked_v397_artifacts_missing."""

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

    artifact = exp4299.run_hygiene(tmp_path)
    exp4299.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_v397_artifacts_missing"
    assert artifact["regression_guard_passed"] is False
    assert artifact["registry_reconciled"] is False
    assert artifact["manifest_reconciled"] is False
    assert artifact["gaps_logged"] == []
    assert artifact["reproducibility_checksum"].startswith("blocked:")
    assert exp4299.GAP_CROSS_GENERATOR_SELECTION not in (
        tmp_path / "ops" / "verifier_gaps.md"
    ).read_text(encoding="utf-8")


def test_scenario_4299_results_entrypoint_runs_with_monkeypatched_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-4299: required results entrypoint delegates to Exp 4299."""

    called: list[bool] = []
    monkeypatch.setattr(exp4299, "main", lambda: called.append(True))
    python_root = str(REPO_ROOT / "python")
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if entry != python_root])

    runpy.run_path(RESULTS_WRAPPER_PATH.as_posix(), run_name="__main__")

    assert called == [True]
