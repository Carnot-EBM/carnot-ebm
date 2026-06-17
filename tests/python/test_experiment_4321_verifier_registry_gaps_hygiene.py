"""Tests for Exp 4321 verifier registry/gaps hygiene.

Spec refs: REQ-VERIFY-4321, SCENARIO-VERIFY-4321.
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

from carnot import experiment_4321_verifier_registry_gaps_hygiene as exp4321_wrapper
from carnot.reporting import verifier_registry_gaps_hygiene_4321 as exp4321


REPO_ROOT = Path(__file__).parents[2]
SPEC_PATH = Path("openspec/capabilities/verification/spec.md")
RESULTS_WRAPPER_PATH = Path("results/experiment_4321_verifier_registry_gaps_hygiene.py")


def _minimal_registry() -> dict[str, Any]:
    return {
        "verifiers": [
            {
                "verifier_id": exp4321.GAP4_VERIFIER_ID,
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


def _write_minimal_repo(tmp_path: Path, *, omit: set[str] | None = None) -> None:
    omit = omit or set()
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
    for path in exp4321.REQUIRED_COPY_PATHS:
        if path in omit:
            continue
        source = REPO_ROOT / path
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def test_req_4321_spec_declared() -> None:
    """REQ-VERIFY-4321: OpenSpec declares the .399 hygiene artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-4321",
        "SCENARIO-VERIFY-4321",
        "python/carnot/experiment_4321_verifier_registry_gaps_hygiene.py",
        RESULTS_WRAPPER_PATH.as_posix(),
        exp4321.EXP4321_ARTIFACT_PATH,
        "blocked_ledgers_unparseable",
        "aggregate_available_report_gaps",
        "cross_domain_selection_holds=false",
        "diffusiongemma_guidance_moat=true",
        "cascade_dominates_controls=false",
        "total_levels=23",
        "cross_game_transfer_helps=false",
        "off_arc_demofit_beats_vote=true",
        exp4321.GAP_CROSS_DOMAIN_FAMILY_INVARIANT_SELECTION,
        exp4321.GAP_GAME_INVARIANT_ARC_VALUE,
        exp4321.GAP_CODE_EXEC_DEMOFIT,
    ):
        assert marker in spec
    assert exp4321.FIELD_PRINCIPLES["honest_verdict"] in spec
    assert exp4321.FIELD_PRINCIPLES["regression_guard_passed"] in spec
    assert exp4321.FIELD_PRINCIPLES["gaps_logged"] in spec
    assert exp4321.FIELD_PRINCIPLES["reproducibility_checksum"] in spec
    assert exp4321_wrapper.main is exp4321.main


def test_scenario_4321_preconditions_outcomes_and_robust_availability(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-4321: missing .399 artifacts are per-axis gaps only."""

    _write_minimal_repo(tmp_path, omit={exp4321.EXP4315_PATH})

    preflight = exp4321.check_preconditions(tmp_path)
    assert preflight["ok"] is True
    assert preflight["blocked_resource"] is None

    guard = exp4321.run_gap4_regression_guard(tmp_path)
    assert guard["regression_guard_passed"] is True
    assert guard["prior_artifact_path"] == exp4321.EXP4310_PATH
    assert guard["recorded_arc1_rule_exec"] == {
        "n": 31,
        "vote_pass2": pytest.approx(0.4516),
        "gated_pass2": pytest.approx(0.5806),
        "headroom_recovered": 4,
        "vote_wins_lost": 0,
    }
    assert guard["replayed_arc1_rule_exec"] == guard["recorded_arc1_rule_exec"]

    bundle = exp4321.load_v399_outcomes(tmp_path)
    outcomes = bundle["v399_outcomes"]
    availability = bundle["availability_report"]
    assert availability["axes"]["cross_domain"]["verdict"] is False
    assert availability["axes"]["in_generation"]["missing_artifacts"] == [
        {
            "axis": "in_generation",
            "artifact_key": "4315_in_generation_moat",
            "experiment_id": 4315,
        }
    ]
    assert availability["axes"]["off_arc_execution"]["verdict"] is True
    assert outcomes["cross_domain"]["cross_domain_selection_holds"] is False
    assert outcomes["cross_domain"]["label_ablation_robust"] is True
    assert outcomes["cross_domain"]["cross_domain_delta"] == pytest.approx(0.2307692308)
    assert outcomes["in_generation"]["available"] is False
    assert outcomes["efficiency_cascade"]["accuracy_always_energy"] == pytest.approx(0.6)
    assert outcomes["efficiency_cascade"]["cascade_dominates_controls"] is False
    assert outcomes["arc_progress"]["total_levels"] == 23
    assert outcomes["arc_progress"]["new_levels_solved_this_task"] == 1
    assert outcomes["cross_game_transfer"]["cross_game_transfer_helps"] is False
    assert outcomes["off_arc_execution"]["off_arc_demofit_beats_vote"] is True
    assert exp4321.robust_aggregator_ok(outcomes["robust_aggregator"]) is True


def test_req_4321_ledgers_record_v399_truth_and_gaps(tmp_path: Path) -> None:
    """REQ-VERIFY-4321: registry and gaps carry the .399 truth."""

    _write_minimal_repo(tmp_path)
    guard = exp4321.run_gap4_regression_guard(tmp_path)
    bundle = exp4321.load_v399_outcomes(tmp_path)
    gaps_logged = exp4321.build_gap_entries(bundle)

    registry, gaps_text, manifest, summary = exp4321.ensure_ledgers_record_v399(
        _minimal_registry(),
        "# Verifier Gaps\n\nHistorical note remains.\n",
        _minimal_manifest(),
        guard,
        bundle,
        gaps_logged,
    )

    assert [entry["gap_id"] for entry in gaps_logged] == [
        exp4321.GAP_CROSS_DOMAIN_FAMILY_INVARIANT_SELECTION,
        exp4321.GAP_GAME_INVARIANT_ARC_VALUE,
        exp4321.GAP_CODE_EXEC_DEMOFIT,
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
        "gaps_logged_ids": [
            exp4321.GAP_CROSS_DOMAIN_FAMILY_INVARIANT_SELECTION,
            exp4321.GAP_GAME_INVARIANT_ARC_VALUE,
            exp4321.GAP_CODE_EXEC_DEMOFIT,
        ],
    }

    gap4 = registry["verifiers"][0]
    assert gap4["eval"]["eval_exp_4321"] == exp4321.EXP4321_ARTIFACT_PATH
    assert gap4["eval"]["exp4321_regression_guard_passed"] is True
    assert gap4["eval"]["exp4321_robust_aggregator_used"] is True
    assert gap4["eval"]["exp4321_cross_domain_selection_holds"] is False
    assert gap4["eval"]["exp4321_label_ablation_robust"] is True
    assert gap4["eval"]["exp4321_diffusiongemma_guidance_moat"] is True
    assert gap4["eval"]["exp4321_scorer_leak_recheck_passed"] is True
    assert gap4["eval"]["exp4321_cascade_dominates_controls"] is False
    assert gap4["eval"]["exp4321_accuracy_always_energy"] == pytest.approx(0.6)
    assert gap4["eval"]["exp4321_arc_total_levels"] == 23
    assert gap4["eval"]["exp4321_arc_new_levels_solved"] == 1
    assert gap4["eval"]["exp4321_cross_game_transfer_helps"] is False
    assert gap4["eval"]["exp4321_cross_game_state_reduction"] == pytest.approx(1.0)
    assert gap4["eval"]["exp4321_off_arc_demofit_beats_vote"] is True
    assert gap4["eval"]["exp4321_off_arc_demofit_minus_vote_delta"] == pytest.approx(0.02)
    assert gap4["eval"]["exp4321_v399_state"] == exp4321.V399_STATE
    assert gap4["eval"]["exp4321_gaps_logged"] == [
        exp4321.GAP_CROSS_DOMAIN_FAMILY_INVARIANT_SELECTION,
        exp4321.GAP_GAME_INVARIANT_ARC_VALUE,
        exp4321.GAP_CODE_EXEC_DEMOFIT,
    ]
    role = next(role for role in gap4["registry_roles"] if role["role_id"] == exp4321.V399_ROLE_ID)
    assert role["v399_state"] == exp4321.V399_STATE
    assert role["arc_total_levels"] == 23
    assert role["off_arc_demofit_beats_vote"] is True
    assert exp4321.registry_contains_v399(registry) is True
    assert exp4321.registry_contains_v399({}) is False

    missing_registry: dict[str, Any] = {}
    exp4321._ensure_gap4_eval(missing_registry, guard, bundle, gaps_logged)
    assert missing_registry["verifiers"][0]["verifier_id"] == exp4321.GAP4_VERIFIER_ID
    empty_registry: dict[str, Any] = {}
    exp4321._ensure_v399_role(empty_registry, bundle, gaps_logged)
    assert empty_registry == {}

    assert "Historical note remains." in gaps_text
    assert "exp4321-gap-cross-domain-family-invariant-selection-4305:start" in gaps_text
    assert "exp4321-gap-4318:start" in gaps_text
    assert "exp4321-gap-code-exec-demofit:start" in gaps_text
    assert "status: filled (gap4_code_demo_fit_execution_transfer_4319)" in gaps_text
    assert manifest == _minimal_manifest()

    failed_moat = {
        **bundle,
        "v399_outcomes": {
            **bundle["v399_outcomes"],
            "in_generation": {
                **bundle["v399_outcomes"]["in_generation"],
                "diffusiongemma_guidance_moat": False,
                "scorer_leak_recheck_passed": False,
            },
        },
    }
    leak_gaps = exp4321.build_gap_entries(failed_moat)
    assert [entry["gap_id"] for entry in leak_gaps] == [
        exp4321.GAP_CROSS_DOMAIN_FAMILY_INVARIANT_SELECTION,
        exp4321.GAP_DIFFUSIONGEMMA_LEAK_FREE_STEERING,
        exp4321.GAP_GAME_INVARIANT_ARC_VALUE,
        exp4321.GAP_CODE_EXEC_DEMOFIT,
    ]


def test_req_4321_build_artifact_validates_required_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-4321: terminal artifact exposes the required schema fields."""

    _write_minimal_repo(tmp_path)
    guard = exp4321.run_gap4_regression_guard(tmp_path)
    bundle = exp4321.load_v399_outcomes(tmp_path)
    gaps_logged = exp4321.build_gap_entries(bundle)
    artifact = exp4321.build_artifact(
        regression_guard=guard,
        outcome_bundle=bundle,
        gaps_logged=gaps_logged,
        registry_reconciled=True,
        manifest_reconciled=True,
        reproducibility_checksum="sha256:" + "0" * 64,
        duration_s=0.25,
    )

    exp4321.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["regression_guard_passed"] is True
    assert artifact["registry_reconciled"] is True
    assert artifact["manifest_reconciled"] is True
    assert artifact["gaps_logged"] == gaps_logged
    assert artifact["field_principles"] == exp4321.FIELD_PRINCIPLES
    assert artifact["model_specs"]["method"] == "cached_v399_ledger_reconciliation"
    assert artifact["inference_substrate"] == exp4321.INFERENCE_SUBSTRATE

    for field in exp4321.REQUIRED_ARTIFACT_FIELDS:
        malformed = dict(artifact)
        malformed.pop(field)
        with pytest.raises(ValueError, match=field):
            exp4321.validate_artifact(malformed)
    with pytest.raises(ValueError, match="terminal prefix"):
        exp4321.validate_artifact({**artifact, "honest_verdict": "not terminal"})
    with pytest.raises(ValueError, match="BARE bool"):
        exp4321.validate_artifact({**artifact, "regression_guard_passed": "yes"})
    with pytest.raises(ValueError, match="registry_reconciled"):
        exp4321.validate_artifact({**artifact, "registry_reconciled": "yes"})
    with pytest.raises(ValueError, match="manifest_reconciled"):
        exp4321.validate_artifact({**artifact, "manifest_reconciled": "yes"})
    with pytest.raises(ValueError, match="gaps_logged"):
        exp4321.validate_artifact({**artifact, "gaps_logged": "gap"})
    with pytest.raises(ValueError, match="gap entry"):
        exp4321.validate_artifact({**artifact, "gaps_logged": [{"gap_id": "GAP"}]})
    with pytest.raises(ValueError, match="v399_outcomes"):
        exp4321.validate_artifact({**artifact, "v399_outcomes": []})
    with pytest.raises(ValueError, match="availability_report"):
        exp4321.validate_artifact({**artifact, "availability_report": []})
    with pytest.raises(ValueError, match="random_seed"):
        exp4321.validate_artifact({**artifact, "random_seed": True})
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp4321.validate_artifact({**artifact, "reproducibility_checksum": ""})
    with pytest.raises(ValueError, match="model_specs"):
        exp4321.validate_artifact({**artifact, "model_specs": {}})
    with pytest.raises(ValueError, match="field_principles"):
        exp4321.validate_artifact({**artifact, "field_principles": {}})
    with pytest.raises(ValueError, match="spec_refs"):
        exp4321.validate_artifact({**artifact, "spec_refs": ["REQ-VERIFY-4321"]})
    with pytest.raises(ValueError, match="inference_substrate"):
        exp4321.validate_artifact({**artifact, "inference_substrate": "live"})


def test_req_4321_blocks_only_unparseable_ledgers(tmp_path: Path) -> None:
    """REQ-VERIFY-4321: ledger parse failure blocks honestly before mutation."""

    _write_minimal_repo(tmp_path)
    (tmp_path / "ops" / "verifier_registry.yaml").write_text("[not: registry]\n", encoding="utf-8")

    preflight = exp4321.check_preconditions(tmp_path)
    assert preflight["ok"] is False
    assert preflight["blocked_resource"] == "verifier_registry"

    artifact = exp4321.run_hygiene(tmp_path)
    assert artifact["honest_verdict"] == "blocked_ledgers_unparseable"
    assert artifact["regression_guard_passed"] is False
    assert artifact["gaps_logged"] == []
    assert artifact["registry_reconciled"] is False
    assert artifact["manifest_reconciled"] is False


def test_req_4321_defensive_helpers_cover_missing_and_malformed_shapes(tmp_path: Path) -> None:
    """REQ-VERIFY-4321: malformed optional artifacts remain axis-local."""

    _write_minimal_repo(
        tmp_path,
        omit={
            exp4321.EXP4314_PATH,
            exp4321.EXP4316_PATH,
            exp4321.EXP4317_PATH,
            exp4321.EXP4318_PATH,
            exp4321.EXP4319_PATH,
        },
    )

    bundle = exp4321.load_v399_outcomes(tmp_path)
    assert bundle["v399_outcomes"]["cross_domain"] == {
        "artifact_path": exp4321.EXP4314_PATH,
        "available": False,
        "missing_verifier_gaps": [],
    }
    assert bundle["v399_outcomes"]["efficiency_cascade"] == {
        "artifact_path": exp4321.EXP4316_PATH,
        "available": False,
    }
    assert bundle["v399_outcomes"]["arc_progress"] == {
        "artifact_path": exp4321.EXP4317_PATH,
        "available": False,
    }
    assert bundle["v399_outcomes"]["cross_game_transfer"] == {
        "artifact_path": exp4321.EXP4318_PATH,
        "available": False,
        "missing_verifier_gaps": [],
    }
    assert bundle["v399_outcomes"]["off_arc_execution"] == {
        "artifact_path": exp4321.EXP4319_PATH,
        "available": False,
        "missing_verifier_gaps": [],
    }

    corrupt = tmp_path / exp4321.EXP4316_PATH
    corrupt.parent.mkdir(parents=True, exist_ok=True)
    corrupt.write_text("{not-json", encoding="utf-8")
    assert exp4321._load_optional_json(tmp_path, exp4321.EXP4316_PATH)[1].startswith(
        "JSONDecodeError"
    )
    corrupt.write_text("[1, 2, 3]", encoding="utf-8")
    assert exp4321._load_optional_json(tmp_path, exp4321.EXP4316_PATH) == (
        None,
        "artifact must parse as an object",
    )

    blank_gaps = tmp_path / "ops" / "verifier_gaps.md"
    original_gaps = blank_gaps.read_text(encoding="utf-8")
    blank_gaps.write_text("", encoding="utf-8")
    assert exp4321.check_preconditions(tmp_path)["blocked_resource"] == "verifier_gaps"
    blank_gaps.write_text(original_gaps, encoding="utf-8")
    manifest = tmp_path / "ops" / "exclusion_manifest.yaml"
    manifest.write_text("- not-a-mapping\n", encoding="utf-8")
    assert exp4321.check_preconditions(tmp_path)["blocked_resource"] == "exclusion_manifest"

    normal = exp4321.load_v399_outcomes(REPO_ROOT)
    invalid_upstream = {
        **normal,
        "v399_outcomes": {
            **normal["v399_outcomes"],
            "cross_domain": {
                **normal["v399_outcomes"]["cross_domain"],
                "cross_domain_selection_holds": False,
                "missing_verifier_gaps": ["bad"],
            },
            "cross_game_transfer": {
                **normal["v399_outcomes"]["cross_game_transfer"],
                "cross_game_transfer_helps": False,
                "missing_verifier_gaps": ["bad"],
            },
        },
    }
    assert [gap["gap_id"] for gap in exp4321.build_gap_entries(invalid_upstream)] == [
        exp4321.GAP_CROSS_DOMAIN_FAMILY_INVARIANT_SELECTION,
        exp4321.GAP_GAME_INVARIANT_ARC_VALUE,
        exp4321.GAP_CODE_EXEC_DEMOFIT
    ]


def test_req_4321_results_entrypoint_writes_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-VERIFY-4321: results entrypoint calls the package runner."""

    _write_minimal_repo(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(RESULTS_WRAPPER_PATH)])
    monkeypatch.setattr(exp4321, "REPO_ROOT", tmp_path)
    python_root = str(REPO_ROOT / "python")
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if entry != python_root])

    runpy.run_path(str(REPO_ROOT / RESULTS_WRAPPER_PATH), run_name="__main__")

    payload = json.loads((tmp_path / exp4321.EXP4321_ARTIFACT_PATH).read_text(encoding="utf-8"))
    assert payload["honest_verdict"].startswith("complete:")
    assert payload["regression_guard_passed"] is True
    assert [gap["gap_id"] for gap in payload["gaps_logged"]] == [
        exp4321.GAP_CROSS_DOMAIN_FAMILY_INVARIANT_SELECTION,
        exp4321.GAP_GAME_INVARIANT_ARC_VALUE,
        exp4321.GAP_CODE_EXEC_DEMOFIT,
    ]
