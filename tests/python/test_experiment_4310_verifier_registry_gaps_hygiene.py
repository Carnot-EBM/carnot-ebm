"""Tests for Exp 4310 verifier registry/gaps hygiene.

Spec refs: REQ-VERIFY-4310, SCENARIO-VERIFY-4310.
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

from carnot import experiment_4310_verifier_registry_gaps_hygiene as exp4310_wrapper
from carnot.reporting import verifier_registry_gaps_hygiene_4310 as exp4310


REPO_ROOT = Path(__file__).parents[2]
SPEC_PATH = Path("openspec/capabilities/verification/spec.md")
RESULTS_WRAPPER_PATH = Path("results/experiment_4310_verifier_registry_gaps_hygiene.py")


def _minimal_registry() -> dict[str, Any]:
    return {
        "verifiers": [
            {
                "verifier_id": exp4310.GAP4_VERIFIER_ID,
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
    for path in exp4310.REQUIRED_COPY_PATHS:
        if path in omit:
            continue
        source = REPO_ROOT / path
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def test_req_4310_spec_declared() -> None:
    """REQ-VERIFY-4310: OpenSpec declares the .398 hygiene artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-4310",
        "SCENARIO-VERIFY-4310",
        "python/carnot/experiment_4310_verifier_registry_gaps_hygiene.py",
        RESULTS_WRAPPER_PATH.as_posix(),
        exp4310.EXP4310_ARTIFACT_PATH,
        "blocked_ledgers_unparseable",
        "aggregate_available_report_gaps",
        "efficiency_pareto_holds=true",
        "diffusiongemma_guidance_moat=false",
        "controls_differentiated=true",
        "cross_domain_selection_holds=false",
        "label_ablation_robust=true",
        "online_adaptation_helps=true",
        "total_levels=22",
        exp4310.GAP_CROSS_DOMAIN_FAMILY_INVARIANT_SELECTION,
    ):
        assert marker in spec
    assert exp4310.FIELD_PRINCIPLES["honest_verdict"] in spec
    assert exp4310.FIELD_PRINCIPLES["regression_guard_passed"] in spec
    assert exp4310.FIELD_PRINCIPLES["gaps_logged"] in spec
    assert exp4310.FIELD_PRINCIPLES["reproducibility_checksum"] in spec
    assert exp4310_wrapper.main is exp4310.main


def test_scenario_4310_preconditions_outcomes_and_robust_availability(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-4310: missing .398 artifacts are per-axis gaps only."""

    _write_minimal_repo(tmp_path, omit={exp4310.EXP4304_PATH})

    preflight = exp4310.check_preconditions(tmp_path)
    assert preflight["ok"] is True
    assert preflight["blocked_resource"] is None

    guard = exp4310.run_gap4_regression_guard(tmp_path)
    assert guard["regression_guard_passed"] is True
    assert guard["prior_artifact_path"] == exp4310.EXP4287_PATH
    assert guard["blocked_exp4299_seen"] is True
    assert guard["recorded_arc1_rule_exec"] == {
        "n": 31,
        "vote_pass2": pytest.approx(0.4516),
        "gated_pass2": pytest.approx(0.5806),
        "headroom_recovered": 4,
        "vote_wins_lost": 0,
    }
    assert guard["replayed_arc1_rule_exec"] == guard["recorded_arc1_rule_exec"]

    bundle = exp4310.load_v398_outcomes(tmp_path)
    outcomes = bundle["v398_outcomes"]
    availability = bundle["availability_report"]
    assert availability["axes"]["efficiency"]["verdict"] is True
    assert availability["axes"]["in_generation"]["missing_artifacts"] == [
        {"axis": "in_generation", "artifact_key": "4304_in_generation", "experiment_id": 4304}
    ]
    assert availability["axes"]["arc_progress"]["flagged_artifacts"] == [
        {
            "axis": "arc_progress",
            "artifact_key": "4307_arc_progress",
            "experiment_id": 4307,
            "reason": "flagged_adversarial",
        }
    ]
    assert outcomes["efficiency"]["efficiency_pareto_holds"] is True
    assert outcomes["efficiency"]["cost_ratio"] == pytest.approx(1.03e-8)
    assert outcomes["in_generation"]["available"] is False
    assert outcomes["cross_domain"]["cross_domain_selection_holds"] is False
    assert outcomes["cross_domain"]["label_ablation_robust"] is True
    assert outcomes["cross_domain"]["cross_domain_delta"] == pytest.approx(0.2307692308)
    assert outcomes["self_learning"]["online_adaptation_helps"] is True
    assert outcomes["self_learning"]["best_adaptive_minus_static_delta"] == pytest.approx(
        0.5292929293
    )
    assert outcomes["arc_progress"]["total_levels"] == 22
    assert outcomes["arc_progress"]["new_levels_solved_this_task"] == 0
    assert outcomes["arc_progress"]["flagged_adversarial"] is True


def test_req_4310_ledgers_record_v398_truth_and_gap(tmp_path: Path) -> None:
    """REQ-VERIFY-4310: registry and gaps carry the .398 truth."""

    _write_minimal_repo(tmp_path)
    guard = exp4310.run_gap4_regression_guard(tmp_path)
    bundle = exp4310.load_v398_outcomes(tmp_path)
    gaps_logged = exp4310.build_gap_entries(bundle)

    registry, gaps_text, manifest, summary = exp4310.ensure_ledgers_record_v398(
        _minimal_registry(),
        "# Verifier Gaps\n\nHistorical note remains.\n",
        _minimal_manifest(),
        guard,
        bundle,
        gaps_logged,
    )

    assert [entry["gap_id"] for entry in gaps_logged] == [
        exp4310.GAP_CROSS_DOMAIN_FAMILY_INVARIANT_SELECTION
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
        "gaps_logged_ids": [exp4310.GAP_CROSS_DOMAIN_FAMILY_INVARIANT_SELECTION],
    }

    gap4 = registry["verifiers"][0]
    assert gap4["eval"]["eval_exp_4310"] == exp4310.EXP4310_ARTIFACT_PATH
    assert gap4["eval"]["exp4310_regression_guard_passed"] is True
    assert gap4["eval"]["exp4310_robust_aggregator_used"] is True
    assert gap4["eval"]["exp4310_efficiency_pareto_holds"] is True
    assert gap4["eval"]["exp4310_efficiency_cost_ratio"] == pytest.approx(1.03e-8)
    assert gap4["eval"]["exp4310_diffusiongemma_guidance_moat"] is False
    assert gap4["eval"]["exp4310_controls_differentiated"] is True
    assert gap4["eval"]["exp4310_scorer_leak_recheck_passed"] is True
    assert gap4["eval"]["exp4310_cross_domain_selection_holds"] is False
    assert gap4["eval"]["exp4310_label_ablation_robust"] is True
    assert gap4["eval"]["exp4310_online_adaptation_helps"] is True
    assert gap4["eval"]["exp4310_arc_total_levels"] == 22
    assert gap4["eval"]["exp4310_arc_flagged_adversarial"] is True
    assert gap4["eval"]["exp4310_v398_state"] == exp4310.V398_STATE
    assert gap4["eval"]["exp4310_gaps_logged"] == [
        exp4310.GAP_CROSS_DOMAIN_FAMILY_INVARIANT_SELECTION
    ]
    role = next(role for role in gap4["registry_roles"] if role["role_id"] == exp4310.V398_ROLE_ID)
    assert role["v398_state"] == exp4310.V398_STATE
    assert role["cross_domain_selection_holds"] is False
    assert role["arc_total_levels"] == 22
    assert exp4310.registry_contains_v398(registry) is True
    assert exp4310.registry_contains_v398({}) is False

    missing_registry: dict[str, Any] = {}
    exp4310._ensure_gap4_eval(missing_registry, guard, bundle, gaps_logged)
    assert missing_registry["verifiers"][0]["verifier_id"] == exp4310.GAP4_VERIFIER_ID
    empty_registry: dict[str, Any] = {}
    exp4310._ensure_v398_role(empty_registry, bundle, gaps_logged)
    assert empty_registry == {}

    assert "Historical note remains." in gaps_text
    assert exp4310.GAP_CROSS_DOMAIN_FAMILY_INVARIANT_SELECTION in gaps_text
    assert "domain-invariant selector features" in gaps_text
    assert manifest == _minimal_manifest()

    failed_leak = {
        **bundle,
        "v398_outcomes": {
            **bundle["v398_outcomes"],
            "in_generation": {
                **bundle["v398_outcomes"]["in_generation"],
                "scorer_leak_recheck_passed": False,
            },
        },
    }
    leak_gaps = exp4310.build_gap_entries(failed_leak)
    assert [entry["gap_id"] for entry in leak_gaps] == [
        exp4310.GAP_DIFFUSIONGEMMA_LEAK_FREE_PARTIAL_STATE,
        exp4310.GAP_CROSS_DOMAIN_FAMILY_INVARIANT_SELECTION,
    ]


def test_req_4310_build_artifact_validates_required_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-4310: terminal artifact exposes the required schema fields."""

    _write_minimal_repo(tmp_path)
    guard = exp4310.run_gap4_regression_guard(tmp_path)
    bundle = exp4310.load_v398_outcomes(tmp_path)
    gaps_logged = exp4310.build_gap_entries(bundle)
    artifact = exp4310.build_artifact(
        regression_guard=guard,
        outcome_bundle=bundle,
        gaps_logged=gaps_logged,
        registry_reconciled=True,
        manifest_reconciled=True,
        reproducibility_checksum="sha256:" + "0" * 64,
        duration_s=0.25,
    )

    exp4310.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["regression_guard_passed"] is True
    assert artifact["registry_reconciled"] is True
    assert artifact["manifest_reconciled"] is True
    assert artifact["gaps_logged"] == gaps_logged
    assert artifact["field_principles"] == exp4310.FIELD_PRINCIPLES
    assert artifact["model_specs"]["method"] == "cached_v398_ledger_reconciliation"
    assert artifact["inference_substrate"] == exp4310.INFERENCE_SUBSTRATE

    for field in exp4310.REQUIRED_ARTIFACT_FIELDS:
        malformed = dict(artifact)
        malformed.pop(field)
        with pytest.raises(ValueError, match=field):
            exp4310.validate_artifact(malformed)
    with pytest.raises(ValueError, match="terminal prefix"):
        exp4310.validate_artifact({**artifact, "honest_verdict": "not terminal"})
    with pytest.raises(ValueError, match="BARE bool"):
        exp4310.validate_artifact({**artifact, "regression_guard_passed": "yes"})
    with pytest.raises(ValueError, match="registry_reconciled"):
        exp4310.validate_artifact({**artifact, "registry_reconciled": "yes"})
    with pytest.raises(ValueError, match="manifest_reconciled"):
        exp4310.validate_artifact({**artifact, "manifest_reconciled": "yes"})
    with pytest.raises(ValueError, match="gaps_logged"):
        exp4310.validate_artifact({**artifact, "gaps_logged": "gap"})
    with pytest.raises(ValueError, match="gap entry"):
        exp4310.validate_artifact({**artifact, "gaps_logged": [{"gap_id": "GAP"}]})
    with pytest.raises(ValueError, match="v398_outcomes"):
        exp4310.validate_artifact({**artifact, "v398_outcomes": []})
    with pytest.raises(ValueError, match="availability_report"):
        exp4310.validate_artifact({**artifact, "availability_report": []})
    with pytest.raises(ValueError, match="random_seed"):
        exp4310.validate_artifact({**artifact, "random_seed": True})
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp4310.validate_artifact({**artifact, "reproducibility_checksum": ""})
    with pytest.raises(ValueError, match="model_specs"):
        exp4310.validate_artifact({**artifact, "model_specs": {}})
    with pytest.raises(ValueError, match="field_principles"):
        exp4310.validate_artifact({**artifact, "field_principles": {}})
    with pytest.raises(ValueError, match="spec_refs"):
        exp4310.validate_artifact({**artifact, "spec_refs": ["REQ-VERIFY-4310"]})
    with pytest.raises(ValueError, match="inference_substrate"):
        exp4310.validate_artifact({**artifact, "inference_substrate": "live"})


def test_req_4310_blocks_only_unparseable_ledgers(tmp_path: Path) -> None:
    """REQ-VERIFY-4310: ledger parse failure blocks honestly before mutation."""

    _write_minimal_repo(tmp_path)
    (tmp_path / "ops" / "verifier_registry.yaml").write_text("[not: registry]\n", encoding="utf-8")

    preflight = exp4310.check_preconditions(tmp_path)
    assert preflight["ok"] is False
    assert preflight["blocked_resource"] == "verifier_registry"

    artifact = exp4310.run_hygiene(tmp_path)
    assert artifact["honest_verdict"] == "blocked_ledgers_unparseable"
    assert artifact["regression_guard_passed"] is False
    assert artifact["gaps_logged"] == []
    assert artifact["registry_reconciled"] is False
    assert artifact["manifest_reconciled"] is False


def test_req_4310_defensive_helpers_cover_missing_and_malformed_shapes(tmp_path: Path) -> None:
    """REQ-VERIFY-4310: malformed optional artifacts remain axis-local."""

    _write_minimal_repo(
        tmp_path,
        omit={
            exp4310.EXP4299_PATH,
            exp4310.EXP4303_PATH,
            exp4310.EXP4305_PATH,
            exp4310.EXP4306_PATH,
            exp4310.EXP4307_PATH,
        },
    )

    guard = exp4310.run_gap4_regression_guard(tmp_path)
    assert guard["regression_guard_passed"] is True
    assert guard["blocked_exp4299_seen"] is True
    assert guard["prior_artifact_path"] == exp4310.EXP4287_PATH

    bundle = exp4310.load_v398_outcomes(tmp_path)
    assert bundle["v398_outcomes"]["efficiency"] == {
        "artifact_path": exp4310.EXP4303_PATH,
        "available": False,
    }
    assert bundle["v398_outcomes"]["cross_domain"] == {
        "artifact_path": exp4310.EXP4305_PATH,
        "available": False,
        "missing_verifier_gaps": [],
    }
    assert bundle["v398_outcomes"]["self_learning"] == {
        "artifact_path": exp4310.EXP4306_PATH,
        "available": False,
    }
    assert bundle["v398_outcomes"]["arc_progress"] == {
        "artifact_path": exp4310.EXP4307_PATH,
        "available": False,
    }

    corrupt = tmp_path / exp4310.EXP4303_PATH
    corrupt.parent.mkdir(parents=True, exist_ok=True)
    corrupt.write_text("{not-json", encoding="utf-8")
    assert exp4310._load_optional_json(tmp_path, exp4310.EXP4303_PATH)[1].startswith(
        "JSONDecodeError"
    )
    corrupt.write_text("[1, 2, 3]", encoding="utf-8")
    assert exp4310._load_optional_json(tmp_path, exp4310.EXP4303_PATH) == (
        None,
        "artifact must parse as an object",
    )

    blank_gaps = tmp_path / "ops" / "verifier_gaps.md"
    original_gaps = blank_gaps.read_text(encoding="utf-8")
    blank_gaps.write_text("", encoding="utf-8")
    assert exp4310.check_preconditions(tmp_path)["blocked_resource"] == "verifier_gaps"
    blank_gaps.write_text(original_gaps, encoding="utf-8")
    manifest = tmp_path / "ops" / "exclusion_manifest.yaml"
    manifest.write_text("- not-a-mapping\n", encoding="utf-8")
    assert exp4310.check_preconditions(tmp_path)["blocked_resource"] == "exclusion_manifest"

    normal = exp4310.load_v398_outcomes(REPO_ROOT)
    invalid_upstream = {
        **normal,
        "v398_outcomes": {
            **normal["v398_outcomes"],
            "cross_domain": {
                **normal["v398_outcomes"]["cross_domain"],
                "cross_domain_selection_holds": False,
                "missing_verifier_gaps": ["bad", *normal["v398_outcomes"]["cross_domain"]["missing_verifier_gaps"]],
            },
        },
    }
    assert [gap["gap_id"] for gap in exp4310.build_gap_entries(invalid_upstream)] == [
        exp4310.GAP_CROSS_DOMAIN_FAMILY_INVARIANT_SELECTION
    ]

    fallback_gap = {
        **normal,
        "v398_outcomes": {
            **normal["v398_outcomes"],
            "cross_domain": {
                **normal["v398_outcomes"]["cross_domain"],
                "cross_domain_selection_holds": False,
                "missing_verifier_gaps": [],
            },
        },
    }
    assert exp4310.build_gap_entries(fallback_gap)[0]["gap_id"] == (
        exp4310.GAP_CROSS_DOMAIN_FAMILY_INVARIANT_SELECTION
    )


def test_req_4310_results_entrypoint_writes_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-VERIFY-4310: results entrypoint calls the package runner."""

    _write_minimal_repo(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(RESULTS_WRAPPER_PATH)])
    monkeypatch.setattr(exp4310, "REPO_ROOT", tmp_path)
    python_root = str(REPO_ROOT / "python")
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if entry != python_root])

    runpy.run_path(str(REPO_ROOT / RESULTS_WRAPPER_PATH), run_name="__main__")

    payload = json.loads((tmp_path / exp4310.EXP4310_ARTIFACT_PATH).read_text(encoding="utf-8"))
    assert payload["honest_verdict"].startswith("complete:")
    assert payload["regression_guard_passed"] is True
    assert payload["gaps_logged"][0]["gap_id"] == exp4310.GAP_CROSS_DOMAIN_FAMILY_INVARIANT_SELECTION
