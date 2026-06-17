"""Tests for Exp 4333 verifier registry/gaps hygiene.

Spec refs: REQ-VERIFY-4333, SCENARIO-VERIFY-4333.
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

from carnot import experiment_4333_verifier_registry_gaps_hygiene as exp4333_wrapper
from carnot.reporting import verifier_registry_gaps_hygiene_4333 as exp4333


REPO_ROOT = Path(__file__).parents[2]
SPEC_PATH = Path("openspec/capabilities/verification/spec.md")
RESULTS_WRAPPER_PATH = Path("results/experiment_4333_verifier_registry_gaps_hygiene.py")


def _minimal_registry() -> dict[str, Any]:
    return {
        "verifiers": [
            {
                "verifier_id": exp4333.GAP4_VERIFIER_ID,
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
    for path in exp4333.REQUIRED_COPY_PATHS:
        if path in omit:
            continue
        source = REPO_ROOT / path
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def test_req_4333_spec_declared() -> None:
    """REQ-VERIFY-4333: OpenSpec declares the .400 hygiene artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-4333",
        "SCENARIO-VERIFY-4333",
        "python/carnot/experiment_4333_verifier_registry_gaps_hygiene.py",
        RESULTS_WRAPPER_PATH.as_posix(),
        exp4333.EXP4333_ARTIFACT_PATH,
        "blocked_ledgers_unparseable",
        "aggregate_available_report_gaps",
        "in_generation_moat_replicates=false",
        "adaptive_guidance_beats_control=false",
        "reproducible_total_levels=13",
        "learned_encoder_transfer_helps=false",
        exp4333.GAP_ARC_GRID_GENERATION_SCORER,
        exp4333.GAP_E3_WORLD_MODEL_RULE_AR25,
        exp4333.GAP_GAME_INVARIANT_ARC_VALUE_4331,
    ):
        assert marker in spec
    assert exp4333.FIELD_PRINCIPLES["honest_verdict"] in spec
    assert exp4333.FIELD_PRINCIPLES["regression_guard_passed"] in spec
    assert exp4333.FIELD_PRINCIPLES["gaps_logged"] in spec
    assert exp4333.FIELD_PRINCIPLES["reproducibility_checksum"] in spec
    assert exp4333_wrapper.main is exp4333.main


def test_scenario_4333_preconditions_outcomes_and_robust_availability(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-4333: missing .400 artifacts are per-axis gaps only."""

    _write_minimal_repo(tmp_path, omit={exp4333.EXP4328_PATH})

    preflight = exp4333.check_preconditions(tmp_path)
    assert preflight["ok"] is True
    assert preflight["blocked_resource"] is None

    guard = exp4333.run_gap4_regression_guard(tmp_path)
    assert guard["regression_guard_passed"] is True
    assert guard["prior_artifact_path"] == exp4333.EXP4321_PATH
    assert guard["recorded_arc1_rule_exec"] == {
        "n": 31,
        "vote_pass2": pytest.approx(0.4516),
        "gated_pass2": pytest.approx(0.5806),
        "headroom_recovered": 4,
        "vote_wins_lost": 0,
    }
    assert guard["replayed_arc1_rule_exec"] == guard["recorded_arc1_rule_exec"]

    bundle = exp4333.load_v400_outcomes(tmp_path)
    outcomes = bundle["v400_outcomes"]
    availability = bundle["availability_report"]
    assert availability["axes"]["in_generation_replication"]["verdict"] is False
    assert availability["axes"]["adaptive_scaleup"]["verdict"] is False
    assert availability["axes"]["e3_deep_tail"]["missing_artifacts"] == [
        {
            "axis": "e3_deep_tail",
            "artifact_key": "4328_e3_ka59",
            "experiment_id": 4328,
        }
    ]
    assert availability["axes"]["learned_encoder_transfer"]["verdict"] is False
    assert outcomes["in_generation_replication"]["in_generation_moat_replicates"] is False
    assert outcomes["in_generation_replication"]["scorer_leak_recheck_passed"] is False
    assert outcomes["adaptive_scaleup"]["adaptive_guidance_beats_control"] is False
    assert outcomes["adaptive_scaleup"]["domain_used"] == "reasoning_corpus_fallback"
    assert outcomes["e3_deep_tail"]["games"]["ar25"]["residual_mismatch_class"] == (
        "missing_world_model_rule_gap_hidden_undo_stack_action7"
    )
    assert outcomes["e3_deep_tail"]["games"]["ka59"]["available"] is False
    assert outcomes["shallow_tail_sweep"]["reproducible_total_levels"] == 13
    assert outcomes["shallow_tail_sweep"]["tn36_schema_finding"]["normalizer"] == (
        "normalise_tn36_click_payload"
    )
    assert outcomes["learned_encoder_transfer"]["learned_encoder_transfer_helps"] is False
    assert outcomes["learned_encoder_transfer"]["n_held_out_levels"] == 13
    assert exp4333.robust_aggregator_ok(outcomes["robust_aggregator"]) is True


def test_req_4333_ledgers_record_v400_truth_gaps_and_retirement(tmp_path: Path) -> None:
    """REQ-VERIFY-4333: registry, gaps, and manifest carry the .400 truth."""

    _write_minimal_repo(tmp_path)
    guard = exp4333.run_gap4_regression_guard(tmp_path)
    bundle = exp4333.load_v400_outcomes(tmp_path)
    gaps_logged = exp4333.build_gap_entries(bundle)

    registry, gaps_text, manifest, summary = exp4333.ensure_ledgers_record_v400(
        _minimal_registry(),
        "# Verifier Gaps\n\nHistorical note remains.\n",
        _minimal_manifest(),
        guard,
        bundle,
        gaps_logged,
    )

    assert [entry["gap_id"] for entry in gaps_logged] == [
        exp4333.GAP_DIFFUSIONGEMMA_SECOND_CORPUS_SCORER,
        exp4333.GAP_ARC_GRID_GENERATION_SCORER,
        exp4333.GAP_E3_WORLD_MODEL_RULE_AR25,
        exp4333.GAP_E3_WORLD_MODEL_RULE_KA59,
        exp4333.GAP_E3_WORLD_MODEL_RULE_TR87,
        exp4333.GAP_E3_WORLD_MODEL_RULE_FT09,
        exp4333.GAP_GAME_INVARIANT_ARC_VALUE_4331,
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
        "gaps_logged_ids": [entry["gap_id"] for entry in gaps_logged],
    }

    gap4 = registry["verifiers"][0]
    assert gap4["eval"]["eval_exp_4333"] == exp4333.EXP4333_ARTIFACT_PATH
    assert gap4["eval"]["exp4333_regression_guard_passed"] is True
    assert gap4["eval"]["exp4333_robust_aggregator_used"] is True
    assert gap4["eval"]["exp4333_in_generation_moat_replicates"] is False
    assert gap4["eval"]["exp4333_scorer_leak_recheck_passed"] is False
    assert gap4["eval"]["exp4333_adaptive_guidance_beats_control"] is False
    assert gap4["eval"]["exp4333_adaptive_domain_used"] == "reasoning_corpus_fallback"
    assert gap4["eval"]["exp4333_e3_reproduced_levels_total"] == 0
    assert gap4["eval"]["exp4333_e3_ar25_residual_mismatch_class"] == (
        "missing_world_model_rule_gap_hidden_undo_stack_action7"
    )
    assert gap4["eval"]["exp4333_shallow_reproducible_total_levels"] == 13
    assert gap4["eval"]["exp4333_tn36_schema"] == "ACTION6 data must be top-level {\"x\": int, \"y\": int}"
    assert gap4["eval"]["exp4333_learned_encoder_transfer_helps"] is False
    assert gap4["eval"]["exp4333_cross_game_state_reduction"] == pytest.approx(1.0084925690021231)
    assert gap4["eval"]["exp4333_v400_state"] == exp4333.V400_STATE
    assert gap4["eval"]["exp4333_gaps_logged"] == [entry["gap_id"] for entry in gaps_logged]
    role = next(role for role in gap4["registry_roles"] if role["role_id"] == exp4333.V400_ROLE_ID)
    assert role["v400_state"] == exp4333.V400_STATE
    assert role["learned_encoder_transfer_helps"] is False
    assert exp4333.registry_contains_v400(registry) is True
    assert exp4333.registry_contains_v400({}) is False

    missing_registry: dict[str, Any] = {}
    exp4333._ensure_gap4_eval(missing_registry, guard, bundle, gaps_logged)
    assert missing_registry["verifiers"][0]["verifier_id"] == exp4333.GAP4_VERIFIER_ID
    empty_registry: dict[str, Any] = {}
    exp4333._ensure_v400_role(empty_registry, bundle, gaps_logged)
    assert empty_registry == {}

    assert "Historical note remains." in gaps_text
    assert "exp4333-gap-diffusiongemma-second-corpus-leak-free-scorer-4325:start" in gaps_text
    assert "exp4333-gap-arc-grid-generation-scorer-4326:start" in gaps_text
    assert "exp4333-gap-e3-world-model-rule-ar25-4327:start" in gaps_text
    assert "exp4333-gap-4331:start" in gaps_text
    assert exp4333.manifest_contains_cross_domain_retirement(manifest) is True


def test_req_4333_build_artifact_validates_required_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-4333: terminal artifact exposes the required schema fields."""

    _write_minimal_repo(tmp_path)
    guard = exp4333.run_gap4_regression_guard(tmp_path)
    bundle = exp4333.load_v400_outcomes(tmp_path)
    gaps_logged = exp4333.build_gap_entries(bundle)
    artifact = exp4333.build_artifact(
        regression_guard=guard,
        outcome_bundle=bundle,
        gaps_logged=gaps_logged,
        registry_reconciled=True,
        manifest_reconciled=True,
        reproducibility_checksum="sha256:" + "0" * 64,
        duration_s=0.25,
    )

    exp4333.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["regression_guard_passed"] is True
    assert artifact["registry_reconciled"] is True
    assert artifact["manifest_reconciled"] is True
    assert artifact["gaps_logged"] == gaps_logged
    assert artifact["field_principles"] == exp4333.FIELD_PRINCIPLES
    assert artifact["model_specs"]["method"] == "cached_v400_ledger_reconciliation"
    assert artifact["inference_substrate"] == exp4333.INFERENCE_SUBSTRATE

    for field in exp4333.REQUIRED_ARTIFACT_FIELDS:
        malformed = dict(artifact)
        malformed.pop(field)
        with pytest.raises(ValueError, match=field):
            exp4333.validate_artifact(malformed)
    with pytest.raises(ValueError, match="terminal prefix"):
        exp4333.validate_artifact({**artifact, "honest_verdict": "not terminal"})
    with pytest.raises(ValueError, match="BARE bool"):
        exp4333.validate_artifact({**artifact, "regression_guard_passed": "yes"})
    with pytest.raises(ValueError, match="registry_reconciled"):
        exp4333.validate_artifact({**artifact, "registry_reconciled": "yes"})
    with pytest.raises(ValueError, match="manifest_reconciled"):
        exp4333.validate_artifact({**artifact, "manifest_reconciled": "yes"})
    with pytest.raises(ValueError, match="gaps_logged"):
        exp4333.validate_artifact({**artifact, "gaps_logged": "gap"})
    with pytest.raises(ValueError, match="gap entry"):
        exp4333.validate_artifact({**artifact, "gaps_logged": [{"gap_id": "GAP"}]})
    with pytest.raises(ValueError, match="v400_outcomes"):
        exp4333.validate_artifact({**artifact, "v400_outcomes": []})
    with pytest.raises(ValueError, match="availability_report"):
        exp4333.validate_artifact({**artifact, "availability_report": []})
    with pytest.raises(ValueError, match="random_seed"):
        exp4333.validate_artifact({**artifact, "random_seed": True})
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp4333.validate_artifact({**artifact, "reproducibility_checksum": ""})
    with pytest.raises(ValueError, match="model_specs"):
        exp4333.validate_artifact({**artifact, "model_specs": {}})
    with pytest.raises(ValueError, match="field_principles"):
        exp4333.validate_artifact({**artifact, "field_principles": {}})
    with pytest.raises(ValueError, match="spec_refs"):
        exp4333.validate_artifact({**artifact, "spec_refs": ["REQ-VERIFY-4333"]})
    with pytest.raises(ValueError, match="inference_substrate"):
        exp4333.validate_artifact({**artifact, "inference_substrate": "live"})


def test_req_4333_blocks_only_unparseable_ledgers(tmp_path: Path) -> None:
    """REQ-VERIFY-4333: ledger parse failure blocks honestly before mutation."""

    _write_minimal_repo(tmp_path)
    (tmp_path / "ops" / "verifier_registry.yaml").write_text("[not: registry]\n", encoding="utf-8")

    preflight = exp4333.check_preconditions(tmp_path)
    assert preflight["ok"] is False
    assert preflight["blocked_resource"] == "verifier_registry"

    artifact = exp4333.run_hygiene(tmp_path)
    assert artifact["honest_verdict"] == "blocked_ledgers_unparseable"
    assert artifact["regression_guard_passed"] is False
    assert artifact["gaps_logged"] == []
    assert artifact["registry_reconciled"] is False
    assert artifact["manifest_reconciled"] is False


def test_req_4333_results_entrypoint_writes_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-4333: results entrypoint calls the package runner."""

    _write_minimal_repo(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(RESULTS_WRAPPER_PATH)])
    monkeypatch.setattr(exp4333, "REPO_ROOT", tmp_path)
    python_root = str(REPO_ROOT / "python")
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if entry != python_root])

    runpy.run_path(str(REPO_ROOT / RESULTS_WRAPPER_PATH), run_name="__main__")

    payload = json.loads((tmp_path / exp4333.EXP4333_ARTIFACT_PATH).read_text(encoding="utf-8"))
    assert payload["honest_verdict"].startswith("complete:")
    assert payload["regression_guard_passed"] is True
    assert [gap["gap_id"] for gap in payload["gaps_logged"]] == [
        exp4333.GAP_DIFFUSIONGEMMA_SECOND_CORPUS_SCORER,
        exp4333.GAP_ARC_GRID_GENERATION_SCORER,
        exp4333.GAP_E3_WORLD_MODEL_RULE_AR25,
        exp4333.GAP_E3_WORLD_MODEL_RULE_KA59,
        exp4333.GAP_E3_WORLD_MODEL_RULE_TR87,
        exp4333.GAP_E3_WORLD_MODEL_RULE_FT09,
        exp4333.GAP_GAME_INVARIANT_ARC_VALUE_4331,
    ]
