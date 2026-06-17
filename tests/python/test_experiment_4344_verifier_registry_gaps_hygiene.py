"""Tests for Exp 4344 verifier registry/gaps hygiene.

Spec refs: REQ-VERIFY-4344, SCENARIO-VERIFY-4344.
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

from carnot import experiment_4344_verifier_registry_gaps_hygiene as exp4344_wrapper
from carnot.reporting import verifier_registry_gaps_hygiene_4344 as exp4344


REPO_ROOT = Path(__file__).parents[2]
SPEC_PATH = Path("openspec/capabilities/verification/spec.md")
RESULTS_WRAPPER_PATH = Path("results/experiment_4344_verifier_registry_gaps_hygiene.py")


def _minimal_registry() -> dict[str, Any]:
    return {
        "verifiers": [
            {
                "verifier_id": exp4344.GAP4_VERIFIER_ID,
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
    for path in exp4344.REQUIRED_COPY_PATHS:
        if path in omit:
            continue
        source = REPO_ROOT / path
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def test_req_4344_spec_declared() -> None:
    """REQ-VERIFY-4344: OpenSpec declares the .401 hygiene artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-4344",
        "SCENARIO-VERIFY-4344",
        "python/carnot/experiment_4344_verifier_registry_gaps_hygiene.py",
        RESULTS_WRAPPER_PATH.as_posix(),
        exp4344.EXP4344_ARTIFACT_PATH,
        "blocked_<file>_unparseable",
        "aggregate_available_report_gaps",
        "in_generation_moat_replicates=true",
        "hidden_step_counter_hud_gap",
        exp4344.GAP_E3_WORLD_MODEL_RULE_AR25_4339,
        exp4344.GAP_E3_WORLD_MODEL_RULE_KA59_4340,
        exp4344.GAP_ACTION_ROLE_TRANSFER_4342,
        "gaps_logged=3",
    ):
        assert marker in spec
    for principle in exp4344.FIELD_PRINCIPLES.values():
        assert principle in spec
    assert exp4344_wrapper.main is exp4344.main


def test_scenario_4344_preconditions_outcomes_and_robust_availability(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-4344: missing .401 artifacts are per-axis gaps only."""

    _write_minimal_repo(tmp_path, omit={exp4344.EXP4340_PATH})

    preflight = exp4344.check_preconditions(tmp_path)
    assert preflight["ok"] is True
    assert preflight["blocked_resource"] is None

    guard = exp4344.run_gap4_regression_guard(tmp_path)
    assert guard["regression_guard_passed"] is True
    assert guard["prior_artifact_path"] == exp4344.EXP4333_PATH
    assert guard["recorded_arc1_rule_exec"] == {
        "n": 31,
        "vote_pass2": pytest.approx(0.4516),
        "gated_pass2": pytest.approx(0.5806),
        "headroom_recovered": 4,
        "vote_wins_lost": 0,
    }
    assert guard["replayed_arc1_rule_exec"] == guard["recorded_arc1_rule_exec"]

    bundle = exp4344.load_v401_outcomes(tmp_path)
    outcomes = bundle["v401_outcomes"]
    availability = bundle["availability_report"]
    assert availability["axes"]["in_generation_moat"]["verdict"] is True
    assert availability["axes"]["e3_ar25"]["verdict"] is True
    assert availability["axes"]["e3_ka59"]["missing_artifacts"] == [
        {"axis": "e3_ka59", "artifact_key": "4340_e3_ka59", "experiment_id": 4340}
    ]
    assert availability["axes"]["e3_sc25"]["verdict"] is True
    assert availability["axes"]["cross_game_transfer"]["verdict"] is False
    assert outcomes["in_generation_moat"]["in_generation_moat_replicates"] is True
    assert outcomes["in_generation_moat"]["scorer_leak_recheck_passed"] is True
    assert outcomes["e3"]["games"]["ar25"]["offline_reproduced"] is True
    assert outcomes["e3"]["games"]["ar25"]["residual_mismatch_class"] == (
        "missing_world_model_rule_gap_hidden_undo_stack_action7"
    )
    assert outcomes["e3"]["games"]["ka59"]["available"] is False
    assert outcomes["e3"]["games"]["sc25"]["residual_mismatch_class"] == "none"
    assert outcomes["cross_game_transfer"]["learned_encoder_transfer_helps"] is False


def test_req_4344_ledgers_record_v401_truth_gaps_and_retirement(tmp_path: Path) -> None:
    """REQ-VERIFY-4344: registry, gaps, and manifest carry the .401 truth."""

    _write_minimal_repo(tmp_path)
    guard = exp4344.run_gap4_regression_guard(tmp_path)
    bundle = exp4344.load_v401_outcomes(tmp_path)
    gap_entries = exp4344.build_gap_entries(bundle)

    registry, gaps_text, manifest, summary = exp4344.ensure_ledgers_record_v401(
        _minimal_registry(),
        "# Verifier Gaps\n\nHistorical note remains.\n",
        _minimal_manifest(),
        guard,
        bundle,
        gap_entries,
    )

    assert [entry["gap_id"] for entry in gap_entries] == [
        exp4344.GAP_E3_WORLD_MODEL_RULE_AR25_4339,
        exp4344.GAP_E3_WORLD_MODEL_RULE_KA59_4340,
        exp4344.GAP_ACTION_ROLE_TRANSFER_4342,
    ]
    assert summary == {
        "registry_reconciled": True,
        "manifest_reconciled": True,
        "gaps_logged_ids": [entry["gap_id"] for entry in gap_entries],
    }

    gap4 = registry["verifiers"][0]
    assert gap4["eval"]["eval_exp_4344"] == exp4344.EXP4344_ARTIFACT_PATH
    assert gap4["eval"]["exp4344_regression_guard_passed"] is True
    assert gap4["eval"]["exp4344_v401_state"] == exp4344.V401_STATE
    assert gap4["eval"]["exp4344_in_generation_moat_replicates"] is True
    assert gap4["eval"]["exp4344_in_generation_replication_ci95"] == [0.283333, 0.4375]
    assert gap4["eval"]["exp4344_e3_reproduced_levels_total"] == 2
    assert gap4["eval"]["exp4344_e3_ar25_offline_reproduced"] is True
    assert gap4["eval"]["exp4344_e3_ka59_residual_mismatch_class"] == (
        "hidden_step_counter_hud_gap"
    )
    assert gap4["eval"]["exp4344_e3_sc25_offline_reproduced"] is True
    assert gap4["eval"]["exp4344_learned_encoder_transfer_helps"] is False
    assert gap4["eval"]["exp4344_cross_game_state_reduction"] == pytest.approx(
        1.00635593220339
    )
    assert gap4["eval"]["exp4344_gaps_logged"] == [
        entry["gap_id"] for entry in gap_entries
    ]
    role = next(role for role in gap4["registry_roles"] if role["role_id"] == exp4344.V401_ROLE_ID)
    assert role["v401_state"] == exp4344.V401_STATE
    assert exp4344.registry_contains_v401(registry) is True
    assert exp4344.registry_contains_v401({}) is False

    assert "Historical note remains." in gaps_text
    assert "exp4344-gap-e3-world-model-rule-ar25-4339:start" in gaps_text
    assert "exp4344-gap-e3-world-model-rule-ka59-4340:start" in gaps_text
    assert "exp4344-gap-4342:start" in gaps_text
    assert exp4344.manifest_contains_cross_game_transfer_retirement(manifest) is True
    assert exp4344.manifest_contains_in_generation_moat_retirement(manifest) is False


def test_req_4344_build_artifact_validates_required_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-4344: terminal artifact exposes the required schema fields."""

    _write_minimal_repo(tmp_path)
    guard = exp4344.run_gap4_regression_guard(tmp_path)
    bundle = exp4344.load_v401_outcomes(tmp_path)
    gap_entries = exp4344.build_gap_entries(bundle)
    artifact = exp4344.build_artifact(
        regression_guard=guard,
        outcome_bundle=bundle,
        gap_entries=gap_entries,
        registry_reconciled=True,
        manifest_reconciled=True,
        reproducibility_checksum="sha256:" + "0" * 64,
        duration_s=0.25,
    )

    exp4344.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["regression_guard_passed"] is True
    assert artifact["registry_reconciled"] is True
    assert artifact["manifest_reconciled"] is True
    assert artifact["gaps_logged"] == 3
    assert artifact["field_principles"] == exp4344.FIELD_PRINCIPLES
    assert artifact["model_specs"]["method"] == "cached_v401_ledger_reconciliation"
    assert artifact["inference_substrate"] == exp4344.INFERENCE_SUBSTRATE

    for field in exp4344.REQUIRED_ARTIFACT_FIELDS:
        malformed = dict(artifact)
        malformed.pop(field)
        with pytest.raises(ValueError, match=field):
            exp4344.validate_artifact(malformed)
    with pytest.raises(ValueError, match="terminal prefix"):
        exp4344.validate_artifact({**artifact, "honest_verdict": "not terminal"})
    with pytest.raises(ValueError, match="BARE bool"):
        exp4344.validate_artifact({**artifact, "regression_guard_passed": "yes"})
    with pytest.raises(ValueError, match="registry_reconciled"):
        exp4344.validate_artifact({**artifact, "registry_reconciled": "yes"})
    with pytest.raises(ValueError, match="manifest_reconciled"):
        exp4344.validate_artifact({**artifact, "manifest_reconciled": "yes"})
    with pytest.raises(ValueError, match="gaps_logged"):
        exp4344.validate_artifact({**artifact, "gaps_logged": [{"gap_id": "GAP"}]})
    with pytest.raises(ValueError, match="v401_outcomes"):
        exp4344.validate_artifact({**artifact, "v401_outcomes": []})
    with pytest.raises(ValueError, match="availability_report"):
        exp4344.validate_artifact({**artifact, "availability_report": []})
    with pytest.raises(ValueError, match="random_seed"):
        exp4344.validate_artifact({**artifact, "random_seed": True})
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp4344.validate_artifact({**artifact, "reproducibility_checksum": ""})
    with pytest.raises(ValueError, match="model_specs"):
        exp4344.validate_artifact({**artifact, "model_specs": {}})
    with pytest.raises(ValueError, match="field_principles"):
        exp4344.validate_artifact({**artifact, "field_principles": {}})
    with pytest.raises(ValueError, match="spec_refs"):
        exp4344.validate_artifact({**artifact, "spec_refs": ["REQ-VERIFY-4344"]})
    with pytest.raises(ValueError, match="inference_substrate"):
        exp4344.validate_artifact({**artifact, "inference_substrate": "live"})


def test_req_4344_blocks_only_unparseable_ledgers(tmp_path: Path) -> None:
    """REQ-VERIFY-4344: ledger parse failure blocks honestly before mutation."""

    _write_minimal_repo(tmp_path)
    (tmp_path / "ops" / "verifier_registry.yaml").write_text(
        "[not: registry]\n",
        encoding="utf-8",
    )

    preflight = exp4344.check_preconditions(tmp_path)
    assert preflight["ok"] is False
    assert preflight["blocked_resource"] == "verifier_registry"

    artifact = exp4344.run_hygiene(tmp_path)
    assert artifact["honest_verdict"] == "blocked_verifier_registry_unparseable"
    assert artifact["regression_guard_passed"] is False
    assert artifact["gaps_logged"] == 0
    assert artifact["registry_reconciled"] is False
    assert artifact["manifest_reconciled"] is False


def test_req_4344_results_entrypoint_writes_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-4344: results entrypoint calls the package runner."""

    _write_minimal_repo(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(RESULTS_WRAPPER_PATH)])
    monkeypatch.setattr(exp4344, "REPO_ROOT", tmp_path)
    python_root = str(REPO_ROOT / "python")
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if entry != python_root])

    runpy.run_path(str(REPO_ROOT / RESULTS_WRAPPER_PATH), run_name="__main__")

    payload = json.loads((tmp_path / exp4344.EXP4344_ARTIFACT_PATH).read_text(encoding="utf-8"))
    assert payload["honest_verdict"].startswith("complete:")
    assert payload["regression_guard_passed"] is True
    assert payload["registry_reconciled"] is True
    assert payload["manifest_reconciled"] is True
    assert payload["gaps_logged"] == 3
