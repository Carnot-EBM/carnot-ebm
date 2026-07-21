"""Tests for Exp5766 ARC LOO component interaction attribution audit.

Spec refs: REQ-ARC-WMTE-5766,
SCENARIO-ARC-WMTE-5766-REGISTRY-AND-INVENTORY-PRECHECK,
SCENARIO-ARC-WMTE-5766-LOO-PAIRED-ATTRIBUTION,
SCENARIO-ARC-WMTE-5766-CONTROLS-GATES-AND-PRODUCER-FIELDS.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5766_arc_loo_component_interaction_audit as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH
pytestmark = pytest.mark.memory_watchdog_skip
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5766_arc_loo_component_interaction_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5766_arc_loo_component_interaction_audit.py "
    "-m pytest tests/python/test_experiment_5766_arc_loo_component_interaction_audit.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5766_arc_loo_component_interaction_audit.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = ".venv/bin/python scripts/check_spec_coverage.py"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5766_arc_loo_component_interaction_audit.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
TEST_COMMANDS = [
    TEST_COMMAND,
    COVERAGE_COMMAND,
    FULL_TEST_COMMAND,
    SPEC_COMMAND,
    ADVERSARIAL_COMMAND,
    ROOT_CLUTTER_COMMAND,
]
TEST_EXIT_CODES = {command: 0 for command in TEST_COMMANDS}


def _trace_row(game: str, index: int) -> dict[str, Any]:
    return {
        "game": game,
        "seed": mod.RANDOM_SEEDS[0],
        "action_budget": mod.ACTION_BUDGET,
        "baseline": {
            "game": game,
            "arm": "baseline",
            "seed": mod.RANDOM_SEEDS[0],
            "action_budget": mod.ACTION_BUDGET,
            "actions_used": 200 + index,
            "levels_reproduced": 1 if index % 6 == 0 else 0,
            "action_effect_predictions": 100,
            "action_effect_correct": 80,
            "valid_actions": 190,
            "invalid_actions": 0,
            "repeated_actions": index % 5,
            "unique_states": 40 + index,
            "planning_reachable": index % 4 == 0,
            "planning_attempts": 1 if index % 4 == 0 else 0,
            "budget_exhausted": False,
            "crashed": False,
            "duration_s": 0.5 + index / 100.0,
            "receipts": [
                {
                    "step": 0,
                    "action": None,
                    "data": None,
                    "reward": 0.0,
                    "level": 0,
                    "state_hash": f"sha256:{game}0".ljust(71, "0")[:71],
                    "observation_hash": f"sha256:{game}0".ljust(71, "0")[:71],
                },
                {
                    "step": 1,
                    "action": 1,
                    "data": None,
                    "reward": 0.0,
                    "level": 1 if index % 6 == 0 else 0,
                    "state_hash": f"sha256:{game}1".ljust(71, "1")[:71],
                    "observation_hash": f"sha256:{game}1".ljust(71, "1")[:71],
                },
            ],
            "failed_reason": None,
        },
        "primitive": {
            "game": game,
            "arm": "primitive",
            "seed": mod.RANDOM_SEEDS[0],
            "action_budget": mod.ACTION_BUDGET,
            "actions_used": 200 + index,
            "levels_reproduced": 1 if index % 6 == 0 else 0,
            "action_effect_predictions": 100,
            "action_effect_correct": 80,
            "valid_actions": 190,
            "invalid_actions": 0,
            "repeated_actions": index % 5,
            "unique_states": 40 + index,
            "planning_reachable": index % 4 == 0,
            "planning_attempts": 1 if index % 4 == 0 else 0,
            "budget_exhausted": False,
            "crashed": False,
            "duration_s": 0.5 + index / 100.0,
            "receipts": [
                {
                    "step": 0,
                    "action": None,
                    "data": None,
                    "reward": 0.0,
                    "level": 0,
                    "state_hash": f"sha256:{game}0".ljust(71, "0")[:71],
                    "observation_hash": f"sha256:{game}0".ljust(71, "0")[:71],
                },
                {
                    "step": 1,
                    "action": 1,
                    "data": None,
                    "reward": 0.0,
                    "level": 1 if index % 6 == 0 else 0,
                    "state_hash": f"sha256:{game}1".ljust(71, "1")[:71],
                    "observation_hash": f"sha256:{game}1".ljust(71, "1")[:71],
                },
            ],
            "failed_reason": None,
        },
        "receipts_preserved": True,
        "failed_reason": None,
    }


def _trace_rows() -> list[dict[str, Any]]:
    return [_trace_row(f"g{i:02d}", i) for i in range(mod.PUBLIC_GAME_COUNT)]


def _preconditions_fixture(games: list[str] | None = None) -> dict[str, Any]:
    roster = games or [row["game"] for row in _trace_rows()]
    registry = {
        "source": str(mod.REGISTRY_RELATIVE_PATH),
        "registry_hash": "sha256:" + "a" * 64,
        "checked_before_attribution": True,
        "public_game_count": mod.PUBLIC_GAME_COUNT,
        "registry_level_count": mod.REGISTRY_LEVEL_COUNT,
        "full_game_clear_count": mod.PUBLIC_GAME_COUNT,
        "all_public_games_complete": True,
        "no_public_level_can_be_credited_as_new": True,
        "games": roster,
        "ok": True,
    }
    return {
        "ok": True,
        "failures": [],
        "upstream_artifact_hashes": {
            "exp5753": {"present": True, "sha256": "sha256:" + "1" * 64},
            "exp5727": {"present": True, "sha256": "sha256:" + "2" * 64},
            "registry": {"present": True, "sha256": registry["registry_hash"]},
        },
        "registry_precheck": registry,
        "registry_hash": registry["registry_hash"],
        "exp5753_trace_rows": mod.PUBLIC_GAME_COUNT,
        "exp5753_agent_owned_trace_provenance": True,
        "exp5753_budgets_and_seeds_match": True,
        "live_environment_reachable": True,
        "resource_precheck": {"ok": True, "disk_free_mb": 999, "ram_free_mb": 999},
        "forbidden_inputs": {
            "source_required": False,
            "game_adapter_required": False,
            "banked_plan_required": False,
            "game_identity_runtime_feature_required": False,
            "outer_loop_ground_truth_required": False,
        },
    }


def test_req_arc_wmte_5766_spec_declares_component_interaction_contract() -> None:
    """REQ-ARC-WMTE-5766: OpenSpec lists every required field and guard."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-WMTE-5766") :]
    section = section[: section.index("### REQ-ARC-WMTE-4738")]
    normalized = " ".join(section.split())

    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized
    for marker in (
        str(mod.RESULT_RELATIVE_PATH),
        "SCENARIO-ARC-WMTE-5766-REGISTRY-AND-INVENTORY-PRECHECK",
        "SCENARIO-ARC-WMTE-5766-LOO-PAIRED-ATTRIBUTION",
        "SCENARIO-ARC-WMTE-5766-CONTROLS-GATES-AND-PRODUCER-FIELDS",
        "leave-one-game-out folds",
        "400-action budgets",
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section or marker in normalized


def test_scenario_arc_wmte_5766_registry_and_inventory_precheck() -> None:
    """SCENARIO-ARC-WMTE-5766-REGISTRY-AND-INVENTORY-PRECHECK."""

    inventory = mod.component_inventory()
    source_hashes = mod.component_source_hashes(REPO)
    preconditions = mod.structured_preconditions(
        root=REPO,
        check_arc_environment=False,
        check_resources=False,
    )
    trace_rows = mod.load_agent_owned_trace_rows(REPO)

    assert len(inventory) >= 5
    assert all(row["live_path_reachable"] is True for row in inventory)
    assert all(row["existing_component"] is True for row in inventory)
    assert all(receipt["present"] for receipt in source_hashes.values())
    assert all(receipt["sha256"].startswith("sha256:") for receipt in source_hashes.values())
    assert preconditions["ok"] is True
    assert preconditions["registry_precheck"]["public_game_count"] == 25
    assert preconditions["registry_precheck"]["registry_level_count"] == 183
    assert preconditions["registry_precheck"]["all_public_games_complete"] is True
    assert preconditions["exp5753_trace_rows"] == 25
    assert len(trace_rows) == 25
    assert preconditions["forbidden_inputs"]["source_required"] is False
    assert preconditions["forbidden_inputs"]["game_identity_runtime_feature_required"] is False


def test_scenario_arc_wmte_5766_loo_paired_attribution_is_fold_disjoint() -> None:
    """SCENARIO-ARC-WMTE-5766-LOO-PAIRED-ATTRIBUTION."""

    rows = _trace_rows()
    games = [row["game"] for row in rows]
    inventory = mod.component_inventory()
    folds = mod.build_loo_fold_manifest(games)
    analysis = mod.run_component_attribution(rows, inventory, folds)

    assert len(folds) == 25
    assert sorted(fold["heldout_game"] for fold in folds) == sorted(games)
    assert all(len(fold["development_games"]) == 24 for fold in folds)
    assert all(fold["heldout_game"] not in fold["development_games"] for fold in folds)
    assert all(fold["runtime_features_used_for_selection"] == [] for fold in folds)
    assert len(analysis["per_game_metrics"]) == 25
    assert len(analysis["marginal_effects"]) == len(inventory)
    assert len(analysis["pairwise_interaction_effects"]) == len(mod.PAIRWISE_SPECS)
    assert analysis["fold_disjointness_receipts"]["all_disjoint"] is True
    assert analysis["development_selected_composition"]["runtime_features_used"] == []
    assert analysis["loo_generalization_delta"] == pytest.approx(0.0)
    assert analysis["loo_generalization_delta_lcb"] == pytest.approx(0.0)
    assert analysis["causal_interaction_count"] == 0


def test_scenario_arc_wmte_5766_controls_replay_and_gate_fields() -> None:
    """SCENARIO-ARC-WMTE-5766-CONTROLS-GATES-AND-PRODUCER-FIELDS."""

    rows = _trace_rows()
    exact = mod.exact_replay_receipts(rows)
    positive = mod.positive_control_receipt()
    leaks = mod.negative_leak_canary_receipts()
    manifest = mod.paired_trial_manifest(rows, mod.component_inventory())
    one_sample_interval = mod.paired_confidence_interval([0.25])

    assert exact["all_exact_replay_passed"] is True
    assert exact["game_count"] == 25
    assert all(row["exact_replay_passed"] for row in exact["per_game"])
    assert positive["non_degenerate"] is True
    assert positive["positive_control_delta"] > 0.0
    assert one_sample_interval == {"mean": 0.25, "ci95_low": 0.25, "ci95_high": 0.25, "n": 1}
    assert leaks["source"]["detected_canary_count"] > 0
    assert leaks["game_identity"]["detected_canary_count"] > 0
    assert leaks["source"]["admitted_leak_count"] == 0
    assert leaks["game_identity"]["admitted_leak_count"] == 0
    assert manifest["action_budget"] == 400
    assert manifest["observations_matched"] is True
    assert manifest["cache_policy_matched"] is True
    assert manifest["arms"][0] == "baseline"
    assert any(arm.startswith("delete:") or arm.startswith("add:") for arm in manifest["arms"])


def test_scenario_arc_wmte_5766_builds_valid_complete_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-ARC-WMTE-5766: complete artifact has schema, gate fields, and no solve credit."""

    rows = _trace_rows()
    monkeypatch.setattr(mod, "structured_preconditions", lambda **_kw: _preconditions_fixture())
    monkeypatch.setattr(mod, "load_agent_owned_trace_rows", lambda *_args, **_kw: rows)

    artifact = mod.build_artifact(
        root=tmp_path,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
    )
    saved_path = mod.write_output(tmp_path, artifact)
    saved = json.loads(saved_path.read_text(encoding="utf-8"))

    assert saved == artifact
    assert tuple(saved) == mod.REQUIRED_ARTIFACT_FIELDS
    assert set(saved["field_principles"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert saved["status"] == "complete"
    assert saved["spec_refs"] == list(mod.SPEC_REFS)
    assert saved["public_game_count"] == 25
    assert saved["registry_level_count"] == 183
    assert len(saved["component_inventory"]) >= 5
    assert len(saved["loo_fold_manifest"]) == 25
    assert saved["fold_disjointness_receipts"]["all_disjoint"] is True
    assert saved["exact_replay_receipts"]["all_exact_replay_passed"] is True
    assert saved["positive_control_receipt"]["non_degenerate"] is True
    assert saved["negative_leak_canary_receipts"]["source"]["admitted_leak_count"] == 0
    assert saved["producer_gate_fields"] == list(mod.PRODUCER_GATE_FIELDS)
    for field in mod.PRODUCER_GATE_FIELDS:
        assert field in saved
        assert not isinstance(saved[field], dict)
    assert saved["loo_generalization_delta"] == pytest.approx(0.0)
    assert saved["loo_generalization_delta_lcb"] == pytest.approx(0.0)
    assert saved["causal_interaction_count"] == 0
    assert saved["source_leak_count"] == 0
    assert saved["game_identity_leak_count"] == 0
    assert saved["solve_provenance"] == "development_proxy"
    assert saved["arc_registry_delta"] == 0
    assert saved["arc_solve_credited"] is False
    assert saved["outer_loop_re_used"] is False
    assert saved["per_game_adapter_used"] is False
    assert saved["source_read_used"] is False
    assert saved["production_default_enabled"] is False
    assert saved["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert saved["test_commands"] == TEST_COMMANDS
    assert saved["test_exit_codes"] == TEST_EXIT_CODES
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)
    assert saved["honest_verdict"].startswith("complete:")
    mod.validate_artifact(saved)


def test_req_arc_wmte_5766_blocked_precondition_bails_before_trace_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-ARC-WMTE-5766: missing preconditions emit blocked artifact before attribution."""

    blocked = _preconditions_fixture()
    blocked["ok"] = False
    blocked["failures"] = ["live_environment_reachable"]
    blocked["registry_precheck"]["ok"] = False
    monkeypatch.setattr(mod, "structured_preconditions", lambda **_kw: blocked)

    def _fail_if_loaded(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        raise AssertionError("trace rows must not load after a failed precondition")

    monkeypatch.setattr(mod, "load_agent_owned_trace_rows", _fail_if_loaded)

    artifact = mod.build_artifact(root=REPO)

    assert artifact["status"] == "blocked"
    assert artifact["honest_verdict"] == "blocked: live_environment_reachable"
    assert artifact["per_game_metrics"] == []
    assert artifact["loo_generalization_delta"] == pytest.approx(0.0)
    assert artifact["loo_generalization_delta_lcb"] == pytest.approx(0.0)
    assert artifact["causal_interaction_count"] == 0
    assert artifact["source_leak_count"] == 0
    assert artifact["game_identity_leak_count"] == 0
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    mod.validate_artifact(artifact)


def test_req_arc_wmte_5766_validation_rejects_manual_overclaims(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-ARC-WMTE-5766: schema, gates, provenance, and checksum edits fail closed."""

    rows = _trace_rows()
    monkeypatch.setattr(mod, "structured_preconditions", lambda **_kw: _preconditions_fixture())
    monkeypatch.setattr(mod, "load_agent_owned_trace_rows", lambda *_args, **_kw: rows)
    artifact = mod.build_artifact(root=tmp_path)

    mutators = [
        ("required field order", lambda data: data.__setitem__("status", data.pop("status"))),
        ("field_principles", lambda data: data["field_principles"].pop("status")),
        ("producer_gate_fields", lambda data: data.__setitem__("producer_gate_fields", [])),
        ("producer_gate_fields", lambda data: data.__setitem__("source_leak_count", {})),
        ("public_game_count", lambda data: data.__setitem__("public_game_count", 24)),
        ("registry_level_count", lambda data: data.__setitem__("registry_level_count", 182)),
        ("component_inventory", lambda data: data.__setitem__("component_inventory", [])),
        ("loo_fold_manifest", lambda data: data.__setitem__("loo_fold_manifest", [])),
        ("exact_replay_receipts", lambda data: data["exact_replay_receipts"].__setitem__("all_exact_replay_passed", False)),
        ("positive_control_receipt", lambda data: data["positive_control_receipt"].__setitem__("non_degenerate", False)),
        ("negative_leak_canary_receipts", lambda data: data.__setitem__("source_leak_count", 1)),
        ("registry credit", lambda data: data.__setitem__("arc_registry_delta", 1)),
        ("registry credit", lambda data: data.__setitem__("arc_solve_credited", True)),
        ("forbidden provenance", lambda data: data.__setitem__("outer_loop_re_used", True)),
        ("forbidden provenance", lambda data: data.__setitem__("per_game_adapter_used", True)),
        ("forbidden provenance", lambda data: data.__setitem__("source_read_used", True)),
        ("production_default_enabled", lambda data: data.__setitem__("production_default_enabled", True)),
        ("inference_substrate", lambda data: data.__setitem__("inference_substrate", "live_llm_inference")),
        ("honest_verdict", lambda data: data.__setitem__("honest_verdict", "ok")),
        ("reproducibility_checksum", lambda data: data.__setitem__("loo_generalization_delta", 0.5)),
    ]

    for message, mutate in mutators:
        bad = deepcopy(artifact)
        mutate(bad)
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(bad)


def test_req_arc_wmte_5766_repository_artifact_is_schema_valid() -> None:
    """REQ-ARC-WMTE-5766: checked-in JSON is the stable terminal audit receipt."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["producer_gate_fields"] == list(mod.PRODUCER_GATE_FIELDS)
    assert artifact["loo_generalization_delta_lcb"] == pytest.approx(0.0)
    assert artifact["causal_interaction_count"] == 0
    assert artifact["source_leak_count"] == 0
    assert artifact["game_identity_leak_count"] == 0
    assert artifact["arc_registry_delta"] == 0
    assert artifact["arc_solve_credited"] is False
