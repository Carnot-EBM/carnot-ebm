"""Tests for Exp6122 ARC solver-kit primitive reachability LOO audit.

Spec refs: REQ-ARC-WMTE-6122,
SCENARIO-ARC-WMTE-6122-REGISTRY-AND-NO-CREDIT-PRECHECK,
SCENARIO-ARC-WMTE-6122-REACHABILITY-SELECTION-GATE,
SCENARIO-ARC-WMTE-6122-HELD-OUT-LOO-OR-CLEAN-NULL.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6122_arc_primitive_reachability_loo as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH
pytestmark = pytest.mark.memory_watchdog_skip


def _fixture_pair(game: str, *, action6_count: int, level: int = 0) -> dict[str, Any]:
    receipts = [
        {
            "step": 0,
            "action": None,
            "data": None,
            "state_hash": f"sha256:{game}reset".ljust(71, "0")[:71],
            "observation_hash": f"sha256:{game}reset".ljust(71, "0")[:71],
            "level": 0,
            "reward": 0.0,
        }
    ]
    for index in range(action6_count):
        receipts.append(
            {
                "step": index + 1,
                "action": 6,
                "data": {"x": index, "y": index + 1},
                "state_hash": f"sha256:{game}{index}".ljust(71, "1")[:71],
                "observation_hash": f"sha256:{game}{index}".ljust(71, "1")[:71],
                "level": level,
                "reward": 0.0,
            }
        )
    row = {
        "game": game,
        "arm": "baseline",
        "seed": 20260804,
        "action_budget": 400,
        "actions_used": len(receipts) - 1,
        "levels_reproduced": level,
        "unique_states": len({r["state_hash"] for r in receipts}),
        "duration_s": 0.01,
        "budget_exhausted": False,
        "crashed": False,
        "failed_reason": None,
        "receipts": receipts,
    }
    return {
        "game": game,
        "seed": 20260804,
        "baseline": row,
        "primitive": dict(row, arm="primitive"),
        "levels_delta": 0,
        "unique_state_coverage_delta": 0,
    }


def _artifact_fixture(games: list[str]) -> dict[str, Any]:
    return {
        "paired_trial_manifest": {
            "games": games,
            "random_seeds": [20260804],
            "action_budget": 400,
        },
        "per_game_metrics": [
            _fixture_pair(game, action6_count=2 if index < 4 else 0)
            for index, game in enumerate(games)
        ],
    }


def test_req_arc_wmte_6122_spec_declares_required_schema() -> None:
    """REQ-ARC-WMTE-6122: OpenSpec lists fields, scenarios, and null contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-ARC-WMTE-6122") :]
    section = section[: section.index("## REQ-ARC-WMTE-6091")]

    for marker in (
        str(mod.RESULT_RELATIVE_PATH),
        "SCENARIO-ARC-WMTE-6122-REGISTRY-AND-NO-CREDIT-PRECHECK",
        "SCENARIO-ARC-WMTE-6122-REACHABILITY-SELECTION-GATE",
        "SCENARIO-ARC-WMTE-6122-HELD-OUT-LOO-OR-CLEAN-NULL",
        "target_level_solve_claim_count=0",
        "complete_null:",
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_arc_wmte_6122_registry_and_no_credit_precheck() -> None:
    """SCENARIO-ARC-WMTE-6122-REGISTRY-AND-NO-CREDIT-PRECHECK."""

    precheck = mod.registry_precheck_and_postcheck(REPO)
    hashes = mod.agent_owned_hashes(REPO, precheck["precheck"]["games"])

    assert precheck["precheck"]["public_game_count"] == 25
    assert precheck["postcheck"]["public_game_count"] == 25
    assert precheck["target_level_solve_claim_count"] == 0
    assert precheck["no_already_reproduced_level_proposed_for_resolve"] is True
    assert precheck["registry_delta"] == 0
    assert hashes["registry"]["sha256"].startswith("sha256:")
    assert hashes["live_agent"]["path"] == str(mod.LIVE_AGENT_RELATIVE_PATH)
    assert hashes["solver_kit"]["path"] == str(mod.SOLVER_KIT_RELATIVE_PATH)
    assert hashes["submitted_defaults"]["sha256"].startswith("sha256:")


def test_scenario_arc_wmte_6122_reachability_selection_gate_cleanly_nulls() -> None:
    """SCENARIO-ARC-WMTE-6122-REACHABILITY-SELECTION-GATE."""

    games = [f"g{i:02d}" for i in range(8)]
    tape = _artifact_fixture(games)
    inventory = mod.primitive_inventory_game_id_free_audit()
    reachability = mod.audit_live_reachability_and_consumption(
        inventory,
        tape,
        development_games=games[:4],
    )
    contract = mod.development_support_and_selection_contract(
        reachability,
        development_games=games[:4],
        held_out_games=games[4:],
    )

    digest_row = next(row for row in reachability if row["operator"] == "object_centric_digest")
    assert digest_row["live_path_reachable"] is True
    assert digest_row["development_downstream_consumption_game_count"] == 4
    assert digest_row["returned_decision_receipts_observed"] is False
    assert digest_row["eligible_for_loo_selection"] is False
    assert contract["selected_primitive_or_none"] is None
    assert contract["selection_status"] == "none"
    assert "no_primitive_with_direct_returned_decision_and_causal_arm_receipts" in contract[
        "selection_reason"
    ]


def test_scenario_arc_wmte_6122_null_artifact_has_required_no_credit_fields() -> None:
    """SCENARIO-ARC-WMTE-6122-HELD-OUT-LOO-OR-CLEAN-NULL."""

    artifact = mod.build_artifact(
        root=REPO,
        test_commands=["unit: exp6122 fixture"],
        test_exit_codes={"unit: exp6122 fixture": 0},
    )

    mod.validate_artifact(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert set(artifact["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"] == "complete_null"
    assert artifact["target_level_solve_claim_count"] == 0
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["selected_primitive_or_none"] is None
    assert artifact["held_out_leave_one_game_out_arm_counts"]["baseline_cells"] == 0
    assert artifact["held_out_leave_one_game_out_arm_counts"]["treatment_cells"] == 0
    assert artifact["duplicate_level_and_unreachable_solver_credit_counts"] == {
        "duplicate_level_credit_count": 0,
        "unreachable_solver_credit_count": 0,
        "development_proxy_credit_count": 0,
        "outer_loop_re_credit_count": 0,
    }
    assert artifact["submitted_defaults_unchanged"]["unchanged"] is True
    assert artifact["protected_files_unchanged"]["unchanged"] is True
    assert artifact["offline_reproduced_new_level"] is False
    assert artifact["verifier_is_oracle"] is False
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete_null:")
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)


def test_req_arc_wmte_6122_repository_artifact_is_schema_valid() -> None:
    """REQ-ARC-WMTE-6122: checked-in JSON is the stable terminal receipt."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["status"] in {"complete_positive", "complete_null", "underpowered", "retired", "blocked"}
    assert artifact["target_level_solve_claim_count"] == 0
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["submitted_defaults_unchanged"]["unchanged"] is True
    assert artifact["offline_reproduced_new_level"] is False
