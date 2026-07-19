"""Tests for Exp5727 full-registry ARC live-vs-oracle generalization gap.

Spec refs: REQ-ARC-WMTE-5727,
SCENARIO-ARC-WMTE-5727-PRECONDITIONS-AND-RUN-PROVENANCE,
SCENARIO-ARC-WMTE-5727-WORST-GAP-CHARACTERIZATION.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from carnot import experiment_5727_arc_generalization_live_oracle_gap_v511 as mod
from carnot.agentic.arc_competition_agent import E3AgentPolicy
from carnot.agentic.arc_executable_world_model import LocalGGUFProposer


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
HARNESS_PATH = REPO / "scripts" / "arc_leaderboard_eval.py"


def _load_harness():
    spec = importlib.util.spec_from_file_location("arc_leaderboard_eval_for_5727", HARNESS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_req_arc_wmte_5727_spec_declares_artifact_contract() -> None:
    """REQ-ARC-WMTE-5727: OpenSpec anchors the live-oracle gap artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-WMTE-5727") :]
    section = section[: section.index("### REQ-ARC-WMTE-4738")]

    for marker in (
        "SCENARIO-ARC-WMTE-5727-PRECONDITIONS-AND-RUN-PROVENANCE",
        "SCENARIO-ARC-WMTE-5727-WORST-GAP-CHARACTERIZATION",
        mod.RESULT_RELATIVE_PATH,
        "scripts/arc_leaderboard_eval.py --games oracle --policy e3 --budget 400",
        "CARNOT_ARC_GENERATOR_CUDA_GPU",
        "non-default proposer port",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PROVENANCE.items():
        assert f"`{field}`" in section
        assert principle["principle"] in section


def _write_registry(path: Path) -> None:
    rows = [
        {"game": "big", "levels_reproduced": 10, "reproducibility": "reproduced"},
        {
            "game": "devtie",
            "levels_reproduced": 9,
            "reproducibility": "reproduced",
            "latest_win": {"solve_provenance": "development_proxy"},
        },
        {
            "game": "livetie",
            "levels_reproduced": 9,
            "reproducibility": "reproduced",
            "latest_win": {"solve_provenance": "live_agent_self_discovery"},
        },
        {
            "game": "small",
            "levels_reproduced": 4,
            "reproducibility": "reproduced",
            "latest_win": {"solve_provenance": "live_agent_self_discovery"},
        },
    ]
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "reproducible_total_games": len(rows),
                "games": rows,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _live_row(
    game: str,
    *,
    levels: int,
    oracle: int,
    actions: int = 400,
    escalated: bool = True,
    skipped: str = "world_model_accuracy_below_threshold",
) -> dict:
    return {
        "game": game,
        "levels": levels,
        "reached": levels,
        "actions": actions,
        "oracle_levels": oracle,
        "gap_vs_oracle": max(0, oracle - levels),
        "actions_to_first_levelup": None if levels == 0 else actions // 2,
        "navigation_diagnostics": {"reset_replay_steps": 17, "forward_walk_hit_rate": 0.125},
        "frame_sequence": [
            {
                "frame_index": 0,
                "action_count": 0,
                "levels_completed": 0,
                "grid_shape": [3, 3],
                "grid_hash": "sha256:first",
            },
            {
                "frame_index": actions - 1,
                "action_count": actions,
                "levels_completed": levels,
                "grid_shape": [3, 3],
                "grid_hash": f"sha256:{game}",
            },
        ],
        "policy_diagnostics": {
            "explore_budget": 24,
            "target_levels": 3,
            "phase": "explore",
            "level_induction_events": [],
            "induction_attempts": (
                [
                    {
                        "reason": "stall",
                        "transition_count": 24,
                        "planned": False,
                        "skipped": skipped,
                        "verify_accuracy": 0.0,
                        "refinement_rounds_used": 1,
                    }
                ]
                if escalated
                else []
            ),
            "proposer": {
                "instantiated": escalated,
                "repo_substr": "Qwen3.5-9B-MTP" if escalated else None,
                "port": 8922 if escalated else None,
                "mtp": True if escalated else None,
            },
        },
    }


def test_scenario_arc_wmte_5727_builds_gap_artifact_and_tiebreaks_provenance(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-5727-WORST-GAP-CHARACTERIZATION: largest gaps are grounded."""

    registry = tmp_path / "ops" / "arc_solve_registry.yaml"
    registry.parent.mkdir(parents=True)
    _write_registry(registry)
    verifier_gaps = tmp_path / "ops" / "verifier_gaps.md"
    verifier_gaps.write_text("### GAP-3: learned / model-native ARC energy\n", encoding="utf-8")
    live = tmp_path / "results" / "arc_live_oracle_gap.json"
    live.parent.mkdir()
    live.write_text(
        json.dumps(
            {
                "experiment": "arc_live_oracle_gap",
                "games_mode": "oracle",
                "policy": "e3",
                "budget": 400,
                "live_levels": 3,
                "oracle_levels": 32,
                "gap": 29,
                "per_game": [
                    _live_row("small", levels=3, oracle=4),
                    _live_row("livetie", levels=1, oracle=9),
                    _live_row("devtie", levels=1, oracle=9),
                    _live_row("big", levels=0, oracle=10, skipped="proposer_failed_or_missing_root"),
                ],
            }
        ),
        encoding="utf-8",
    )

    artifact = mod.build_artifact(
        root=tmp_path,
        live_gap_path=live,
        registry_path=registry,
        verifier_gaps_path=verifier_gaps,
        preconditions_checked={
            "qwen35_9b_gguf_cached": True,
            "registry_reproducible_total_games": 4,
            "registry_premise_ok": True,
            "cuda_gpu_env": "0",
            "proposer_port": 8922,
            "proposer_port_non_default": True,
        },
        random_seed=20260719,
    )

    assert artifact["harness_used"] == "scripts/arc_leaderboard_eval.py"
    assert artifact["policy_kind"] == "e3"
    assert artifact["budget_per_game"] == 400
    assert artifact["games_measured"] == 4
    assert artifact["live_levels_total"] == 5
    assert artifact["oracle_levels_total"] == 32
    assert artifact["gap_total"] == 27
    assert [row["game"] for row in artifact["worst_gap_games"]] == ["big", "devtie", "livetie"]
    assert artifact["worst_gap_games"][1]["oracle_win_solve_provenance"] == "development_proxy"
    assert artifact["worst_gap_games"][1]["stall_class"] == "INDUCTION QUALITY"
    assert "not_attempted_to_fix" in artifact["worst_gap_games"][1]
    assert artifact["verifier_gaps_entries_added"] == []
    assert artifact["any_new_level_found"] is False
    assert artifact["inference_substrate_by_game"]["big"]["tier3_qwen35_mtp_escalated"] is True
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_provenance"])
    assert mod.validate_artifact(artifact) == []


def test_req_arc_wmte_5727_blocked_preconditions_emit_terminal_artifact(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-5727: missing GGUF blocks before result compilation."""

    artifact = mod.blocked_artifact(
        "blocked_qwen35_9b_gguf_not_cached",
        preconditions_checked={"qwen35_9b_gguf_cached": False},
        random_seed=20260719,
    )

    assert artifact["honest_verdict"] == "blocked_qwen35_9b_gguf_not_cached"
    assert artifact["games_measured"] == 0
    assert artifact["per_game_gap"] == []
    assert artifact["worst_gap_games"] == []
    assert mod.validate_artifact(artifact) == []


def test_req_arc_wmte_5727_harness_exposes_policy_and_frame_diagnostics() -> None:
    """REQ-ARC-WMTE-5727: the harness keeps public-frame and induction evidence."""

    harness = _load_harness()
    frame = SimpleNamespace(
        levels_completed=2,
        frame=[[1, 1], [0, 2]],
        available_actions=[SimpleNamespace(name="ACTION1")],
    )
    policy = SimpleNamespace(
        phase="induce",
        explore_budget=24,
        target_levels=3,
        induction_attempts=[{"reason": "stall", "planned": False}],
        level_induction_events=[],
        proposer=SimpleNamespace(repo_substr="Qwen3.5-9B-MTP", port=8922, mtp=True),
        explorer=SimpleNamespace(
            navigation_diagnostics=lambda: {
                "reset_replay_steps": 3,
                "forward_walk_hit_rate": 0.5,
            }
        ),
    )

    public_frame = harness._frame_public_summary(frame, frame_index=7, action_count=11)
    diagnostics = harness._policy_diagnostics(policy)

    assert public_frame["frame_index"] == 7
    assert public_frame["action_count"] == 11
    assert public_frame["levels_completed"] == 2
    assert public_frame["grid_shape"] == [2, 2]
    assert public_frame["grid_hash"].startswith("sha256:")
    assert diagnostics["phase"] == "induce"
    assert diagnostics["induction_attempts"] == [{"reason": "stall", "planned": False}]
    assert diagnostics["proposer"]["repo_substr"] == "Qwen3.5-9B-MTP"
    assert diagnostics["proposer"]["port"] == 8922


def test_req_arc_wmte_5727_e3_proposer_port_env_override(monkeypatch) -> None:
    """REQ-ARC-WMTE-5727: local measurement can avoid the default 8919 server."""

    monkeypatch.setenv("CARNOT_ARC_PROPOSER_PORT", "8922")

    policy = E3AgentPolicy("paritytest", proposer=None, value_head=lambda _frame: 0.0)
    proposer = policy._proposer()

    assert isinstance(proposer, LocalGGUFProposer)
    assert proposer.repo_substr == "Qwen3.5-9B-MTP"
    assert proposer.port == 8922
    assert proposer.mtp is True
    assert proposer.kv_quant == "q8_0"


def test_req_arc_wmte_5727_defensive_branches_are_explicit(monkeypatch, tmp_path: Path) -> None:
    """REQ-ARC-WMTE-5727: defensive artifact paths fail closed and remain covered."""

    assert mod.oracle_win_solve_provenance(None) == "unknown_registry_missing"
    assert (
        mod.oracle_win_solve_provenance({"notes": ["solve_provenance: outer_loop_re"]})
        == "outer_loop_re"
    )
    assert mod._int_field({"x": object()}, "x", 7) == 7

    rows = mod.per_game_gap_rows(
        {"per_game": [[], {"game": ""}, {"game": "fallback", "levels": "bad"}]},
        {
            "games": [
                {
                    "game": "fallback",
                    "levels_reproduced": 2,
                    "reproducibility": "reproduced",
                }
            ]
        },
    )
    assert rows == [
        {
            "game": "fallback",
            "live_levels": 0,
            "oracle_levels": 2,
            "gap": 2,
            "oracle_win_solve_provenance": "unknown_registry_unstructured",
        }
    ]

    no_attempt_budget = {
        "game": "budget",
        "levels": 0,
        "actions": 400,
        "frame_sequence": [],
    }
    no_attempt_perception = {"game": "perception", "levels": 0, "actions": 3}
    partial = {"game": "partial", "levels": 1, "actions": 3}
    assert mod.stall_class_for_row(no_attempt_budget, 400) == "SEARCH/BUDGET"
    assert mod.stall_class_for_row(no_attempt_perception, 400) == "PERCEPTION"
    assert mod.stall_class_for_row(partial, 400).startswith("OTHER:")
    assert "unavailable" in mod.grounded_evidence_for_row(no_attempt_perception, 400)[1]
    assert mod.grounded_evidence_for_row(no_attempt_perception, 400)[2] == (
        "policy_diagnostics.induction_attempts: []"
    )

    assert mod.verifier_gap_reference("SEARCH/BUDGET", "") == (
        "not_a_new_missing_discriminator_search_or_budget_gap"
    )
    assert mod.verifier_gap_reference("PERCEPTION", "perception backlog") == "existing_perception_gap"
    assert mod.verifier_gap_reference("PERCEPTION", "") == "no_new_missing_discriminator_logged"
    assert mod.inference_substrate_by_game(
        [
            {"game": "none"},
            {"game": "attempt", "policy_diagnostics": {"induction_attempts": [{"planned": False}]}},
        ]
    )["attempt"]["tier3_qwen35_mtp_escalated"] is True

    bad = mod.blocked_artifact(
        "not_terminal",
        preconditions_checked={},
        random_seed=20260719,
    )
    bad.pop("harness_used")
    bad["field_provenance"] = {}
    bad["policy_kind"] = "explorer"
    bad["budget_per_game"] = 20000
    bad["worst_gap_games"] = [{"game": "a"}]
    bad["reproducibility_checksum"] = "sha256:bad"
    errors = mod.validate_artifact(bad)
    assert "missing required field: harness_used" in errors
    assert "field_provenance mismatch: harness_used" in errors
    assert "honest_verdict lacks terminal prefix" in errors
    assert "reproducibility_checksum mismatch" in errors
    assert "policy_kind must be e3" in errors
    assert "budget_per_game must be 400" in errors
    assert "worst_gap_games must contain exactly three games" in errors
    missing_provenance = mod.blocked_artifact(
        "blocked_missing_field_provenance",
        preconditions_checked={},
        random_seed=20260719,
    )
    missing_provenance["field_provenance"] = None
    missing_provenance["reproducibility_checksum"] = mod._checksum_payload(missing_provenance)
    assert "field_provenance missing" in mod.validate_artifact(missing_provenance)

    fake_home = tmp_path / "home"
    model = (
        fake_home
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--unsloth--Qwen3.5-9B-MTP-GGUF"
        / "snapshots"
        / "abc"
        / "Qwen3.5-9B-Q4_K_M.gguf"
    )
    model.parent.mkdir(parents=True)
    model.write_text("gguf", encoding="utf-8")

    class FakeRoot(Path):
        _flavour = type(Path())._flavour

        def home(self):
            return fake_home

    monkeypatch.setenv("CARNOT_ARC_PROPOSER_PORT", "8922")
    monkeypatch.setenv("CARNOT_ARC_GENERATOR_CUDA_GPU", "0")
    preconditions = mod._default_preconditions(FakeRoot(tmp_path))
    assert preconditions["qwen35_9b_gguf_cached"] is True
    assert preconditions["proposer_port_non_default"] is True
