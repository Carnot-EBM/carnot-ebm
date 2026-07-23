"""Tests for Exp 5860 bounded game-blind active observation.

Spec refs: REQ-ARC-WMTE-5860,
SCENARIO-ARC-WMTE-5860-TAPE-IS-AGENT-OWNED,
SCENARIO-ARC-WMTE-5860-BUDGET-PARITY-AND-READY-GATE,
SCENARIO-ARC-WMTE-5860-STABLE-ARTIFACT.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_5860_live_active_observation_ab as exp5860


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _obs(hash_suffix: str, *, level: int = 0) -> dict[str, object]:
    return {
        "observation_id": f"obs-{hash_suffix}",
        "frame_hash": f"frame-{hash_suffix}",
        "grid_hash": f"grid-{hash_suffix}",
        "grid_shape": [4, 4],
        "available_actions": [1, 2, 6],
        "level": level,
        "raw_observation_hash": f"raw-{hash_suffix}",
    }


def _item(
    step: int,
    *,
    arm: str = "active_observer",
    action: dict[str, object] | None = None,
    before: str = "a",
    after: str = "b",
    before_level: int = 0,
    after_level: int = 0,
) -> dict[str, object]:
    return {
        "source": "agent_runtime_observation",
        "game": "sc25",
        "arm": arm,
        "step_index": step,
        "action": action or {"a": 1, "data": None},
        "before": _obs(before, level=before_level),
        "after": _obs(after, level=after_level),
        "latency_s": 0.01,
        "model_call_id": "qwen-call-0" if arm == "active_observer" else None,
    }


def _preconditions() -> dict[str, object]:
    return {
        "agents_md_read": True,
        "codex_md_read": True,
        "spec_req_5860_present": True,
        "registry_precheck_complete": True,
        "not_a_solve_task": True,
        "live_environment_available": True,
        "sota_qwen_cached": True,
        "embedded_tokenizer_verified": True,
        "gpu_vram_checked": True,
        "atomic_output_path_checked": True,
        "forbidden_controller_channels_excluded": True,
    }


def _registry_precheck() -> dict[str, object]:
    return {
        "candidate_games": ["sc25"],
        "all_candidate_games_registry_complete": True,
        "registry_total_public_games": 25,
        "registry_total_reproducible_levels": 183,
        "not_a_solve_task": True,
        "per_game": {"sc25": {"levels_reproduced": 6, "full_game_clear": True}},
    }


def _receipts() -> dict[str, object]:
    return {
        "canonical_scored_entrypoint": "python/carnot/agentic/arc_competition_agent.py:E3AgentPolicy",
        "standing_eval_entrypoint": "scripts/arc_leaderboard_eval.py",
        "arc_sdk_version": "0.1.0",
        "entrypoint_hashes": {"arc_competition_agent.py": "sha256:a"},
        "requested_missing_entrypoints": {"scripts/arc_live_agent.py": False},
    }


def _exclusions() -> dict[str, object]:
    return {
        "game_adapters_enabled": False,
        "public_source_read_enabled": False,
        "offline_ground_truth_bfs_enabled": False,
        "registry_trajectory_enabled": False,
        "per_game_model_enabled": False,
        "hand_rule_enabled": False,
        "outer_loop_counterexample_channel_enabled": False,
    }


def _gpu() -> dict[str, object]:
    return {
        "nvidia_smi": ["NVIDIA GeForce RTX 3090, 24576 MiB, 23000 MiB"],
        "llama_cpp_server_binary": "/tmp/llama-server",
        "embedded_tokenizer": {
            "unsloth/Qwen3.6-35B-A3B-GGUF": {
                "ok": True,
                "detail": "embedded GGUF tokenizer OK",
            }
        },
    }


def _model_specs() -> list[dict[str, object]]:
    return [
        {
            "name": "Qwen3.6-35B-A3B",
            "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "role": "moe",
            "quantization": "Q4_K_M",
        }
    ]


def test_req_arc_wmte_5860_spec_declares_artifact_contract() -> None:
    """REQ-ARC-WMTE-5860: OpenSpec anchors the artifact and control contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in exp5860.SPEC_REFS + [exp5860.RESULT_RELATIVE_PATH]:
        assert marker in spec
    for field in exp5860.REQUIRED_FIELDS:
        assert field in spec


def test_scenario_arc_wmte_5860_tape_is_append_only_and_agent_owned() -> None:
    """SCENARIO-ARC-WMTE-5860-TAPE-IS-AGENT-OWNED: tape accepts only runtime evidence."""

    empty = exp5860.AgentOwnedTape()
    tape = empty.append(_item(0, after="a"))

    assert len(empty.items) == 0
    assert len(tape.items) == 1
    original_hash = tape.content_hash()

    external_view = tape.items[0]
    external_view["action"] = {"a": 99}
    assert tape.items[0]["action"] == {"a": 1, "data": None}
    assert tape.content_hash() == original_hash

    forbidden = _item(1)
    forbidden["goal_label"] = "win"
    with pytest.raises(ValueError, match="forbidden tape key"):
        tape.append(forbidden)

    wrong_source = dict(_item(1), source="registry_trajectory")
    with pytest.raises(ValueError, match="agent_runtime_observation"):
        tape.append(wrong_source)


def test_scenario_arc_wmte_5860_history_views_are_game_blind() -> None:
    """SCENARIO-ARC-WMTE-5860-TAPE-IS-AGENT-OWNED: views partition runtime tape only."""

    tape = (
        exp5860.AgentOwnedTape()
        .append(_item(0, before="a", after="a"))
        .append(_item(1, before="a", after="b"))
        .append(_item(2, before="b", after="c", after_level=1))
    )

    views = exp5860.history_views(tape, local_window=2)

    assert [row["step_index"] for row in views["global_history"]] == [0, 1, 2]
    assert [row["step_index"] for row in views["local_active"]] == [1, 2]
    assert [row["step_index"] for row in views["event_boundary"]] == [1, 2]
    rendered = json.dumps(views, sort_keys=True)
    assert "goal_label" not in rendered
    assert "adapter_fact" not in rendered
    assert "event_caption" not in rendered


def test_scenario_arc_wmte_5860_metrics_count_evidence_not_levels() -> None:
    """REQ-ARC-WMTE-5860: evidence metrics are based on action/outcome tape."""

    tape = (
        exp5860.AgentOwnedTape()
        .append(_item(0, before="a", after="a", action={"a": 1, "data": None}))
        .append(_item(1, before="a", after="b", action={"a": 1, "data": None}))
        .append(_item(2, before="b", after="c", action={"a": 6, "data": {"x": 1, "y": 2}}))
    )

    metrics = exp5860.score_tape(tape)
    controls = exp5860.null_control_metrics(tape, seed=5860)

    assert metrics["actions"] == 3
    assert metrics["no_op_actions"] == 1
    assert metrics["novel_causal_relation_confirmations"] == 2
    assert metrics["transition_alias_disambiguation"] == 1
    assert metrics["ambiguity_resolved_per_action"] > 0.0
    assert set(controls) == {
        "shuffled_tape",
        "view_ablation",
        "random_priority",
        "no_memory",
    }


def test_scenario_arc_wmte_5860_budget_parity_and_ready_gate() -> None:
    """SCENARIO-ARC-WMTE-5860-BUDGET-PARITY-AND-READY-GATE: score needs strict parity."""

    arms = exp5860.build_arm_definitions(
        games=("sc25",),
        action_budget=4,
        wall_clock_budget_s=30.0,
        model_call_budget=2,
        token_budget=512,
        reset_budget=1,
    )
    assert exp5860.budgets_have_parity(arms) is True

    positive = {
        "active_observer": {
            "ambiguity_resolved_per_action": 0.5,
            "proposal_support": {"short": 0.5, "medium": 0.4, "long": 0.3},
        },
        "current_e3": {
            "ambiguity_resolved_per_action": 0.1,
            "proposal_support": {"short": 0.1, "medium": 0.1, "long": 0.1},
        },
        "random_legal": {
            "ambiguity_resolved_per_action": 0.2,
            "proposal_support": {"short": 0.2, "medium": 0.2, "long": 0.2},
        },
        "periodic": {
            "ambiguity_resolved_per_action": 0.0,
            "proposal_support": {"short": 0.0, "medium": 0.0, "long": 0.0},
        },
    }
    accounting = {arm: {"actions": 4, "model_calls": 2, "tokens": 100, "resets": 1, "latency_s": 1.0} for arm in arms}

    assert exp5860.active_observation_ready_score(positive, arms, accounting) == 1.0

    over_budget = dict(accounting)
    over_budget["active_observer"] = dict(over_budget["active_observer"], actions=5)
    assert exp5860.active_observation_ready_score(positive, arms, over_budget) == 0.0

    null_metrics = dict(positive)
    null_metrics["active_observer"] = {
        "ambiguity_resolved_per_action": 0.1,
        "proposal_support": {"short": 0.1, "medium": 0.1, "long": 0.1},
    }
    assert exp5860.active_observation_ready_score(null_metrics, arms, accounting) == 0.0


def test_scenario_arc_wmte_5860_helper_edges_fail_closed() -> None:
    """SCENARIO-ARC-WMTE-5860-STABLE-ARTIFACT: helper edge cases fail closed."""

    class _ItemValue:
        def item(self):
            return 7

    class _BrokenItemValue:
        def item(self):
            raise RuntimeError("not scalar")

    safe = exp5860._canonical(  # noqa: SLF001 - white-box guard for checksum stability
        {
            "path": Path("x"),
            "tuple": ("a", "b"),
            "set": {"b", "a"},
            "item": _ItemValue(),
        }
    )
    assert safe["path"] == "x"
    assert safe["tuple"] == ["a", "b"]
    assert safe["set"] == ["a", "b"]
    assert safe["item"] == 7
    broken = _BrokenItemValue()
    assert exp5860._json_safe(broken) is broken  # noqa: SLF001

    nested = {"outer": [{"adapter_fact": "forbidden"}]}
    assert (
        exp5860._contains_forbidden_key(nested) == "adapter_fact"  # noqa: SLF001
    )

    with pytest.raises(ValueError, match="missing tape item fields"):
        exp5860.AgentOwnedTape().append({"source": "agent_runtime_observation"})

    arms = exp5860.build_arm_definitions(
        games=("sc25",),
        action_budget=4,
        wall_clock_budget_s=30.0,
        model_call_budget=2,
        token_budget=512,
        reset_budget=1,
    )
    non_parity = dict(arms)
    non_parity["periodic"] = {
        **non_parity["periodic"],
        "budgets": {**non_parity["periodic"]["budgets"], "legal_action_budget": 5},
    }
    accounting = {
        arm: {"actions": 0, "model_calls": 0, "tokens": 0, "resets": 0, "latency_s": 0.0}
        for arm in arms
    }
    assert exp5860.active_observation_ready_score({}, non_parity, accounting) == 0.0
    assert exp5860.active_observation_ready_score({}, arms, accounting) == 0.0

    tape = exp5860.AgentOwnedTape().append(_item(0, action={"a": 1, "data": None}))
    choice = exp5860._heuristic_active_choice(  # noqa: SLF001
        [{"a": 1, "data": None}, {"a": 2, "data": None}],
        tape,
    )
    assert choice == 1

    assert exp5860._honest_verdict("blocked_precondition", 0.0).startswith(  # noqa: SLF001
        "blocked:"
    )
    assert exp5860._honest_verdict("complete_null", 1.0).startswith(  # noqa: SLF001
        "complete_positive:"
    )


def test_scenario_arc_wmte_5860_stable_artifact_and_checksum(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-5860-STABLE-ARTIFACT: result JSON is schema-stable."""

    active_tape = (
        exp5860.AgentOwnedTape()
        .append(_item(0, before="a", after="a"))
        .append(_item(1, before="a", after="b"))
    )
    random_tape = (
        exp5860.AgentOwnedTape()
        .append(_item(0, arm="random_legal", before="a", after="a"))
        .append(_item(1, arm="random_legal", before="a", after="b"))
    )
    tapes = {
        "current_e3": exp5860.AgentOwnedTape(),
        "random_legal": random_tape,
        "periodic": exp5860.AgentOwnedTape(),
        "active_observer": active_tape,
    }
    arms = exp5860.build_arm_definitions(
        games=("sc25",),
        action_budget=2,
        wall_clock_budget_s=30.0,
        model_call_budget=2,
        token_budget=512,
        reset_budget=1,
    )
    accounting = {
        arm: {"actions": len(tapes[arm].items), "model_calls": int(arm == "active_observer"), "tokens": 12, "resets": 1, "latency_s": 0.2}
        for arm in arms
    }
    levels = {arm: {"sc25": {"start_level": 0, "reached_level": 0, "levels_gained": 0}} for arm in arms}

    artifact = exp5860.build_artifact(
        status="complete_null",
        preconditions_checked=_preconditions(),
        registry_precheck=_registry_precheck(),
        live_path_and_sdk_receipts=_receipts(),
        adapter_source_bfs_and_registry_exclusion_receipts=_exclusions(),
        model_specs=_model_specs(),
        models_used=["unsloth/Qwen3.6-35B-A3B-GGUF"],
        gpu_and_llama_cpp_receipts=_gpu(),
        tapes_by_arm=tapes,
        arm_definitions_and_budget_parity=arms,
        action_model_call_and_latency_accounting=accounting,
        descriptive_level_outcomes=levels,
        duration_s=0.5,
        test_commands=["pytest test_experiment_5860"],
        test_exit_codes={"pytest test_experiment_5860": 0},
    )

    exp5860.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete_null:")
    assert artifact["registry_modified"] is False
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["verifier_is_oracle"] is False
    assert isinstance(artifact["active_observation_ready_score"], float)

    out = tmp_path / "experiment_5860_live_active_observation_ab.json"
    exp5860.write_artifact(out, artifact)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded == artifact

    drifted = dict(artifact, models_used=["unsloth/gemma-4-31B-it-GGUF"])
    assert exp5860.reproducibility_checksum(drifted) != artifact["reproducibility_checksum"]

    bad = dict(artifact, registry_modified=True)
    bad["reproducibility_checksum"] = exp5860.reproducibility_checksum(bad)
    with pytest.raises(ValueError, match="registry_modified must be false"):
        exp5860.validate_artifact(bad)

    for mutation, message in (
        (lambda row: {k: v for k, v in row.items() if k != "status"}, "missing required"),
        (lambda row: dict(row, solve_provenance="development_proxy"), "solve_provenance"),
        (lambda row: dict(row, verifier_is_oracle=True), "verifier_is_oracle"),
        (
            lambda row: dict(row, active_observation_ready_score=0.5),
            "active_observation_ready_score",
        ),
        (lambda row: dict(row, honest_verdict="maybe later"), "honest_verdict"),
        (lambda row: dict(row, field_provenance={}), "field_provenance"),
        (lambda row: dict(row, reproducibility_checksum="sha256:bad"), "checksum"),
    ):
        mutated = mutation(artifact)
        if "reproducibility_checksum" in mutated and message != "checksum":
            mutated["reproducibility_checksum"] = exp5860.reproducibility_checksum(mutated)
        with pytest.raises(ValueError, match=message):
            exp5860.validate_artifact(mutated)
