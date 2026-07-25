"""Tests for Exp5916 held live structured ARC memory A/B.

Spec refs: REQ-ARC-LRHL-5916,
SCENARIO-ARC-LRHL-5916-PRECONDITION-BLOCK,
SCENARIO-ARC-LRHL-5916-MATCHED-HELD-LIVE-AB,
SCENARIO-ARC-LRHL-5916-CAUSAL-CONTROLS,
SCENARIO-ARC-LRHL-5916-NO-SOLVE-CREDIT.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.agentic import arc_structured_memory_live_held_ab as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/agentic-harness/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def test_req_arc_lrhl_5916_spec_declares_held_live_contract() -> None:
    """REQ-ARC-LRHL-5916: OpenSpec freezes the held live capability A/B."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-LRHL-5916") :]

    for marker in (
        "SCENARIO-ARC-LRHL-5916-PRECONDITION-BLOCK",
        "SCENARIO-ARC-LRHL-5916-MATCHED-HELD-LIVE-AB",
        "SCENARIO-ARC-LRHL-5916-CAUSAL-CONTROLS",
        "SCENARIO-ARC-LRHL-5916-NO-SOLVE-CREDIT",
        mod.RESULT_RELATIVE_PATH,
        "cached_sota_pair()",
        "Exp5915",
        "AutoTokenizer",
        "live_llm_inference",
    ):
        assert marker in section

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section

    for field, principle in mod.REQUIRED_FIELD_PROVENANCE.items():
        assert f"`{field}`" in section
        assert principle["principle"] in section


def _ok_preconditions(root: Path = REPO) -> dict:
    return {
        "registry_precheck": {
            "ok": True,
            "registry_hash_before": "sha256:registry",
            "all_public_games_cleared": True,
        },
        "upstream_capability_gate": {
            "ok": True,
            "capability_ready": True,
            "checked_before_model_load": True,
            "exp5915_sha256": "sha256:5915",
        },
        "model_resolution": {"ok": True, "resolved_count": 2},
        "model_hashes": {"ok": True},
        "gguf_tokenizers": {"ok": True, "used_hf_autotokenizer": False},
        "llama_cpp_cuda": {"ok": True, "public_cuda_build": True},
        "dual_rtx3090_health": {"ok": True, "healthy_rtx3090_count": 2},
        "resources": {"ok": True, "ram_ok": True, "disk_ok": True, "vram_ok": True},
        "real_offload_utilization": {"ok": True, "checked_before_inference": True},
        "output_path": {"ok": True},
        "protected_workloads": {"ok": True},
        "submitted_e3_adapter_disabled": {"ok": True},
        "state_isolation_teardown": {
            "ok": True,
            "persistent_cross_cell_state_detected": False,
            "teardown_called_count": 2,
        },
        "live_runner_execution_binding": {
            "ok": True,
            "allow_live_env": True,
            "conductor_live_runner_bound": True,
        },
        "ok": True,
    }


def _live_row(
    *,
    model: str,
    game: str,
    episode: str,
    arm: str,
    accurate: bool,
    actions: int,
    event_hash: str,
    evidence_utilization_count: int,
    query_count: int,
) -> dict:
    return {
        "model": model,
        "game": game,
        "episode": episode,
        "arm": arm,
        "seed": mod.RANDOM_SEEDS[0],
        "arm_order_index": list(mod.ARM_NAMES).index(arm),
        "environment_score": 1.0 if accurate else 0.0,
        "progress": 1.0 if accurate else 0.3,
        "held_objective_correct": bool(accurate),
        "held_episode_correct": bool(accurate),
        "levels_completed": 1 if accurate else 0,
        "actions": actions,
        "invalid_actions": 0,
        "noop_actions": 1 if arm == mod.NO_MEMORY_ARM else 0,
        "repeated_actions": 0,
        "tokens": 320,
        "latency_s": 1.0,
        "gpu_receipt_id": f"{model}:{game}:{episode}:{arm}",
        "query_count": query_count,
        "event_bytes": 4096 if arm != mod.NO_MEMORY_ARM else 0,
        "bytes_read": 4096 if arm != mod.NO_MEMORY_ARM else 0,
        "evidence_utilization_count": evidence_utilization_count,
        "event_tape_hash": event_hash,
        "prompt_hash": "sha256:prompt",
        "decoding_hash": "sha256:decoding",
        "budget_receipt": {
            "action_budget": mod.BUDGETS["max_actions_per_episode_arm"],
            "token_budget": mod.BUDGETS["max_tokens_per_episode_arm"],
            "wall_clock_s": mod.BUDGETS["max_wall_clock_s_per_episode_arm"],
            "query_budget": mod.BUDGETS["max_queries_per_episode_arm"],
            "byte_budget": mod.BUDGETS["max_event_bytes_per_episode_arm"],
        },
        "source_bfs_adapter_prior_game_hidden_access_count": 0,
    }


def _positive_live_held_ab() -> dict:
    rows = []
    for model in ("Qwen3.6-35B-A3B", "Gemma4-26B-A4B-it"):
        for game, episode in (("held-gamma", "held-ep-0001"), ("held-delta", "held-ep-0002")):
            event_hash = f"sha256:{model}:{game}:{episode}:events"
            rows.extend(
                [
                    _live_row(
                        model=model,
                        game=game,
                        episode=episode,
                        arm=mod.NO_MEMORY_ARM,
                        accurate=False,
                        actions=124,
                        event_hash=event_hash,
                        evidence_utilization_count=0,
                        query_count=0,
                    ),
                    _live_row(
                        model=model,
                        game=game,
                        episode=episode,
                        arm=mod.RAW_TAPE_ARM,
                        accurate=False,
                        actions=116,
                        event_hash=event_hash,
                        evidence_utilization_count=1,
                        query_count=4,
                    ),
                    _live_row(
                        model=model,
                        game=game,
                        episode=episode,
                        arm=mod.STRUCTURED_INDEX_ARM,
                        accurate=True,
                        actions=82,
                        event_hash=event_hash,
                        evidence_utilization_count=3,
                        query_count=4,
                    ),
                ]
            )
    return {
        "rows": rows,
        "duration_s": 65.0,
        "gpu_receipts": [
            {
                "gpu": 0,
                "model": "Qwen3.6-35B-A3B",
                "gpu_utilization_pct": 71,
                "vram_used_mb": 23100,
                "offload": "llama_cpp_cuda",
            },
            {
                "gpu": 1,
                "model": "Gemma4-26B-A4B-it",
                "gpu_utilization_pct": 68,
                "vram_used_mb": 18400,
                "offload": "llama_cpp_cuda",
            },
        ],
    }


def _positive_controls(rows: list[dict]) -> dict:
    return {
        "subset_size": 2,
        "structured_baseline_accuracy": 1.0,
        "shuffle_accuracy": 0.0,
        "relevant_deletion_accuracy": 0.0,
        "irrelevant_deletion_accuracy": 1.0,
        "shuffle_effect_delta": -1.0,
        "relevant_deletion_effect_delta": -1.0,
        "connected_to_exp5901_causal_mechanism": True,
        "controls_passed": bool(rows),
        "budget_matched": True,
        "safety_regression": False,
    }


def test_scenario_arc_lrhl_5916_blocks_before_inference(monkeypatch, tmp_path: Path) -> None:
    """SCENARIO-ARC-LRHL-5916-PRECONDITION-BLOCK: failed gates stop live inference."""

    monkeypatch.setattr(
        mod,
        "preconditions",
        lambda root=mod.REPO_ROOT: {
            **_ok_preconditions(root),
            "upstream_capability_gate": {"ok": False, "reason": "Exp5915 gate failed"},
            "ok": False,
        },
    )

    def _fail_if_called(*_args, **_kwargs):
        raise AssertionError("live held A/B must not run after a failed precondition")

    monkeypatch.setattr(mod, "run_live_held_ab", _fail_if_called)
    artifact = mod.build_artifact(root=tmp_path)

    assert artifact["status"] == "blocked_precondition"
    assert artifact["honest_verdict"].startswith("blocked_precondition:")
    assert artifact["public_level_solve_claimed"] is False
    assert artifact["source_bfs_adapter_prior_game_and_hidden_state_access_count"] == 0
    assert artifact["structured_memory_live_ready_score"] == 0.0
    assert artifact["no_memory_raw_and_structured_live_metrics"]["live_row_count"] == 0
    assert artifact["registry_unchanged"] is True
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["verifier_is_oracle"] is False
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
    mod.validate_artifact(artifact)


def test_scenario_arc_lrhl_5916_positive_held_live_ready_score(
    monkeypatch, tmp_path: Path
) -> None:
    """SCENARIO-ARC-LRHL-5916-MATCHED-HELD-LIVE-AB: positive lower bounds promote."""

    monkeypatch.setattr(mod, "preconditions", _ok_preconditions)
    monkeypatch.setattr(mod, "run_live_held_ab", lambda *_args, **_kwargs: _positive_live_held_ab())
    monkeypatch.setattr(mod, "run_confirmatory_controls", _positive_controls)

    artifact = mod.build_artifact(root=tmp_path)
    metrics = artifact["no_memory_raw_and_structured_live_metrics"]
    lower = artifact["per_model_game_episode_lower_bounds"]
    parity = artifact["identical_event_byte_and_budget_parity"]

    assert artifact["status"] == "complete_positive"
    assert artifact["honest_verdict"].startswith("complete_positive:")
    assert metrics[mod.STRUCTURED_INDEX_ARM]["held_episode_accuracy"] > metrics[
        mod.RAW_TAPE_ARM
    ]["held_episode_accuracy"]
    assert metrics[mod.STRUCTURED_INDEX_ARM]["held_episode_accuracy"] > metrics[
        mod.NO_MEMORY_ARM
    ]["held_episode_accuracy"]
    assert lower["structured_over_raw_accuracy_lower_bound"] > 0.0
    assert lower["structured_over_none_accuracy_lower_bound"] > 0.0
    assert parity["all_raw_structured_event_bytes_identical"] is True
    assert parity["principle"] == mod.REQUIRED_FIELD_PROVENANCE[
        "identical_event_byte_and_budget_parity"
    ]["principle"]
    assert parity["budget_violations"] == []
    safety = artifact["held_accuracy_progress_efficiency_and_safety_metrics"]
    assert safety["safety_regression"] is False
    assert safety["budget_regression"] is False
    assert artifact["shuffled_and_deletion_confirmatory_controls"]["controls_passed"] is True
    assert artifact["structured_memory_live_ready_score"] == 1.0
    mod.validate_artifact(artifact)


def test_scenario_arc_lrhl_5916_controls_or_safety_regression_block_ready(
    monkeypatch, tmp_path: Path
) -> None:
    """SCENARIO-ARC-LRHL-5916-CAUSAL-CONTROLS: control failures prevent promotion."""

    run = _positive_live_held_ab()
    for row in run["rows"]:
        if row["arm"] == mod.STRUCTURED_INDEX_ARM:
            row["invalid_actions"] = 1

    monkeypatch.setattr(mod, "preconditions", _ok_preconditions)
    monkeypatch.setattr(mod, "run_live_held_ab", lambda *_args, **_kwargs: run)
    monkeypatch.setattr(
        mod,
        "run_confirmatory_controls",
        lambda rows: {
            **_positive_controls(rows),
            "controls_passed": False,
            "safety_regression": True,
        },
    )

    artifact = mod.build_artifact(root=tmp_path)

    assert artifact["structured_memory_live_ready_score"] == 0.0
    assert artifact["held_accuracy_progress_efficiency_and_safety_metrics"][
        "safety_regression"
    ] is True
    assert artifact["honest_verdict"].startswith("unsafe:")


def test_req_arc_lrhl_5916_helper_negative_paths() -> None:
    """REQ-ARC-LRHL-5916: parity, controls, and bounds expose missing evidence."""

    rows = _positive_live_held_ab()["rows"]
    controls = mod.run_confirmatory_controls(rows)
    registry_path = REPO / "ops" / "arc_solve_registry.yaml"

    assert controls["controls_passed"] is True
    assert controls["relevant_deletion_accuracy"] < controls["structured_baseline_accuracy"]
    assert mod.per_model_game_episode_lower_bounds([rows[0]])["group_count"] == 0
    assert mod._registry_unchanged(
        REPO, {"registry_hash_before": mod._sha256_file(registry_path)}
    ) is True

    bad_rows = json.loads(json.dumps(rows))
    for row in bad_rows:
        if row["arm"] == mod.RAW_TAPE_ARM:
            row["event_tape_hash"] = "sha256:different"
            row["prompt_hash"] = "sha256:different-prompt"
            row["decoding_hash"] = "sha256:different-decoding"
            row["budget_receipt"] = {"action_budget": 1}
            row["actions"] = mod.BUDGETS["max_actions_per_episode_arm"] + 1
            break

    parity = mod.identical_event_byte_and_budget_parity(bad_rows)
    assert parity["all_raw_structured_event_bytes_identical"] is False
    assert parity["prompts_identical"] is False
    assert parity["decoding_identical"] is False
    assert parity["budgets_identical"] is False
    assert parity["budget_violations"]


def test_req_arc_lrhl_5916_runtime_block_is_terminal(monkeypatch, tmp_path: Path) -> None:
    """REQ-ARC-LRHL-5916: a live-runner failure is reported as blocked, not fabricated."""

    monkeypatch.setattr(mod, "preconditions", _ok_preconditions)

    def _runner_failure(*_args, **_kwargs):
        raise RuntimeError("runner unavailable")

    monkeypatch.setattr(mod, "run_live_held_ab", _runner_failure)
    artifact = mod.build_artifact(root=tmp_path)

    assert artifact["status"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["structured_memory_live_ready_score"] == 0.0
    mod.validate_artifact(artifact)


def test_scenario_arc_lrhl_5916_no_solve_credit_and_null(
    monkeypatch, tmp_path: Path
) -> None:
    """SCENARIO-ARC-LRHL-5916-NO-SOLVE-CREDIT: completions are telemetry only."""

    run = _positive_live_held_ab()
    for row in run["rows"]:
        if row["arm"] == mod.STRUCTURED_INDEX_ARM:
            row["held_episode_correct"] = False
            row["held_objective_correct"] = False
            row["environment_score"] = 0.0

    monkeypatch.setattr(mod, "preconditions", _ok_preconditions)
    monkeypatch.setattr(mod, "run_live_held_ab", lambda *_args, **_kwargs: run)
    monkeypatch.setattr(mod, "run_confirmatory_controls", _positive_controls)

    artifact = mod.build_artifact(root=tmp_path)

    assert mod._first_precondition_failure({"raw_bool_failure": False}) == "raw_bool_failure"
    assert artifact["status"] == "complete_null"
    assert artifact["honest_verdict"].startswith("complete_null:")
    assert artifact["public_level_solve_claimed"] is False
    assert artifact["incidental_completion_receipts"]["registry_credit_requested"] is False
    assert artifact["incidental_completion_receipts"]["new_completion_headline_allowed"] is False
    assert "all_unchanged" in mod.protected_files_unchanged(REPO)


def test_req_arc_lrhl_5916_validator_rejects_scope_and_checksum(
    monkeypatch, tmp_path: Path
) -> None:
    """REQ-ARC-LRHL-5916: schema validation protects scope and checksum."""

    monkeypatch.setattr(mod, "preconditions", _ok_preconditions)
    monkeypatch.setattr(mod, "run_live_held_ab", lambda *_args, **_kwargs: _positive_live_held_ab())
    monkeypatch.setattr(mod, "run_confirmatory_controls", _positive_controls)
    artifact = mod.build_artifact(root=tmp_path)

    with pytest.raises(ValueError, match="public_level_solve_claimed"):
        mod.validate_artifact({**artifact, "public_level_solve_claimed": True})
    with pytest.raises(ValueError, match="source_bfs_adapter"):
        mod.validate_artifact(
            {
                **artifact,
                "source_bfs_adapter_prior_game_and_hidden_state_access_count": 1,
            }
        )
    with pytest.raises(ValueError, match="registry_unchanged"):
        mod.validate_artifact({**artifact, "registry_unchanged": False})
    with pytest.raises(ValueError, match="missing required fields"):
        bad = dict(artifact)
        del bad["status"]
        mod.validate_artifact(bad)
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact({**artifact, "inference_substrate": "offline"})
    with pytest.raises(ValueError, match="verifier_is_oracle"):
        mod.validate_artifact({**artifact, "verifier_is_oracle": True})
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact({**artifact, "honest_verdict": "ready: bad_prefix"})
    with pytest.raises(ValueError, match="structured_memory_live_ready_score"):
        mod.validate_artifact(
            {
                **artifact,
                "structured_memory_live_ready_score": 1.0,
                "per_model_game_episode_lower_bounds": {
                    "structured_over_raw_accuracy_lower_bound": 0.0,
                    "structured_over_none_accuracy_lower_bound": 0.0,
                },
            }
        )
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact({**artifact, "reproducibility_checksum": "sha256:bad"})


def test_req_arc_lrhl_5916_writer_roundtrip(monkeypatch, tmp_path: Path) -> None:
    """REQ-ARC-LRHL-5916: write_artifact emits a validated checksum-stable file."""

    monkeypatch.setattr(mod, "preconditions", _ok_preconditions)
    monkeypatch.setattr(mod, "run_live_held_ab", lambda *_args, **_kwargs: _positive_live_held_ab())
    monkeypatch.setattr(mod, "run_confirmatory_controls", _positive_controls)

    output = tmp_path / "experiment_5916.json"
    artifact = mod.write_artifact(root=tmp_path, output_path=output)
    reread = json.loads(output.read_text(encoding="utf-8"))

    assert reread == artifact
    assert reread["status"] == "complete_positive"
    mod.validate_artifact(reread)


def test_req_arc_lrhl_5916_repository_artifact_is_schema_valid() -> None:
    """REQ-ARC-LRHL-5916: checked-in artifact is the stable held live A/B receipt."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
    assert artifact["schema"] == mod.SCHEMA
    assert artifact["public_level_solve_claimed"] is False
    assert artifact["source_bfs_adapter_prior_game_and_hidden_state_access_count"] == 0
    assert artifact["registry_unchanged"] is True
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["verifier_is_oracle"] is False
    assert artifact["honest_verdict"].startswith(
        ("complete_positive:", "complete_null:", "unsafe:", "blocked_precondition:", "blocked:")
    )
    assert len(artifact["reproducibility_checksum"].removeprefix("sha256:")) == 64
    mod.validate_artifact(artifact)
