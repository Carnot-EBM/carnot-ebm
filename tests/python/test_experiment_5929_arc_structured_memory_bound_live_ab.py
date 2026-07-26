"""Tests for Exp5929 capability-bound structured ARC memory live A/B.

Spec refs: REQ-ARC-LRBH-5929,
SCENARIO-ARC-LRBH-5929-PRECONDITION-BLOCK,
SCENARIO-ARC-LRBH-5929-BOUND-MATCHED-HELD-LIVE-AB,
SCENARIO-ARC-LRBH-5929-NO-SOLVE-CREDIT.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.agentic import arc_structured_memory_bound_live_ab as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/agentic-harness/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def test_req_arc_lrbh_5929_spec_declares_bound_live_contract() -> None:
    """REQ-ARC-LRBH-5929: OpenSpec freezes the bound live A/B contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-LRBH-5929") :]

    for marker in (
        "SCENARIO-ARC-LRBH-5929-PRECONDITION-BLOCK",
        "SCENARIO-ARC-LRBH-5929-BOUND-MATCHED-HELD-LIVE-AB",
        "SCENARIO-ARC-LRBH-5929-NO-SOLVE-CREDIT",
        "REQ-ARC-LRBH-5929-CAPABILITY-REPLAY",
        "REQ-ARC-LRBH-5929-HELD-CELL",
        "REQ-ARC-LRBH-5929-ARM-ISOLATION",
        "REQ-ARC-LRBH-5929-BYTE-BUDGET-PARITY",
        "REQ-ARC-LRBH-5929-ADAPTER-DISABLED",
        "REQ-ARC-LRBH-5929-LIVE-PROVENANCE",
        "REQ-ARC-LRBH-5929-TEARDOWN",
        "REQ-ARC-LRBH-5929-REGISTRY-IMMUTABILITY",
        mod.RESULT_RELATIVE_PATH,
        "cached_sota_pair()",
        "AutoTokenizer",
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section

    for field, principle in mod.REQUIRED_FIELD_PROVENANCE.items():
        assert f"`{field}`" in section
        assert principle["principle"] in section


def _ok_preconditions(_root: Path = REPO) -> dict:
    return {
        "registry_precheck_and_selected_held_cells": {
            "ok": True,
            "registry_hash_before": "sha256:registry",
            "registry_hash_after_precheck": "sha256:registry",
            "all_public_games_cleared": True,
            "selected_cells_not_public_solve_targets": True,
        },
        "gate_and_capability_replay": {
            "ok": True,
            "checked_before_model_load": True,
            "exp5928_stored_ready_score": 1.0,
            "exp5928_replayed_ready_score": 1.0,
        },
        "model_resolution": {"ok": True, "resolved_count": 3},
        "model_hashes": {"ok": True, "models": []},
        "gguf_tokenizers": {"ok": True, "used_hf_autotokenizer": False},
        "llama_cpp_cuda": {"ok": True, "public_cuda_build": True},
        "dual_rtx3090_health": {"ok": True, "healthy_rtx3090_count": 2},
        "resources": {"ok": True, "ram_ok": True, "disk_ok": True, "vram_ok": True},
        "real_offload_utilization": {"ok": True, "nonzero_offload_required": True},
        "output_path": {"ok": True},
        "protected_workloads": {"ok": True},
        "checkpoint_resume": {"ok": True, "resume_verified": True},
        "actual_bound_e3_entrypoint": {
            "ok": True,
            "capability_consumed_before_environment_action": True,
            "actual_live_entrypoint": "carnot.agentic.arc_competition_agent:consume_process_bound_capability_preflight",
        },
        "adapter_disabled": {"ok": True, "adapter_disabled": True},
        "capability_teardown": {
            "ok": True,
            "nonce_replay_denied_before_teardown": True,
            "child_process_orphaned": False,
            "issuer_secret_persisted": False,
        },
        "ok": True,
    }


def _row(
    *,
    model: str,
    episode: str,
    arm: str,
    correct: bool,
    actions: int,
    event_hash: str,
    relevance: float,
    abstained: bool = False,
) -> dict:
    return {
        "model": model,
        "held_cell": episode,
        "episode": episode,
        "game": episode.split("/", 1)[0],
        "arm": arm,
        "seed": mod.RANDOM_SEEDS[0],
        "prompt_hash": "sha256:prompt",
        "decoding_hash": "sha256:decoding",
        "event_tape_hash": event_hash,
        "event_bytes_sha256": event_hash,
        "context_budget": mod.BUDGETS["max_context_tokens_per_episode_arm"],
        "token_budget": mod.BUDGETS["max_tokens_per_episode_arm"],
        "action_budget": mod.BUDGETS["max_actions_per_episode_arm"],
        "query_budget": mod.BUDGETS["max_queries_per_episode_arm"],
        "byte_budget": mod.BUDGETS["max_event_bytes_per_episode_arm"],
        "latency_budget_s": mod.BUDGETS["max_wall_clock_s_per_episode_arm"],
        "budget_receipt": dict(mod.BUDGETS),
        "held_objective_correct": correct,
        "verified_progress_events": 2 if correct else 1,
        "progress": 1.0 if correct else 0.25,
        "retrieval_relevance": relevance,
        "retrieval_relevance_score": relevance,
        "action_legality_rate": 1.0,
        "invalid_actions": 0,
        "illegal_actions": 0,
        "noop_actions": 0,
        "repeated_actions": 0,
        "actions": actions,
        "tokens": 512,
        "context_tokens": 1024,
        "latency_s": 2.5,
        "gpu_memory_mb": 18000,
        "vram_used_mb": 18000,
        "query_count": 3 if arm != mod.NO_MEMORY_ARM else 0,
        "event_bytes": 4096 if arm != mod.NO_MEMORY_ARM else 0,
        "bytes_read": 4096 if arm != mod.NO_MEMORY_ARM else 0,
        "memory_representation": arm,
        "abstained": abstained,
        "levels_completed": 1 if correct and arm == mod.STRUCTURED_INDEX_ARM else 0,
        "source_bfs_adapter_prior_game_hidden_access_count": 0,
        "live_agent_row": True,
        "adapter_disabled": True,
        "solve_provenance": "live_agent_self_discovery"
        if correct and arm == mod.STRUCTURED_INDEX_ARM
        else None,
    }


def _positive_run() -> dict:
    rows = []
    model_ids = [row["name"] for row in mod.MODEL_SPECS]
    for model in model_ids:
        for held_cell in ("held-e3-alpha/episode-0001", "held-e3-beta/episode-0002"):
            event_hash = f"sha256:{model}:{held_cell}:events"
            rows.extend(
                [
                    _row(
                        model=model,
                        episode=held_cell,
                        arm=mod.NO_MEMORY_ARM,
                        correct=False,
                        actions=120,
                        event_hash=event_hash,
                        relevance=0.0,
                    ),
                    _row(
                        model=model,
                        episode=held_cell,
                        arm=mod.RAW_TAPE_ARM,
                        correct=False,
                        actions=110,
                        event_hash=event_hash,
                        relevance=0.4,
                    ),
                    _row(
                        model=model,
                        episode=held_cell,
                        arm=mod.STRUCTURED_INDEX_ARM,
                        correct=True,
                        actions=78,
                        event_hash=event_hash,
                        relevance=0.9,
                    ),
                ]
            )
    return {
        "rows": rows,
        "duration_s": 75.0,
        "gpu_receipts": [
            {"gpu": 0, "utilization_pct": 72, "vram_used_mb": 23100},
            {"gpu": 1, "utilization_pct": 69, "vram_used_mb": 18400},
        ],
    }


def test_scenario_arc_lrbh_5929_blocks_before_inference(monkeypatch, tmp_path: Path) -> None:
    """SCENARIO-ARC-LRBH-5929-PRECONDITION-BLOCK: gates stop live rows."""

    monkeypatch.setattr(
        mod,
        "preconditions",
        lambda root=mod.REPO_ROOT: {
            **_ok_preconditions(root),
            "gate_and_capability_replay": {"ok": False, "reason": "Exp5928 replay failed"},
            "ok": False,
        },
    )

    def _fail_if_called(*_args, **_kwargs):
        raise AssertionError("bound live runner must not execute after a failed precondition")

    monkeypatch.setattr(mod, "run_bound_live_ab", _fail_if_called)
    artifact = mod.build_artifact(root=tmp_path)

    assert artifact["status"] == "blocked_precondition"
    assert artifact["honest_verdict"].startswith("blocked_precondition:")
    assert artifact["structured_memory_live_ready_score"] == 0.0
    assert artifact["per_model_episode_retrieval_progress_legality_efficiency_and_abstention"][
        "live_row_count"
    ] == 0
    assert artifact["actual_bound_e3_entrypoint_receipt"]["live_inference_started"] is False
    assert artifact["solve_provenance"] is None
    assert artifact["registry_unchanged"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
    mod.validate_artifact(artifact)


def test_scenario_arc_lrbh_5929_positive_bound_live_ready_score(
    monkeypatch, tmp_path: Path
) -> None:
    """SCENARIO-ARC-LRBH-5929-BOUND-MATCHED-HELD-LIVE-AB: live rows promote."""

    monkeypatch.setattr(mod, "preconditions", _ok_preconditions)
    monkeypatch.setattr(mod, "run_bound_live_ab", lambda *_args, **_kwargs: _positive_run())

    artifact = mod.build_artifact(root=tmp_path)

    assert artifact["status"] == "complete_positive"
    assert artifact["honest_verdict"].startswith("complete_positive:")
    assert artifact["gate_and_capability_replay_receipt"]["exp5928_replayed_ready_score"] == 1.0
    assert artifact["actual_bound_e3_entrypoint_receipt"][
        "capability_consumed_before_environment_action"
    ] is True
    assert artifact["adapter_disabled"] is True
    assert artifact["no_per_game_adapter_or_public_solve_target"]["ok"] is True
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["identical_event_bytes_and_arm_budget_parity"]["ok"] is True
    intervals = artifact["primary_live_utility_comparison_and_intervals"]
    assert intervals["structured_over_raw_interval_lower"] > 0.0
    assert intervals["structured_over_none_interval_lower"] > 0.0
    assert artifact["structured_memory_live_ready_score"] == 1.0
    accounting = artifact["token_context_latency_gpu_and_memory_accounting"]
    assert accounting["total_tokens"] > 0
    assert accounting["gpu_receipts"]
    mod.validate_artifact(artifact)


def test_scenario_arc_lrbh_5929_parity_or_proxy_rows_prevent_promotion(
    monkeypatch, tmp_path: Path
) -> None:
    """REQ-ARC-LRBH-5929-BYTE-BUDGET-PARITY: off-path rows void live claim."""

    run = _positive_run()
    run["rows"][1]["event_tape_hash"] = "sha256:different"
    run["rows"][2]["source_bfs_adapter_prior_game_hidden_access_count"] = 1
    run["rows"][2]["live_agent_row"] = False

    monkeypatch.setattr(mod, "preconditions", _ok_preconditions)
    monkeypatch.setattr(mod, "run_bound_live_ab", lambda *_args, **_kwargs: run)
    artifact = mod.build_artifact(root=tmp_path)

    assert artifact["status"] == "complete_null"
    assert artifact["structured_memory_live_ready_score"] == 0.0
    assert artifact["identical_event_bytes_and_arm_budget_parity"]["ok"] is False
    assert artifact["no_per_game_adapter_or_public_solve_target"]["ok"] is False
    mod.validate_artifact(artifact)


def test_req_arc_lrbh_5929_helper_negative_branches(monkeypatch, tmp_path: Path) -> None:
    """REQ-ARC-LRBH-5929-LIVE-PROVENANCE: pure helpers expose voided evidence."""

    assert mod._first_precondition_failure({"raw_bool_failure": False}) == "raw_bool_failure"

    run = _positive_run()
    one_row = [run["rows"][0]]
    assert mod.identical_event_bytes_and_arm_budget_parity(one_row)["paired_raw_structured_cell_count"] == 0
    assert mod.primary_live_utility_comparison_and_intervals(one_row)["group_count"] == 0

    over_budget = json.loads(json.dumps(run["rows"]))
    over_budget[0]["actions"] = mod.BUDGETS["max_actions_per_episode_arm"] + 1
    assert mod.identical_event_bytes_and_arm_budget_parity(over_budget)["budget_violations"]

    proxy_outcome = json.loads(json.dumps(run["rows"]))
    proxy_outcome[2]["solve_provenance"] = "development_proxy"
    assert mod.solve_provenance(proxy_outcome) == "off_path_or_proxy_voids_live_claim"

    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"not a real gguf but enough to hash")
    hashes = mod.model_file_hashes_from_resolution(
        {
            "ok": True,
            "resolved_model_specs": [
                {
                    "name": "Unit",
                    "hf_id": "unit/GGUF",
                    "model_path": str(gguf),
                    "resolved_via": "unit",
                }
            ],
        }
    )
    assert hashes["ok"] is True
    assert hashes["models"][0]["sha256"].startswith("sha256:")

    missing_hashes = mod.model_file_hashes_from_resolution(
        {"ok": True, "resolved_model_specs": [{"model_path": str(tmp_path / "missing.gguf")}]}
    )
    assert missing_hashes["ok"] is False

    monkeypatch.setattr(mod, "preconditions", _ok_preconditions)

    def _runner_failure(*_args, **_kwargs):
        raise RuntimeError("runner unavailable")

    monkeypatch.setattr(mod, "run_bound_live_ab", _runner_failure)
    artifact = mod.build_artifact(root=tmp_path)
    assert artifact["status"] == "blocked_precondition"
    assert artifact["honest_verdict"].startswith("blocked_precondition: bound_live_runner_unavailable")
    mod.validate_artifact(artifact)


def test_req_arc_lrbh_5929_validation_rejects_overclaims(monkeypatch, tmp_path: Path) -> None:
    """REQ-ARC-LRBH-5929-REGISTRY-IMMUTABILITY: validator rejects unsafe claims."""

    monkeypatch.setattr(mod, "preconditions", _ok_preconditions)
    monkeypatch.setattr(mod, "run_bound_live_ab", lambda *_args, **_kwargs: _positive_run())
    artifact = mod.build_artifact(root=tmp_path)

    with pytest.raises(ValueError, match="missing required fields"):
        bad = dict(artifact)
        del bad["status"]
        mod.validate_artifact(bad)
    with pytest.raises(ValueError, match="registry_unchanged"):
        mod.validate_artifact({**artifact, "registry_unchanged": False})
    with pytest.raises(ValueError, match="adapter_disabled"):
        mod.validate_artifact({**artifact, "adapter_disabled": False})
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact({**artifact, "inference_substrate": "live_llm_inference"})
    with pytest.raises(ValueError, match="verifier_is_oracle"):
        mod.validate_artifact({**artifact, "verifier_is_oracle": False})
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact({**artifact, "honest_verdict": "blocked: wrong_prefix"})
    with pytest.raises(ValueError, match="ready score"):
        bad = json.loads(json.dumps(artifact))
        bad["identical_event_bytes_and_arm_budget_parity"]["ok"] = False
        bad["reproducibility_checksum"] = mod._checksum(bad)
        mod.validate_artifact(bad)
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact({**artifact, "reproducibility_checksum": "sha256:bad"})


def test_req_arc_lrbh_5929_writer_and_repository_artifact(monkeypatch, tmp_path: Path) -> None:
    """SCENARIO-ARC-LRBH-5929-NO-SOLVE-CREDIT: writer is stable and repo artifact validates."""

    monkeypatch.setattr(mod, "preconditions", _ok_preconditions)
    monkeypatch.setattr(mod, "run_bound_live_ab", lambda *_args, **_kwargs: _positive_run())

    output = tmp_path / "experiment_5929.json"
    artifact = mod.write_artifact(root=tmp_path, output_path=output)
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["registry_unchanged"] is True
    mod.validate_artifact(artifact)

    if RESULT_PATH.exists():
        repo_artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        for field in mod.REQUIRED_ARTIFACT_FIELDS:
            assert field in repo_artifact
        assert repo_artifact["schema"] == mod.SCHEMA
        assert repo_artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
        assert repo_artifact["honest_verdict"].startswith(
            ("complete_positive:", "complete_null:", "retired:", "blocked_precondition:")
        )
        mod.validate_artifact(repo_artifact)
