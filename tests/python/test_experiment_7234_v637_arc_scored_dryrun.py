"""Tests for the V637 local scored-stack ARC dry run.

Spec refs: REQ-ARC-WMTE-7234 and SCENARIO-ARC-WMTE-7234-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import runpy

import pytest

from carnot import experiment_7234_v637_arc_scored_dryrun as exp

pytestmark = pytest.mark.memory_watchdog_skip


def _backend(
    name: str,
    *,
    invoked: bool,
    dispatches: int = 0,
    engine_writes: int = 0,
    consumed: int = 0,
) -> dict[str, object]:
    return {
        "backend": name,
        "disposition": "complete" if invoked else "blocked_external_absence",
        "model_invoked": invoked,
        "model_spec": (
            {
                "hf_id": exp.PRIMARY_MODEL_ID if name == "llamacpp" else exp.COMPARATOR_MODEL_ID,
                "quantization": exp.PRIMARY_QUANTIZATION if name == "llamacpp" else "INT4",
            }
            if invoked
            else None
        ),
        "transport_completed": invoked,
        "semantic_usable": bool(dispatches and engine_writes and consumed),
        "tool_dispatches": dispatches,
        "engine_writes": engine_writes,
        "policy_engine_consumptions": consumed,
        "valid_engine_count": engine_writes,
        "actions": 25 if invoked else 0,
        "levels": 0,
        "action_rows": [],
        "induction_rows": [],
        "raw_request_manifest": [],
        "phase_spans": [],
        "runner_receipt": {},
        "gpu_receipts": {},
        "error": None if invoked else "comparator_runtime_absent",
    }


def test_req_7234_spec_and_frozen_contract() -> None:
    """REQ-ARC-WMTE-7234 freezes model, budget, seed, and output identity."""

    spec = (exp.REPO_ROOT / exp.SPEC_PATH).read_text(encoding="utf-8")
    assert "## REQ-ARC-WMTE-7234:" in spec
    assert "SCENARIO-ARC-WMTE-7234-TRANSPORT" in spec
    assert exp.TASK_ID == "exp7234-arc-scored-dryrun"
    assert exp.EXPERIMENT_ID == 7234
    assert exp.MILESTONE == "2026.09.637"
    assert exp.RUN_DATE == "20260912"
    assert exp.RANDOM_SEED == 7_234_001
    assert exp.ACTION_BUDGET == 256
    assert exp.BACKEND_TIMEOUT_S == 1500
    assert exp.TOTAL_RUNTIME_CAP_S == 3300
    assert exp.STARTUP_TIMEOUT_S == 240
    assert exp.PRIMARY_MODEL_ID == "unsloth/Qwen3.8-27B-GGUF"
    assert exp.PRIMARY_QUANTIZATION == "Q4_K_M"
    assert exp.COMPARATOR_MODEL_ID == "RedHatAI/Qwen3.8-27B-INT4"


def test_scenario_7234_preflight_unwraps_only_exact_wrappers() -> None:
    """SCENARIO-ARC-WMTE-7234-PREFLIGHT keeps ordinary mappings intact."""

    assert exp.unwrap_evidence_value({"principle": "why", "value": 3}) == 3
    missing = {"value": 3}
    extra = {"principle": "why", "value": 3, "evidence": "ordinary"}
    assert exp.unwrap_evidence_value(missing) is missing
    assert exp.unwrap_evidence_value(extra) is extra


def test_scenario_7234_factory_uses_submitted_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-ARC-WMTE-7234-FACTORY uses the real factory and no solution input."""

    from carnot.agentic import arc_competition_agent as agent

    seen: dict[str, object] = {}

    class FakePolicy:
        def __init__(self, game_id: str, **kwargs: object) -> None:
            seen["game_id"] = game_id
            seen["kwargs"] = kwargs

    monkeypatch.setattr(agent, "E3AgentPolicy", FakePolicy)
    policy, receipt = exp.build_disposable_submitted_policy("lp85", proposer=object())

    assert isinstance(policy, FakePolicy)
    assert seen["game_id"] == "lp85"
    assert "solutions" not in seen["kwargs"]
    assert receipt["factory"] == "make_carnot_agent"
    assert receipt["policy_class"] == "E3AgentPolicy"
    assert receipt["adapter_disabled"] is True
    assert receipt["denied_inputs"] == [
        "banked_solutions",
        "game_adapter",
        "game_source",
        "ground_truth_state",
        "registry_contents",
        "solved_trajectories",
    ]


def test_req_7234_rotation_excludes_prior_task_target() -> None:
    """REQ-ARC-WMTE-7234 selects from metadata before outcomes are visible."""

    selected, receipt = exp.select_game_from_rotation(
        ("r11l", "lp85", "ls20"), credited_task_targets={"r11l"}
    )
    assert selected == "lp85"
    assert receipt["rotation"] == ["r11l", "lp85", "ls20"]
    assert receipt["excluded_credited_task_targets"] == ["r11l"]
    assert receipt["selection_basis"] == "first_metadata_rotation_entry_not_used_by_prior_task"


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        (
            "llamacpp",
            {
                "CARNOT_ARC_LLM_BACKEND": "llamacpp",
                "CARNOT_ARC_INDUCE_TOOL_LOOP": "selfparse",
                "CARNOT_FORCE_LIVE": "1",
                "CARNOT_ARC_ACTION_PROVENANCE": "1",
            },
        ),
        (
            "vllm",
            {
                "CARNOT_ARC_LLM_BACKEND": "vllm",
                "CARNOT_ARC_INDUCE_TOOL_LOOP": "selfparse",
                "CARNOT_FORCE_LIVE": "1",
                "CARNOT_ARC_ACTION_PROVENANCE": "1",
            },
        ),
    ],
)
def test_scenario_7234_transport_environment_is_scoped(
    backend: str, expected: dict[str, str], tmp_path: Path
) -> None:
    """SCENARIO-ARC-WMTE-7234-TRANSPORT preserves equal policy limits."""

    env = exp.session_environment(
        {},
        backend=backend,
        model_path=tmp_path / "model",
        gpu_index=1,
        port=9911,
        raw_dir=tmp_path / "raw",
    )
    for key, value in expected.items():
        assert env[key] == value
    assert env["CARNOT_ARC_INDUCE_MAX_TOKENS"] == str(exp.COMPLETION_BUDGET)
    assert env["CARNOT_ARC_INDUCE_TIMEOUT"] == str(exp.INDUCTION_TIMEOUT_S)
    assert env["CARNOT_ARC_RANDOM_SEED"] == str(exp.RANDOM_SEED)
    assert "CARNOT_ARC_SUPERVISOR_TOOL_ARM" not in env
    if backend == "vllm":
        assert env["CARNOT_ARC_VLLM_MODEL_DIR"] == str(tmp_path / "model")
        assert "CARNOT_ARC_GGUF_PATH" not in env
    else:
        assert env["CARNOT_ARC_GGUF_PATH"] == str(tmp_path / "model")
        assert "CARNOT_ARC_VLLM_MODEL_DIR" not in env


def test_scenario_7234_receipts_reduce_real_policy_channels() -> None:
    """SCENARIO-ARC-WMTE-7234-RECEIPTS separates transport from useful execution."""

    session = {
        "backend": "llamacpp",
        "model_invoked": True,
        "model_spec": {"hf_id": exp.PRIMARY_MODEL_ID, "quantization": "Q4_K_M"},
        "completions": [
            {"endpoint": "/v1/chat/completions", "response_sha256": "sha256:" + "a" * 64}
        ],
        "action_rows": [
            {"i": 0, "top_branch": "induce.plan_needs_reset"},
            {"i": 1, "top_branch": "execute.plan_step", "heldout_accuracy": 0.8},
        ],
        "induction_rows": [
            {
                "tool_calls_total": 2,
                "tool_calls_by_name": {"run_engine_on_transitions": 1},
                "engine_written": True,
                "engine_functionally_identity": False,
                "verify_accuracy": 0.8,
                "heldout_accuracy": 0.75,
                "planned": True,
                "skipped": "",
            }
        ],
        "run_row": {"actions": 2, "levels": 0, "reached": 0},
        "runner_receipt": {},
        "gpu_receipts": {},
        "phase_spans": [],
        "error": None,
    }
    row = exp.reduce_backend_session(session)
    assert row["transport_completed"] is True
    assert row["tool_dispatches"] == 2
    assert row["engine_writes"] == 1
    assert row["policy_engine_consumptions"] == 1
    assert row["valid_engine_count"] == 1
    assert row["non_identity_prediction_count"] == 1
    assert row["trust_acceptance_count"] == 1
    assert row["model_planned_actions"] == 1
    assert row["semantic_usable"] is True


def test_scenario_7234_terminal_null_allows_comparator_absence(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7234-COMPARATOR-BLOCK completes without inflating readiness."""

    checks = [exp.gate_check("sources", "repo", "paths", True, True)]
    artifact = exp.build_terminal_artifact(
        run_date=exp.RUN_DATE,
        duration_s=61.0,
        started_at_utc="2026-09-12T00:00:00+00:00",
        ended_at_utc="2026-09-12T00:01:01+00:00",
        checks=checks,
        source_hashes={"source": "sha256:" + "b" * 64},
        selection_receipt={"selected_game": "lp85"},
        backend_rows=[_backend("llamacpp", invoked=True), _backend("vllm", invoked=False)],
        validation_receipts=[],
    )
    assert artifact["status"] == "complete"
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("complete_null_")
    assert artifact["scored_dryrun_complete_score"] == 1
    assert artifact["local_scored_path_ready_score"] == 0
    assert artifact["model_invoked"] is True
    assert artifact["MODEL_SPECS"] == [
        {"hf_id": exp.PRIMARY_MODEL_ID, "quantization": exp.PRIMARY_QUANTIZATION}
    ]
    assert exp.validate_artifact(artifact) == []
    path = tmp_path / "artifact.json"
    exp.atomic_write(path, artifact)
    assert exp.validate_artifact(path) == []


def test_scenario_7234_ready_requires_dispatch_write_and_consumption() -> None:
    """SCENARIO-ARC-WMTE-7234-TERMINAL-NULL rejects partial plumbing evidence."""

    rows = [_backend("llamacpp", invoked=True, dispatches=1, engine_writes=1, consumed=1)]
    rows.append(_backend("vllm", invoked=True, dispatches=1, engine_writes=0, consumed=0))
    artifact = exp.build_terminal_artifact(
        run_date=exp.RUN_DATE,
        duration_s=61.0,
        started_at_utc="2026-09-12T00:00:00+00:00",
        ended_at_utc="2026-09-12T00:01:01+00:00",
        checks=[exp.gate_check("sources", "repo", "paths", True, True)],
        source_hashes={},
        selection_receipt={"selected_game": "lp85"},
        backend_rows=rows,
        validation_receipts=[],
    )
    assert artifact["local_scored_path_ready_score"] == 1
    assert artifact["verdict_class"] == "positive"
    broken = deepcopy(artifact)
    broken["per_game_results"][0]["policy_engine_consumptions"] = 0
    broken["reproducibility_checksum"] = exp.artifact_checksum(broken)
    assert "local_scored_path_ready_score_inconsistent" in exp.validate_artifact(broken)


def test_entrypoint_is_thin(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-ARC-WMTE-7234 exposes the runnable script without adding logic there."""

    monkeypatch.setattr(exp, "main", lambda argv=None: 0)
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(exp.REPO_ROOT / exp.WRAPPER_PATH), run_name="__main__")
    assert stopped.value.code == 0


def test_validation_rejects_missing_principle_and_checksum() -> None:
    """SCENARIO-ARC-WMTE-7234-RECEIPTS binds all terminal values."""

    artifact = exp.build_terminal_artifact(
        run_date=exp.RUN_DATE,
        duration_s=0.1,
        started_at_utc="2026-09-12T00:00:00+00:00",
        ended_at_utc="2026-09-12T00:00:01+00:00",
        checks=[exp.gate_check("model", "cache", "path", "present", None, False)],
        source_hashes={},
        selection_receipt={},
        backend_rows=[],
        validation_receipts=[],
    )
    assert artifact["status"] == "blocked"
    assert artifact["inference_substrate"] == "blocked_no_run"
    broken = deepcopy(artifact)
    broken["field_principles"].pop("status")
    broken["reproducibility_checksum"] = "sha256:" + "0" * 64
    errors = exp.validate_artifact(broken)
    assert "field_principles_must_cover_every_top_level_field" in errors
    assert "reproducibility_checksum_mismatch" in errors


def test_task_contract_matches_frozen_roadmap() -> None:
    """REQ-ARC-WMTE-7234 authenticates the exact V637 task row."""

    observed = exp.task_contract(exp.REPO_ROOT / exp.ROADMAP_PATH)
    assert json.loads(json.dumps(observed)) == json.loads(json.dumps(exp.EXPECTED_TASK_CONTRACT))


def test_defensive_receipt_and_validation_paths(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7234-RECEIPTS rejects malformed terminal evidence."""

    assert exp.task_contract(tmp_path / "absent.yaml") == {}
    nested = exp.reduce_backend_session(
        {
            "backend": "llamacpp",
            "model_invoked": True,
            "model_spec": {"hf_id": exp.PRIMARY_MODEL_ID},
            "completions": [],
            "action_rows": [],
            "induction_rows": [
                {
                    "tool_gap": {
                        "tool_calls_total": 3,
                        "tool_calls_by_name": {"diff_grids": 3},
                    }
                }
            ],
        }
    )
    assert nested["tool_dispatches"] == 3
    assert exp._tool_names(
        {"tool_gap": {"tool_calls_by_name": {"diff_grids": 3}}}
    ) == {"diff_grids": 3}
    assert exp._tool_names({"tool_calls_by_name": "invalid"}) == {}
    assert exp.validate_artifact(tmp_path / "absent.json") == [
        "artifact_unreadable:FileNotFoundError"
    ]

    valid = exp.build_terminal_artifact(
        run_date=exp.RUN_DATE,
        duration_s=61,
        started_at_utc="2026-09-12T00:00:00+00:00",
        ended_at_utc="2026-09-12T00:01:01+00:00",
        checks=[exp.gate_check("sources", "repo", "paths", True, True)],
        source_hashes={},
        selection_receipt={},
        backend_rows=[_backend("llamacpp", invoked=True), _backend("vllm", invoked=False)],
        validation_receipts=[],
    )
    broken = deepcopy(valid)
    broken.update(
        {
            "schema": "wrong",
            "run_date": "wrong",
            "status": "complete",
            "verdict_class": "wrong",
            "honest_verdict": "wrong",
            "MODEL_SPECS": [],
            "model_invoked": True,
            "duration_s": 1,
        }
    )
    errors = exp.validate_artifact(broken)
    assert "schema_or_experiment_identity_mismatch" in errors
    assert "run_date_mismatch" in errors
    assert "verdict_class_invalid" in errors
    assert "complete_honest_verdict_prefix_invalid" in errors
    assert "model_specs_do_not_match_invocations" in errors
    assert "model_full_generation_duration_floor_failed" in errors

    blocked = deepcopy(valid)
    blocked.update(
        {
            "status": "blocked",
            "honest_verdict": "wrong",
            "model_invoked": False,
        }
    )
    errors = exp.validate_artifact(blocked)
    assert "blocked_honest_verdict_prefix_invalid" in errors
    assert "model_invoked_inconsistent" in errors

    invalid_status = deepcopy(valid)
    invalid_status["status"] = "running"
    assert "status_not_terminal" in exp.validate_artifact(invalid_status)
