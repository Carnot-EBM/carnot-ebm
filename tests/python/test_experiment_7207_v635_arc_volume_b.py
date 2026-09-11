"""Tests for the second independently bounded V635 selfparse session.

Spec refs: REQ-ARC-WMTE-7207 and SCENARIO-ARC-WMTE-7207-*.
"""

from __future__ import annotations

import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest

from carnot import experiment_7206_v635_arc_volume_a as base
from carnot import experiment_7207_v635_arc_volume_b as exp


def test_scenario_7207_configuration_is_scoped_and_restored() -> None:
    """SCENARIO-ARC-WMTE-7207-CONFIGURATION changes only the delegated process scope."""

    original = {
        name: getattr(base, name)
        for name in (
            "TASK_ID",
            "EXPERIMENT_ID",
            "RANDOM_SEED",
            "SCHEMA",
            "DRIVING_REQUIREMENT",
            "MODULE_PATH",
            "WRAPPER_PATH",
            "TEST_PATH",
            "RESULT_PATH",
            "CHECKPOINT_PATH",
            "CHECKPOINT_SCHEMA",
            "RAW_DIR",
            "SIBLING_PATH",
            "SIBLING_TASK_ID",
        )
    }

    with exp.configured_runtime() as configured:
        assert configured.TASK_ID == "exp7207-arc-volume-b"
        assert configured.EXPERIMENT_ID == 7207
        assert configured.RANDOM_SEED == 7_207_001
        assert configured.SCHEMA == "carnot.experiment_7207.arc_volume_b.v1"
        assert configured.DRIVING_REQUIREMENT == "REQ-ARC-WMTE-7207"
        assert configured.RESULT_PATH == Path("results/experiment_7207_v635_arc_volume_b.json")
        assert configured.CHECKPOINT_PATH.parent == Path(
            "results/checkpoints/experiment_7207_v635_arc_volume_b"
        )
        assert configured.RAW_DIR == Path("results/raw/experiment_7207")
        assert configured.SIBLING_PATH == Path("results/experiment_7206_v635_arc_volume_a.json")
        assert configured.SIBLING_TASK_ID == "exp7206-arc-volume-a"
        assert configured.EXPECTED_TASK_CONTRACT == exp.EXPECTED_TASK_CONTRACT
        assert exp.MODULE_PATH in configured.REQUIRED_SOURCE_PATHS
        assert exp.WRAPPER_PATH in configured.REQUIRED_SOURCE_PATHS
        assert exp.TEST_PATH in configured.REQUIRED_SOURCE_PATHS

    assert {name: getattr(base, name) for name in original} == original


def test_scenario_7207_configuration_restores_after_failure() -> None:
    """SCENARIO-ARC-WMTE-7207-CONFIGURATION restores shared state after an exception."""

    original_seed = base.RANDOM_SEED
    with pytest.raises(RuntimeError, match="fixture"):
        with exp.configured_runtime():
            assert base.RANDOM_SEED == exp.RANDOM_SEED
            raise RuntimeError("fixture")
    assert base.RANDOM_SEED == original_seed


def test_req_7207_freezes_model_budget_and_selfparse_environment() -> None:
    """REQ-ARC-WMTE-7207 keeps the mandated model, bounds, and direct selfparse mode."""

    assert exp.MODEL_SPECS == [{"hf_id": "unsloth/Qwen3.8-27B-GGUF", "quantization": "Q4_K_M"}]
    assert exp.ACTION_BUDGET == 4000
    assert exp.SESSION_TIMEOUT_S == 3600
    assert exp.INDUCTION_TIMEOUT_S == 2400
    assert exp.N_CTX == 49152
    assert exp.COMPLETION_BUDGET == 4096
    with exp.configured_runtime() as configured:
        configured.configure_reused_driver()
        env = configured.reused.session_environment(
            {},
            model_path="/cache/model.gguf",
            gpu_index=1,
            port=9123,
            raw_dir=Path("/tmp/exp7207-raw"),
        )
        assert env["CARNOT_FORCE_LIVE"] == "1"
        assert env["CARNOT_ARC_INDUCE_TOOL_LOOP"] == "selfparse"
        assert env["CARNOT_ARC_INDUCE_N_CTX"] == "49152"
        assert env["CARNOT_ARC_INDUCE_MAX_TOKENS"] == "4096"
        assert "CARNOT_ARC_SUPERVISOR_TOOL_ARM" not in env


def test_scenario_7207_sibling_is_exp7206_and_nonblocking(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7207-SIBLING labels absent session A without blocking session B."""

    with exp.configured_runtime() as configured:
        receipt, rows = configured._optional_sibling(tmp_path, {})
    assert receipt == {"source": "exp7206-arc-volume-a", "state": "absent_nonblocking"}
    assert rows == []


def test_scenario_7207_projection_resolves_current_seed_at_call_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-ARC-WMTE-7207-CONFIGURATION labels current rows as session B."""

    monkeypatch.setattr(
        base,
        "terminal_tool_event_receipt",
        lambda **kwargs: {
            "aggregation_consistent": True,
            "recorded_tool_calls_total": 1,
            "tool_calls_total": 1,
            "tool_calls_by_name": {"list_transitions": 1},
            "tool_call_events": [{"tool_name": "list_transitions"}],
            "error": None,
        },
    )
    run_row = {
        "policy_diagnostics": {
            "induction_attempts": [
                {
                    "started_at": "2026-09-11T10:00:00+00:00",
                    "reason": "stall",
                    "tool_gap": {
                        "selfparse": True,
                        "terminated_by": "early_stop_non_improving",
                        "tool_calls_total": 1,
                    },
                }
            ]
        }
    }
    with exp.configured_runtime() as configured:
        rows = configured.project_tool_inductions(Path("/tmp"), run_row, [])

    assert rows[0]["seed"] == 7_207_001
    assert rows[0]["source_session_id"] == "exp7207-arc-volume-b"
    assert rows[0]["unit_id"] == "r11l:7207001:induction:0"


def test_scenario_7207_terminal_delegate_uses_scoped_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-ARC-WMTE-7207-TERMINAL delegates one run with session B identity."""

    seen: dict[str, object] = {}

    def fake_run_experiment(**kwargs: object) -> dict[str, object]:
        seen.update(
            task_id=base.TASK_ID,
            seed=base.RANDOM_SEED,
            result=kwargs["result_path"],
            sibling=base.SIBLING_TASK_ID,
        )
        return {"status": "complete", "arc_session_complete_score": 1}

    monkeypatch.setattr(base, "run_experiment", fake_run_experiment)
    result = exp.run_experiment(
        root=tmp_path,
        run_date="20260911",
        result_path=tmp_path / "result.json",
        checkpoint_path=tmp_path / "checkpoint.json",
        raw_dir=tmp_path / "raw",
    )
    assert result["arc_session_complete_score"] == 1
    assert seen == {
        "task_id": "exp7207-arc-volume-b",
        "seed": 7_207_001,
        "result": tmp_path / "result.json",
        "sibling": "exp7206-arc-volume-a",
    }
    assert base.TASK_ID == "exp7206-arc-volume-a"


def test_req_7207_validation_and_session_child_delegate_in_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-ARC-WMTE-7207 uses the session B schema for validation and child execution."""

    monkeypatch.setattr(
        base,
        "validate_artifact",
        lambda value: [] if base.SCHEMA == exp.SCHEMA and value == {"ok": True} else ["wrong"],
    )
    monkeypatch.setattr(
        base,
        "run_session_child",
        lambda args: 0 if base.RANDOM_SEED == exp.RANDOM_SEED and args.role == "session" else 1,
    )
    assert exp.validate_artifact({"ok": True}) == []
    assert exp.run_session_child(SimpleNamespace(role="session")) == 0


def test_req_7207_main_and_script_entrypoint_delegate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-ARC-WMTE-7207 exposes the requested executable entrypoint."""

    calls: list[tuple[str, int, object]] = []

    def fake_main(argv: object = None) -> int:
        calls.append((base.TASK_ID, base.RANDOM_SEED, argv))
        return 0

    monkeypatch.setattr(base, "main", fake_main)
    assert exp.main(["--validate", "/tmp/result.json"]) == 0
    with pytest.raises(SystemExit) as module_stopped:
        runpy.run_path(str(exp.REPO_ROOT / exp.MODULE_PATH), run_name="__main__")
    assert module_stopped.value.code == 0
    monkeypatch.setattr(exp, "main", lambda argv=None: calls.append(("script", 7207, argv)) or 0)
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(exp.REPO_ROOT / exp.WRAPPER_PATH), run_name="__main__")
    assert stopped.value.code == 0
    assert calls == [
        ("exp7207-arc-volume-b", 7_207_001, ["--validate", "/tmp/result.json"]),
        ("exp7207-arc-volume-b", 7_207_001, None),
        ("script", 7207, None),
    ]


def test_req_7207_task_contract_matches_roadmap() -> None:
    """REQ-ARC-WMTE-7207 authenticates the exact producer contract before model work."""

    with exp.configured_runtime() as configured:
        observed = configured._task_contract(exp.REPO_ROOT / configured.ROADMAP_PATH)
    assert json.loads(json.dumps(observed)) == json.loads(json.dumps(exp.EXPECTED_TASK_CONTRACT))
