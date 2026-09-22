"""CPU tests for REQ-ARC-WMTE-7530 B2 live measurement orchestration."""

from __future__ import annotations

import json

from carnot import experiment_7531_b2_induction_gate_measurement as exp7531


def test_schedule_reuses_only_the_frozen_e6_panel() -> None:
    """REQ-ARC-WMTE-7530 keeps game selection frozen before outcomes."""

    frozen = json.loads((exp7531.REPO_ROOT / exp7531.FROZEN_PANEL_PATH).read_text())
    rows = exp7531.build_schedule(frozen)

    assert len(rows) == len(exp7531.PANEL_GAMES) * len(exp7531.EPISODE_SEEDS)
    assert tuple(dict.fromkeys(row["game"] for row in rows)) == exp7531.PANEL_GAMES
    assert {row["game"] for row in rows} == set(exp7531.PANEL_GAMES)
    assert all(row["adapter_disabled"] is True for row in rows)
    assert all(row["game_source_read"] is False for row in rows)


def test_static_precondition_requires_gpu_one_already_selected(monkeypatch) -> None:
    """REQ-ARC-WMTE-7530 checks CUDA visibility before runtime initialization."""

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(exp7531, "STAGE1_PATH", exp7531.FROZEN_PANEL_PATH)
    checks, _cited, _environment_dir = exp7531.static_preconditions()

    check = next(row for row in checks if row["check"] == "cuda_visible_devices_already_gpu_1")
    assert check["passed"] is False
    assert check["observed"] is None
    assert check["path"] == "process_environment"


def test_blocked_artifact_keeps_every_scheduled_unit() -> None:
    """SCENARIO-ARC-WMTE-7530-FEASIBILITY never drops blocked units."""

    schedule = [
        {"episode_id": "sb26:seed-1", "game": "sb26", "seed": 1},
        {"episode_id": "vc33:seed-1", "game": "vc33", "seed": 1},
    ]
    failed = {
        "check": "physical_gpu_1_idle_before_runtime_preflight",
        "passed": False,
        "observed": {"used_memory_mb": 700},
    }
    artifact = exp7531.blocked_artifact(
        failed_check=failed,
        checks=[failed],
        cited=[],
        schedule=schedule,
        duration_s=0.5,
    )

    assert artifact["honest_verdict"].startswith("blocked_")
    assert artifact["inference_substrate"] == "no_model_load"
    assert artifact["gate_opportunity_count"] == 0
    assert artifact["induction_attempt_count"] == 0
    assert artifact["sample_floor"]["met"] is False
    assert [row["disposition"] for row in artifact["episode_rows"]] == [
        "unstarted",
        "unstarted",
    ]
    assert artifact["gate_ready_to_ship"] is False
