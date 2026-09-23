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
    assert {row["max_new_tokens_per_call"] for row in rows} == {4096}


def test_b2_induction_generation_receives_4096_without_changing_7471(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-10008-INDUCTION-BUDGET overrides only the B2 child."""

    observed: dict[str, int] = {}

    def mocked_generation_boundary(_args: object, *, induction_max_tokens: int) -> int:
        observed["max_tokens"] = induction_max_tokens
        return 0

    monkeypatch.setattr(exp7531.e6, "run_live_session", mocked_generation_boundary)

    assert exp7531.run_live_session(exp7531.argparse.Namespace()) == 0
    assert observed == {"max_tokens": 4096}
    assert exp7531.exp7471.MAX_NEW_TOKENS == 256


def test_e6_induction_proposer_constructor_receives_4096() -> None:
    """SCENARIO-ARC-WMTE-10008-INDUCTION-BUDGET reaches the generation constructor."""

    observed: dict[str, object] = {}

    class MockProposer:
        def __init__(self, **kwargs: object) -> None:
            observed.update(kwargs)

    proposer = exp7531.e6._construct_induction_proposer(
        exp7531.argparse.Namespace(
            model_path="/tmp/mock-model.gguf",
            model_revision="mock-revision",
            port=59999,
        ),
        max_tokens=4096,
        proposer_type=MockProposer,
    )

    assert isinstance(proposer, MockProposer)
    assert observed["max_tokens"] == 4096


def test_completion_token_distribution_exposes_nonuniform_lengths() -> None:
    """SCENARIO-ARC-WMTE-10008-TOKEN-DISTRIBUTION reports cap diagnostics."""

    rows = [
        {"completion_tokens": 4096},
        {"completion_tokens": 1024},
        {"completion_tokens": 3072},
        {"completion_tokens": 1024},
    ]

    distribution = exp7531.completion_token_distribution(rows)

    assert distribution == {
        "count": 4,
        "histogram": {"1024": 2, "3072": 1, "4096": 1},
        "minimum": 1024,
        "maximum": 4096,
        "mean": 2304.0,
        "median": 2048.0,
        "unique_count": 3,
        "uniform": False,
        "possible_remaining_hard_cap": False,
    }


def test_durable_completion_receipts_exclude_stale_attempt_usage(tmp_path) -> None:
    """REQ-ARC-WMTE-10008 publishes transport-backed completion lengths."""

    request_dir = tmp_path / "sp80__seed-1" / "requests"
    request_dir.mkdir(parents=True)
    (request_dir / "00_response.json").write_text(
        json.dumps({"usage": {"completion_tokens": 4096}})
    )
    attempts = [
        {
            "episode_id": "sp80:seed-1",
            "attempt_id": "sp80:seed-1:induction:1",
            "completion_tokens": 4096,
        },
        {
            "episode_id": "sp80:seed-1",
            "attempt_id": "sp80:seed-1:induction:2",
            "completion_tokens": 4096,
        },
    ]

    evidence = exp7531.durable_completion_token_evidence(attempts, tmp_path)

    assert evidence["distribution"]["histogram"] == {"4096": 1}
    assert evidence["per_attempt_reported_distribution"]["histogram"] == {"4096": 2}
    assert evidence["rows_without_distinct_completed_response"] == ["sp80:seed-1:induction:2"]


def test_induction_model_spec_records_the_effective_budget() -> None:
    """REQ-ARC-WMTE-10008 does not retain E6's historical 256-token receipt."""

    original = {
        "decoding": {"max_new_tokens": 256, "retry_budget": 0},
        "runtime_settings": {"max_new_tokens_per_call": 256, "n_ctx": 49_152},
    }

    corrected = exp7531.induction_model_spec(original)

    assert corrected["decoding"]["max_new_tokens"] == 4096
    assert corrected["runtime_settings"]["max_new_tokens_per_call"] == 4096
    assert original["decoding"]["max_new_tokens"] == 256
    assert original["runtime_settings"]["max_new_tokens_per_call"] == 256


def test_positive_control_diagnostic_flags_a_saturated_progress_proxy() -> None:
    """REQ-ARC-WMTE-10008 preserves the known positive-control limitation."""

    attempts = [
        {
            "planned": False,
            "verifier_result": "not_observed",
            "progress_within_window": True,
        },
        {
            "planned": False,
            "verifier_result": "not_observed",
            "progress_within_window": True,
        },
    ]

    diagnostic = exp7531.positive_control_diagnostic(attempts)

    assert diagnostic["planned_true_count"] == 0
    assert diagnostic["verifier_observed_count"] == 0
    assert diagnostic["progress_within_window_true_count"] == 2
    assert diagnostic["progress_signal_saturated"] is True


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
