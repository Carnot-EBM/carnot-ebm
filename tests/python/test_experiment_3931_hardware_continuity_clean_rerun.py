"""Tests for Exp 3931 clean hardware continuity rerun.

Spec refs: REQ-HW-3931, SCENARIO-HW-3931.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_3931_hardware_continuity_clean_rerun as mod


def _base_artifact(
    *,
    gatemate_reachable: bool = False,
    polarfire_reachable: bool = True,
    kv260_reachable: bool = False,
    gatemate_state: str = "blocked_gatemate_toolchain_missing",
    polarfire_state: str = "terminal_hash_verified_soft_cpu_ssh_dispatch",
    kv260_state: str = "blocked_kv260_ssh_unreachable",
) -> dict[str, Any]:
    return {
        "schema": "carnot.hardware_continuity_consolidated.v1",
        "experiment": 3922,
        "spec_refs": ["REQ-HW-3922", "SCENARIO-HW-3922"],
        "honest_verdict": "success: stale_exp3922_zero_timer",
        "inference_substrate": "hardware_smoke",
        "gatemate_reachable": gatemate_reachable,
        "polarfire_reachable": polarfire_reachable,
        "kv260_reachable": kv260_reachable,
        "gatemate_state": gatemate_state,
        "gatemate_terminal_state_reached": False,
        "duration_s": 0.0,
        "run_duration_s": 0.0,
        "polarfire_state": polarfire_state,
        "kv260_state": kv260_state,
        "fabric_acceleration_claimed": False,
        "preconditions_checked": [
            {
                "resource": "polarfire_ssh",
                "command": "ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire true",
                "available": polarfire_reachable,
                "checked_before_board_operations": True,
            },
            {
                "resource": "kv260_ssh",
                "command": "ssh -o ConnectTimeout=5 -o BatchMode=yes kria true",
                "available": kv260_reachable,
                "checked_before_board_operations": True,
            },
        ],
        "gatemate_summary": None,
        "polarfire_dispatch_summary": {
            "polarfire_workload_validated": polarfire_reachable,
            "result_hash_match": polarfire_reachable,
            "run_duration_s": 5.25 if polarfire_reachable else 0.0,
            "claim_boundary": "soft_cpu_ssh_dispatch_no_fpga_fabric_acceleration",
        }
        if polarfire_reachable
        else None,
        "kv260_loaded_overlay": None,
        "kv260_carnot_ising_active": False,
        "kv260_uio_devices": [],
        "kv260_command_transcripts": {
            "xmutil_listapps": None,
            "uio_list": None,
        },
        "audit_duration_s": 0.0,
        "field_principles": {
            "duration_s": "old marker GGUF CUDA text should not survive",
        },
        "random_seed": 3922,
        "reproducibility_checksum": "0" * 64,
        "flagged_adversarial": True,
        "corrigendum_pending": [{"detail": "old marker GGUF CUDA text should not survive"}],
    }


class FakeBaseBuilder:
    """REQ-HW-3931 synthetic Exp 3922 builder with optional command timing."""

    def __init__(
        self,
        artifact: dict[str, Any],
        commands: list[mod.CommandResult] | None = None,
    ) -> None:
        self.artifact = artifact
        self.commands = list(commands or [])
        self.calls: list[Path] = []

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(Path(kwargs["repo_root"]))
        runner = kwargs["command_runner"]
        for result in self.commands:
            runner(result.command, 10.0)
        return json.loads(json.dumps(self.artifact))


def _command(
    command: tuple[str, ...],
    returncode: int = 0,
    duration_s: float = 0.25,
) -> mod.CommandResult:
    return mod.CommandResult(command, returncode, duration_s=duration_s)


class RecordingRunner:
    """SCENARIO-HW-3931 command runner returning fixed command results."""

    def __init__(self, results: dict[tuple[str, ...], mod.CommandResult]) -> None:
        self.results = results
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, command: tuple[str, ...], timeout_s: float) -> mod.CommandResult:
        del timeout_s
        self.commands.append(command)
        if command not in self.results:
            raise AssertionError(f"unexpected command: {command!r}")
        return self.results[command]


def _assert_required_fields_are_bare(payload: dict[str, Any]) -> None:
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in payload
        assert field in payload["field_principles"]
        assert not (
            isinstance(payload[field], dict) and set(payload[field]) == {"value", "principle"}
        )


def test_req_hw_3931_spec_entry_declares_clean_rerun_contract() -> None:
    """REQ-HW-3931: OpenSpec anchors the clean rerun contract."""
    spec = Path("openspec/capabilities/fpga/spec.md").read_text(encoding="utf-8")

    assert "REQ-HW-3931" in spec
    assert "SCENARIO-HW-3931" in spec
    assert "experiment_3931_hardware_continuity_clean_rerun.json" in spec
    assert "openFPGALoader -c dirtyJtag --detect" in spec
    assert "ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire" in spec
    assert "ssh -o ConnectTimeout=5 -o BatchMode=yes kria" in spec
    assert "ssh kria 'xmutil listapps'" in spec
    assert "ssh kria 'ls /dev/uio*'" in spec
    assert "fabric_acceleration_claimed=false" in spec
    assert "blocked_all_boards_unreachable" in spec


def test_req_hw_3931_clean_success_repairs_zero_timers_and_markers(tmp_path: Path) -> None:
    """REQ-HW-3931: a reachable board requires distinct clean top-level timers."""
    artifact = mod.build_artifact(
        repo_root=tmp_path,
        base_builder=FakeBaseBuilder(_base_artifact()),
        duration_s=8.5,
        run_duration_s=5.25,
    )

    assert artifact["schema"] == mod.SCHEMA
    assert artifact["experiment"] == mod.EXPERIMENT_ID
    assert artifact["spec_refs"] == mod.SPEC_REFS
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert artifact["honest_verdict"] == (
        "success: hardware_continuity_clean_"
        "gatemateblocked_gatemate_toolchain_missing_"
        "pfterminal_hash_verified_soft_cpu_ssh_dispatch_"
        "kvblocked_kv260_ssh_unreachable_"
        "distinct_timers_no_fabric_claim"
    )
    assert artifact["duration_s"] == 8.5
    assert artifact["run_duration_s"] == 5.25
    assert artifact["duration_s"] != artifact["run_duration_s"]
    assert artifact["fabric_acceleration_claimed"] is False
    assert artifact["inference_substrate"] == "hardware_smoke"
    assert artifact["prior_exp3922_diagnosis"].startswith("Exp 3922 recorded zero")
    assert "flagged_adversarial" not in artifact
    assert "corrigendum_pending" not in artifact
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert "gguf" not in json.dumps(artifact).lower()
    assert "cuda" not in json.dumps(artifact).lower()
    assert "mmcblk" not in json.dumps(artifact).lower()
    _assert_required_fields_are_bare(artifact)
    mod.validate_artifact(artifact)


def test_scenario_hw_3931_all_boards_unreachable_records_blocked_verdict(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-3931: all board misses emit the all-boards blocked verdict."""
    artifact = mod.build_artifact(
        repo_root=tmp_path,
        base_builder=FakeBaseBuilder(
            _base_artifact(
                polarfire_reachable=False,
                kv260_reachable=False,
                polarfire_state="blocked_polarfire_ssh_unreachable",
            )
        ),
        duration_s=0.7,
        run_duration_s=0.2,
    )

    assert artifact["honest_verdict"] == "blocked_all_boards_unreachable"
    assert artifact["gatemate_reachable"] is False
    assert artifact["polarfire_reachable"] is False
    assert artifact["kv260_reachable"] is False
    assert artifact["duration_s"] == 0.7
    assert artifact["run_duration_s"] == 0.2
    assert artifact["fabric_acceleration_claimed"] is False
    mod.validate_artifact(artifact)


def test_scenario_hw_3931_measures_command_runner_duration_when_not_injected(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-3931: board-operation time comes from measured command results."""
    polarfire_preflight = ("ssh", "polarfire", "true")
    kv260_preflight = ("ssh", "kria", "true")
    base_builder = FakeBaseBuilder(
        _base_artifact(kv260_reachable=True, kv260_state="nonterminal_carnot_ising_inactive_uio_present"),
        commands=[
            _command(polarfire_preflight, duration_s=0.4),
            _command(kv260_preflight, duration_s=0.3),
        ],
    )
    runner = RecordingRunner(
        {
            polarfire_preflight: _command(polarfire_preflight, duration_s=0.4),
            kv260_preflight: _command(kv260_preflight, duration_s=0.3),
        }
    )
    ticks = iter([0.0, 1.4])

    artifact = mod.build_artifact(
        repo_root=tmp_path,
        command_runner=runner,
        base_builder=base_builder,
        clock=lambda: next(ticks),
    )

    assert runner.commands == [polarfire_preflight, kv260_preflight]
    assert artifact["duration_s"] == 1.4
    assert artifact["run_duration_s"] == 0.7
    assert artifact["honest_verdict"].startswith("success: hardware_continuity_clean_gatemate")
    mod.validate_artifact(artifact)


def test_req_hw_3931_run_experiment_writes_json_and_validates(tmp_path: Path) -> None:
    """REQ-HW-3931: run_experiment writes the requested deliverable JSON."""
    out_path = mod.run_experiment(
        repo_root=tmp_path,
        base_builder=FakeBaseBuilder(_base_artifact()),
        duration_s=9.0,
        run_duration_s=5.25,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path == tmp_path / mod.OUTPUT_REL_PATH
    assert payload["schema"] == mod.SCHEMA
    assert payload["experiment"] == mod.EXPERIMENT_ID
    assert payload["spec_refs"] == mod.SPEC_REFS
    assert payload["random_seed"] == mod.RANDOM_SEED
    assert payload["reproducibility_checksum"] == mod.payload_checksum(payload)
    _assert_required_fields_are_bare(payload)
    mod.validate_artifact(payload)


def test_req_hw_3931_validate_artifact_reports_schema_and_gate_errors(
    tmp_path: Path,
) -> None:
    """REQ-HW-3931: validation rejects stale schema, markers, and timer errors."""
    good = mod.build_artifact(
        repo_root=tmp_path,
        base_builder=FakeBaseBuilder(_base_artifact()),
        duration_s=8.5,
        run_duration_s=5.25,
    )

    for mutation, expected in [
        (lambda item: item.update(schema="wrong"), "schema"),
        (lambda item: item.update(experiment=3922), "experiment"),
        (lambda item: item.update(spec_refs=["REQ-HW-3922"]), "spec_refs"),
        (lambda item: item.update(random_seed=3922), "random_seed"),
        (lambda item: item.pop("kv260_state"), "missing required fields"),
        (lambda item: item.update(field_principles=[]), "field_principles"),
        (
            lambda item: item["field_principles"].pop("kv260_state"),
            "field_principles missing",
        ),
        (
            lambda item: item.update(kv260_state={"value": "x", "principle": "wrapped"}),
            "bare scalar",
        ),
        (lambda item: item.update(fabric_acceleration_claimed=True), "must be false"),
        (lambda item: item.update(inference_substrate="live_model"), "hardware_smoke"),
        (
            lambda item: item.update(
                honest_verdict="success: stale_exp3922_zero_timer",
            ),
            "clean success prefix",
        ),
        (
            lambda item: item.update(
                gatemate_reachable=False,
                polarfire_reachable=False,
                kv260_reachable=False,
                honest_verdict="success: hardware_continuity_clean_gatematex_pfx_kvx_distinct_timers_no_fabric_claim",
            ),
            "blocked_all_boards_unreachable",
        ),
        (lambda item: item.update(duration_s=0.0), "positive"),
        (lambda item: item.update(run_duration_s=0.0), "positive"),
        (lambda item: item.update(run_duration_s=8.5), "distinct timers"),
        (
            lambda item: item.update(kv260_command_transcripts={"retired": "/dev/mmcblk0"}),
            "forbidden substrate marker",
        ),
        (
            lambda item: item.update(polarfire_dispatch_summary={"marker": "torch.cuda"}),
            "forbidden substrate marker",
        ),
        (lambda item: item.update(reproducibility_checksum="0" * 64), "does not match"),
    ]:
        bad = json.loads(json.dumps(good))
        mutation(bad)
        try:
            mod.validate_artifact(bad)
        except ValueError as exc:
            assert expected in str(exc)
        else:  # pragma: no cover - assertion guard
            raise AssertionError(f"{expected} mutation was accepted")


def test_req_hw_3931_measurement_fallbacks_are_deterministic() -> None:
    """REQ-HW-3931: timer fallback uses board summaries when commands lack durations."""
    base = _base_artifact(gatemate_reachable=True)
    base["gatemate_summary"] = {"run_duration_s": 1.25}
    base["kv260_command_transcripts"] = {
        "xmutil_listapps": {"duration_s": 0.2},
        "uio_list": {"duration_s": 0.3},
    }
    base["polarfire_dispatch_summary"] = {"run_duration_s": 2.0}

    assert mod.measured_board_duration(base, []) == 3.75
    assert mod.measured_board_duration({"gatemate_summary": {"run_duration_s": "bad"}}, []) == 0.0
    assert mod.state_token("Terminal Hash Verified!") == "terminal_hash_verified"
