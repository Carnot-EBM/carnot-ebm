"""Tests for Exp 3774 KV260 opportunistic terminal-state audit.

Spec refs: REQ-HW-3774, SCENARIO-HW-3774.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_3774_kv260_opportunistic_continuity_audit as mod


class RecordingRunner:
    """Synthetic command runner so tests never depend on a live board."""

    def __init__(self, probes: dict[tuple[str, ...], list[mod.CommandProbe]]) -> None:
        self.probes = {command: list(values) for command, values in probes.items()}
        self.commands: list[tuple[str, ...]] = []
        self.timeouts: list[float] = []

    def __call__(
        self,
        command: tuple[str, ...],
        timeout_s: float = 60.0,
    ) -> mod.CommandProbe:
        self.commands.append(command)
        self.timeouts.append(timeout_s)
        if command not in self.probes or not self.probes[command]:
            raise AssertionError(f"unexpected command: {command!r}")
        return self.probes[command].pop(0)


def _probe(
    command: tuple[str, ...],
    exit_code: int,
    stdout: str = "",
    stderr: str = "",
    duration_s: float = 0.01,
) -> mod.CommandProbe:
    return mod.CommandProbe(
        command=command,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        duration_s=duration_s,
    )


def _assert_required_fields_and_principles(payload: dict[str, object]) -> None:
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in payload
        assert field in payload["field_principles"]
        assert "principle" not in str(payload[field]).lower()


def test_req_hw_3774_spec_anchor_declares_opportunistic_audit_contract() -> None:
    """REQ-HW-3774: OpenSpec declares SSH-only opportunistic terminal audit."""
    spec = Path("openspec/capabilities/fpga/spec.md").read_text(encoding="utf-8")
    assert "REQ-HW-3774" in spec
    assert "SCENARIO-HW-3774" in spec
    assert "experiment_3774_kv260_opportunistic_continuity_audit.json" in spec
    assert "ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'" in spec
    assert "ssh kria 'xmutil listapps'" in spec
    assert "blocked_kv260_ssh_unreachable" in spec
    assert mod.SUCCESS_VERDICT in spec
    assert "host SD-card device-node precondition is permanently retired" in spec


@pytest.mark.parametrize(
    ("outcome", "probes", "expected"),
    [
        pytest.param(
            "terminal_holds",
            {
                mod.KV260_SSH_COMMAND: [_probe(mod.KV260_SSH_COMMAND, 0)],
                mod.KV260_LISTAPPS_COMMAND: [
                    _probe(
                        mod.KV260_LISTAPPS_COMMAND,
                        0,
                        stdout="carnot_ising_v4    XRT_FLAT    id_ok\n",
                    )
                ],
            },
            {
                "commands": [mod.KV260_SSH_COMMAND, mod.KV260_LISTAPPS_COMMAND],
                "verdict": mod.SUCCESS_VERDICT,
                "ssh": True,
                "terminal": True,
                "overlay": True,
                "overlay_name": "carnot_ising_v4",
            },
            id="terminal_holds",
        ),
        pytest.param(
            "ssh_blocked",
            {
                mod.KV260_SSH_COMMAND: [
                    _probe(
                        mod.KV260_SSH_COMMAND,
                        255,
                        stderr="ssh: connect to host kria port 22: timed out\n",
                    )
                ],
            },
            {
                "commands": [mod.KV260_SSH_COMMAND],
                "verdict": mod.BLOCKED_VERDICT,
                "ssh": False,
                "terminal": False,
                "overlay": False,
                "overlay_name": None,
            },
            id="ssh_blocked",
        ),
        pytest.param(
            "overlay_regressed",
            {
                mod.KV260_SSH_COMMAND: [_probe(mod.KV260_SSH_COMMAND, 0)],
                mod.KV260_LISTAPPS_COMMAND: [
                    _probe(mod.KV260_LISTAPPS_COMMAND, 0, stdout="k26-starter-kits\n")
                ],
            },
            {
                "commands": [mod.KV260_SSH_COMMAND, mod.KV260_LISTAPPS_COMMAND],
                "verdict": mod.REGRESSION_VERDICT,
                "ssh": True,
                "terminal": False,
                "overlay": False,
                "overlay_name": None,
            },
            id="overlay_regressed",
        ),
    ],
)
def test_scenario_hw_3774_honest_outcomes_are_parametrized(
    outcome: str,
    probes: dict[tuple[str, ...], list[mod.CommandProbe]],
    expected: dict[str, object],
) -> None:
    """SCENARIO-HW-3774: terminal, SSH-blocked, and regressed outcomes are honest."""
    runner = RecordingRunner(probes)

    payload = mod.build_artifact(command_runner=runner, duration_s=2.25)

    assert outcome in {"terminal_holds", "ssh_blocked", "overlay_regressed"}
    assert runner.commands == expected["commands"]
    assert payload["honest_verdict"] == expected["verdict"]
    assert payload["inference_substrate"] == "hardware_smoke"
    assert payload["kv260_ssh_reachable"] is expected["ssh"]
    assert payload["terminal_state_holds"] is expected["terminal"]
    assert payload["kv260_overlay_loadable"] is expected["overlay"]
    assert payload["kv260_overlay_name"] == expected["overlay_name"]
    assert payload["duration_s"] == pytest.approx(2.25)
    assert payload["random_seed"] == mod.RANDOM_SEED
    assert "gguf" not in json.dumps(payload).lower()
    assert "cuda" not in json.dumps(payload).lower()
    assert "live_inference" not in json.dumps(payload).lower()
    assert "/dev/mmcblk" not in json.dumps(runner.commands)
    assert "/dev/disk" not in json.dumps(runner.commands)
    assert all("python3 -" not in " ".join(command) for command in runner.commands)

    preconditions = payload["preconditions_checked"]
    assert preconditions[0]["resource"] == "kv260_ssh"
    assert preconditions[0]["command"] == mod.command_to_string(mod.KV260_SSH_COMMAND)
    assert preconditions[0]["checked_before_board_operations"] is True
    _assert_required_fields_and_principles(payload)

    if outcome == "ssh_blocked":
        assert payload["command_probes"]["kv260_xmutil_listapps"] is None
        assert payload["operator_regression_note"] == "kv260_ssh_unreachable"
    if outcome == "overlay_regressed":
        assert payload["operator_regression_note"] == "carnot_overlay_not_listed"


def test_scenario_hw_3774_root_required_xmutil_fallback_stays_over_ssh() -> None:
    """SCENARIO-HW-3774: sudo xmutil fallback preserves the required first probe."""
    root_error = "xmutil should be called with root privileges. Please try again using 'sudo'.\n"
    runner = RecordingRunner(
        {
            mod.KV260_SSH_COMMAND: [_probe(mod.KV260_SSH_COMMAND, 0)],
            mod.KV260_LISTAPPS_COMMAND: [_probe(mod.KV260_LISTAPPS_COMMAND, 1, stderr=root_error)],
            mod.KV260_LISTAPPS_SUDO_COMMAND: [
                _probe(
                    mod.KV260_LISTAPPS_SUDO_COMMAND,
                    0,
                    stdout="carnot_ising_v2_n64    XRT_FLAT    id_ok\n",
                )
            ],
        }
    )

    payload = mod.build_artifact(command_runner=runner, duration_s=3.0)

    assert runner.commands == [
        mod.KV260_SSH_COMMAND,
        mod.KV260_LISTAPPS_COMMAND,
        mod.KV260_LISTAPPS_SUDO_COMMAND,
    ]
    assert payload["honest_verdict"] == mod.SUCCESS_VERDICT
    assert payload["terminal_state_holds"] is True
    assert payload["kv260_overlay_name"] == "carnot_ising_v2_n64"
    assert payload["command_probes"]["kv260_xmutil_listapps"]["exit_code"] == 1
    assert payload["command_probes"]["kv260_xmutil_listapps_sudo"]["exit_code"] == 0


def test_req_hw_3774_field_principles_match_required_why_annotations() -> None:
    """REQ-HW-3774: required fields store values and field_principles stores why."""
    assert mod.FIELD_PRINCIPLES == {
        "honest_verdict": (
            "Terminal prefix; blocked_kv260_ssh_unreachable if the board is unreachable."
        ),
        "inference_substrate": "an SSH board check, not live inference.",
        "kv260_ssh_reachable": (
            "The SSH-reachability fact (NOT host SD-card presence -- the retired wrong mechanism)."
        ),
        "terminal_state_holds": (
            "Confirms the .340 terminal state did not regress; opportunistic, not mandated."
        ),
        "preconditions_checked": (
            "Records the SSH check was actually run before any board operation."
        ),
        "random_seed": "Determinism precondition.",
        "reproducibility_checksum": "Content hash catches drift.",
        "duration_s": "Wall-clock plausibility floor.",
    }


def test_req_hw_3774_run_experiment_writes_checksum_and_schema(tmp_path: Path) -> None:
    """REQ-HW-3774: result JSON has schema, required values, and stable checksum."""
    runner = RecordingRunner(
        {
            mod.KV260_SSH_COMMAND: [_probe(mod.KV260_SSH_COMMAND, 0)],
            mod.KV260_LISTAPPS_COMMAND: [
                _probe(mod.KV260_LISTAPPS_COMMAND, 0, stdout="carnot_ising\n")
            ],
        }
    )

    out_path = mod.run_experiment(
        repo_root=tmp_path,
        command_runner=runner,
        duration_s=4.0,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path == tmp_path / mod.OUTPUT_REL_PATH
    assert payload["schema"] == mod.SCHEMA
    assert payload["experiment_id"] == mod.EXPERIMENT_ID
    assert payload["task_id"] == mod.TASK_ID
    assert payload["prior_terminal_confirmation_experiments"] == [
        "exp3730",
        "exp3741",
        "exp3762",
    ]
    assert len(payload["reproducibility_checksum"]) == 64
    _assert_required_fields_and_principles(payload)

    checksum_payload = dict(payload)
    expected_checksum = checksum_payload.pop("reproducibility_checksum")
    assert mod.sha256_payload(checksum_payload) == expected_checksum


def test_req_hw_3774_validate_rejects_missing_required_fields() -> None:
    """REQ-HW-3774: artifact validation rejects missing required schema fields."""
    payload = {field: "present" for field in mod.REQUIRED_ARTIFACT_FIELDS if field != "duration_s"}

    with pytest.raises(ValueError, match="duration_s"):
        mod.validate_artifact(payload)


def test_req_hw_3774_run_command_records_successful_subprocess() -> None:
    """REQ-HW-3774: command probes retain stdout, stderr, exit code, and duration."""
    probe = mod.run_command(
        ("python", "-c", "import sys; print('ok'); print('err', file=sys.stderr)"),
        timeout_s=10.0,
    )

    assert probe.exit_code == 0
    assert probe.stdout == "ok\n"
    assert probe.stderr == "err\n"
    assert probe.combined_output == "ok\nerr\n"
    assert probe.as_dict()["command"] == (
        "python -c 'import sys; print('\"'\"'ok'\"'\"'); print('\"'\"'err'\"'\"', file=sys.stderr)'"
    )
    assert probe.duration_s >= 0.0
