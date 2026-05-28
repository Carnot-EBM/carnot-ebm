"""Tests for Exp 3282 Garak install/probe manifest.

Spec refs: REQ-REPORT-3282, SCENARIO-REPORT-3282.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Sequence

import pytest

from carnot.reporting import garak_install_probe_manifest_3282 as mod


REQUIRED_FIELDS = {
    "garak_install_probe_manifest_ready",
    "garak_runner_ready",
    "garak_available",
    "garak_version",
    "garak_import_command",
    "garak_cli_command",
    "probe_inventory",
    "promptinject_probe_count",
    "local_target_adapter_plan",
    "install_blockers",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class FakeRunner:
    def __init__(self, responses: dict[str, mod.CommandResult]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def __call__(self, command: Sequence[str], timeout_s: int) -> mod.CommandResult:
        rendered = mod.command_to_string(command)
        self.calls.append(rendered)
        if mod.INVENTORY_MARKER in rendered and mod.INVENTORY_MARKER in self.responses:
            return self.responses[mod.INVENTORY_MARKER]
        for needle, response in sorted(self.responses.items(), key=lambda item: len(item[0]), reverse=True):
            if needle in rendered:
                return response
        return mod.CommandResult(returncode=127, stdout="", stderr=f"missing fake for {rendered}")


def _prior_exp3274(*, garak_available: bool) -> dict[str, Any]:
    return {
        "experiment_id": "exp3274",
        "garak_available": garak_available,
        "blocked_reasons": [] if garak_available else ["blocked_garak_unavailable"],
        "honest_verdict": "complete: prior garak state",
    }


def test_req_report_3282_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3282: OpenSpec declares the Garak toolchain contract first."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3282" in spec
    assert "SCENARIO-REPORT-3282" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "full Garak red-team benchmark" in spec
    assert "scripts/research_conductor.py" in spec
    for field in REQUIRED_FIELDS:
        assert field in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3282_blocked_manifest_when_garak_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-REPORT-3282: missing Garak becomes a precise blocked artifact."""

    _write_json(tmp_path, mod.EXP3274_REL_PATH, _prior_exp3274(garak_available=False))
    runner = FakeRunner(
        {
            mod.IMPORT_MARKER: mod.CommandResult(
                returncode=1,
                stdout="",
                stderr="ModuleNotFoundError: No module named 'garak'",
            ),
            "garak --version": mod.CommandResult(
                returncode=127,
                stdout="",
                stderr="FileNotFoundError: garak",
            ),
        }
    )
    monkeypatch.setattr(mod.shutil, "which", lambda name: None)

    artifact = mod.run_experiment(
        project_root=tmp_path,
        output_path=mod.OUTPUT_REL_PATH,
        runner=runner,
        monotonic=iter([100.0, 101.25]).__next__,
    )
    second = mod.run_experiment(
        project_root=tmp_path,
        output_path=mod.OUTPUT_REL_PATH,
        runner=runner,
        monotonic=iter([200.0, 201.0]).__next__,
    )
    saved = json.loads((tmp_path / mod.OUTPUT_REL_PATH).read_text(encoding="utf-8"))

    assert saved == second
    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["garak_install_probe_manifest_ready"] is True
    assert artifact["garak_runner_ready"] is False
    assert artifact["garak_available"] is False
    assert artifact["garak_version"] == ""
    assert "import garak" in artifact["garak_import_command"]
    assert artifact["garak_cli_command"] == "garak --version"
    assert artifact["promptinject_probe_count"] == 0
    assert artifact["duration_s"] == pytest.approx(1.25)
    assert artifact["reproducibility_checksum"] == second["reproducibility_checksum"]
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["honest_verdict"].startswith("complete:")
    assert "garak_runner_ready=false" in artifact["honest_verdict"]

    blocker_reasons = {item["reason"] for item in artifact["install_blockers"]}
    assert {
        "blocked_garak_import_unavailable",
        "blocked_garak_cli_unavailable",
        "blocked_probe_inventory_unavailable",
    } <= blocker_reasons
    assert any("ModuleNotFoundError" in item["stderr_summary"] for item in artifact["install_blockers"])
    assert all(item["next_action"] for item in artifact["install_blockers"])

    assert artifact["probe_inventory"]
    assert {item["family"] for item in artifact["probe_inventory"]} >= {
        "promptinject",
        "jailbreak",
        "encoding",
        "leakage",
        "hallucination",
    }
    assert all(item["available"] is False for item in artifact["probe_inventory"])
    assert all(item["source"] == "static_expected_recheck" for item in artifact["probe_inventory"])

    plan = artifact["local_target_adapter_plan"]
    assert plan["adapter_kind"] == "llama_cpp_openai_compatible_rest"
    assert "llama-server" in plan["llama_cpp_server_command"][0]
    assert "--target_type openai.OpenAICompatible" in plan["garak_generator_command"]
    assert set(plan["mandated_targets"]) == set(mod.MANDATED_TARGET_MODELS)
    assert plan["does_not_run_model"] is True

    prior = [
        item
        for item in artifact["preconditions_checked"]
        if item["name"] == "prior_exp3274_garak_available"
    ][0]
    assert prior["passed"] is False
    assert prior["blocked_reasons"] == ["blocked_garak_unavailable"]


def test_req_report_3282_isolated_uv_runner_can_supply_live_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-3282: an isolated command path can satisfy the runner contract."""

    _write_json(tmp_path, mod.EXP3274_REL_PATH, _prior_exp3274(garak_available=False))
    inventory_stdout = json.dumps(
        [
            {
                "module": "garak.probes.promptinject",
                "classes": ["HijackHateHumans", "HijackKillHumans"],
            },
            {"module": "garak.probes.jailbreak", "classes": ["DAN"]},
            {"module": "garak.probes.encoding", "classes": ["InjectBase64"]},
            {"module": "garak.probes.leakreplay", "classes": ["ReplayLeak"]},
            {"module": "garak.probes.hallucination", "classes": ["Snowball"]},
            {"module": "garak.probes.unrelated", "classes": ["Ignored"]},
        ]
    )
    runner = FakeRunner(
        {
            f"{mod.IMPORT_MARKER}": mod.CommandResult(
                returncode=1,
                stdout="",
                stderr="ModuleNotFoundError: No module named 'garak'",
            ),
            "garak --version": mod.CommandResult(returncode=127, stdout="", stderr="not found"),
            f"uv run --no-project --with garak python -c": mod.CommandResult(
                returncode=0,
                stdout='{"version": "1.2.3"}\n',
                stderr="",
            ),
            f"{mod.INVENTORY_MARKER}": mod.CommandResult(
                returncode=0,
                stdout=inventory_stdout,
                stderr="",
            ),
            "uv run --no-project --with garak garak --version": mod.CommandResult(
                returncode=0,
                stdout="garak 1.2.3\n",
                stderr="",
            ),
        }
    )
    monkeypatch.setattr(mod.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)

    artifact = mod.run_experiment(
        project_root=tmp_path,
        output_path=mod.OUTPUT_REL_PATH,
        runner=runner,
        monotonic=iter([1.0, 3.5]).__next__,
    )

    assert artifact["garak_install_probe_manifest_ready"] is True
    assert artifact["garak_runner_ready"] is True
    assert artifact["garak_available"] is True
    assert artifact["garak_version"] == "1.2.3"
    assert artifact["install_blockers"] == []
    assert artifact["promptinject_probe_count"] == 2
    assert artifact["garak_import_command"].startswith("uv run --no-project --with garak")
    assert artifact["garak_cli_command"].startswith("uv run --no-project --with garak")
    assert artifact["honest_verdict"].startswith("complete: garak_install_probe_manifest_ready=true")

    inventory = artifact["probe_inventory"]
    assert [item for item in inventory if item["family"] == "promptinject"]
    assert {item["family"] for item in inventory} == {
        "encoding",
        "hallucination",
        "jailbreak",
        "leakage",
        "promptinject",
    }
    assert all(item["available"] is True for item in inventory)
    assert any(item["class_name"] == "HijackHateHumans" for item in inventory)

    precondition_names = {item["name"] for item in artifact["preconditions_checked"]}
    assert {
        "project_garak_import",
        "project_garak_cli",
        "isolated_uv_garak_import",
        "isolated_uv_garak_cli",
        "garak_probe_inventory",
        "prior_exp3274_garak_available",
    } <= precondition_names


def test_req_report_3282_helper_edges() -> None:
    """REQ-REPORT-3282: helper parsing, classification, and validation are strict."""

    assert mod.classify_probe_family("garak.probes.promptinject", "Hijack") == "promptinject"
    assert mod.classify_probe_family("garak.probes.dan", "DAN") == "jailbreak"
    assert mod.classify_probe_family("garak.probes.encoding", "Base64") == "encoding"
    assert mod.classify_probe_family("garak.probes.leakreplay", "Leak") == "leakage"
    assert mod.classify_probe_family("garak.probes.snowball", "Snowball") == "hallucination"
    assert mod.classify_probe_family("garak.probes.other", "Other") == "other"
    assert mod.stderr_summary("a\n" * 20, limit=12).endswith("...")

    parsed = mod.parse_import_probe_stdout('{"version": "9.9.9"}')
    assert parsed == "9.9.9"
    assert mod.parse_import_probe_stdout("garak 1.0.0") == "garak 1.0.0"
    assert mod.parse_import_probe_stdout("{") == "{"
    assert mod.parse_import_probe_stdout("[1]") == "[1]"
    assert mod.first_nonempty_line("", " \n ") == ""

    raw_inventory = json.dumps(
        [
            {"module": "garak.probes.promptinject", "classes": ["A", "B"]},
            {"module": "garak.probes.other", "classes": ["Ignored"]},
            {"module": "garak.probes.encoding", "classes": "Bad"},
            "skip",
        ]
    )
    inventory = mod.parse_probe_inventory(raw_inventory)
    assert [item["class_name"] for item in inventory] == ["A", "B"]
    assert mod.parse_probe_inventory("{") == []
    assert mod.parse_probe_inventory("{}") == []
    assert mod.inventory_command_from_import({"passed": False}) == ()
    project_inventory_command = mod.inventory_command_from_import(
        {"passed": True, "command": f"{sys.executable} -c {mod.IMPORT_MARKER}"}
    )
    assert project_inventory_command[0] == sys.executable
    assert mod.GARAK_INVENTORY_CODE in project_inventory_command

    failed_uv_checks = [
        {
            "name": "isolated_uv_garak_import",
            "passed": False,
            "command": "uv run --no-project --with garak python -c import",
            "returncode": 1,
            "stderr_summary": "build failed",
            "blocked_reason": "blocked_garak_import_unavailable",
        }
    ]
    blockers = mod.install_blockers(False, False, failed_uv_checks)
    assert blockers[0]["reason"] == "blocked_garak_import_unavailable"
    assert (
        mod.next_action_for("blocked_other")
        == "inspect the command row and rerun the failed toolchain probe"
    )
    assert mod.run_command((sys.executable, "-c", "print('ok')"), 10).stdout.strip() == "ok"

    missing_or_bad = Path("bad-exp3282.json")
    assert mod.read_json_object(missing_or_bad) == {}
    missing_or_bad.write_text("{", encoding="utf-8")
    try:
        assert mod.read_json_object(missing_or_bad) == {}
    finally:
        missing_or_bad.unlink()

    artifact = {
        "garak_install_probe_manifest_ready": True,
        "garak_runner_ready": False,
        "garak_available": False,
        "garak_version": "",
        "garak_import_command": "python -c import garak",
        "garak_cli_command": "garak --version",
        "probe_inventory": mod.static_probe_inventory(),
        "promptinject_probe_count": 0,
        "local_target_adapter_plan": mod.local_target_adapter_plan(),
        "install_blockers": [],
        "preconditions_checked": [],
        "random_seed": 3282,
        "reproducibility_checksum": "",
        "duration_s": 0.0,
        "honest_verdict": "complete: ok",
    }
    artifact["reproducibility_checksum"] = mod.reproducibility_checksum(artifact)
    mod.validate_artifact(artifact)
    with pytest.raises(ValueError, match="missing required"):
        mod.validate_artifact({key: artifact[key] for key in REQUIRED_FIELDS - {"duration_s"}})
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(artifact | {"honest_verdict": "blocked"})
    with pytest.raises(ValueError, match="promptinject_probe_count"):
        mod.validate_artifact(artifact | {"promptinject_probe_count": -1})
    assert mod.duration(10.0, 8.0) == pytest.approx(0.0)
