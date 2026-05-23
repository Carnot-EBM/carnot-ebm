"""Tests for Exp 2914 GateMate toolchain preflight.

Spec refs: REQ-HW-066, SCENARIO-HW-066.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from carnot.experiment_2914_gatemate_toolchain_preflight import (
    ARTIFACT_FILENAME,
    CommandResult,
    build_artifact,
    command_result_text,
    parse_version_text,
    run_experiment,
    _default_run_command,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "fpga" / "spec.md"

REQUIRED_FIELDS = (
    "honest_verdict",
    "gatemate_toolchain_ready",
    "yosys_path",
    "yosys_version",
    "nextpnr_gatemate_path",
    "nextpnr_gatemate_version",
    "openfpgaloader_path",
    "openfpgaloader_version",
    "missing_toolchain",
    "rtl_sources_present",
    "constraints_present",
    "no_flash_attempted",
    "inference_substrate",
    "duration_s",
    "run_date",
)


def _write_rtl(repo_root: Path, *, valid: bool = True) -> None:
    rtl_dir = repo_root / "hardware" / "gatemate"
    rtl_dir.mkdir(parents=True, exist_ok=True)
    body = (
        "module ising_n16_gatemate;\n"
        "localparam integer N_VARIABLES = 16;\n"
        "endmodule\n"
    )
    if not valid:
        body = "module wrong_top; endmodule\n"
    (rtl_dir / "ising_n16_gatemate.v").write_text(body, encoding="utf-8")


def _write_constraint(repo_root: Path) -> None:
    constraint_dir = repo_root / "hardware" / "gatemate"
    constraint_dir.mkdir(parents=True, exist_ok=True)
    (constraint_dir / "ising_n16_gatemate.ccf").write_text(
        "# GateMate pin constraints placeholder for test\n",
        encoding="utf-8",
    )


def _fake_runner(results: dict[str, CommandResult]):
    calls: list[tuple[str, ...]] = []

    def run(args: list[str], timeout_s: float) -> CommandResult:
        calls.append(tuple(args))
        name = Path(args[0]).name
        return results[name]

    return run, calls


def _which_from(paths: dict[str, str]):
    def which(name: str) -> str | None:
        return paths.get(name)

    return which


def _clock(values: list[float]):
    state = iter(values)

    def monotonic() -> float:
        return next(state)

    return monotonic


def test_exp2914_spec_entry_present() -> None:
    """REQ-HW-066: the FPGA capability spec anchors this preflight."""
    spec = SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-HW-066" in spec
    assert "SCENARIO-HW-066" in spec
    assert ARTIFACT_FILENAME in spec


def test_exp2914_blocks_when_nextpnr_gatemate_is_missing(tmp_path: Path) -> None:
    """SCENARIO-HW-066: absent nextpnr-gatemate yields a blocked artifact."""
    _write_rtl(tmp_path)
    runner, calls = _fake_runner(
        {
            "yosys": CommandResult(0, "Yosys 0.64+149\n", ""),
            "openFPGALoader": CommandResult(0, "openFPGALoader v1.1.1\n", ""),
        }
    )

    artifact = build_artifact(
        repo_root=tmp_path,
        home_dir=tmp_path / "home",
        run_command=runner,
        which_func=_which_from(
            {
                "yosys": "/opt/oss-cad-suite/bin/yosys",
                "openFPGALoader": "/opt/oss-cad-suite/bin/openFPGALoader",
            }
        ),
        monotonic=_clock([10.0, 10.25]),
    )

    assert artifact["honest_verdict"] == "blocked_gatemate_toolchain_missing"
    assert artifact["gatemate_toolchain_ready"] is False
    assert artifact["missing_toolchain"] == ["nextpnr-gatemate"]
    assert artifact["yosys_path"] == "/opt/oss-cad-suite/bin/yosys"
    assert artifact["yosys_version"] == "Yosys 0.64+149"
    assert artifact["nextpnr_gatemate_path"] == ""
    assert artifact["nextpnr_gatemate_version"] == ""
    assert artifact["openfpgaloader_version"] == "openFPGALoader v1.1.1"
    assert artifact["rtl_sources_present"] is True
    assert artifact["constraints_present"] is False
    assert artifact["no_flash_attempted"] is True
    assert artifact["inference_substrate"] == "hardware_preflight"
    assert artifact["duration_s"] == 0.25
    assert artifact["run_date"] == "20260523"
    assert not any("-b" in call for call in calls)
    assert not any(any("synth_gatemate" in arg for arg in call) for call in calls)


def test_exp2914_ready_path_requires_all_requested_tools_and_constraints(
    tmp_path: Path,
) -> None:
    """REQ-HW-066: the ready verdict requires all requested binaries and sources."""
    _write_rtl(tmp_path)
    _write_constraint(tmp_path)
    runner, _calls = _fake_runner(
        {
            "yosys": CommandResult(0, "Yosys 0.64+149\n", ""),
            "nextpnr-gatemate": CommandResult(0, "nextpnr-gatemate 0.10-test\n", ""),
            "openFPGALoader": CommandResult(0, "openFPGALoader v1.1.1\n", ""),
        }
    )

    artifact = build_artifact(
        repo_root=tmp_path,
        home_dir=tmp_path / "home",
        run_command=runner,
        which_func=_which_from(
            {
                "yosys": "/suite/bin/yosys",
                "nextpnr-gatemate": "/suite/bin/nextpnr-gatemate",
                "openFPGALoader": "/suite/bin/openFPGALoader",
            }
        ),
        monotonic=_clock([20.0, 21.0]),
    )

    assert artifact["honest_verdict"] == "complete_gatemate_toolchain_preflight_ready"
    assert artifact["gatemate_toolchain_ready"] is True
    assert artifact["missing_toolchain"] == []
    assert artifact["constraints_present"] is True
    assert artifact["nextpnr_gatemate_version"] == "nextpnr-gatemate 0.10-test"


def test_exp2914_rejects_missing_or_malformed_sources(tmp_path: Path) -> None:
    """REQ-HW-066: source presence means the expected n=16 top exists."""
    _write_constraint(tmp_path)
    runner, _calls = _fake_runner(
        {
            "yosys": CommandResult(0, "Yosys test\n", ""),
            "nextpnr-gatemate": CommandResult(0, "nextpnr test\n", ""),
            "openFPGALoader": CommandResult(0, "loader test\n", ""),
        }
    )

    missing_artifact = build_artifact(
        repo_root=tmp_path,
        home_dir=tmp_path / "home",
        run_command=runner,
        which_func=_which_from(
            {
                "yosys": "/bin/yosys",
                "nextpnr-gatemate": "/bin/nextpnr-gatemate",
                "openFPGALoader": "/bin/openFPGALoader",
            }
        ),
        monotonic=_clock([28.0, 29.0]),
    )

    assert missing_artifact["honest_verdict"] == (
        "blocked_gatemate_sources_or_constraints_missing"
    )
    assert missing_artifact["rtl_sources_present"] is False

    _write_rtl(tmp_path, valid=False)
    _write_constraint(tmp_path)
    runner, _calls = _fake_runner(
        {
            "yosys": CommandResult(0, "Yosys test\n", ""),
            "nextpnr-gatemate": CommandResult(0, "nextpnr test\n", ""),
            "openFPGALoader": CommandResult(0, "loader test\n", ""),
        }
    )

    artifact = build_artifact(
        repo_root=tmp_path,
        home_dir=tmp_path / "home",
        run_command=runner,
        which_func=_which_from(
            {
                "yosys": "/bin/yosys",
                "nextpnr-gatemate": "/bin/nextpnr-gatemate",
                "openFPGALoader": "/bin/openFPGALoader",
            }
        ),
        monotonic=_clock([30.0, 31.5]),
    )

    assert artifact["honest_verdict"] == "blocked_gatemate_sources_or_constraints_missing"
    assert artifact["gatemate_toolchain_ready"] is False
    assert artifact["rtl_sources_present"] is False
    assert artifact["constraints_present"] is True


def test_exp2914_records_modern_alternatives_without_satisfying_legacy_binary(
    tmp_path: Path,
) -> None:
    """REQ-HW-066: nextpnr-himbaechel is recorded but does not replace nextpnr-gatemate."""
    _write_rtl(tmp_path)
    home_bin = tmp_path / "home" / "tools" / "oss-cad-suite" / "bin"
    home_bin.mkdir(parents=True)
    for name in ("nextpnr-himbaechel", "gmpack"):
        (home_bin / name).write_text("#!/bin/sh\n", encoding="utf-8")
    runner, _calls = _fake_runner(
        {
            "yosys": CommandResult(0, "Yosys test\n", ""),
            "openFPGALoader": CommandResult(0, "loader test\n", ""),
            "nextpnr-himbaechel": CommandResult(0, "nextpnr-himbaechel 0.10\n", ""),
            "gmpack": CommandResult(
                1,
                "",
                "Open Source Tools for GateMate FPGAs Version v1.13\n",
            ),
        }
    )

    artifact = build_artifact(
        repo_root=tmp_path,
        home_dir=tmp_path / "home",
        run_command=runner,
        which_func=_which_from(
            {
                "yosys": "/suite/bin/yosys",
                "openFPGALoader": "/suite/bin/openFPGALoader",
            }
        ),
        monotonic=_clock([40.0, 41.0]),
    )

    alternatives = {
        entry["name"]: entry for entry in artifact["detected_alternative_toolchain"]
    }
    assert artifact["honest_verdict"] == "blocked_gatemate_toolchain_missing"
    assert artifact["missing_toolchain"] == ["nextpnr-gatemate"]
    assert alternatives["nextpnr-himbaechel"]["path"] == str(home_bin / "nextpnr-himbaechel")
    assert alternatives["gmpack"]["version"] == (
        "Open Source Tools for GateMate FPGAs Version v1.13"
    )


def test_exp2914_versions_every_detected_candidate_path(tmp_path: Path) -> None:
    """REQ-HW-066: every detected executable path gets a version probe."""
    _write_rtl(tmp_path)
    home_bin = tmp_path / "home" / "tools" / "oss-cad-suite" / "bin"
    home_bin.mkdir(parents=True)
    (home_bin / "yosys").write_text("#!/bin/sh\n", encoding="utf-8")
    runner, calls = _fake_runner({"yosys": CommandResult(0, "Yosys test\n", "")})

    artifact = build_artifact(
        repo_root=tmp_path,
        home_dir=tmp_path / "home",
        run_command=runner,
        which_func=_which_from({"yosys": "/suite/bin/yosys"}),
        monotonic=_clock([45.0, 46.0]),
    )

    yosys = next(entry for entry in artifact["required_toolchain"] if entry["name"] == "yosys")
    assert yosys["candidate_paths"] == ["/suite/bin/yosys", str(home_bin / "yosys")]
    assert [entry["path"] for entry in yosys["candidate_version_results"]] == (
        yosys["candidate_paths"]
    )
    assert ("/suite/bin/yosys", "-V") in calls
    assert (str(home_bin / "yosys"), "-V") in calls


def test_exp2914_run_experiment_writes_required_json(tmp_path: Path) -> None:
    """SCENARIO-HW-066: run_experiment writes the deliverable JSON."""
    _write_rtl(tmp_path)
    runner, _calls = _fake_runner(
        {
            "yosys": CommandResult(0, "Yosys 0.64+149\n", ""),
            "openFPGALoader": CommandResult(0, "openFPGALoader v1.1.1\n", ""),
        }
    )
    artifact_path = tmp_path / "results" / ARTIFACT_FILENAME

    artifact = run_experiment(
        repo_root=tmp_path,
        artifact_path=artifact_path,
        home_dir=tmp_path / "home",
        run_command=runner,
        which_func=_which_from(
            {
                "yosys": "/opt/oss-cad-suite/bin/yosys",
                "openFPGALoader": "/opt/oss-cad-suite/bin/openFPGALoader",
            }
        ),
        monotonic=_clock([50.0, 50.5]),
    )

    written = json.loads(artifact_path.read_text(encoding="utf-8"))
    missing = [field for field in REQUIRED_FIELDS if field not in written]
    assert not missing
    assert written == artifact
    assert artifact_path.name == ARTIFACT_FILENAME


def test_exp2914_version_text_helpers() -> None:
    """REQ-HW-066: version capture tolerates stdout, stderr, and empty results."""
    assert parse_version_text(CommandResult(0, "\nYosys test\n", "")) == "Yosys test"
    assert parse_version_text(
        CommandResult(
            1,
            "",
            "Error: unrecognised option\nOpen Source Tools Version v1.13\n",
        )
    ) == "Open Source Tools Version v1.13"
    assert parse_version_text(CommandResult(1, "", "\nloader error\n")) == "loader error"
    assert parse_version_text(CommandResult(127, "", "")) == ""
    assert command_result_text(CommandResult(1, "out\n", "err\n")) == "out\nerr"


def test_exp2914_default_runner_executes_version_command() -> None:
    """REQ-HW-066: the real runner captures return code, stdout, and stderr."""
    result = _default_run_command(
        [sys.executable, "-c", "import sys; print('stdout'); print('stderr', file=sys.stderr)"],
        timeout_s=5.0,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "stdout"
    assert result.stderr.strip() == "stderr"
