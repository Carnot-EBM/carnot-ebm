"""Tests for Exp 2927 GateMate himbaechel/gmpack constraints preflight.

Spec refs: REQ-HW-068, SCENARIO-HW-068.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.experiment_2927_gatemate_himbaechel_constraints_preflight import (
    ARTIFACT_FILENAME,
    CommandResult,
    build_artifact,
    parse_version_text,
    run_experiment,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "fpga" / "spec.md"

REQUIRED_FIELDS = (
    "honest_verdict",
    "gatemate_himbaechel_ready",
    "constraints_ready",
    "tool_paths",
    "tool_versions",
    "device",
    "nextpnr_command_template",
    "gmpack_command_template",
    "rtl_top",
    "constraints_path",
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
        body = "module wrong_width_gatemate; endmodule\n"
    (rtl_dir / "ising_n16_gatemate.v").write_text(body, encoding="utf-8")


def _write_constraint(repo_root: Path) -> Path:
    constraint_dir = repo_root / "hardware" / "gatemate"
    constraint_dir.mkdir(parents=True, exist_ok=True)
    path = constraint_dir / "ising_n16_gatemate.ccf"
    path.write_text("# test-only GateMate constraints\n", encoding="utf-8")
    return path


def _fake_runner(results: dict[tuple[str, ...], CommandResult]):
    calls: list[tuple[str, ...]] = []

    def run(args: list[str], timeout_s: float) -> CommandResult:
        del timeout_s
        call = tuple([Path(args[0]).name, *args[1:]])
        calls.append(call)
        return results[call]

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


def _successful_tool_results() -> dict[tuple[str, ...], CommandResult]:
    return {
        ("yosys", "-V"): CommandResult(0, "Yosys 0.64+149\n", ""),
        (
            "nextpnr-himbaechel",
            "--version",
        ): CommandResult(0, '"nextpnr-himbaechel" Version nextpnr-0.10\n', ""),
        (
            "nextpnr-himbaechel",
            "--device",
            "CCGM1A1",
        ): CommandResult(
            0,
            "Info: Using uarch 'gatemate' for device 'CCGM1A1'\n",
            "",
        ),
        (
            "gmpack",
            "--version",
        ): CommandResult(
            1,
            "",
            "Open Source Tools for GateMate FPGAs Version v1.13\n",
        ),
        ("openFPGALoader", "-V"): CommandResult(0, "openFPGALoader v1.1.1\n", ""),
    }


def _successful_paths() -> dict[str, str]:
    return {
        "yosys": "/opt/oss-cad-suite/bin/yosys",
        "nextpnr-himbaechel": "/opt/oss-cad-suite/bin/nextpnr-himbaechel",
        "gmpack": "/opt/oss-cad-suite/bin/gmpack",
        "openFPGALoader": "/opt/oss-cad-suite/bin/openFPGALoader",
    }


def test_exp2927_spec_entry_present() -> None:
    """REQ-HW-068: the FPGA capability spec anchors this corrected preflight."""
    spec = SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-HW-068" in spec
    assert "SCENARIO-HW-068" in spec
    assert ARTIFACT_FILENAME in spec


def test_exp2927_blocks_on_missing_constraints_without_inventing_pins(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-068: ready tools plus missing constraints emit a blocked artifact."""
    _write_rtl(tmp_path)
    runner, calls = _fake_runner(_successful_tool_results())

    artifact = build_artifact(
        repo_root=tmp_path,
        home_dir=tmp_path / "home",
        run_command=runner,
        which_func=_which_from(_successful_paths()),
        monotonic=_clock([10.0, 10.5]),
        path_env="/opt/oss-cad-suite/bin:/usr/bin",
    )

    assert artifact["honest_verdict"] == "blocked_constraints_missing"
    assert artifact["gatemate_himbaechel_ready"] is True
    assert artifact["constraints_ready"] is False
    assert artifact["constraints_path"] == ""
    assert artifact["rtl_top"] == "ising_n16_gatemate"
    assert artifact["device"] == "CCGM1A1"
    assert artifact["tool_paths"]["nextpnr-himbaechel"].endswith("nextpnr-himbaechel")
    assert artifact["tool_versions"]["gmpack"] == (
        "Open Source Tools for GateMate FPGAs Version v1.13"
    )
    assert "--device CCGM1A1" in artifact["nextpnr_command_template"]
    assert "--vopt ccf=<constraints_path>" in artifact["nextpnr_command_template"]
    assert artifact["gmpack_command_template"].endswith(
        "build/gatemate/ising_n16_gatemate.bit"
    )
    assert artifact["no_flash_attempted"] is True
    assert artifact["inference_substrate"] == "hardware_toolchain_preflight"
    assert artifact["duration_s"] == 0.5
    assert artifact["run_date"] == "20260523"
    assert ("nextpnr-himbaechel", "--device", "CCGM1A1") in calls
    assert not any("synth_gatemate" in arg for call in calls for arg in call)
    assert not any("-b" in call for call in calls)


def test_exp2927_ready_path_requires_tools_device_rtl_and_constraints(
    tmp_path: Path,
) -> None:
    """REQ-HW-068: the ready verdict requires the current toolchain and constraints."""
    _write_rtl(tmp_path)
    constraint_path = _write_constraint(tmp_path)
    runner, _calls = _fake_runner(_successful_tool_results())

    artifact = build_artifact(
        repo_root=tmp_path,
        home_dir=tmp_path / "home",
        run_command=runner,
        which_func=_which_from(_successful_paths()),
        monotonic=_clock([20.0, 21.25]),
    )

    assert artifact["honest_verdict"] == "ready_gatemate_himbaechel_constraints_preflight"
    assert artifact["gatemate_himbaechel_ready"] is True
    assert artifact["constraints_ready"] is True
    assert artifact["constraints_path"] == str(constraint_path)
    assert f"--vopt ccf={constraint_path}" in artifact["nextpnr_command_template"]
    assert artifact["rtl_top"] == "ising_n16_gatemate"


def test_exp2927_uses_first_non_ccf_constraint_when_no_ccf_exists(
    tmp_path: Path,
) -> None:
    """REQ-HW-068: explicit non-CCF constraints are recorded without pin invention."""
    _write_rtl(tmp_path)
    constraint_dir = tmp_path / "hardware" / "gatemate"
    constraint_dir.mkdir(parents=True, exist_ok=True)
    constraint_path = constraint_dir / "ising_n16_gatemate.pcf"
    constraint_path.write_text("# explicit test constraints\n", encoding="utf-8")
    runner, _calls = _fake_runner(_successful_tool_results())

    artifact = build_artifact(
        repo_root=tmp_path,
        home_dir=tmp_path / "home",
        run_command=runner,
        which_func=_which_from(_successful_paths()),
        monotonic=_clock([22.0, 22.5]),
    )

    assert artifact["honest_verdict"] == "ready_gatemate_himbaechel_constraints_preflight"
    assert artifact["constraints_ready"] is True
    assert artifact["constraints_path"] == str(constraint_path)
    assert artifact["constraint_paths_present"] == [str(constraint_path)]


def test_exp2927_missing_build_tool_blocks_toolchain_readiness(tmp_path: Path) -> None:
    """REQ-HW-068: nextpnr-himbaechel must be present for toolchain readiness."""
    _write_rtl(tmp_path)
    _write_constraint(tmp_path)
    runner, calls = _fake_runner(
        {
            ("yosys", "-V"): CommandResult(0, "Yosys test\n", ""),
            ("gmpack", "--version"): CommandResult(0, "gmpack test\n", ""),
            ("openFPGALoader", "-V"): CommandResult(0, "loader test\n", ""),
        }
    )

    artifact = build_artifact(
        repo_root=tmp_path,
        home_dir=tmp_path / "home",
        run_command=runner,
        which_func=_which_from(
            {
                "yosys": "/suite/bin/yosys",
                "gmpack": "/suite/bin/gmpack",
                "openFPGALoader": "/suite/bin/openFPGALoader",
            }
        ),
        monotonic=_clock([30.0, 30.25]),
        known_bin_dirs=[],
    )

    assert artifact["honest_verdict"] == "blocked_gatemate_toolchain_missing"
    assert artifact["gatemate_himbaechel_ready"] is False
    assert artifact["missing_toolchain"] == ["nextpnr-himbaechel"]
    assert not any(call[:3] == ("nextpnr-himbaechel", "--device", "CCGM1A1") for call in calls)


def test_exp2927_nextpnr_device_probe_must_accept_ccgm1a1(tmp_path: Path) -> None:
    """REQ-HW-068: a present nextpnr binary is not ready until CCGM1A1 is accepted."""
    _write_rtl(tmp_path)
    _write_constraint(tmp_path)
    results = _successful_tool_results()
    results[("nextpnr-himbaechel", "--device", "CCGM1A1")] = CommandResult(
        1,
        "",
        "Error: unsupported device CCGM1A1\n",
    )
    runner, _calls = _fake_runner(results)

    artifact = build_artifact(
        repo_root=tmp_path,
        home_dir=tmp_path / "home",
        run_command=runner,
        which_func=_which_from(_successful_paths()),
        monotonic=_clock([40.0, 41.0]),
    )

    assert artifact["honest_verdict"] == "blocked_nextpnr_device_unsupported"
    assert artifact["gatemate_himbaechel_ready"] is False
    assert artifact["nextpnr_device_supported"] is False


def test_exp2927_rejects_missing_n16_rtl_top(tmp_path: Path) -> None:
    """REQ-HW-068: the build plan must name an actual n=16 GateMate top."""
    _write_rtl(tmp_path, valid=False)
    _write_constraint(tmp_path)
    runner, _calls = _fake_runner(_successful_tool_results())

    artifact = build_artifact(
        repo_root=tmp_path,
        home_dir=tmp_path / "home",
        run_command=runner,
        which_func=_which_from(_successful_paths()),
        monotonic=_clock([50.0, 51.0]),
    )

    assert artifact["honest_verdict"] == "blocked_rtl_top_missing"
    assert artifact["rtl_top"] == ""


def test_exp2927_parse_version_prefers_non_error_version_line() -> None:
    """REQ-HW-068: gmpack reports its version after a non-fatal option error."""
    result = CommandResult(
        1,
        "",
        "Error: unrecognised option '--version'\n"
        "Open Source Tools for GateMate FPGAs Version v1.13-1-gdc48418\n",
    )

    assert parse_version_text(result) == (
        "Open Source Tools for GateMate FPGAs Version v1.13-1-gdc48418"
    )
    assert parse_version_text(CommandResult(0, "", "")) == ""


def test_exp2927_run_experiment_writes_required_json(tmp_path: Path) -> None:
    """SCENARIO-HW-068: run_experiment writes the required v3 deliverable JSON."""
    _write_rtl(tmp_path)
    runner, _calls = _fake_runner(_successful_tool_results())
    artifact_path = tmp_path / "results" / ARTIFACT_FILENAME

    artifact = run_experiment(
        repo_root=tmp_path,
        artifact_path=artifact_path,
        home_dir=tmp_path / "home",
        run_command=runner,
        which_func=_which_from(_successful_paths()),
        monotonic=_clock([60.0, 60.75]),
        path_env="/opt/oss-cad-suite/bin:/usr/bin",
    )

    written = json.loads(artifact_path.read_text(encoding="utf-8"))
    missing = [field for field in REQUIRED_FIELDS if field not in written]
    assert not missing
    assert written == artifact
    assert artifact_path.name == ARTIFACT_FILENAME
