#!/usr/bin/env python3
"""Experiment 827 — KV260 nextpnr-xilinx Attempt and iCE40 Bitstream Fallback.

**Research question:**
    Exp 816 produced synthesis_clean_n32 on iCE40 HX8K using OSS-CAD-Suite yosys
    (3952 LUTs for N=32).  The physical KV260 board uses Zynq UltraScale+ (xczu5eg),
    not iCE40.  OSS-CAD-Suite includes nextpnr-xilinx (still experimental), which can
    target some Xilinx parts.  This experiment:

    1. Attempts nextpnr-xilinx targeting xczu5eg — records whether the binary exists
       and whether synthesis completes cleanly.
    2. Falls back to generating a concrete iCE40 HX8K bitstream (.bin) via nextpnr-ice40
       + icepack so we have a hardware-programmable artifact even if Zynq synthesis fails.
    3. Validates the bitstream by checking the iCE40 magic header (0xFF 0x00).

**Why this matters:**
    The KV260 board arrives soon and we need to know which tool path leads to a
    programmable artifact.  If nextpnr-xilinx is not ready for xczu5eg, the iCE40
    bitstream proves our RTL is fully synthesizable to physical hardware and gives
    us something concrete to measure timing/power on an iCE40 eval board.

**Honest verdict mapping:**
    xilinx_synthesis_clean:     nextpnr-xilinx ran successfully on xczu5eg.
    ice40_bitstream_generated:  icepack produced a .bin with valid iCE40 magic header.
    bitstream_invalid_header:   icepack ran but produced a file with wrong magic bytes.
    synthesis_blocked:          neither path produced any output artifact.

Spec: REQ-HW-039, SCENARIO-HW-035
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from experiment_template import ExperimentTemplate  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

DELIVERABLE = "results/experiment_827_kv260_nextpnr_xilinx_v3.json"
EXP816_ARTIFACT = "results/experiment_816_kv260_synthesis_v2.json"
RTL_SOURCE = "hardware/kv260/ising_sampler_v3.v"
BITSTREAM_DEST = "hardware/kv260/ising_n32.bin"

# OSS-CAD-Suite installs to ~/tools/oss-cad-suite/bin by convention (Exp 807/816).
OSS_CAD_BIN = Path.home() / "tools" / "oss-cad-suite" / "bin"

# Exp 816 must have produced synthesis_clean_n32 with this LUT count.
EXPECTED_LUT_COUNT_N32 = 3952


def _patch_n_spins(rtl_text: str, n_spins: int) -> str:
    """Return RTL text with the N parameter default overridden to n_spins.

    The RTL defaults to N=64.  nextpnr-ice40 needs N=32 to fit within the
    iCE40 HX8K's 7680 LUT budget (Exp 816 confirmed 3952 LUTs at N=32).
    We patch the source text rather than using yosys chparam — same approach
    as Exp 816 which validated this works across yosys versions.

    Spec: REQ-HW-039-3
    """
    import re

    return re.sub(
        r"(parameter\s+integer\s+N\s*=\s*)\d+",
        rf"\g<1>{n_spins}",
        rtl_text,
    )


def _run(cmd: list[str], timeout: int = 300) -> tuple[int, str, str]:
    """Run a subprocess and return (returncode, stdout, stderr).

    FileNotFoundError means the binary does not exist — returned as rc=-1 so
    callers can treat it as 'tool unavailable' without a separate exception path.
    This mirrors the _run() pattern established across Exps 791, 804, 816.
    """
    try:
        result = subprocess.run(
            [str(c) for c in cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"Timed out after {timeout}s: {cmd}"
    except FileNotFoundError:
        return -1, "", f"Executable not found: {cmd[0]}"


def check_xilinx_available(oss_cad_bin: Path) -> bool:
    """Return True if nextpnr-xilinx is present and responds to --help.

    nextpnr-xilinx is still experimental in OSS-CAD-Suite and was not included
    in builds prior to late 2025.  We probe --help because --version is not
    always supported; --help exits 0 on known working builds.

    Spec: REQ-HW-039-1
    """
    binary = oss_cad_bin / "nextpnr-xilinx"
    if not binary.exists():
        return False
    rc, _, _ = _run([binary, "--help"], timeout=15)
    return rc == 0


def run_xilinx_synthesis(
    oss_cad_bin: Path,
    rtl_source: Path,
    tmp_dir: Path,
) -> tuple[bool, str]:
    """Attempt nextpnr-xilinx synthesis targeting xczu5eg.

    We first run yosys to produce a JSON netlist (the format nextpnr-xilinx
    expects), then pass it to nextpnr-xilinx.  If either step fails we record
    xilinx_synthesis_clean=False.

    Returns (synthesis_clean, combined_log).

    Spec: REQ-HW-039-2
    """
    json_netlist = tmp_dir / "ising_xilinx_netlist.json"
    asc_out = tmp_dir / "ising_xilinx.asc"

    # Step A: yosys synthesis to JSON netlist (Xilinx requires JSON, not iCE40 primitives).
    yosys_bin = oss_cad_bin / "yosys"
    ys_script = tmp_dir / "synth_xilinx.ys"
    ys_script.write_text(
        f"read_verilog {rtl_source}\n"
        f"synth -top ising_sampler_v3 -flatten -json {json_netlist}\n"
        "stat\n"
    )
    rc_yosys, out_yosys, err_yosys = _run([yosys_bin, str(ys_script)], timeout=180)
    combined_yosys = out_yosys + "\n" + err_yosys

    if rc_yosys != 0 or "ERROR" in combined_yosys.upper():
        return False, f"[yosys step failed]\n{combined_yosys}"

    # Step B: nextpnr-xilinx — attempt routing for xczu5eg.
    # WHY no --pcf: xczu5eg chip-db may be absent; we want to detect that failure
    # separately from a legitimate place-and-route failure.
    xilinx_bin = oss_cad_bin / "nextpnr-xilinx"
    cmd = [
        xilinx_bin,
        "--chipdb",
        "xczu5eg",
        "--json",
        str(json_netlist),
        "--asc",
        str(asc_out),
    ]
    rc_pnr, out_pnr, err_pnr = _run(cmd, timeout=300)
    combined_pnr = out_pnr + "\n" + err_pnr

    synthesis_clean = (rc_pnr == 0) and ("ERROR" not in combined_pnr.upper())
    return synthesis_clean, f"[yosys]\n{combined_yosys}\n[nextpnr-xilinx]\n{combined_pnr}"


def run_ice40_bitstream(
    oss_cad_bin: Path,
    rtl_source: Path,
    tmp_dir: Path,
) -> tuple[bool, bool, int, str]:
    """Generate an iCE40 HX8K bitstream from ising_sampler_v3.v using nextpnr-ice40 + icepack.

    Steps:
      1. yosys synth_ice40 → JSON netlist (same approach as Exp 816, but we keep the file).
      2. nextpnr-ice40 --hx8k --json <netlist> --asc <asc_out> — place and route.
      3. icepack <asc_out> <bin_out> — pack ASC into binary bitstream.
      4. Validate first 4 bytes: iCE40 magic is 0xFF 0x00 (and conventionally 0x00 0xFF).

    Returns (bitstream_generated, valid_header, bitstream_size_bytes, log).

    WHY re-run yosys here: Exp 816's yosys run deleted the netlist JSON on completion
    to keep temp files clean.  We must regenerate it.

    Spec: REQ-HW-039-3, REQ-HW-039-4
    """
    json_netlist = tmp_dir / "ising_ice40_netlist.json"
    asc_out = tmp_dir / "ising_n32.asc"
    bin_out = tmp_dir / "ising_n32.bin"

    # Patch RTL to N=32 — same technique as Exp 816.  HX8K has 7680 LUTs;
    # N=64 (the RTL default) uses ~10000+ LUTs and cannot be placed.
    rtl_patched = tmp_dir / "ising_sampler_v3_n32.v"
    rtl_patched.write_text(_patch_n_spins(rtl_source.read_text(), 32))

    yosys_bin = oss_cad_bin / "yosys"
    ys_script = tmp_dir / "synth_ice40.ys"
    ys_script.write_text(
        f"read_verilog {rtl_patched}\n"
        f"synth_ice40 -top ising_sampler_v3 -json {json_netlist}\n"
        "stat\n"
    )
    rc_yosys, out_yosys, err_yosys = _run([yosys_bin, str(ys_script)], timeout=180)
    combined_yosys = out_yosys + "\n" + err_yosys
    if rc_yosys != 0 or "ERROR" in combined_yosys.upper():
        return False, False, 0, f"[yosys synth_ice40 failed]\n{combined_yosys}"

    # nextpnr-ice40: place and route to produce ASC file.
    nextpnr_bin = oss_cad_bin / "nextpnr-ice40"
    rc_pnr, out_pnr, err_pnr = _run(
        [nextpnr_bin, "--hx8k", "--json", str(json_netlist), "--asc", str(asc_out)],
        timeout=300,
    )
    combined_pnr = out_pnr + "\n" + err_pnr
    if rc_pnr != 0:
        return (
            False,
            False,
            0,
            (f"[yosys]\n{combined_yosys}\n[nextpnr-ice40 failed]\n{combined_pnr}"),
        )

    # icepack: convert ASC → binary bitstream.
    icepack_bin = oss_cad_bin / "icepack"
    rc_pack, out_pack, err_pack = _run(
        [icepack_bin, str(asc_out), str(bin_out)],
        timeout=60,
    )
    combined_pack = out_pack + "\n" + err_pack
    full_log = (
        f"[yosys]\n{combined_yosys}\n[nextpnr-ice40]\n{combined_pnr}\n[icepack]\n{combined_pack}"
    )

    if not bin_out.exists():
        return False, False, 0, full_log

    raw = bin_out.read_bytes()
    # iCE40 bitstream magic: first byte 0xFF, second byte 0x00.
    # (Full 4-byte sync word is 0xFF 0x00 0x00 0xFF but we check the leading two
    # as the reliable discriminator — icepack sometimes varies bytes 2-3 by variant.)
    valid_header = len(raw) >= 2 and raw[0] == 0xFF and raw[1] == 0x00
    return True, valid_header, len(raw), full_log


def validate_exp816_gate(repo_root: Path) -> tuple[bool, int | None]:
    """Check that Exp 816 produced synthesis_clean_n32 with correct LUT count.

    Returns (gate_passed, lut_count_n32).
    A missing or malformed artifact is treated as gate failure (conservative).

    Spec: REQ-HW-039
    """
    import json

    artifact_path = repo_root / EXP816_ARTIFACT
    if not artifact_path.exists():
        return False, None
    try:
        data = json.loads(artifact_path.read_text())
    except Exception:
        return False, None
    verdict = data.get("honest_verdict", "")
    lut = data.get("lut_count_n32")
    ok = verdict == "synthesis_clean_n32" and lut == EXPECTED_LUT_COUNT_N32
    return ok, lut


def run_experiment(tmpl: ExperimentTemplate) -> tuple[dict, str]:
    """Run the nextpnr-xilinx attempt and iCE40 bitstream fallback experiment.

    Steps:
      1. Gate: confirm Exp 816 produced synthesis_clean_n32 with lut_count_n32=3952.
      2. Confirm OSS-CAD-Suite is installed.
      3. Probe nextpnr-xilinx; attempt Xilinx synthesis if available.
      4. Fallback: generate iCE40 bitstream with nextpnr-ice40 + icepack.
      5. Validate bitstream magic header; copy to hardware/kv260/.
      6. Determine honest_verdict.

    Spec: REQ-HW-039, SCENARIO-HW-035
    """
    repo_root = Path(tmpl._repo_root)

    gate_ok, lut_count_n32 = validate_exp816_gate(repo_root)
    if not gate_ok:
        return {
            "gate_exp816_passed": False,
            "gate_exp816_lut_count_n32": lut_count_n32,
            "honest_verdict": "synthesis_blocked",
            "block_reason": (
                f"Exp 816 gate failed: expected honest_verdict='synthesis_clean_n32' "
                f"and lut_count_n32={EXPECTED_LUT_COUNT_N32}, got lut_count_n32={lut_count_n32}"
            ),
            "xilinx_available": False,
            "xilinx_synthesis_clean": False,
            "ice40_bitstream_generated": False,
            "valid_header": False,
            "bitstream_size_bytes": 0,
            "bitstream_path": None,
            "oss_cad_bin": str(OSS_CAD_BIN),
        }, "blocked"

    # Verify OSS-CAD-Suite at expected path (nextpnr-ice40 and icepack are mandatory).
    if not (OSS_CAD_BIN / "nextpnr-ice40").exists():
        return {
            "gate_exp816_passed": True,
            "gate_exp816_lut_count_n32": lut_count_n32,
            "honest_verdict": "synthesis_blocked",
            "block_reason": f"OSS-CAD-Suite nextpnr-ice40 not found at {OSS_CAD_BIN}",
            "xilinx_available": False,
            "xilinx_synthesis_clean": False,
            "ice40_bitstream_generated": False,
            "valid_header": False,
            "bitstream_size_bytes": 0,
            "bitstream_path": None,
            "oss_cad_bin": str(OSS_CAD_BIN),
        }, "blocked"

    rtl_source = repo_root / RTL_SOURCE
    if not rtl_source.exists():
        return {
            "gate_exp816_passed": True,
            "gate_exp816_lut_count_n32": lut_count_n32,
            "honest_verdict": "synthesis_blocked",
            "block_reason": f"RTL source not found: {rtl_source}",
            "xilinx_available": False,
            "xilinx_synthesis_clean": False,
            "ice40_bitstream_generated": False,
            "valid_header": False,
            "bitstream_size_bytes": 0,
            "bitstream_path": None,
            "oss_cad_bin": str(OSS_CAD_BIN),
        }, "blocked"

    # Primary path: nextpnr-xilinx.
    xilinx_available = check_xilinx_available(OSS_CAD_BIN)
    xilinx_synthesis_clean = False
    xilinx_log = ""

    with tempfile.TemporaryDirectory(prefix="exp827_") as tmp_str:
        tmp_dir = Path(tmp_str)

        if xilinx_available:
            xilinx_synthesis_clean, xilinx_log = run_xilinx_synthesis(
                OSS_CAD_BIN, rtl_source, tmp_dir
            )

        # Fallback (or complement): generate iCE40 bitstream regardless of Xilinx outcome,
        # so we always have a hardware-programmable artifact when nextpnr-xilinx fails.
        (
            ice40_bitstream_generated,
            valid_header,
            bitstream_size_bytes,
            ice40_log,
        ) = run_ice40_bitstream(OSS_CAD_BIN, rtl_source, tmp_dir)

        # Copy bitstream to repo if valid.
        bitstream_path: str | None = None
        if ice40_bitstream_generated and valid_header:
            bin_src = tmp_dir / "ising_n32.bin"
            dest = repo_root / BITSTREAM_DEST
            shutil.copy2(str(bin_src), str(dest))
            bitstream_path = str(dest)

    # Determine honest verdict.
    if xilinx_available and xilinx_synthesis_clean:
        honest_verdict = "xilinx_synthesis_clean"
    elif ice40_bitstream_generated and valid_header:
        honest_verdict = "ice40_bitstream_generated"
    elif ice40_bitstream_generated and not valid_header:
        honest_verdict = "bitstream_invalid_header"
    else:
        honest_verdict = "synthesis_blocked"

    return {
        "gate_exp816_passed": True,
        "gate_exp816_lut_count_n32": lut_count_n32,
        "xilinx_available": xilinx_available,
        "xilinx_synthesis_clean": xilinx_synthesis_clean,
        "xilinx_log": xilinx_log[:4000] if xilinx_log else "",
        "ice40_bitstream_generated": ice40_bitstream_generated,
        "valid_header": valid_header,
        "bitstream_size_bytes": bitstream_size_bytes,
        "bitstream_path": bitstream_path,
        "ice40_log": ice40_log[:8000] if ice40_log else "",
        "oss_cad_bin": str(OSS_CAD_BIN),
        "honest_verdict": honest_verdict,
    }, "success"


def main() -> None:
    """Entry point: set up experiment lifecycle and run.

    Spec: REQ-HW-039, SCENARIO-HW-035
    """
    apply_env_autofix()

    tmpl = ExperimentTemplate(
        exp_id=827,
        title="KV260 nextpnr-xilinx Attempt and iCE40 Bitstream Fallback",
        deliverable=DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()

    with ExperimentTimeoutWatchdog(827, timeout_minutes=45):
        result_fields, status = run_experiment(tmpl)

    artifact = tmpl.build_result(result_fields, status=status)
    out_path = tmpl._repo_root / DELIVERABLE
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))
    print(
        f"[Exp 827] honest_verdict={artifact.get('honest_verdict')}  "
        f"ice40_bitstream={artifact.get('ice40_bitstream_generated')}  "
        f"valid_header={artifact.get('valid_header')}"
    )
    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
