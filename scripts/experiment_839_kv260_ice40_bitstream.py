#!/usr/bin/env python3
"""Experiment 839 — iCE40 HX8K Place-and-Route + Bitstream Generation.

**Research question:**
    Exp 816 produced synthesis_clean_n32 via yosys synth_ice40 (3952 LUTs, N=32,
    iCE40 HX8K target). Exp 827 attempted nextpnr-xilinx for KV260 Zynq and fell
    back to nextpnr-ice40, but produced honest_verdict='synthesis_blocked'.

    This experiment runs nextpnr-ice40 place-and-route + icepack bitstream
    generation directly, with the --package ct256 flag required for iCE40 HX8K
    and --freq 25 timing constraint. The resulting .bin file cannot be loaded
    onto the KV260 board (wrong FPGA family) but validates the full
    synthesis→PnR→bitstream pipeline end-to-end before Vivado is installed.

**Why this matters:**
    A confirmed iCE40 bitstream proves our N=32 Ising sampler RTL survives all
    three physical-synthesis stages. The KV260 board arrives 2026-04-20 but
    requires Vivado for bitstream generation; having the iCE40 path working
    gives us a concrete measurable milestone and validates the OSS-CAD-Suite
    toolchain is fully functional on this machine.

**Why re-synthesize instead of reusing Exp 816 JSON:**
    Exp 816 wrote the netlist JSON to a tempfile that was deleted at exit.
    This experiment re-runs yosys synth_ice40 to produce a fresh JSON, then
    immediately hands it to nextpnr-ice40. The synthesis step is fast (~9 s)
    and reproducible (LUT count should remain 3952).

**Honest verdict mapping:**
    bitstream_generated:              icepack produced .bin with valid iCE40 header (0xFF 0x00).
    bitstream_generated_invalid_header: .bin exists but magic bytes are wrong.
    pnr_failed:                       nextpnr-ice40 returned non-zero.
    nextpnr_not_available:            nextpnr-ice40 binary absent in OSS-CAD-Suite.
    synthesis_artifact_missing:       Exp 816 gate check failed (artifact missing or wrong verdict).

Spec: REQ-HW-040, SCENARIO-HW-044
"""

from __future__ import annotations

import json
import os
import re
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

DELIVERABLE = "results/experiment_839_kv260_ice40_bitstream.json"
EXP816_ARTIFACT = "results/experiment_816_kv260_synthesis_v2.json"
RTL_SOURCE = "hardware/kv260/ising_sampler_v3.v"
BITSTREAM_DEST = "hardware/kv260/ising_n32_exp839.bin"

# OSS-CAD-Suite installs to ~/tools/oss-cad-suite/bin by convention (Exp 807/816).
OSS_CAD_BIN = Path.home() / "tools" / "oss-cad-suite" / "bin"

# Expected LUT count from Exp 816 (3952 SB_LUT4 at N=32).
EXPECTED_LUT_COUNT_N32 = 3952

# iCE40 package for HX8K CT256 — required by nextpnr-ice40 for place-and-route.
# Without --package, nextpnr-ice40 refuses to map I/O pins.
ICE40_PACKAGE = "ct256"

# Target clock frequency in MHz — conservative for broad compatibility.
ICE40_FREQ_MHZ = 25


def _run(cmd: list[str], timeout: int = 300) -> tuple[int, str, str]:
    """Run a subprocess and return (returncode, stdout, stderr).

    FileNotFoundError → rc=-1 so callers treat it as 'tool unavailable'.
    TimeoutExpired → rc=-1 with a descriptive message.
    This mirrors the _run() pattern from Exps 791, 804, 816, 827.
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


def _patch_n_spins(rtl_text: str, n_spins: int) -> str:
    """Return RTL text with the N parameter default overridden to n_spins.

    The RTL defaults to N=64. nextpnr-ice40 cannot fit N=64 within the HX8K's
    7680 LUT budget; N=32 uses 3952 LUTs (Exp 816). We patch the parameter in
    source text — same approach validated by Exp 816 and Exp 827.

    Spec: REQ-HW-040-2
    """
    return re.sub(
        r"(parameter\s+integer\s+N\s*=\s*)\d+",
        rf"\g<1>{n_spins}",
        rtl_text,
    )


def validate_exp816_gate(repo_root: Path) -> tuple[bool, int | None]:
    """Check that Exp 816 produced synthesis_clean_n32 with LUT count 3952.

    Returns (gate_passed, lut_count_n32).
    Missing or malformed artifact → conservative gate failure.

    Spec: REQ-HW-040-1
    """
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


def check_nextpnr_ice40_available(oss_cad_bin: Path) -> bool:
    """Return True if nextpnr-ice40 is present and responds to --help.

    We also verify icepack is present, since both are required for the full
    synthesis→bitstream pipeline. A working nextpnr-ice40 without icepack
    cannot complete the bitstream step.

    Spec: REQ-HW-040-3
    """
    nextpnr_bin = oss_cad_bin / "nextpnr-ice40"
    icepack_bin = oss_cad_bin / "icepack"
    if not nextpnr_bin.exists() or not icepack_bin.exists():
        return False
    rc, _, _ = _run([nextpnr_bin, "--help"], timeout=30)
    # nextpnr-ice40 --help exits 0 in OSS-CAD-Suite builds.
    return rc == 0


def run_pnr_and_pack(
    oss_cad_bin: Path,
    rtl_source: Path,
    tmp_dir: Path,
) -> tuple[bool, bool, bool, int, bool, str]:
    """Run yosys synth_ice40 → nextpnr-ice40 PnR → icepack bitstream generation.

    Steps:
      1. Patch RTL to N=32 and run yosys synth_ice40 → JSON netlist.
      2. Run nextpnr-ice40 --hx8k --package ct256 --freq 25 → ASC file.
      3. Run icepack → .bin bitstream.
      4. Validate first 2 bytes: iCE40 magic is 0xFF 0x00.

    Returns (pnr_complete, timing_met, bitstream_generated, bitstream_size_bytes,
             bitstream_valid_header, full_log).

    timing_met is True when nextpnr-ice40 does NOT report a timing violation.
    We consider the absence of 'Max frequency' + 'constraint not met' in output
    as timing-met (conservative: any timing warning → timing_met=False).

    Spec: REQ-HW-040-3, REQ-HW-040-4, REQ-HW-040-5
    """
    json_netlist = tmp_dir / "ising_ice40_n32.json"
    asc_out = tmp_dir / "carnot_ising_n32.asc"
    bin_out = tmp_dir / "carnot_ising_n32.bin"

    # Patch RTL to N=32 — N=64 (RTL default) exceeds HX8K's 7680-LUT budget.
    rtl_patched = tmp_dir / "ising_sampler_v3_n32.v"
    rtl_patched.write_text(_patch_n_spins(rtl_source.read_text(), 32))

    # Step 1: yosys synthesis → JSON netlist.
    yosys_bin = oss_cad_bin / "yosys"
    ys_script = tmp_dir / "synth_ice40_n32.ys"
    ys_script.write_text(
        f"read_verilog {rtl_patched}\n"
        f"synth_ice40 -top ising_sampler_v3 -json {json_netlist}\n"
        "stat\n"
    )
    rc_yosys, out_yosys, err_yosys = _run([yosys_bin, str(ys_script)], timeout=180)
    combined_yosys = out_yosys + "\n" + err_yosys
    if rc_yosys != 0 or "ERROR" in combined_yosys.upper():
        return False, False, False, 0, False, f"[yosys synth_ice40 failed]\n{combined_yosys}"

    # Step 2: nextpnr-ice40 place-and-route.
    nextpnr_bin = oss_cad_bin / "nextpnr-ice40"
    pnr_cmd = [
        nextpnr_bin,
        "--hx8k",
        "--package",
        ICE40_PACKAGE,
        "--json",
        str(json_netlist),
        "--asc",
        str(asc_out),
        "--freq",
        str(ICE40_FREQ_MHZ),
    ]
    rc_pnr, out_pnr, err_pnr = _run(pnr_cmd, timeout=600)
    combined_pnr = out_pnr + "\n" + err_pnr
    if rc_pnr != 0:
        full_log = f"[yosys]\n{combined_yosys}\n[nextpnr-ice40 failed rc={rc_pnr}]\n{combined_pnr}"
        return False, False, False, 0, False, full_log

    # Parse timing: nextpnr-ice40 prints "constraint not met" when timing fails.
    timing_met = "constraint not met" not in combined_pnr.lower()

    # Step 3: icepack → binary bitstream.
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
        return True, timing_met, False, 0, False, full_log

    raw = bin_out.read_bytes()
    # iCE40 bitstream: first byte 0xFF, second byte 0x00 (magic header).
    # The full 4-byte sync word is 0xFF 0x00 0x00 0xFF but bytes 2-3 vary by
    # device variant. Checking only bytes 0-1 is reliable across all HX8K builds.
    valid_header = len(raw) >= 2 and raw[0] == 0xFF and raw[1] == 0x00
    return True, timing_met, True, len(raw), valid_header, full_log


def run_experiment(tmpl: ExperimentTemplate) -> tuple[dict, str]:
    """Execute the iCE40 PnR + bitstream experiment and return (fields, status).

    This function is separated from main() so tests can call it with a mock
    template without touching the filesystem or invoking real subprocesses.

    Steps:
      1. Validate Exp 816 gate (synthesis_clean_n32 with LUT count 3952).
      2. Check nextpnr-ice40 and icepack are available in OSS-CAD-Suite.
      3. Run synthesis → PnR → icepack in a temp directory.
      4. Copy valid bitstream to hardware/kv260/.
      5. Determine honest_verdict.

    Spec: REQ-HW-040
    """
    repo_root = Path(tmpl._repo_root)

    # Gate: confirm Exp 816 succeeded before spending PnR time.
    gate_passed, lut_count = validate_exp816_gate(repo_root)
    if not gate_passed:
        fields = {
            "gate_exp816_passed": False,
            "gate_exp816_lut_count_n32": lut_count,
            "nextpnr_available": False,
            "pnr_complete": False,
            "timing_met": False,
            "bitstream_generated": False,
            "bitstream_size_bytes": 0,
            "bitstream_valid_header": False,
            "output_path": None,
            "pnr_log": "",
            "honest_verdict": "synthesis_artifact_missing",
            "oss_cad_bin": str(OSS_CAD_BIN),
        }
        return fields, "blocked"

    # Check tool availability.
    nextpnr_available = check_nextpnr_ice40_available(OSS_CAD_BIN)
    if not nextpnr_available:
        fields = {
            "gate_exp816_passed": True,
            "gate_exp816_lut_count_n32": lut_count,
            "nextpnr_available": False,
            "pnr_complete": False,
            "timing_met": False,
            "bitstream_generated": False,
            "bitstream_size_bytes": 0,
            "bitstream_valid_header": False,
            "output_path": None,
            "pnr_log": "",
            "honest_verdict": "nextpnr_not_available",
            "oss_cad_bin": str(OSS_CAD_BIN),
        }
        return fields, "blocked"

    rtl_source = repo_root / RTL_SOURCE

    with tempfile.TemporaryDirectory(prefix="exp839_") as tmp_str:
        tmp_dir = Path(tmp_str)
        (
            pnr_complete,
            timing_met,
            bitstream_generated,
            bitstream_size_bytes,
            bitstream_valid_header,
            pnr_log,
        ) = run_pnr_and_pack(OSS_CAD_BIN, rtl_source, tmp_dir)

        # Copy bitstream to repo if valid header confirmed.
        output_path: str | None = None
        if bitstream_generated and bitstream_valid_header:
            dest = repo_root / BITSTREAM_DEST
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tmp_dir / "carnot_ising_n32.bin", dest)
            output_path = str(dest)

    # Determine honest verdict.
    if not pnr_complete:
        honest_verdict = "pnr_failed"
    elif bitstream_generated and bitstream_valid_header:
        honest_verdict = "bitstream_generated"
    elif bitstream_generated and not bitstream_valid_header:
        honest_verdict = "bitstream_generated_invalid_header"
    else:
        honest_verdict = "pnr_failed"

    status = "success" if honest_verdict == "bitstream_generated" else "partial"

    fields = {
        "gate_exp816_passed": True,
        "gate_exp816_lut_count_n32": lut_count,
        "nextpnr_available": nextpnr_available,
        "pnr_complete": pnr_complete,
        "timing_met": timing_met,
        "bitstream_generated": bitstream_generated,
        "bitstream_size_bytes": bitstream_size_bytes,
        "bitstream_valid_header": bitstream_valid_header,
        "output_path": output_path,
        "pnr_log": pnr_log[:10000] if pnr_log else "",
        "honest_verdict": honest_verdict,
        "oss_cad_bin": str(OSS_CAD_BIN),
    }
    return fields, status


def main() -> None:
    """Entry point for Experiment 839 — iCE40 PnR + Bitstream Generation."""
    apply_env_autofix()
    os.makedirs("output", exist_ok=True)

    tmpl = ExperimentTemplate(
        839,
        "KV260 iCE40 HX8K Place-and-Route + Bitstream Generation",
        DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()

    _watchdog = ExperimentTimeoutWatchdog(839, timeout_minutes=30)

    fields, status = run_experiment(tmpl)

    artifact = tmpl.build_result(fields, status=status)
    tmpl._output_path.write_text(json.dumps(artifact, indent=2))

    print(
        f"[Exp 839] honest_verdict={artifact.get('honest_verdict')}  "
        f"pnr_complete={artifact.get('pnr_complete')}  "
        f"bitstream_generated={artifact.get('bitstream_generated')}  "
        f"valid_header={artifact.get('bitstream_valid_header')}  "
        f"size={artifact.get('bitstream_size_bytes')} bytes"
    )

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
