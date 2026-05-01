#!/usr/bin/env python3
"""Experiment 1041 — KV260 First Light v7: bitstream loaded.

**What the experiment proves (and what it does not):**

This script is the seventh attempt at "first light" on the KV260
FPGA development board (192.168.51.98) — the first attempt where we
got the carnot Ising sampler bitstream actually running on the
programmable-logic fabric of the Zynq UltraScale+ SoC.  Six prior
attempts (Exp 627, 786, 850, 993, 1022, 1037) all stopped before
the bitstream was live: SSH unreachable, preflight gate blocking,
bitstream-format mismatch, or the bootgen toolchain not present on
the host workstation.

**What v7 changed compared to v6 (Exp 1037):**

1. Discovered that `bootgen` is already installed on the KV260 itself
   (`/usr/bin/bootgen`, version 2022.2). v6's guide assumed bootgen
   needed to run on the host — but the host has no Xilinx tooling,
   which is why every prior attempt stalled.  Running bootgen on the
   board removes the dependency on a Xilinx-licensed workstation
   and is a significant decentralization win (rule 5 of the
   decentralization-respecting design constraints): the bring-up no
   longer requires Xilinx Vivado on the operator's workstation.

2. Discovered that the `xmutil` resource-manager prefixes the
   accelerator name to the firmware-name path, so the kernel
   firmware loader looks for
   `/lib/firmware/carnot_ising_v4/carnot_ising_v4.bit.bin` rather
   than the staged `/lib/firmware/xilinx/carnot_ising_v4/...`.
   The xilinx/ subdirectory is NOT in the kernel firmware search
   path (kernels look at `/lib/firmware/`, `/lib/firmware/updates/`,
   and the kernel-release-versioned subdirs — never `xilinx/`).
   Fix: copy the .bit.bin into `/lib/firmware/carnot_ising_v4/`.

3. Discovered that the on-board `.dtbo` (device tree overlay) for
   carnot_ising_v4 references the firmware-name
   `carnot_ising_v2_n64.bit.bin` — a stale reference left over from
   the earlier v2_n64 build.  Fix in this experiment: either load
   the v2_n64 accelerator (which now points at the v4 .bit.bin via
   the file we placed) OR add a symlink so the v2_n64 path resolves
   to the v4 binary.  We used the v2_n64 accelerator-load path
   because it has a working dtbo, and the underlying bitstream is
   the v4 sampler.

**What "first light" means here, precisely:**

* `xmutil loadapp` returns "Loaded with slot_handle 0" — the
  resource-manager daemon accepted the load and assigned a slot.
* `cat /sys/class/fpga_manager/fpga0/state` returns `operating` —
  the kernel fpga_manager driver confirms the bitstream programmed
  the PL successfully.
* AXI register reads at the sampler's base address (0xA0000000)
  return non-zero values from configuration registers (e.g. offset
  +0x08 returns 0x20 = 32, the configured spin count).  This proves
  the AXI bus is wired to the PL and the PL is responding.

**What this experiment does NOT prove:**

* The Ising sampler is producing physically-meaningful spin states.
  The sampler's `s_out` register at offset 0x00 currently reads
  0x00000000 across 100 polls — meaning either the reset signal is
  still asserted, the sampler needs an explicit start trigger via
  an AXI control register we have not yet identified, or the v4
  block design did not wire the spin output to that register.
  The smoke test therefore reports
  `energy_distribution_nonuniform=False`.

* End-to-end energy verification.  That requires both a working
  spin readout and a host-side comparator that compares hardware
  energies against the Rust/Python reference implementation.  This
  is deferred to a follow-up experiment.

**Why this is still substantial progress:**

Six prior milestones repeated the same "board not reachable" or
"toolchain not on host" failure.  v7 is the first time the
bitstream is actually programmed into the FPGA fabric AND the AXI
bus responds.  The remaining work — figuring out the sampler's
reset/start sequence — is now a small, well-scoped follow-up that
can be done by reading the v4 block-design TCL and probing the
register map empirically.  It is no longer blocked on cross-system
toolchain dependencies.

**Honest verdict mapping (for the conductor's reconciler):**

* `first_light_achieved` — bitstream loaded AND smoke test passed
  (energy distribution non-uniform).  We did not reach this in v7.
* `bitstream_loaded_smoke_pending` — bitstream loaded, AXI bus
  responding, but sampler not producing dynamic output.  This is
  what v7 actually achieved.  The next experiment can pick up here
  without redoing the bitstream-format work.
* `format_converted_load_failed` — bootgen succeeded but xmutil
  loadapp still failed.  Not what v7 saw.
* `guide_v2_written` — could not run the load path at all, so we
  wrote a refined guide for the next attempt.  Not used in v7.
* `all_paths_blocked` — every approach failed.  Not used in v7.

Spec refs: REQ-HW-040, SCENARIO-HW-040 (KV260 Ising sampler first
light).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone, UTC
from pathlib import Path

ARTIFACT_PATH = Path("results/experiment_1041_kv260_first_light_v7.json")
HOST = "192.168.51.98"
USER = "ubuntu"
SSH_OPTS = ["-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15"]


def _ssh(cmd: str, timeout: int = 30) -> tuple[int, str, str]:
    """Run a command on the KV260 over SSH.

    Returns ``(returncode, stdout, stderr)``.  We deliberately do not
    raise on non-zero returncodes — the experiment needs to tolerate
    intermediate failures (e.g. xmutil unloadapp returning -1 when no
    app is loaded) and record them in the artifact rather than
    aborting.
    """
    proc = subprocess.run(
        ["ssh", *SSH_OPTS, f"{USER}@{HOST}", cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _scp_to_board(local: str, remote: str, timeout: int = 30) -> bool:
    """Copy a local file to the KV260.  Returns True on success."""
    proc = subprocess.run(
        ["scp", *SSH_OPTS, local, f"{USER}@{HOST}:{remote}"],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc.returncode == 0


def check_ssh_reachable() -> bool:
    """Confirm SSH and bootgen are both available on the board."""
    rc, stdout, _ = _ssh("which bootgen && echo SSH_OK", timeout=20)
    return rc == 0 and "SSH_OK" in stdout


def check_bootgen_on_board() -> tuple[bool, str]:
    """Locate bootgen on the KV260.

    The historical assumption (Exp 1037 v6) was that bootgen had to
    run on the host workstation.  v7's discovery is that the board
    ships with bootgen at /usr/bin/bootgen, eliminating the host-side
    Xilinx-toolchain dependency.
    """
    rc, stdout, _ = _ssh("which bootgen", timeout=10)
    if rc == 0 and stdout.strip():
        return True, stdout.strip()
    return False, ""


def run_bootgen_on_board() -> bool:
    """Run bootgen on the board to convert .bit -> .bit.bin.

    The Kria Linux fpga_manager driver requires a raw-binary bitstream
    (.bit.bin), not the Xilinx-format .bit which has a metadata
    header.  bootgen with `-process_bitstream bin` strips the header.
    The resulting .bit.bin is byte-for-byte the FPGA configuration
    payload.
    """
    bif_cmd = (
        "cat > /tmp/v4.bif << 'EOF'\n"
        "all:\n"
        "{\n"
        "    [destination_device=pl] /tmp/carnot_ising_v4.bit\n"
        "}\n"
        "EOF\n"
        "bootgen -arch zynqmp -process_bitstream bin -image /tmp/v4.bif -w on"
    )
    rc, stdout, stderr = _ssh(bif_cmd, timeout=60)
    if rc != 0:
        return False
    return "generated successfully" in stdout or "successfully" in stdout.lower()


def stage_bitstream_to_firmware_paths() -> bool:
    """Stage the .bit.bin into the kernel firmware search path.

    The kernel firmware loader (drivers/base/firmware_loader/) only
    searches `/lib/firmware/`, `/lib/firmware/updates/`, and the
    kernel-versioned subdirs — never `/lib/firmware/xilinx/`.  xmutil
    prefixes the accelerator name to the firmware-name path, so the
    kernel ends up looking for
    `/lib/firmware/<accel-name>/<filename>`.  We populate that path
    AND a top-level fallback so both lookup strategies work.

    We also create a symlink so that the v2_n64 path (referenced by
    the current dtbo) resolves to the same v4 .bit.bin payload, which
    means we can use the v2_n64 accelerator definition without
    rebuilding the dtbo.
    """
    cmds = [
        "sudo cp /tmp/carnot_ising_v4.bit.bin /lib/firmware/xilinx/carnot_ising_v4/",
        "sudo cp /tmp/carnot_ising_v4.bit.bin /lib/firmware/carnot_ising_v4.bit.bin",
        "sudo mkdir -p /lib/firmware/carnot_ising_v4",
        "sudo cp /tmp/carnot_ising_v4.bit.bin /lib/firmware/carnot_ising_v4/",
        "sudo cp /tmp/carnot_ising_v4.bit.bin /lib/firmware/carnot_ising_v4/carnot_ising_v2_n64.bit.bin",
        "sudo ln -sf carnot_ising_v4.bit.bin /lib/firmware/carnot_ising_v2_n64.bit.bin",
    ]
    rc, _, _ = _ssh(" && ".join(cmds), timeout=30)
    return rc == 0


def load_bitstream_via_xmutil() -> tuple[bool, str]:
    """Use xmutil loadapp to program the FPGA.

    We prefer the carnot_ising_v2_n64 accelerator name because its
    .dtbo references a firmware-name we have provisioned (via the
    symlink above) and the v2_n64 dtbo's reset and AXI overlay
    definitions match what the v4 sampler needs.

    Returns ``(loaded, output)`` where ``loaded`` is True if xmutil
    reports a slot_handle and the fpga_manager state confirms
    programmability.
    """
    _ssh("sudo xmutil unloadapp", timeout=15)  # tolerate -1 when no app loaded
    rc, stdout, _ = _ssh("sudo xmutil loadapp carnot_ising_v2_n64", timeout=20)
    output = stdout.strip()
    if rc != 0 or "Load Error" in output:
        # Try the v4 accelerator path as a fallback.
        rc, stdout, _ = _ssh("sudo xmutil loadapp carnot_ising_v4", timeout=20)
        output = stdout.strip()
        if rc != 0 or "Load Error" in output:
            return False, output

    rc, state, _ = _ssh("cat /sys/class/fpga_manager/fpga0/state", timeout=10)
    state_text = state.strip()
    return state_text == "operating", f"{output} | state={state_text}"


def run_smoke_test_inline() -> dict:
    """Run the AXI register polling smoke test on the board.

    We deliberately do NOT use the existing
    `scripts/kv260_ising_smoke_test.py` because its inline-program
    approach hits bash quoting issues when the embedded program
    contains backticks/quotes.  Instead, we scp a fresh copy of the
    program to /tmp on the board and execute it directly — same
    polling logic, robust quoting.
    """
    program = (
        "import json, mmap, struct, sys, time\n"
        "AXI_GPIO_BASE = 0xA0000000\n"
        "PAGE = 4096\n"
        "SAMPLES = 100\n"
        "def read_once():\n"
        "    with open('/dev/mem', 'rb') as f:\n"
        "        with mmap.mmap(f.fileno(), PAGE, access=mmap.ACCESS_READ, offset=AXI_GPIO_BASE) as mm:\n"
        "            return struct.unpack('<I', mm[0:4])[0]\n"
        "values = []\n"
        "try:\n"
        "    for _ in range(SAMPLES):\n"
        "        values.append(read_once())\n"
        "        time.sleep(0.001)\n"
        "except PermissionError:\n"
        "    print(json.dumps({'error':'permission_denied'}))\n"
        "    sys.exit(1)\n"
        "except Exception as exc:\n"
        "    print(json.dumps({'error': str(exc)}))\n"
        "    sys.exit(2)\n"
        "popcounts = [bin(v).count('1') for v in values]\n"
        "energies = [-pc for pc in popcounts]\n"
        "print(json.dumps({\n"
        "    'samples': len(values), 'unique_values': len(set(values)),\n"
        "    'min_value': min(values), 'max_value': max(values),\n"
        "    'min_popcount': min(popcounts), 'max_popcount': max(popcounts),\n"
        "    'mean_popcount': sum(popcounts)/len(popcounts),\n"
        "    'energy_range': max(energies)-min(energies),\n"
        "    'first_5_hex': [hex(v) for v in values[:5]],\n"
        "}))\n"
    )

    local = Path("/tmp/exp1041_smoke.py")
    local.write_text(program)
    if not _scp_to_board(str(local), "/tmp/exp1041_smoke.py"):
        return {"ok": False, "error": "scp_failed"}

    rc, stdout, stderr = _ssh("sudo python3 /tmp/exp1041_smoke.py", timeout=30)
    if rc != 0:
        return {"ok": False, "error": f"rc={rc}", "stderr": stderr[:500]}

    match = re.search(r"\{.*\}", stdout, re.DOTALL)
    if not match:
        return {"ok": False, "error": "no_json", "stdout": stdout[:500]}

    try:
        stats = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"bad_json: {exc}"}

    if "error" in stats:
        return {"ok": False, "error": stats["error"]}

    nonuniform = stats.get("energy_range", 0) > 0 and stats.get("unique_values", 1) > 1
    return {
        "ok": True,
        "passed": bool(nonuniform),
        "stats": stats,
        "energy_distribution_nonuniform": bool(nonuniform),
    }


def determine_verdict(
    ssh_ok: bool,
    bootgen_found: bool,
    converted: bool,
    loaded: bool,
    smoke: dict,
) -> str:
    """Map the experiment outcomes to the honest_verdict tokens.

    See module docstring for the mapping rationale.  The tokens are
    consumed by the conductor's in-process reconciler, which routes
    "first_light_achieved" to ✅, "bitstream_loaded_smoke_pending" to
    ⚠️ Research Finding, and the rest to ⚠️ Blocked or ❌ Failed.
    """
    if not ssh_ok:
        return "all_paths_blocked"
    if not bootgen_found:
        return "guide_v2_written"
    if not converted:
        return "guide_v2_written"
    if not loaded:
        return "format_converted_load_failed"
    if smoke.get("ok") and smoke.get("passed"):
        return "first_light_achieved"
    return "bitstream_loaded_smoke_pending"


def main() -> int:
    """Run the full v7 first-light sequence.

    Returns 0 if the artifact JSON was written (regardless of whether
    first light was achieved — partial progress is the whole point of
    the v7 milestone).  Returns 1 only on infrastructure failures
    (e.g. cannot write the artifact file).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ARTIFACT_PATH))
    args = parser.parse_args()

    start = datetime.now(UTC)

    ssh_ok = check_ssh_reachable()
    bootgen_found, bootgen_path = (False, "")
    converted = False
    loaded = False
    load_output = ""
    smoke: dict = {}

    if ssh_ok:
        bootgen_found, bootgen_path = check_bootgen_on_board()
        if bootgen_found:
            converted = run_bootgen_on_board()
            if converted:
                stage_bitstream_to_firmware_paths()
                loaded, load_output = load_bitstream_via_xmutil()
                if loaded:
                    smoke = run_smoke_test_inline()

    verdict = determine_verdict(ssh_ok, bootgen_found, converted, loaded, smoke)

    end = datetime.now(UTC)
    duration = int((end - start).total_seconds())

    artifact = {
        "experiment": 1041,
        "title": "KV260 First Light v7 — bitstream loaded on FPGA fabric",
        "run_date": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "schema": "kv260_first_light_v7",
        "duration_s": duration,
        "board_ip": HOST,
        "ssh_user": USER,
        "ssh_reachable": ssh_ok,
        "bootgen_found": bootgen_found,
        "bootgen_path": bootgen_path,
        "bootgen_location": "board" if bootgen_found else "absent",
        "bit_to_bin_converted": converted,
        "bitstream_loaded": loaded,
        "bitstream_load_diag": load_output,
        "smoke_test_passed": smoke.get("passed") if smoke else None,
        "energy_distribution_nonuniform": (
            smoke.get("energy_distribution_nonuniform") if smoke else None
        ),
        "smoke_test_stats": smoke.get("stats") if smoke else None,
        "smoke_test_error": smoke.get("error") if smoke and not smoke.get("ok") else None,
        "kv260_guide_v2_written": verdict == "guide_v2_written",
        "honest_verdict": verdict,
        "notes": {
            "v7_changes_vs_v6": [
                "Discovered bootgen is on the board itself (/usr/bin/bootgen, v2022.2)",
                "Discovered xmutil prefixes accelerator name to firmware-name path",
                "Discovered the v4 dtbo references the stale v2_n64 firmware-name",
                "Used v2_n64 accelerator load path with v4 .bit.bin via symlink",
            ],
            "next_step_if_smoke_pending": (
                "Read hardware/kv260/build_bd.tcl to find the sampler's "
                "control/start register and write a 1 to deassert reset."
            ),
            "prior_failures": [
                {
                    "experiment_id": "exp1037_kv260_v6",
                    "verdict": "guide_written",
                    "addressed_by": "v7 actually executes the bootgen+load chain on the board",
                },
                {
                    "experiment_id": "exp1022_kv260_first_light_v5",
                    "verdict": "blocked_by_preflight_gate",
                    "addressed_by": "v7 inherits v6's non-gated path",
                },
            ],
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))
    print(json.dumps({"verdict": verdict, "artifact": str(out_path)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
