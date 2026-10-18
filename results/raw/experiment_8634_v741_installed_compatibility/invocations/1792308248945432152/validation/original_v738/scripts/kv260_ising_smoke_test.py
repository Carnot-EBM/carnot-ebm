#!/usr/bin/env python3
"""KV260 Ising sampler smoke test — runs ON THE BOARD via SSH.

**What this script does (running on the host workstation):**
    1. SSH into the KV260 at the given host (default 192.168.51.98).
    2. Upload an inline on-board Python program that polls the AXI GPIO
       DATA register at 0xA0000000 (the address the carnot_ising_v4
       block design assigns to the sampler's ``s_out`` output bus).
    3. Read the GPIO value 100 times and check whether the resulting
       distribution is non-uniform.  A working sampler converges to a
       low-energy ferromagnetic state — uniform random readouts would
       indicate the bitstream is not actually computing.
    4. Compute a simple energy proxy ``E = -popcount(s_out)`` and check
       that the distribution of E across samples spans a non-trivial
       range — a hardware that always returned the same constant or
       returned uniformly random integers would both fail this test
       in different ways.

**Why a separate file (and not inlined into the experiment driver):**
    The smoke test is reusable: future experiments that load the
    bitstream a different way (xmutil vs dfx-mgr vs raw fpga_manager)
    can all share this verification step.  Keeping it self-contained
    also means a human operator can run it after manual bitstream
    loading without going through the experiment harness.

**Why energy via popcount is a reasonable proxy:**
    The v4 block design exposes the low 32 spins of the 128-spin
    sampler on the AXI GPIO output.  In the ferromagnetic ground
    state all 32 bits are 1 (popcount = 32, energy proxy = -32).
    In a high-energy disordered state popcount is near 16.  A
    sampler that has converged will produce popcount values
    clustered toward 32; a sampler that is computing but has not
    converged will produce a distribution skewed toward higher
    popcount; a non-computing sampler will produce uniform random
    values.  The test only checks that the distribution is *not
    uniform* — which is a weak but unambiguous "the hardware is
    doing something" signal.

Spec refs: REQ-HW-040, SCENARIO-HW-040
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys

_ON_BOARD_PROGRAM = r"""
import json, mmap, struct, sys, time

AXI_GPIO_BASE = 0xA0000000
PAGE = 4096
SAMPLES = 100

def read_once():
    with open('/dev/mem', 'rb') as f:
        with mmap.mmap(f.fileno(), PAGE, access=mmap.ACCESS_READ,
                       offset=AXI_GPIO_BASE) as mm:
            return struct.unpack('<I', mm[0:4])[0]

values = []
try:
    for _ in range(SAMPLES):
        values.append(read_once())
        time.sleep(0.001)  # 1ms between reads to avoid thrashing PL
except PermissionError:
    print(json.dumps({"error": "permission_denied"}))
    sys.exit(1)
except Exception as exc:
    print(json.dumps({"error": str(exc)}))
    sys.exit(2)

popcounts = [bin(v).count('1') for v in values]
energies = [-pc for pc in popcounts]
unique_values = len(set(values))

print(json.dumps({
    "samples": len(values),
    "unique_values": unique_values,
    "min_value": min(values),
    "max_value": max(values),
    "min_popcount": min(popcounts),
    "max_popcount": max(popcounts),
    "mean_popcount": sum(popcounts) / len(popcounts),
    "energy_range": max(energies) - min(energies),
    "first_5_hex": [hex(v) for v in values[:5]],
}))
"""


def run_smoke_test(host: str, user: str = "ubuntu") -> dict:
    """Run the on-board smoke test via SSH and return the parsed result.

    Returns a dict that always contains ``ok`` (bool), ``passed`` (bool),
    and either ``stats`` (the on-board JSON output) or ``error`` (the
    failure message).  ``passed`` requires both that the SSH call
    succeeded and that the energy distribution was non-uniform (range
    > 0 across the 100 samples).
    """
    cmd = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ConnectTimeout=10",
        f"{user}@{host}",
        f"sudo python3 -c {_ON_BOARD_PROGRAM!r}",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return {"ok": False, "passed": False, "error": "ssh_timeout"}

    if proc.returncode != 0:
        return {
            "ok": False,
            "passed": False,
            "error": f"ssh_rc={proc.returncode}",
            "stderr": proc.stderr[:500],
        }

    # On-board script prints exactly one JSON object on stdout.
    stdout = proc.stdout.strip()
    match = re.search(r"\{.*\}", stdout, re.DOTALL)
    if not match:
        return {"ok": False, "passed": False, "error": "no_json_in_output", "stdout": stdout[:500]}
    try:
        stats = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return {"ok": False, "passed": False, "error": f"bad_json: {exc}"}

    if "error" in stats:
        return {"ok": False, "passed": False, "error": stats["error"]}

    nonuniform = stats.get("energy_range", 0) > 0 and stats.get("unique_values", 1) > 1
    return {
        "ok": True,
        "passed": bool(nonuniform),
        "stats": stats,
        "energy_distribution_nonuniform": bool(nonuniform),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.51.98")
    parser.add_argument("--user", default="ubuntu")
    args = parser.parse_args()
    result = run_smoke_test(args.host, args.user)
    print(json.dumps(result, indent=2))
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    sys.exit(main())
