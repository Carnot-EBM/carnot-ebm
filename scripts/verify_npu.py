#!/usr/bin/python3
"""End-to-end NPU stack verification for AMD XDNA (Ryzen AI).

**Why this exists:** Exp 511 (milestone 2026.04.38) reported
``npu_not_available`` based on one check -- whether ``onnxruntime`` lists
a ``VitisAIExecutionProvider``. That check cannot distinguish between
"the hardware is absent", "the driver is missing", "the XRT userspace
is missing", "memlock is too low", and "ONNX's VitisAI EP wasn't
compiled in" -- all five produce the same ``npu_not_available``
verdict. On the Strix Point host used during milestone 2026.04.38 the
real state is that the hardware and driver are fine, XRT is installed,
and only the ONNX+VitisAI wrapper is missing; the experiment's
detection conflates that last (relatively minor) missing piece with a
complete absence of NPU capability, which is wrong.

**What this script does:** exercises the stack from the bottom up,
reports each layer's state independently, and emits a structured JSON
artifact so the research record distinguishes real hardware absence
from missing software.

**How this script is used intentionally (not imported):**
    The script is invoked with the *system* Python (not the project
    venv) because ``pyxrt`` is an ABI-tied C extension shipped by the
    Arch ``xrt`` package into the system site-packages. Installing
    ``pyxrt`` into the venv would mean rebuilding it against the venv's
    Python version, which is brittle. See
    ``docs/npu-setup-linux.md`` for the full stack setup procedure.

**Layers checked, in order:**
    1. Kernel: ``/dev/accel/accel0`` exists and the current user can
       ``os.access(path, os.R_OK | os.W_OK)`` it.
    2. Firmware: ``dmesg`` shows the ``amdnpu`` firmware loaded at
       boot; NPU Firmware Version extractable from ``xrt-smi examine``.
    3. Userspace libs: ``/usr/lib/libxrt_driver_xdna.so.2`` present.
    4. Python bindings: ``import pyxrt`` succeeds and
       ``pyxrt.enumerate_devices()`` returns >= 1.
    5. User-session memlock: ``prlimit --memlock`` soft >= 64 MiB.
       (XRT needs to mmap ~64 MiB of BAR region; if ``memlock`` is below
       that, ``pyxrt.device(0)`` fails with EAGAIN.)
    6. Device open: ``pyxrt.device(0)`` returns a live handle.
    7. ONNX providers (optional): whether ``VitisAIExecutionProvider``
       is present.  On Linux it is effectively never present today
       (AMD's Linux ONNX+VitisAI bundle is not publicly released);
       this layer is reported as ``expected_missing`` so it does not
       fail the overall verdict when it is absent.

**Overall honest_verdict values:**
    - ``npu_fully_available``        -- layers 1-6 all pass.
    - ``npu_fully_available_pyxrt_only`` -- 1-6 pass, 7 absent (current
      expected state on Linux; pyxrt path works, ONNX path needs AMD
      binary wheel).
    - ``npu_blocked_memlock``        -- 1-4 pass but 5 fails.  One-line
      fix: install ``/etc/security/limits.d/xrt.conf`` raising memlock
      for the ``render`` group, then re-login.
    - ``npu_userspace_missing``      -- 1-2 pass, 3-4 fail. Install
      ``xrt`` + ``xrt-plugin-amdxdna`` from the Arch repo.
    - ``npu_kernel_missing``         -- 1 fails.  Driver isn't loaded;
      check ``modprobe amdxdna`` and firmware package.
    - ``npu_hardware_absent``        -- no XDNA lspci entry at all.

Run directly:
    /usr/bin/python3 scripts/verify_npu.py

Or via ``--json-only`` for machine parsing:
    /usr/bin/python3 scripts/verify_npu.py --json-only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import resource
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "results" / "npu_stack_verification.json"
XRT_XDNA_LIB = "/usr/lib/libxrt_driver_xdna.so.2"
ACCEL_DEV = "/dev/accel/accel0"
# XRT needs to mmap ~64 MiB of NPU BAR; memlock must be at least that much.
# Round to 128 MiB to leave headroom for additional BO allocations.
MIN_MEMLOCK_BYTES = 128 * 1024 * 1024


def _check_hardware() -> dict:
    """Layer 1 check -- does lspci see an XDNA device?"""
    lspci = shutil.which("lspci")
    if not lspci:
        return {"ok": False, "reason": "lspci missing; cannot enumerate PCI"}
    out = subprocess.run(
        [lspci, "-nn"], capture_output=True, text=True,
    ).stdout
    match = re.search(r"(?im)^(\S+).*(Signal processing|XDNA|Neural Processing Unit).*\[1022:17f0\]", out)
    if not match:
        return {"ok": False, "reason": "No AMD XDNA device found on PCI bus"}
    return {"ok": True, "pci_bdf": match.group(1)}


def _check_kernel_device() -> dict:
    """Layer 1b check -- is /dev/accel/accel0 present and accessible?"""
    if not Path(ACCEL_DEV).exists():
        return {"ok": False, "reason": f"{ACCEL_DEV} missing; amdxdna driver not attached"}
    readable = os.access(ACCEL_DEV, os.R_OK)
    writable = os.access(ACCEL_DEV, os.W_OK)
    if not (readable and writable):
        return {"ok": False, "reason": f"{ACCEL_DEV} not accessible to user; need 'render' group membership"}
    return {"ok": True, "path": ACCEL_DEV}


def _check_firmware_version() -> dict:
    """Layer 2 check -- is the firmware loaded and what version is it?"""
    # dmesg is gated to root on many systems; try it and fall back to xrt-smi.
    dmesg = shutil.which("dmesg")
    if dmesg:
        out = subprocess.run(
            [dmesg], capture_output=True, text=True,
        ).stdout
        fw_line = next((ln for ln in out.splitlines()
                        if "amdxdna" in ln.lower() and "firmware" in ln.lower()), None)
        if fw_line:
            return {"ok": True, "dmesg_firmware_line": fw_line.strip()}
    # Fall back: xrt-smi examine parses the firmware version out of device
    # registers, so it succeeds even without dmesg access.
    xrt_smi = shutil.which("xrt-smi")
    if xrt_smi:
        out = subprocess.run(
            [xrt_smi, "examine"], capture_output=True, text=True,
        ).stdout
        match = re.search(r"NPU Firmware Version\s*:\s*([\d.]+)", out)
        if match:
            return {"ok": True, "npu_firmware_version": match.group(1)}
    return {"ok": False, "reason": "Could not confirm firmware load (dmesg gated + xrt-smi absent/failed)"}


def _check_xrt_userspace() -> dict:
    """Layer 3 check -- is the XRT XDNA driver library installed?"""
    path = Path(XRT_XDNA_LIB)
    if not path.exists():
        return {"ok": False, "reason": f"{XRT_XDNA_LIB} missing; install xrt + xrt-plugin-amdxdna"}
    return {"ok": True, "path": str(path)}


def _check_pyxrt_import() -> dict:
    """Layer 4 check -- is the pyxrt Python binding installed?"""
    try:
        import pyxrt  # type: ignore[import-not-found]
    except ImportError as exc:
        return {"ok": False, "reason": f"pyxrt not importable: {exc}"}
    n_devices = pyxrt.enumerate_devices()
    return {"ok": n_devices >= 1,
            "enumerated_devices": n_devices,
            "pyxrt_module": getattr(pyxrt, "__file__", "unknown")}


def _check_memlock() -> dict:
    """Layer 5 check -- does the current session have enough RLIMIT_MEMLOCK?"""
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    if soft == resource.RLIM_INFINITY:
        return {"ok": True, "soft": "unlimited", "hard": "unlimited"}
    if soft < MIN_MEMLOCK_BYTES:
        return {"ok": False,
                "soft_bytes": soft,
                "hard_bytes": hard,
                "required_bytes": MIN_MEMLOCK_BYTES,
                "reason": ("RLIMIT_MEMLOCK too low; install "
                           "/etc/security/limits.d/xrt.conf raising memlock "
                           "to unlimited for @render, then re-login")}
    return {"ok": True, "soft_bytes": soft, "hard_bytes": hard}


def _check_device_open() -> dict:
    """Layer 6 check -- can pyxrt actually open the device?

    Done inside a try/except because pyxrt raises RuntimeError for
    both hardware-absent and memlock-too-low cases; the caller
    correlates with layer 5 to distinguish the two.
    """
    try:
        import pyxrt  # type: ignore[import-not-found]
    except ImportError:
        return {"ok": False, "reason": "pyxrt not importable"}
    try:
        d = pyxrt.device(0)
    except RuntimeError as exc:
        msg = str(exc)
        hint = ("memlock-related; see layer 5" if "mmap" in msg.lower()
                and "err=-11" in msg.lower()
                else "unknown open failure")
        return {"ok": False, "reason": msg, "hint": hint}
    bdf = None
    name = None
    try:
        import pyxrt  # re-import for xrt_info_device enum
        bdf = d.get_info(pyxrt.xrt_info_device.bdf)
        name = d.get_info(pyxrt.xrt_info_device.name)
    except Exception:
        pass
    return {"ok": True, "bdf": bdf, "name": name}


def _check_onnxruntime_vitisai() -> dict:
    """Layer 7 check -- does onnxruntime list VitisAIExecutionProvider?

    Informational only. On Linux this provider is essentially never
    available in the stock onnxruntime package; AMD ships it as a
    separate patched build. A ``False`` here is ``expected_missing``,
    not a blocker.
    """
    try:
        import onnxruntime as ort
    except ImportError as exc:
        return {"available": False, "reason": f"onnxruntime not installed: {exc}",
                "expected_missing": True}
    providers = list(ort.get_available_providers())
    vitisai = "VitisAIExecutionProvider" in providers
    return {"available": vitisai,
            "providers": providers,
            "expected_missing": not vitisai,
            "onnxruntime_version": getattr(ort, "__version__", "unknown")}


def classify_overall(layers: dict) -> str:
    """Collapse per-layer results to a single honest_verdict string."""
    if not layers["hardware"]["ok"]:
        return "npu_hardware_absent"
    if not layers["kernel_device"]["ok"]:
        return "npu_kernel_missing"
    if not layers["xrt_userspace"]["ok"] or not layers["pyxrt_import"]["ok"]:
        return "npu_userspace_missing"
    if not layers["memlock"]["ok"]:
        return "npu_blocked_memlock"
    if not layers["device_open"]["ok"]:
        return "npu_device_open_failed"
    if layers["onnxruntime_vitisai"]["available"]:
        return "npu_fully_available"
    return "npu_fully_available_pyxrt_only"


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the AMD XDNA NPU stack.")
    parser.add_argument("--json-only", action="store_true",
                        help="Suppress human-readable output; just write the JSON.")
    args = parser.parse_args()

    layers = {
        "hardware": _check_hardware(),
        "kernel_device": _check_kernel_device(),
        "firmware": _check_firmware_version(),
        "xrt_userspace": _check_xrt_userspace(),
        "pyxrt_import": _check_pyxrt_import(),
        "memlock": _check_memlock(),
        "device_open": _check_device_open(),
        "onnxruntime_vitisai": _check_onnxruntime_vitisai(),
    }
    verdict = classify_overall(layers)

    artifact = {
        "schema": "carnot.npu_stack_verification.v1",
        "honest_verdict": verdict,
        "layers": layers,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(artifact, indent=2) + "\n")

    if not args.json_only:
        print(f"NPU stack verification -> {verdict}")
        print(f"Artifact: {OUTPUT_PATH}")
        for name, info in layers.items():
            ok = info.get("ok", info.get("available", None))
            marker = {True: "[OK]", False: "[--]", None: "[??]"}.get(ok, "[??]")
            print(f"  {marker} {name}: {info}")
    # Exit code: 0 on any "available" verdict so CI does not panic if ONNX
    # is expected-missing on Linux; non-zero only on genuine blockers.
    return 0 if verdict.startswith("npu_fully_available") else 1


if __name__ == "__main__":
    sys.exit(main())
