"""KV260 bitstream generation + board-flash experiment driver (exp2477).

Why this script exists
======================

`exp2465` (milestone 2026.05.238) proved that the open-source yosys flow
can elaborate the full 20-file ``hardware/kv260/`` RTL suite with
``synthesis_errors=0`` once the ``carnot_ising_top`` wrapper and the
``LUT6`` UNISIM stub were committed. yosys alone, however, cannot emit
a Xilinx ``.bit`` file -- the bitstream format is proprietary and
requires Vivado (or a future nextpnr-xilinx port for this part).

This driver therefore covers the remaining steps needed to graduate the
KV260 from `kv260_synthesis_succeeded=True` to the terminal state
defined in CLAUDE.md ``Hardware-Task Continuity Discipline``:

    kv260_bitstream_flashed=True + latency_ns recorded

The script's job is mechanical: it runs Vivado in batch mode against
the ``carnot_ising_top`` wrapper, copies the produced ``.bit`` into the
repository's ``output/`` tree, computes a SHA-256 for auditability, and
then probes the local USB bus for an attached KV260 programmer. If no
board is attached, the script records ``kv260_bitstream_flashed=False``
together with the explicit reason so the CLAUDE.md adversarial-verify
linter cannot misread the artifact as a fabrication.

A small set of pure helper functions (``compute_bitstream_sha256``,
``detect_kv260_programmer``, ``build_artifact``) are exposed so the
unit tests can exercise the bookkeeping logic without needing Vivado
itself.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

# Vendor + product IDs that correspond to a USB-attached KV260 programmer.
# The KV260's onboard JTAG is an FTDI FT4232H accessible at vid:pid
# 0x0403:0x6011 once the board is powered and connected via the front USB-C
# port. The Xilinx Platform Cable USB II appears at 0x03fd:0x0008. Both are
# recognised by openFPGALoader. If neither shows up in `lsusb`, the board
# is physically not connected and a flash attempt cannot succeed.
_KV260_USB_IDS: Tuple[Tuple[str, str], ...] = (
    ("0403", "6011"),
    ("03fd", "0008"),
)


def compute_bitstream_sha256(bit_path: Path) -> str:
    """Return the SHA-256 hex digest of the bitstream at ``bit_path``.

    Used as the ``reproducibility_checksum`` field of the experiment
    artifact; downstream auditors can re-run the Vivado flow and confirm
    the produced ``.bit`` matches the digest recorded here.
    """

    if not bit_path.exists():
        raise FileNotFoundError(f"bitstream not found: {bit_path}")
    h = hashlib.sha256()
    with bit_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_lsusb(text: str) -> Iterable[Tuple[str, str]]:
    """Yield ``(vid, pid)`` tuples extracted from ``lsusb`` short output.

    ``lsusb`` short lines look like ``1209:c0ca (bus 3, device 6) path: 2.3``
    or the older ``Bus 003 Device 006: ID 1209:c0ca DirtyJTAG``. Both
    forms include ``vid:pid`` somewhere, so the parser just searches for
    the first four-hex-digit pair separated by a colon on each line.
    """

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        for token in line.split():
            if (
                len(token) == 9
                and token[4] == ":"
                and all(c in "0123456789abcdefABCDEF" for c in token[:4] + token[5:])
            ):
                yield token[:4].lower(), token[5:].lower()
                break


def detect_kv260_programmer(lsusb_text: str) -> Tuple[bool, str]:
    """Return ``(detected, reason)`` for KV260 USB programmer presence.

    ``detected`` is True when the ``lsusb`` output contains one of the
    known programmer VID/PID pairs. ``reason`` is a human-readable note
    explaining the decision -- the artifact stores it verbatim so the
    flash failure mode is unambiguous post-hoc.
    """

    seen = list(_parse_lsusb(lsusb_text))
    matched = [pair for pair in seen if pair in _KV260_USB_IDS]
    if matched:
        vid, pid = matched[0]
        return True, f"matched KV260 programmer vid:pid={vid}:{pid}"
    return (
        False,
        "no KV260 programmer found on local USB bus "
        f"(scanned {len(seen)} devices; required one of "
        + ", ".join(f"{v}:{p}" for v, p in _KV260_USB_IDS)
        + ")",
    )


def run_vivado(
    tcl_path: Path,
    log_path: Path,
    vivado_bin: str = "/tools/Xilinx/2025.2.1/Vivado/bin/vivado",
    timeout_s: float = 900.0,
) -> Tuple[int, float]:
    """Invoke Vivado in batch mode against ``tcl_path``.

    Returns ``(exit_code, wall_time_s)``. The function does NOT raise on
    a non-zero exit; the caller decides whether to treat that as fatal.
    """

    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = _dt.datetime.utcnow()
    proc = subprocess.run(
        [
            vivado_bin,
            "-mode",
            "batch",
            "-source",
            str(tcl_path),
            "-log",
            str(log_path),
            "-notrace",
        ],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    dur = (_dt.datetime.utcnow() - start).total_seconds()
    return proc.returncode, dur


def build_artifact(
    *,
    experiment_id: str,
    milestone: str,
    honest_verdict: str,
    kv260_bitstream_flashed: bool,
    bitstream_path: Optional[str],
    bitstream_tool_used: str,
    vivado_available: bool,
    nextpnr_xilinx_available: bool,
    rtl_file_count: int,
    latency_ns: Optional[int],
    duration_s: float,
    preconditions_checked: Iterable[Dict[str, Any]],
    extras: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the experiment_2477 JSON artifact.

    Centralising the schema here means the tests can exercise the same
    builder the runtime uses, and any future planner regenerating the
    deliverable cannot drop a required field by accident.
    """

    artifact: Dict[str, Any] = {
        "experiment_id": experiment_id,
        "experiment": "exp2477-kv260-bitstream-flash",
        "milestone": milestone,
        "run_date": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "schema": "results/v1",
        "honest_verdict": honest_verdict,
        "kv260_bitstream_flashed": bool(kv260_bitstream_flashed),
        "bitstream_path": bitstream_path,
        "bitstream_tool_used": bitstream_tool_used,
        "vivado_available": bool(vivado_available),
        "nextpnr_xilinx_available": bool(nextpnr_xilinx_available),
        "rtl_file_count": int(rtl_file_count),
        "latency_ns": latency_ns,
        "duration_s": float(duration_s),
        "preconditions_checked": list(preconditions_checked),
    }
    if extras:
        for k, v in extras.items():
            artifact.setdefault(k, v)
    return artifact


REQUIRED_ARTIFACT_FIELDS: Tuple[str, ...] = (
    "experiment_id",
    "experiment",
    "milestone",
    "run_date",
    "schema",
    "honest_verdict",
    "kv260_bitstream_flashed",
    "bitstream_path",
    "bitstream_tool_used",
    "vivado_available",
    "nextpnr_xilinx_available",
    "rtl_file_count",
    "latency_ns",
    "duration_s",
    "preconditions_checked",
)


TERMINAL_VERDICT_PREFIXES: Tuple[str, ...] = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)


def validate_artifact(artifact: Dict[str, Any]) -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for a candidate artifact.

    Checks that all REQUIRED_ARTIFACT_FIELDS are present and that
    ``honest_verdict`` starts with one of the terminal prefixes mandated
    by the CLAUDE.md ``Verdict Terminal-Prefix Discipline``.
    """

    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            return False, f"missing required field: {field}"
    verdict = artifact.get("honest_verdict", "")
    if not isinstance(verdict, str):
        return False, "honest_verdict must be a string"
    if not any(verdict.startswith(p) for p in TERMINAL_VERDICT_PREFIXES):
        return (
            False,
            f"honest_verdict {verdict!r} does not start with a terminal prefix "
            f"(one of {TERMINAL_VERDICT_PREFIXES})",
        )
    return True, "ok"


def write_artifact(artifact: Dict[str, Any], path: Path) -> None:
    """Persist ``artifact`` to ``path`` as pretty-printed JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=False) + "\n")


__all__ = [
    "compute_bitstream_sha256",
    "detect_kv260_programmer",
    "run_vivado",
    "build_artifact",
    "validate_artifact",
    "write_artifact",
    "REQUIRED_ARTIFACT_FIELDS",
    "TERMINAL_VERDICT_PREFIXES",
]


if __name__ == "__main__":  # pragma: no cover - exercised manually
    # The actual Vivado invocation lives in /tmp/exp2477/kv260_synth.tcl;
    # this entrypoint is intentionally a thin shell so re-runs from the
    # repo can recompute the SHA and validate the deliverable.
    repo = Path(__file__).resolve().parents[1]
    bit = repo / "output" / "exp2477_kv260_bitstream" / "carnot_kv260.bit"
    print("bitstream_present=", bit.exists())
    if bit.exists():
        print("sha256=", compute_bitstream_sha256(bit))
    out = subprocess.run(["lsusb"], capture_output=True, text=True, check=False)
    detected, reason = detect_kv260_programmer(out.stdout)
    print("kv260_programmer_detected=", detected, "reason=", reason)
