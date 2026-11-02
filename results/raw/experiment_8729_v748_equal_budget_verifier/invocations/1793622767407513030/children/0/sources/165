"""The hardware and context an ARC run actually used, read from the driver and the process.

Spec: REQ-ARC-WMTE-7022.

Kept in its own module, with no carnot imports, for two reasons. It is producer-agnostic --
any run that wants to record its envelope can import it, not only the leaderboard eval. And it
stays testable: importing `scripts/arc_leaderboard_eval.py` pulls in the whole carnot stack,
about half a gigabyte, which the suite's per-test memory watchdog correctly refuses.
"""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from typing import Any

_ENVELOPE_CACHE: dict[str, Any] | None = None


def run_envelope() -> dict[str, Any]:
    """The hardware and context this run actually used, read from the driver and the process.

    WHY THIS EXISTS (REQ-ARC-WMTE-7022, 2026-09-05). Thirteen of thirteen eval-run artifacts
    recorded no GPU, no VRAM, no `n_ctx`, no model path and no layer split. The `n_ctx=98304`
    conclusion, the 33-38 tok/s decode figure and the engine comparison all rested on runs whose
    artifacts could not say what produced them; the numbers lived in session logs and in
    recollection, neither of which is the record. A third party cannot check a hardware claim
    against an artifact that omits the hardware, and G2 asks exactly that.

    Read lazily and cached: called on the first payload write, by which time the model is
    resident, so the GPU rows show the real footprint rather than an empty card. Every field
    records absence explicitly rather than being omitted, because a missing key and a measured
    "none" are different facts.
    """

    global _ENVELOPE_CACHE
    if _ENVELOPE_CACHE is not None:
        return _ENVELOPE_CACHE

    def _sh(*args: str) -> str:
        try:
            return subprocess.run(
                args, capture_output=True, text=True, timeout=20, check=False
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return ""

    gpu_inventory: dict[str, dict[str, Any]] = {}
    for row in _sh(
        "nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"
    ).splitlines():
        parts = [c.strip() for c in row.split(",")]
        if len(parts) >= 3 and parts[0].isdigit():
            gpu_inventory[parts[1]] = {"index": int(parts[0]), "gpu_model": ",".join(parts[2:])}

    # Attribute GPU memory to THIS process tree only. A concurrent conductor experiment on the
    # other card is not this run's footprint, and recording it would overstate the envelope.
    mine = {os.getpid()}
    for _ in range(4):  # bounded: pick up llama-server children as they appear
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                ppid = int((entry / "stat").read_text().split()[3])
            except (OSError, IndexError, ValueError):
                continue
            if ppid in mine:
                mine.add(int(entry.name))

    gpus: list[dict[str, Any]] = []
    for row in _sh(
        "nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader"
    ).splitlines():
        parts = [c.strip() for c in row.split(",")]
        if len(parts) >= 3 and parts[0].isdigit() and int(parts[0]) in mine:
            gpus.append(
                {
                    "index": gpu_inventory.get(parts[1], {}).get("index"),
                    "gpu_uuid": parts[1],
                    "gpu_model": gpu_inventory.get(parts[1], {}).get("gpu_model"),
                    "pid": int(parts[0]),
                    "used_memory": parts[2],
                }
            )

    server_path = os.environ.get("CARNOT_LLAMA_SERVER")
    model_path = None
    for pid in sorted(mine):
        try:
            argv = (Path("/proc") / str(pid) / "cmdline").read_bytes().decode().split("\0")
        except (OSError, UnicodeError):
            continue
        if argv and argv[0].endswith("llama-server") and "-m" in argv:
            server_path = argv[0]
            model_path = argv[argv.index("-m") + 1]
            break

    _ENVELOPE_CACHE = {
        "n_ctx_requested": os.environ.get("CARNOT_ARC_INDUCE_N_CTX"),
        "generator_cuda_gpu_requested": os.environ.get("CARNOT_ARC_GENERATOR_CUDA_GPU"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "llama_server_path": server_path,
        "model_path": model_path,
        # The list the DRIVER reports for this process tree. An env var says what was asked
        # for; this says what happened. They disagreed for a whole run on 2026-09-04, when
        # CARNOT_ARC_GENERATOR_CUDA_GPU=1 still split a 27B model across both cards.
        "gpus_held": gpus,
        "gpu_indices_held": sorted({g["index"] for g in gpus if g["index"] is not None}),
        "hostname": platform.node(),
    }
    return _ENVELOPE_CACHE
