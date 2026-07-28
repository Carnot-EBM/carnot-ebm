#!/usr/bin/env python3
"""F13: persist the n_ctx 81920 per-PID VRAM residency readings that back
`_VRAM_MTP_HEAD_*` (head surcharge 1290 MiB at 81920) and
`_VRAM_GEMMA31B_11LAYER_81920_CHECK_MIB = 21730`.

WHY THIS SCRIPT EXISTS. Those two constants were read once off a live
`nvidia-smi` and never written down. The 32768 pair IS corroborated by
t1/shipped_mtp_{on,off}.json; the 81920 pair was not corroborated by
anything, and one of them (21730) is advertised in the source as "a third,
differently-shaped configuration" that independently validates the base
envelope -- i.e. it is load-bearing evidence, resting on an unrecorded
reading. This script re-takes both arms and PERSISTS them.

CARD IDENTITY IS PROVEN FROM PER-PID RESIDENCY, NEVER FROM THE ENV VAR.
`CUDA_VISIBLE_DEVICES=0` renames devices inside the child process; it does
not tell you which physical card the allocation landed on. We join
`nvidia-smi --query-compute-apps=pid,used_gpu_memory,gpu_uuid` against
`--query-gpu=index,uuid` so the recorded number is "this PID holds N MiB on
the card with THIS uuid, which is index I" -- a fact about the driver's own
accounting rather than about our environment.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BINDIR = Path.home() / ".cache" / "shipped_mtp_binary_probe" / "extracted"
MODEL = (
    Path.home()
    / ".cache/huggingface/hub/models--unsloth--gemma-4-31B-it-GGUF/snapshots"
    / "f130ba51393346288f5862e30e9586b9b021513f/gemma-4-31B-it-Q4_K_M.gguf"
)
DRAFT = Path.home() / ".cache/kaggle_mtp_head_upload/mtp-gemma-4-31B-it-Q8_0.gguf"
N_CTX = 81920
FFN_CPU_LAYERS = 11
GPU_INDEX = 0


def ffn_override_regex(n: int) -> str:
    """Same construction as `_ffn_cpu_override_regex` in the live module."""
    if n <= 0:
        return ""
    idx = "|".join(str(i) for i in range(int(n)))
    return rf"blk\.({idx})\.ffn_(gate|up|down)\.weight=CPU"


def gpu_uuid_to_index() -> dict[str, int]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    m: dict[str, int] = {}
    for line in out.splitlines():
        if not line.strip():
            continue
        idx, uuid = (p.strip() for p in line.split(",", 1))
        m[uuid] = int(idx)
    return m


def pid_residency(pid: int) -> dict | None:
    """MiB this PID holds, and WHICH PHYSICAL CARD, from the driver's own books."""
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_gpu_memory,gpu_uuid",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    uuid_idx = gpu_uuid_to_index()
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if int(parts[0]) != pid:
            continue
        uuid = parts[2]
        return {
            "pid": pid,
            "used_mib": int(parts[1]),
            "gpu_uuid": uuid,
            "gpu_index": uuid_idx.get(uuid, -1),
        }
    return None


def run_arm(label: str, mtp: bool, port: int) -> dict:
    log_path = HERE / f"{label}.log"
    args = [
        str(BINDIR / "llama-server"),
        "--model",
        str(MODEL),
        "-ngl",
        "999",
        "--ctx-size",
        str(N_CTX),
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--no-warmup",
    ]
    rx = ffn_override_regex(FFN_CPU_LAYERS)
    if rx:
        args += ["-ot", rx]
    if mtp:
        args += ["--spec-type", "draft-mtp", "--model-draft", str(DRAFT)]

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(GPU_INDEX)
    env["LD_LIBRARY_PATH"] = str(BINDIR)

    with open(log_path, "w") as fh:
        proc = subprocess.Popen(args, stdout=fh, stderr=subprocess.STDOUT, env=env)

    resident = None
    marker_seen = None
    try:
        deadline = time.time() + 600
        while time.time() < deadline:
            time.sleep(5)
            if proc.poll() is not None:
                break
            text = log_path.read_text(errors="replace")
            if "server is listening" in text or "main: server is listening" in text:
                # settle, then take the residency reading
                time.sleep(8)
                resident = pid_residency(proc.pid)
                break
        text = log_path.read_text(errors="replace")
        marker_seen = "common_speculative_impl_draft_mtp: adding speculative implementation" in text
    finally:
        # Tear down by explicit PID. Never pkill a pattern -- it would match this
        # very process's own command line.
        try:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=45)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=20)
            except Exception:
                pass
        time.sleep(6)  # let the driver reclaim before the next arm

    return {
        "label": label,
        "mtp": mtp,
        "n_ctx": N_CTX,
        "ffn_cpu_layers": FFN_CPU_LAYERS,
        "argv": args,
        "residency": resident,
        "mtp_positive_marker_seen": marker_seen,
        "log": str(log_path),
    }


def main() -> int:
    HERE.mkdir(parents=True, exist_ok=True)
    for p in (MODEL, DRAFT, BINDIR / "llama-server"):
        if not p.exists():
            print(f"blocked_missing_input: {p}", file=sys.stderr)
            return 2
    results = [
        run_arm("f13_81920_11layer_mtp_off", False, 8981),
        run_arm("f13_81920_11layer_mtp_on", True, 8982),
    ]
    off = results[0]["residency"]
    on = results[1]["residency"]
    summary = {
        "purpose": "persist the n_ctx 81920 / 11-CPU-FFN-layer per-PID VRAM residency readings",
        "binary_sha256": subprocess.run(
            ["sha256sum", str(BINDIR / "llama-server")],
            capture_output=True,
            text=True,
        ).stdout.split()[0],
        "arms": results,
        "measured_mtp_off_mib": off["used_mib"] if off else None,
        "measured_mtp_on_mib": on["used_mib"] if on else None,
        "measured_head_surcharge_mib": (on["used_mib"] - off["used_mib"]) if (on and off) else None,
        "source_constant_mtp_off_mib": 21730,
        "source_constant_head_surcharge_mib": 1290,
    }
    (HERE / "f13_81920_residency.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
