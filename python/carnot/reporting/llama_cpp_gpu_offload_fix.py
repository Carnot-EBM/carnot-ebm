"""Exp 1207 llama.cpp GPU-offload verification artifact.

GRPO v5 (Exp 1208) needs `llama-cpp-python` to push tokens through the GPU at
production speed; without that the experiment falls back to CPU and the
50 tokens/sec floor is unreachable. Two prior attempts (Exp 1179, Exp 1192)
failed for purely environmental reasons -- a 60-minute pip-install timeout and
a broken pre-test suite -- not because the science behind the GRPO update was
wrong. This module exists so that Exp 1207 can produce a small, machine-
readable artifact that says, in concrete numbers, whether the locally
installed llama.cpp build is in good shape:

- did the wheel ship with CUDA support compiled in,
- does ``llama_cpp.llama_supports_gpu_offload()`` return ``True`` at runtime,
- and does a tiny inference smoke test reach the 50 tokens/sec floor.

The CUDA-support and throughput numbers are observed by the operator and fed
into :func:`build_artifact`; the module deliberately does not import
``llama_cpp`` itself so the artifact builder remains testable on hosts
without the binary library or a GPU.

Spec: REQ-REPORT-015, SCENARIO-REPORT-012.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
DELIVERABLE_PATH = REPO_ROOT / "results" / "experiment_1207_llama_cpp_gpu_offload_fix_v3.json"

THROUGHPUT_FLOOR_TOK_PER_SEC = 50.0
ALLOWED_INSTALL_METHODS = frozenset({"pre-built-wheel", "source-cmake-cuda", "already-installed"})


def gpu_offload_verified(cuda_support_compiled: bool, throughput_tokens_per_sec: float) -> bool:
    """Return whether the install meets the headline GPU-offload bar.

    The contract is intentionally narrow: CUDA must have been compiled into
    the wheel *and* a real inference call must have sustained the 50 tok/s
    floor. Either one missing means the GRPO loop will silently fall back to
    CPU and the artifact must say so honestly rather than pretending the
    install is healthy.
    """

    return bool(cuda_support_compiled) and throughput_tokens_per_sec >= THROUGHPUT_FLOOR_TOK_PER_SEC


def honest_verdict(cuda_support_compiled: bool, throughput_tokens_per_sec: float) -> str:
    """Map (compile-flag, throughput) onto the honest_verdict enum.

    There are exactly three outcomes: the install is healthy
    ("gpu_offload_verified"), the wheel has CUDA compiled in but inference
    falls back to CPU ("partial_offload_cpu_fallback"), or the wheel was
    never compiled with CUDA at all ("gpu_offload_failed"). Anything more
    granular would invent distinctions the artifact does not measure.
    """

    if not cuda_support_compiled:
        return "gpu_offload_failed"
    if throughput_tokens_per_sec < THROUGHPUT_FLOOR_TOK_PER_SEC:
        return "partial_offload_cpu_fallback"
    return "gpu_offload_verified"


def build_artifact(
    *,
    llama_cpp_version: str,
    cuda_version_detected: str,
    cuda_support_compiled: bool,
    install_method: str,
    llama_supports_gpu_offload: bool,
    throughput_tokens_per_sec: float,
    notes: str | None = None,
) -> dict[str, object]:
    """Assemble the Exp 1207 artifact dict from observed environment values.

    Each field maps to a concrete observation the operator records before
    calling this function: the version string from ``pip show``, the CUDA
    runtime detected by torch / nvidia-smi, the result of
    ``llama_cpp.llama_supports_gpu_offload()``, and the tokens-per-second
    measurement from a small smoke run. ``notes`` is an optional free-form
    field for environmental gotchas (for example, a missing
    ``LD_LIBRARY_PATH`` entry that the operator had to set manually).
    """

    if install_method not in ALLOWED_INSTALL_METHODS:
        raise ValueError(
            f"install_method must be one of {sorted(ALLOWED_INSTALL_METHODS)}, "
            f"got {install_method!r}"
        )

    verified = gpu_offload_verified(cuda_support_compiled, throughput_tokens_per_sec)
    artifact: dict[str, object] = {
        "experiment": "1207_llama_cpp_gpu_offload_fix_v3",
        "schema": "llama_cpp_gpu_offload_fix_v3",
        "run_date": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "success" if verified else "blocked",
        "llama_cpp_version": llama_cpp_version,
        "cuda_version_detected": cuda_version_detected,
        "cuda_support_compiled": bool(cuda_support_compiled),
        "install_method": install_method,
        "llama_supports_gpu_offload": bool(llama_supports_gpu_offload),
        "throughput_tokens_per_sec": float(throughput_tokens_per_sec),
        "throughput_floor_tokens_per_sec": THROUGHPUT_FLOOR_TOK_PER_SEC,
        "llama_cpp_gpu_offload_verified": verified,
        "honest_verdict": honest_verdict(cuda_support_compiled, throughput_tokens_per_sec),
    }
    if notes is not None:
        artifact["notes"] = notes
    return artifact


def write_artifact(artifact: dict[str, object], out_path: Path = DELIVERABLE_PATH) -> Path:
    """Write the artifact to disk, creating parent directories as needed.

    The deliverable is the only thing downstream consumers (the conductor's
    completion-detector and any human reading ``results/``) read, so writing
    it last is the act of finishing the experiment.
    """

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return out_path


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: build and write the Exp 1207 artifact.

    Returns a process exit code: 0 when the install is verified end-to-end,
    1 when CUDA support is compiled but throughput failed, and 2 when CUDA
    support is missing entirely. The non-zero codes are distinct so a caller
    can tell "rebuild the wheel" apart from "investigate the runtime".
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-cpp-version", required=True)
    parser.add_argument("--cuda-version-detected", required=True)
    parser.add_argument(
        "--cuda-support-compiled",
        action="store_true",
        help="Set when llama_supports_gpu_offload() returned True at runtime.",
    )
    parser.add_argument("--install-method", choices=sorted(ALLOWED_INSTALL_METHODS), required=True)
    parser.add_argument(
        "--llama-supports-gpu-offload",
        action="store_true",
        help="Mirror of llama_supports_gpu_offload(); kept separate for clarity.",
    )
    parser.add_argument("--throughput-tokens-per-sec", type=float, required=True)
    parser.add_argument("--notes", default=None)
    parser.add_argument("--out", type=Path, default=DELIVERABLE_PATH)
    args = parser.parse_args(argv)

    artifact = build_artifact(
        llama_cpp_version=args.llama_cpp_version,
        cuda_version_detected=args.cuda_version_detected,
        cuda_support_compiled=args.cuda_support_compiled,
        install_method=args.install_method,
        llama_supports_gpu_offload=args.llama_supports_gpu_offload,
        throughput_tokens_per_sec=args.throughput_tokens_per_sec,
        notes=args.notes,
    )
    write_artifact(artifact, args.out)
    print(
        f"[exp1207] verdict={artifact['honest_verdict']} "
        f"throughput={artifact['throughput_tokens_per_sec']:.1f} tok/s "
        f"out={args.out}"
    )
    verdict = artifact["honest_verdict"]
    if verdict == "gpu_offload_verified":
        return 0
    if verdict == "partial_offload_cpu_fallback":
        return 1
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
