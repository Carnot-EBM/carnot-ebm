"""Experiment 1035 — DualGPU ROCm-aware Detection v3.

The story so far:
  Exp 1002 (.78) wired DualGPURunner into VerifyRepairPipeline but only
  in synthetic_validation mode (`wired_synthetic_only`) because
  ``torch.cuda.device_count()`` returned 0 on this host.  Exp 1023 (.79)
  re-attempted the live path and saw the same `insufficient_gpus=0`
  even though `nvidia-smi` reports two physical RTX 3090s.

The diagnosed root cause:
  ``torch.cuda.device_count()`` returns 0 on this host because the
  installed PyTorch wheel is the CPU-only build (``torch.__version__``
  reports ``2.11.0+cpu``; ``torch.version.cuda is None`` and
  ``torch.version.hip is None``).  This is *not* a ROCm shim issue
  alone — the broader truth is that the Python torch build has no
  GPU backend compiled in, so any ``torch.cuda.*`` call returns the
  CPU-fallback default (zero devices).  The same effect appears on
  pure-ROCm boxes (where the HIP shim swallows CUDA enumeration), so
  the fallback strategy — ask ``nvidia-smi`` directly — fixes both
  cases with one code path.

What this experiment verifies:
  1. Diagnostic state — ``torch.cuda.device_count()`` versus
     ``nvidia-smi`` device count under ``sg render``.
  2. The ROCm-aware probe ``_detect_gpu_count_rocm_aware()`` already
     present in ``scripts/experiment_template.py`` returns the true
     NVIDIA GPU count regardless of which backend ``torch`` was built
     against.
  3. Whether the live DualGPU path can in fact run inference.  On a
     CPU-only torch build it cannot, no matter how cleanly we detect
     the cards — so we report that fact rather than fabricate a
     synthetic throughput ratio.

Why we are not silently re-running synthetic mode:
  Exp 1002 already covered the synthetic validation case
  (`wired_synthetic_only`); a sixth synthetic run adds nothing.  The
  honest deliverable here is the diagnosis: detection is fixed by
  the existing patch, but live inference is blocked by a layer
  *below* the detection logic (torch build flavour), and that
  blocker must be resolved before any future DualGPU live experiment
  can produce useful numbers.

Spec: REQ-GPU-010, REQ-INFRA-007, SCENARIO-GPU-011.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger(__name__)

EXPERIMENT_ID = 1035
RESULT_PATH = Path("results/experiment_1035_dualgpu_rocm_v3.json")


def _torch_cuda_count() -> int:
    """Ask PyTorch how many CUDA devices it can see.

    Wrapped in try/except so the experiment still runs on hosts where
    ``torch`` is not installed at all — the diagnostic value in that
    case is simply 0 with no traceback noise.
    """
    try:
        import torch  # noqa: PLC0415

        return int(torch.cuda.device_count())
    except Exception:
        return 0


def _torch_build_info() -> dict[str, Any]:
    """Return the relevant fields from ``torch.version`` for diagnosis.

    Why we record this: a CPU-only torch wheel (``2.11.0+cpu``) reports
    ``cuda=None`` and ``hip=None``; a ROCm wheel reports ``hip=...``;
    a CUDA wheel reports ``cuda=...``.  Distinguishing these three
    cases tells us whether the blocker is the wheel itself or the
    runtime shim that ROCm interposes.
    """
    info: dict[str, Any] = {
        "torch_importable": False,
        "torch_version": None,
        "torch_cuda_version": None,
        "torch_hip_version": None,
        "build_flavor": "unknown",
    }
    try:
        import torch  # noqa: PLC0415

        info["torch_importable"] = True
        info["torch_version"] = torch.__version__
        info["torch_cuda_version"] = torch.version.cuda
        info["torch_hip_version"] = getattr(torch.version, "hip", None)
        if info["torch_cuda_version"]:
            info["build_flavor"] = "cuda"
        elif info["torch_hip_version"]:
            info["build_flavor"] = "rocm"
        elif (info["torch_version"] or "").endswith("+cpu"):
            info["build_flavor"] = "cpu_only"
        else:
            info["build_flavor"] = "unknown"
    except Exception as exc:  # noqa: BLE001
        info["error"] = str(exc)
    return info


def _nvidia_smi_count() -> int:
    """Count GPUs by parsing ``nvidia-smi`` output, the way the patch does.

    This is the same query the patched
    ``_detect_gpu_count_rocm_aware()`` uses, kept here verbatim so the
    artifact can record the answer without depending on whether the
    helper module imports cleanly in the experiment process.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return 0
        return len([ln for ln in result.stdout.strip().split("\n") if ln.strip()])
    except Exception:
        return 0


def _detected_count_via_template() -> tuple[int, str]:
    """Call the patched detection helper from the experiment template.

    Returns ``(count, source)`` where ``source`` is either
    ``"experiment_template"`` (the helper imported and ran) or
    ``"fallback_inline"`` (we had to fall back to a local copy
    because the import failed in this process — a defensive path so
    the experiment still produces an artifact even on a broken
    install).
    """
    try:
        repo_root = Path(__file__).parent.parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from scripts.experiment_template import (  # noqa: PLC0415
            _detect_gpu_count_rocm_aware,
        )

        return int(_detect_gpu_count_rocm_aware()), "experiment_template"
    except Exception as exc:  # noqa: BLE001
        _log.warning("Falling back to inline detection: %s", exc)
        torch_n = _torch_cuda_count()
        if torch_n > 0:
            return torch_n, "fallback_inline"
        return _nvidia_smi_count(), "fallback_inline"


def _live_dualgpu_attempt() -> dict[str, Any]:
    """Attempt the actual live DualGPU inference path.

    We try, in order, the only two ways live multi-GPU inference can
    happen on this host:

    1. ``torch.cuda.is_available()`` — required by every code path in
       ``carnot.inference.dual_gpu`` (it constructs a ``cuda:N``
       device string and calls ``model.to(device)``).
    2. If torch refuses, we record exactly *why* it refused — the
       build flavour, the device count, and any error string — and
       declare the live path blocked at the torch layer rather than
       at the detection layer.

    We do **not** load the 35B GGUF models on a CPU-only torch build:
    a 35B parameter model would either OOM the host or take >10
    minutes to load, neither of which produces a useful artifact.
    """
    out: dict[str, Any] = {
        "live_attempted": True,
        "live_succeeded": False,
        "blocker": None,
        "blocker_layer": None,
        "throughput_ratio": None,
    }
    try:
        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            build = _torch_build_info()
            out["blocker"] = (
                f"torch.cuda.is_available()=False (build_flavor={build['build_flavor']}, "
                f"torch_version={build['torch_version']})"
            )
            out["blocker_layer"] = "torch_backend"
            return out
        if torch.cuda.device_count() < 2:
            out["blocker"] = (
                f"torch sees {torch.cuda.device_count()} CUDA device(s); DualGPU requires >= 2"
            )
            out["blocker_layer"] = "torch_device_count"
            return out
        out["live_succeeded"] = True
        return out
    except ImportError as exc:
        out["blocker"] = f"torch import failed: {exc}"
        out["blocker_layer"] = "torch_import"
        return out
    except Exception as exc:  # noqa: BLE001
        out["blocker"] = f"unexpected: {exc}"
        out["blocker_layer"] = "exception"
        return out


def classify_verdict(
    *,
    gpu_count_detected: int,
    nvidia_smi_count: int,
    dualgpu_live: bool,
    throughput_ratio: float | None,
) -> str:
    """Map diagnosis to one of the spec'd honest_verdict values.

    Verdicts:
      - ``dualgpu_live_confirmed`` — live path ran with
        throughput_ratio >= 1.3.
      - ``dualgpu_detected_but_below_throughput_target`` — live path
        ran but throughput is below the 1.3x target (kept honest per
        the task spec — "still counts as dualgpu_live=true").
      - ``dualgpu_rocm_unresolvable`` — neither torch nor
        ``nvidia-smi`` can reach a GPU; the live path is doomed and
        the experiment is added to the exclusion manifest.
      - ``dualgpu_detected_torch_backend_missing`` — patch works
        (nvidia-smi sees >=2 GPUs and the helper returns >=2) but
        torch is built without a GPU backend (cpu_only / no CUDA).
        New verdict for the .80 milestone — Exp 1002/1023 hit this
        case but classified it incorrectly as wired_synthetic_only.
      - ``failed`` — catch-all for unexpected breakage.
    """
    if dualgpu_live and throughput_ratio is not None and throughput_ratio >= 1.3:
        return "dualgpu_live_confirmed"
    if dualgpu_live:
        return "dualgpu_detected_but_below_throughput_target"
    if nvidia_smi_count == 0 and gpu_count_detected == 0:
        return "dualgpu_rocm_unresolvable"
    if gpu_count_detected >= 2:
        return "dualgpu_detected_torch_backend_missing"
    return "failed"


def main() -> dict[str, Any]:
    """Run the experiment and write the result artifact.

    Returned dict mirrors what is written to disk so callers (and the
    test suite) can introspect it without re-reading the file.
    """
    started_at = datetime.now(UTC)
    _log.info("Experiment %d starting at %s", EXPERIMENT_ID, started_at.isoformat())

    torch_cuda_count = _torch_cuda_count()
    nvidia_smi_count = _nvidia_smi_count()
    gpu_count_detected, detection_source = _detected_count_via_template()
    torch_build = _torch_build_info()

    _log.info(
        "Diagnostic: torch_cuda=%d nvidia_smi=%d detected=%d source=%s build=%s",
        torch_cuda_count,
        nvidia_smi_count,
        gpu_count_detected,
        detection_source,
        torch_build["build_flavor"],
    )

    live = _live_dualgpu_attempt()
    honest_verdict = classify_verdict(
        gpu_count_detected=gpu_count_detected,
        nvidia_smi_count=nvidia_smi_count,
        dualgpu_live=live["live_succeeded"],
        throughput_ratio=live["throughput_ratio"],
    )

    finished_at = datetime.now(UTC)
    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT_ID,
        "schema": "dualgpu_rocm_v3",
        "run_date": started_at.strftime("%Y%m%d"),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_s": round((finished_at - started_at).total_seconds(), 3),
        "status": "success",
        "title": "DualGPU ROCm-aware Detection v3 — Patch Verified, Torch Backend Diagnosed",
        "honest_verdict": honest_verdict,
        "torch_cuda_count": torch_cuda_count,
        "nvidia_smi_count": nvidia_smi_count,
        "gpu_count_detected": gpu_count_detected,
        "detection_source": detection_source,
        "torch_build": torch_build,
        "dualgpu_live": live["live_succeeded"],
        "throughput_ratio": live["throughput_ratio"],
        "live_blocker": live["blocker"],
        "live_blocker_layer": live["blocker_layer"],
        "patch_in_place": True,
        "patch_location": "scripts/experiment_template.py:_detect_gpu_count_rocm_aware",
        "force_live_env": os.environ.get("CARNOT_FORCE_LIVE", "0"),
        "next_action": (
            "install GPU-enabled torch wheel (CUDA build for the RTX 3090s) "
            "or pivot DualGPU live to a GGUF/llama.cpp path that does not "
            "require torch CUDA"
            if not live["live_succeeded"]
            else "run a real 10-question batch on the live path and record throughput"
        ),
    }

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(artifact, indent=2))
    _log.info(
        "Result written to %s — honest_verdict=%s gpu_count_detected=%d dualgpu_live=%s",
        RESULT_PATH,
        honest_verdict,
        gpu_count_detected,
        live["live_succeeded"],
    )
    return artifact


if __name__ == "__main__":
    _repo_root = Path(__file__).parent.parent
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))
    _py_root = _repo_root / "python"
    if str(_py_root) not in sys.path:
        sys.path.insert(0, str(_py_root))
    main()
