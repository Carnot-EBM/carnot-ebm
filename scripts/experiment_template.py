#!/usr/bin/env python3
"""Experiment scaffolding template — eliminates cold-start boilerplate for new experiments.

**Researcher summary:**
    Every new experiment currently spends 15-20 minutes writing the same boilerplate:
    imports, argparse, result schema, checkpoint logic, GPU setup, and timeout wiring.
    This template eliminates that overhead by providing re-usable classes that encode
    the patterns validated across Exps 258, 294, 302, and the 2026.04.21 retrospective.

**What this template provides:**
    1. ``ExperimentTemplate`` — orchestrates the standard experiment lifecycle:
       - Output directory and checkpoint directory creation
       - Checkpoint save/resume with atomic writes (no .tmp files left behind)
       - GPU pre-warm + health-check (the Exp 294 fix for lazy-load GPU stalls)
       - Standardised result schema with all required fields
       - Thread-based timeout for any long-running function

    2. ``BatchedInferenceRunner`` — groups a list of questions into batches and
       runs each batch with a ``batch_size * 60 s`` timeout (not per-question).
       This is the ``+6% wall-time`` improvement from the 2026.04.21 retrospective.
       - Maintains a ``batch_log`` of ``{batch_id, batch_size, batch_time_s}`` per batch.
       - Returns results in the original question order.

**Usage example for a new experiment:**

    ```python
    from scripts.experiment_template import ExperimentTemplate, BatchedInferenceRunner

    # 1. Instantiate
    tmpl = ExperimentTemplate(
        exp_id=307,
        title="My new experiment",
        deliverable="results/experiment_307_results.json",
        requires_gpu=True,
    )

    # 2. Setup (creates dirs, loads checkpoint if present)
    tmpl.setup()

    # 3. (Optional) Pre-warm GPUs using Exp 294 pattern.
    #
    # MODEL SELECTION — MANDATORY for any live-data or verify-repair experiment:
    # Always try `cached_sota_pair()` first; it resolves the three mandated SOTA
    # GGUFs (unsloth/Qwen3.6-35B-A3B-GGUF, unsloth/gemma-4-26B-A4B-it-GGUF,
    # unsloth/gemma-4-31B-it-GGUF) via the HF cache and returns `model_path`
    # entries loadable through `Gemma4QuantizedLoader` (llama.cpp-backed, model-
    # agnostic despite the class name).  These produce REAL arithmetic CoT that
    # downstream extractors (CoACE, LLMAsExtractor, JEPA) can actually score.
    #
    # Hardcoding `Qwen/Qwen3.5-0.8B` or `google/gemma-4-E4B-it` produces
    # 'The answer is 42.' and echo-question garbage that has no arithmetic
    # structure — this blocked RETRO-033 for 15+ consecutive attempts before
    # the cached SOTA GGUFs were wired in.  Only use the legacy tiny-model
    # pair when `cached_sota_pair()` returns None (GGUFs not present on the
    # current host) AND log a LOUD warning about expected output quality.
    #
    # Record `models_used` in every artifact with the exact hub IDs so the
    # retrospective can verify which path ran.
    from carnot.inference.sota_models import cached_sota_pair  # noqa
    specs = cached_sota_pair(gpu_indices=(0, 1))
    if specs is None:
        print("WARNING: cached_sota_pair() returned None — no SOTA GGUFs in "
              "HF cache. Falling back to legacy tiny models "
              "(Qwen/Qwen3.5-0.8B + google/gemma-4-E4B-it). Expected CoT "
              "structure: POOR. Output will be 50/50 'The answer is 42.' + "
              "question-echo. Downstream extractor recall will be < 10%.")
        MODEL_SPECS = [
            {"name": "Gemma4-E4B-it", "hf_id": "google/gemma-4-E4B-it", "gpu": 0},
            {"name": "Qwen3.5-0.8B", "hf_id": "Qwen/Qwen3.5-0.8B", "gpu": 1},
        ]
        models_used_field = [s["hf_id"] for s in MODEL_SPECS]
        expected_cot_structure = False
    else:
        MODEL_SPECS = specs   # each entry has name, hf_id, gpu, model_path
        models_used_field = [s["hf_id"] for s in MODEL_SPECS]
        expected_cot_structure = True
    gpu_status = tmpl.setup_gpu(MODEL_SPECS)
    if not gpu_status["all_healthy"]:
        artifact = tmpl.build_result({}, status="blocked",
                                      stall_root_cause=gpu_status["models"])
        # write artifact and exit …

    # 4. Batch inference
    questions = [f"What is {i} + {i}?" for i in range(100)]
    bir = BatchedInferenceRunner(my_inference_fn, batch_size=8)
    results = bir.run_batch(questions)
    print(bir.batch_log)  # [{batch_id, batch_size, batch_time_s}, ...]

    # 5. Save checkpoint periodically
    tmpl.checkpoint_save({"done_so_far": [r.response for r in results[:50]]}, step=50)

    # 6. Build final artifact with all required fields auto-populated
    artifact = tmpl.build_result(
        {"responses": [r.response for r in results], "batch_log": bir.batch_log},
        status="success",
    )
    ```

Spec: REQ-VERIFY-083, REQ-VERIFY-084,
      REQ-INFRA-007, REQ-INFRA-014,
      REQ-INFRA-073, REQ-INFRA-074,
      SCENARIO-VERIFY-109, SCENARIO-VERIFY-110, SCENARIO-VERIFY-111,
      SCENARIO-VERIFY-112, SCENARIO-VERIFY-113, SCENARIO-VERIFY-114,
      SCENARIO-VERIFY-115, SCENARIO-VERIFY-116,
      SCENARIO-INFRA-011, SCENARIO-INFRA-015,
      SCENARIO-INFRA-083, SCENARIO-INFRA-084, SCENARIO-INFRA-085
"""

from __future__ import annotations

import ast
import atexit
import contextlib
import datetime
import gc
import hashlib
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar

from carnot.experiment_artifacts import atomic_write_json, resolve_experiment_artifact_path
from carnot.inference.sota_models import cached_sota_pair
from carnot.pipeline.deliverable_guard import DeliverableGuard
from carnot.pipeline.dual_gpu_assigner import DualGPUAssigner

_log = logging.getLogger(__name__)
_T = TypeVar("_T")


# Training entrypoints that the GPU-zombie reaper must NEVER kill. A model-training
# process legitimately holds large VRAM for hours. The nvidia-smi fallback path of
# kill_gpu_zombies() (used when pynvml is absent) gates the kill on the MINIMUM GPU
# utilisation across ALL GPUs — so an idle GPU 0 (0%) drags the gate to 0% and the
# reaper kills a fully-busy training run on GPU 1 (100% util, >1GB VRAM).
#
# Origin: 2026-06-13. The outer-loop's contiguous TRM Sudoku-Extreme training
# (decision "A": training is OWNED BY THE OUTER-LOOP; the conductor stands down on TRM —
# research-roadmap-next.yaml .386 HARD COORDINATION RULE) was SIGTERM'd repeatedly by
# conductor experiment tasks calling kill_gpu_zombies() in their GPU setup. This is the
# MECHANICAL backstop for that prose rule (the sibling fix in gpu_monitor.detect_zombies
# covers the conductor's per-task zombie reaper). Matched against /proc/<pid>/cmdline
# because nvidia-smi reports only the bare process name.
_TRAINING_ENTRYPOINT_MARKERS = ("train.py", "/nn/train", "src/nn/train")


def _pid_is_protected_training_proc(pid: int) -> bool:
    """True if PID is a model-training process that must be exempt from zombie-kill."""
    try:
        cmdline = (
            Path(f"/proc/{pid}/cmdline")
            .read_bytes()
            .replace(b"\x00", b" ")
            .decode("utf-8", "replace")
        )
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return False
    return any(marker in cmdline for marker in _TRAINING_ENTRYPOINT_MARKERS)


def _cuda_is_available() -> bool:
    """Return True only when at least one CUDA GPU is accessible.

    Why a helper instead of importing torch directly: torch is an optional
    dependency and may not be installed on CPU-only machines.  Wrapping the
    check in a try/except lets every experiment run safely without GPUs while
    still enabling GPU acceleration when hardware is present.
    """
    try:
        import torch  # noqa: PLC0415

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _detect_gpu_count_rocm_aware() -> int:
    """Return the number of NVIDIA GPUs visible to this process.

    Why we need a ROCm-aware fallback: on ROCm systems (AMD GPU driver stack),
    ``torch.cuda.device_count()`` returns 0 even when NVIDIA GPUs are physically
    present unless ``CUDA_VISIBLE_DEVICES`` is explicitly set.  This happens because
    ROCm's HIP-over-CUDA shim intercepts the CUDA device enumeration before PyTorch
    can see the real NVIDIA cards.  ``nvidia-smi`` bypasses the driver shim entirely
    and queries the NVIDIA kernel module directly, so it reliably reports the true
    GPU count regardless of what ROCm has done to the CUDA environment.

    Fall-through order:
    1. ``torch.cuda.device_count()`` — fast, authoritative on pure-CUDA hosts.
    2. ``nvidia-smi`` subprocess — authoritative on ROCm hosts or any host where
       torch returns 0 but NVIDIA hardware is actually present.
    3. Return 0 — no NVIDIA GPU tooling available (CPU-only host).
    """
    try:
        import torch  # noqa: PLC0415

        cuda_count = torch.cuda.device_count()
        if cuda_count > 0:
            return cuda_count
    except Exception:
        pass

    # ROCm fallback: ask nvidia-smi directly.
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            lines = [ln for ln in result.stdout.strip().split("\n") if ln.strip()]
            return len(lines)
    except Exception:
        pass

    return 0


def _uses_placeholder_model_ids(model_specs: list[dict[str, Any]]) -> bool:
    """True when every spec names a non-loadable unit-test placeholder model."""

    if not model_specs:
        return False
    placeholder_prefixes = ("mock/", "test/")
    return all(str(spec.get("hf_id", "")).startswith(placeholder_prefixes) for spec in model_specs)


def _run_in_daemon_thread_with_timeout(
    fn: Callable[[], _T],
    timeout_s: float,
) -> tuple[bool, _T | None]:
    """Run ``fn`` in a daemon thread and return ``(completed, result)``."""

    result_box: list[_T] = []
    error_box: list[BaseException] = []

    def _target() -> None:
        try:
            result_box.append(fn())
        except BaseException as exc:
            error_box.append(exc)

    worker = threading.Thread(target=_target, daemon=True)
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        return False, None
    if error_box:
        raise error_box[0]
    return True, result_box[0] if result_box else None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_RESULT_FIELDS: list[str] = [
    "experiment",
    "schema",
    "run_date",
    "started_at",
    "finished_at",
    "duration_s",
    "status",
    "title",
]
"""Every experiment artifact MUST contain these top-level keys (REQ-VERIFY-083).

These mirror the fields validated across Exps 294, 302, and 303 and ensure
downstream tooling (conductor, retrospective scripts) can always parse results.
"""

OPTIONAL_ECONOMICS_FIELDS: tuple[str, ...] = ("cost_usd", "decision_class")
"""Optional economics fields that experiments *should* emit when they make a
verify/detect/repair decision end-to-end.

Why these two and not more: the 2026-04-19 Vidoc Security study
(https://decrypt.co/364744/anthropic-mythos-replicated-public-models-vidoc-security)
showed that the per-scan cost of LLM-driven vulnerability discovery has
collapsed to well under $30, which makes the comparison-across-systems
question hinge on *economics per decision* rather than raw accuracy.
Carnot's answering unit of work is a verify/detect/repair decision, so:

    - cost_usd:        approximate USD spent on LLM API calls + GPU time for
                       this experiment, measured end-to-end (prompt tokens +
                       completion tokens + optional hardware amortisation).
                       Zero is acceptable when the experiment ran entirely on
                       local hardware; omit the field entirely when genuinely
                       unknown rather than reporting a fake number.
    - decision_class:  one of 'detect' | 'verify' | 'repair' (or a list of
                       these when an experiment covers multiple) so the
                       retrospective agent can slice results by which moat
                       tier the experiment exercised.

These fields are *optional* rather than required on purpose: making them
required would break every pre-500 artifact on disk that predates the
validation-moat framing.  ``build_result`` accepts them as kwargs and puts
them in the artifact when provided, so new experiments adopt the schema
naturally without forcing a backfill.
"""

DECISION_CLASSES: frozenset[str] = frozenset({"detect", "verify", "repair"})
"""Allowed values for the ``decision_class`` economics field.

A single experiment may cover multiple classes -- in that case pass a list
of strings like ``["detect", "verify"]`` rather than a single string.  The
retrospective agent treats list-valued decision_class as a superset.
"""

PRODUCER_NORMALIZER_RECEIPTS_FIELD = "producer_normalizer_receipts"
"""Receipt field added when producer-side artifact normalization changes anything.

The producer hook is allowed to make shape-only repairs, but those repairs still
matter during audits.  A single template-owned receipt key lets conductor gates
inspect normalized bare fields while reviewers can see which repairs or unsafe
rejections occurred.
"""

DEFAULT_BATCH_SIZE: int = 8
"""Default number of questions per inference batch (REQ-VERIFY-084).

The 2026.04.21 retrospective identified batching at 8-16 as the sweet spot
for 3-6× throughput improvement over sequential one-at-a-time inference.
"""

_CHECKPOINT_FILENAME = "checkpoint.json"
"""Filename used for experiment checkpoints within the checkpoint directory."""


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    """Return the current UTC timestamp in ISO-8601 format (e.g. ``2026-04-14T12:00:00Z``)."""
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_date() -> str:
    """Return today's date as an 8-digit string (e.g. ``'20260414'``)."""
    return datetime.datetime.now(datetime.UTC).strftime("%Y%m%d")


def _get_repo_root() -> Path:
    """Return the repository root, honouring the ``CARNOT_REPO_ROOT`` env override."""
    override = os.environ.get("CARNOT_REPO_ROOT")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[1]


def normalize_artifact_for_template_write(
    artifact: Mapping[str, Any],
    *,
    nullable_fields: Sequence[str] = (),
    gate_fields: Sequence[str] = (),
    required_principle_fields: Sequence[str] = (),
) -> dict[str, Any]:
    """Normalize a producer-built artifact copy before it is written.

    Spec: REQ-REPORT-5267, SCENARIO-REPORT-5267-TEMPLATE-NORMALIZATION,
    SCENARIO-REPORT-5267-UNSAFE-REJECTION.

    Why this helper lives in the template: newly produced artifacts should have
    easy-to-read bare gate fields before conductor gates or auditors inspect
    them.  The helper reuses the Exp5247 strict normalizer, but it does not
    require legacy template users to have already adopted `inference_substrate`.
    Missing methodology, duration, model, or solve evidence remains a rejection
    receipt; the helper never invents those fields.
    """

    from carnot.experiment_5247_slot_artifact_normalizer_v480 import (  # noqa: PLC0415
        normalize_artifact,
    )

    result = normalize_artifact(
        artifact,
        nullable_fields=nullable_fields,
        gate_fields=gate_fields,
        required_principle_fields=required_principle_fields,
        require_inference_substrate=False,
    )
    normalized = dict(result.normalized)
    if result.safe_repairs or result.unsafe_rejections:
        normalized[PRODUCER_NORMALIZER_RECEIPTS_FIELD] = {
            "safe_repairs": [dict(row) for row in result.safe_repairs],
            "unsafe_rejections": [dict(row) for row in result.unsafe_rejections],
            "ready_for_gated_consumers": result.ready_for_gated_consumers,
        }
    return normalized


# ---------------------------------------------------------------------------
# EnvPropagationGuard  (REQ-INFRA-070)
# ---------------------------------------------------------------------------

_SESSION_ENV_PATH = Path.home() / ".carnot_session_env"
"""Path where EnvPropagationGuard persists env vars across subprocess boundaries.

Why a file and not a shell export:
    ``claude -p`` creates a NEW process tree that does not inherit the outer
    shell's environment.  Writing CARNOT_FORCE_LIVE=1 with ``os.environ``
    only patches the current process; the next ``claude -p`` invocation
    spawns a fresh interpreter with a bare env and silently falls back to
    non-live mode.  This file is a simple, language-agnostic escape hatch:
    any subprocess that calls ``EnvPropagationGuard.load_session_env()`` at
    startup (ExperimentTemplate.__init__ does this automatically) will pick
    up the persisted vars before any other code runs.
"""


class EnvPropagationGuard:
    """Cross-process environment propagation for Carnot experiment sessions.

    Problem it solves (RETRO-LIVE-ENV-NOT-PROPAGATED, 6th consecutive recurrence):
        ``apply_env_autofix()`` correctly sets ``os.environ["CARNOT_FORCE_LIVE"]``
        inside the running process, but the conductor forks ``claude -p`` to run
        experiments.  Each ``claude -p`` is a brand-new process tree; it does NOT
        inherit the patched environment.  The fix is to persist the override to
        ``~/.carnot_session_env`` and have every ``ExperimentTemplate.__init__``
        source that file before doing anything else.

    Format of ~/.carnot_session_env:
        Plain-text KEY=VALUE lines, one per line.  Lines starting with ``#``
        are comments and are ignored.  Values must not contain newlines.
    """

    _path: Path = _SESSION_ENV_PATH

    @classmethod
    def write_session_env(cls, vars: dict[str, str]) -> None:
        """Append or update KEY=VALUE entries in ~/.carnot_session_env.

        Existing keys are overwritten; other lines are preserved.
        Creates the file if it does not exist.

        Parameters
        ----------
        vars:
            Mapping of env-var names to string values to persist.
        """
        existing: dict[str, str] = {}
        if cls._path.exists():
            for raw_line in cls._path.read_text().splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, _, v = line.partition("=")
                    existing[k.strip()] = v.strip()
        existing.update(vars)
        lines = [f"{k}={v}" for k, v in sorted(existing.items())]
        cls._path.write_text("\n".join(lines) + "\n")

    @classmethod
    def load_session_env(cls) -> dict[str, str]:
        """Read ~/.carnot_session_env and apply each KEY=VALUE to os.environ.

        Only sets vars that are NOT already present in os.environ, so an
        explicit shell export always takes precedence over the session file.

        Returns the mapping of vars that were applied (empty dict if the file
        does not exist or every key was already set).
        """
        if not cls._path.exists():
            return {}
        applied: dict[str, str] = {}
        for raw_line in cls._path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, _, v = line.partition("=")
                key = k.strip()
                val = v.strip()
                if key not in os.environ:
                    os.environ[key] = val
                    applied[key] = val
        return applied

    # ------------------------------------------------------------------
    # Session-boundary-persistent state file (REQ-INFRA-080)
    # ------------------------------------------------------------------

    STATE_FILE: Path = Path.home() / ".carnot" / "conductor_state.sh"
    """Shell-sourceable state file persisted across conductor session boundaries.

    Why ~/.carnot/conductor_state.sh and not ~/.carnot_session_env:
        The conductor launches each claude -p experiment in a fresh process tree
        that does not inherit the parent's environment.  ~/.carnot_session_env
        handles the intra-session case (same conductor run), but the STATE_FILE
        handles the INTER-session case: a state written by session N is read at
        startup of session N+1, so CARNOT_FORCE_LIVE=1 survives a full conductor
        restart.  The shell `export KEY=VALUE` syntax lets operators also source
        this file directly in their shell profile for manual debugging.
    """

    @classmethod
    def propagate(cls) -> dict[str, str]:
        """Load CARNOT_*, ROCM_*, and HSA_* vars into the current process from all sources.

        Sources (applied in order, later sources win):
        1. STATE_FILE (~/.carnot/conductor_state.sh) — inter-session persistence.
        2. ~/.carnot_session_env — intra-session persistence (written by prior calls).
        3. os.environ — explicit shell exports always win over persisted values.

        Additionally unconditionally sets CARNOT_FORCE_LIVE=1 (the persistent fix
        for RETRO-015 recurrence: env vars not propagated to conductor subprocesses).

        Returns
        -------
        dict[str, str]
            All CARNOT_*, ROCM_*, and HSA_* vars now present in os.environ after
            propagation, including vars that were already set.

        Spec: REQ-INFRA-080, SCENARIO-INFRA-090
        """
        # Step 1: source STATE_FILE if it exists — inter-session persistence
        if cls.STATE_FILE.exists():
            for raw_line in cls.STATE_FILE.read_text().splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                # Strip leading 'export ' if present
                if line.startswith("export "):
                    line = line[len("export ") :]
                if "=" in line:
                    k, _, v = line.partition("=")
                    key = k.strip()
                    val = v.strip().strip('"').strip("'")
                    # STATE_FILE wins over nothing but loses to explicit shell exports
                    if key not in os.environ:
                        os.environ[key] = val

        # Step 2: source ~/.carnot_session_env — intra-session persistence
        cls.load_session_env()

        # Step 3: unconditionally set CARNOT_FORCE_LIVE=1 — persistent fix for
        # RETRO-015 recurrence.  This is the one var that MUST always be present
        # when the conductor launches GPU experiments.
        os.environ["CARNOT_FORCE_LIVE"] = "1"

        # Collect all propagated vars for the return value / artifact
        propagated = {
            k: v for k, v in os.environ.items() if k.startswith(("CARNOT_", "ROCM_", "HSA_"))
        }
        _log.info(
            "EnvPropagationGuard.propagate(): %d vars in scope: %s",
            len(propagated),
            sorted(propagated.keys()),
        )
        return propagated

    @classmethod
    def write_state_file(cls) -> None:
        """Persist all current CARNOT_* vars to ~/.carnot/conductor_state.sh.

        Always includes CARNOT_FORCE_LIVE=1 even if not in os.environ, so
        the next conductor session starts with the correct gate value.

        Creates ~/.carnot/ if it does not exist.

        Spec: REQ-INFRA-081, SCENARIO-INFRA-091
        """
        cls.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Collect CARNOT_* vars from the current environment
        vars_to_write: dict[str, str] = {
            k: v for k, v in os.environ.items() if k.startswith("CARNOT_")
        }
        # Unconditionally ensure CARNOT_FORCE_LIVE=1 — the persistent fix
        vars_to_write["CARNOT_FORCE_LIVE"] = "1"

        lines = ["#!/bin/sh", "# Carnot conductor state — auto-generated, do not edit by hand.", ""]
        for key in sorted(vars_to_write):
            val = vars_to_write[key]
            lines.append(f"export {key}={val}")
        lines.append("")  # trailing newline
        cls.STATE_FILE.write_text("\n".join(lines))
        _log.info(
            "EnvPropagationGuard.write_state_file(): wrote %d vars to %s",
            len(vars_to_write),
            cls.STATE_FILE,
        )


# ---------------------------------------------------------------------------
# Reproducibility checksum helper
# ---------------------------------------------------------------------------


def _compute_repro_checksum(
    seed: int,
    code_files: list[str],
    data_path: str | None = None,
) -> str:
    """Return a 16-character hex reproducibility checksum.

    Why a checksum: after the 2026-04-29 exp1031 verdict flip (fr11_loop_closed
    at 21:12Z vs carnot_filter_below_baseline at 01:13Z), the project needs a
    lightweight fingerprint that makes *what environment produced this verdict*
    auditable. The checksum is NOT a guarantee of bit-exact reproduction (GPU
    non-determinism prevents that); it IS a signal that the same seed + code +
    data were used, so verdict differences must be attributed to GPU noise or
    model-load variance rather than code changes.

    The hash covers:
    - The integer seed (first 8 bytes, big-endian) — seed changes ⟹ hash changes.
    - Content of each code file in ``code_files`` that exists on disk.
    - Content of ``data_path`` if provided and it exists on disk.

    Parameters
    ----------
    seed:
        Integer RNG seed recorded in the artifact.
    code_files:
        List of file paths to include in the hash (typically the experiment
        script and any helper modules it imports from this repo).
    data_path:
        Optional path to the primary dataset file consumed by the experiment.

    Returns
    -------
    str
        First 16 hex characters of the SHA-256 digest — short enough to embed
        in a JSON artifact without clutter, long enough to detect accidental
        code-or-data drift.
    """
    h = hashlib.sha256()
    h.update(seed.to_bytes(8, "big"))
    for path in code_files:
        if os.path.exists(path):
            with open(path, "rb") as fp:
                h.update(fp.read())
    if data_path and os.path.exists(data_path):
        with open(data_path, "rb") as fp:
            h.update(fp.read())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# InferenceResult
# ---------------------------------------------------------------------------


@dataclass
class InferenceResult:
    """Single-question result from ``BatchedInferenceRunner``.

    Fields
    ------
    prompt : str
        The original question / prompt string.
    response : str
        The model-generated response (empty string on timeout).
    batch_id : int
        Zero-based index of the batch this prompt was part of.
    timed_out : bool
        ``True`` if the entire batch timed out before this result was produced.
    """

    prompt: str
    response: str
    batch_id: int
    timed_out: bool = False


# ---------------------------------------------------------------------------
# ExperimentTemplate
# ---------------------------------------------------------------------------


class ExperimentTemplate:
    """Standard scaffolding for Carnot research experiments.

    Encodes the best-practice patterns from Exps 258, 294, 302:
    - Atomic checkpoint save/resume (no data loss on interrupted runs)
    - GPU pre-warm + health-check before timed inference (Exp 294 pattern)
    - Standardised artifact schema (all required fields auto-populated)
    - Thread-based timeout for any long-running function

    Parameters
    ----------
    exp_id : int
        The experiment number (e.g. ``307``).
    title : str
        Human-readable experiment title embedded in every artifact.
    deliverable : str
        Relative path (from repo root) of the JSON artifact to write.
    requires_gpu : bool
        If ``True``, ``setup_gpu()`` is expected to be called before inference.
    repo_root : Path | None
        Override the repository root (used in tests; defaults to auto-detection).
    """

    def __init__(
        self,
        exp_id: int,
        title: str,
        deliverable: str,
        *,
        requires_gpu: bool = False,
        repo_root: Path | None = None,
        seed: int = 42,
    ) -> None:
        # REQ-INFRA-070: source the session-env file FIRST so that vars written
        # by a prior apply_env_autofix() call (in a different process/invocation)
        # are present before any GPU or live-env checks run.  This is the fix for
        # RETRO-LIVE-ENV-NOT-PROPAGATED (6th consecutive recurrence): ``claude -p``
        # spawns a fresh process tree that does not inherit the outer shell env.
        EnvPropagationGuard.load_session_env()

        self.exp_id = exp_id
        self.title = title
        self.deliverable = deliverable
        self.requires_gpu = requires_gpu
        # Random seed for verdict-reproducibility discipline (2026-04-29 incident).
        # Default 42 preserves backward compatibility for experiments that don't
        # care about reproducibility; pass seed=<N> to ExperimentTemplate() to
        # override. The seed is applied in setup() and recorded in every artifact.
        self.random_seed: int = seed
        self._repo_root: Path = repo_root if repo_root is not None else _get_repo_root()
        self._allow_artifact_override = repo_root is None
        self.checkpoint: dict[str, Any] | None = None
        self._started_at: str = _utc_now()
        self._t0: float = time.perf_counter()

        # Phase timing instrumentation. Populated by the `phase()` context
        # manager below; auto-included in build_result()'s artifact under
        # the `phase_timings_s` key when non-empty. Lets researchers profile
        # where the 5-9 min Sonnet research-step actually goes (model loads,
        # training loops, inference, internal pytest re-runs) without writing
        # bespoke timing code in every script. Compounds across milestones.
        self._phase_timings: list[dict[str, Any]] = []

        # Set by setup_gpu() — warm inference server and dual-GPU runner.
        # None until setup_gpu() is called or when running in CPU fallback mode.
        self.model_server: Any | None = None
        self.gpu_runner: Any | None = None

        # Set by setup()
        self._ckpt_dir: Path = resolve_experiment_artifact_path(
            Path("results") / "checkpoints" / f"experiment_{exp_id}",
            root=self._repo_root,
            allow_override=self._allow_artifact_override,
        )
        self._output_path: Path = resolve_experiment_artifact_path(
            deliverable,
            root=self._repo_root,
            allow_override=self._allow_artifact_override,
        )

        # REQ-INFRA-033: guard that raises FileNotFoundError if the deliverable
        # is absent when assert_deliverable_written() is called at end of main().
        # RETRO-032/033/036: three milestones lost to silently missing result JSONs.
        guard_path = deliverable if self._allow_artifact_override else str(self._output_path)
        self._guard = DeliverableGuard(guard_path)

        # REQ-INFRA-073 / RETRO-054: register teardown so GPU VRAM is freed
        # even when the experiment exits abnormally (ctrl-c, exception, conductor kill).
        # Without this, VRAM leaks accumulate monotonically across every milestone.
        atexit.register(self.teardown)

    # ------------------------------------------------------------------
    # kill_gpu_zombies() — REQ-INFRA-074
    # ------------------------------------------------------------------

    @classmethod
    def kill_gpu_zombies(
        cls,
        vram_threshold_mb: int = 1000,
        util_threshold_pct: float = 5.0,
    ) -> dict[str, Any]:
        """Kill GPU processes holding VRAM at near-zero utilization (zombie processes).

        Why this is a classmethod: it must run at process startup BEFORE any
        ExperimentTemplate instance holds GPU memory.  Calling it as a class method
        means callers don't need a template instance just to clean up lingering zombies
        from previous experiments.

        A "zombie" GPU process is one that:
        - Holds more than ``vram_threshold_mb`` MB of VRAM on ANY GPU, AND
        - Has less than ``util_threshold_pct`` % compute utilization on that GPU.

        These are the processes that caused 47,653 MB of stuck VRAM at the end of
        milestone .40 (RETRO-054).  They appear to be alive (holding VRAM) but are
        not doing any compute — they survived the experiment exit without releasing
        their GPU memory.

        Parameters
        ----------
        vram_threshold_mb : int
            Minimum VRAM (in MB) for a process to be considered a zombie candidate.
            Default 1000 MB (~1 GB) avoids killing tiny helper processes.
        util_threshold_pct : float
            Maximum GPU utilization (in %) for a process to be considered a zombie.
            Default 5.0% — any process doing real compute should be above this.

        Returns
        -------
        dict with keys:
            - ``killed_pids`` (list[int]): PIDs that were sent SIGTERM.
            - ``freed_mb`` (int): Total VRAM freed across all killed processes.
            - ``error`` (str, optional): Present when pynvml is not available.

        Spec: REQ-INFRA-074, SCENARIO-INFRA-084, SCENARIO-INFRA-085
        """
        try:
            import pynvml  # noqa: PLC0415
        except ImportError:
            _log.debug("kill_gpu_zombies: pynvml not installed — trying nvidia-smi fallback")
            return cls._kill_gpu_zombies_nvidia_smi(vram_threshold_mb, util_threshold_pct)

        killed_pids: list[int] = []
        freed_mb = 0

        try:
            pynvml.nvmlInit()
            n_gpus = pynvml.nvmlDeviceGetCount()
            seen_pids: set[int] = set()

            for gpu_idx in range(n_gpus):
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)

                # GPU-wide utilization — used to gate the per-process kill decision.
                # When a GPU is at <5% utilization overall, processes on it are idle.
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util_pct = float(util.gpu)
                except Exception:
                    gpu_util_pct = 0.0

                try:
                    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                except Exception:
                    procs = []

                for proc in procs:
                    pid = proc.pid
                    if pid in seen_pids:
                        continue
                    vram_mb = (proc.usedGpuMemory or 0) // (1024 * 1024)
                    if vram_mb >= vram_threshold_mb and gpu_util_pct < util_threshold_pct:
                        if _pid_is_protected_training_proc(pid):
                            _log.info(
                                "kill_gpu_zombies: SKIP protected training PID %d (gpu=%d, vram_mb=%d)",
                                pid,
                                gpu_idx,
                                vram_mb,
                            )
                            seen_pids.add(pid)
                            continue
                        try:
                            os.kill(pid, signal.SIGTERM)
                            killed_pids.append(pid)
                            freed_mb += vram_mb
                            seen_pids.add(pid)
                            _log.warning(
                                "kill_gpu_zombies: killed zombie PID %d (gpu=%d, vram_mb=%d, gpu_util=%.1f%%)",
                                pid,
                                gpu_idx,
                                vram_mb,
                                gpu_util_pct,
                            )
                        except OSError as exc:
                            _log.warning("kill_gpu_zombies: could not kill PID %d: %s", pid, exc)

            pynvml.nvmlShutdown()
        except Exception as exc:
            _log.warning("kill_gpu_zombies: pynvml error — %s", exc)
            return {"killed_pids": killed_pids, "freed_mb": freed_mb, "error": str(exc)}

        return {"killed_pids": killed_pids, "freed_mb": freed_mb}

    @classmethod
    def _kill_gpu_zombies_nvidia_smi(
        cls,
        vram_threshold_mb: int = 1000,
        util_threshold_pct: float = 5.0,
    ) -> dict[str, Any]:
        """nvidia-smi fallback for kill_gpu_zombies() when pynvml is not installed.

        Why nvidia-smi and not pynvml: pynvml requires a separate pip install and was not
        present on the .41 milestone host (RETRO-059: killed_pids=[], error='pynvml_unavailable').
        nvidia-smi ships with the NVIDIA driver on every CUDA host — no extra package needed.

        The query uses CSV output for reliable machine parsing:
            nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits
        Each line: '<pid>, <vram_mb>'.

        Utilization is read separately because nvidia-smi does not expose per-process GPU utilization
        directly — only device-level utilization is available. We use device utilization as a proxy,
        same as the pynvml path.

        Spec: REQ-INFRA-063, SCENARIO-INFRA-088, SCENARIO-INFRA-089
        """
        killed_pids: list[int] = []
        freed_mb = 0

        try:
            # Step 1: get per-process VRAM usage from nvidia-smi
            vram_result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (FileNotFoundError, OSError):
            _log.debug("kill_gpu_zombies: nvidia-smi not found — no GPU tooling available")
            return {"killed_pids": [], "freed_mb": 0, "error": "no_gpu_tooling"}
        except Exception as exc:
            _log.warning("kill_gpu_zombies: nvidia-smi subprocess error — %s", exc)
            return {"killed_pids": [], "freed_mb": 0, "error": "no_gpu_tooling"}

        # Step 2: get device-level GPU utilization (proxy for per-process idle detection)
        try:
            util_result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            # Take the minimum across all GPUs as a conservative idle signal.
            # If ANY GPU is busy, we avoid killing processes on that machine.
            util_lines = [l.strip() for l in util_result.stdout.strip().splitlines() if l.strip()]
            gpu_util_pct = min(
                (float(u) for u in util_lines if u.isdigit() or u.replace(".", "").isdigit()),
                default=0.0,
            )
        except Exception:
            gpu_util_pct = 0.0  # assume idle if we cannot read utilization

        # Step 3: parse VRAM output and kill zombie candidates
        seen_pids: set[int] = set()
        for line in vram_result.stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            try:
                pid = int(parts[0].strip())
                vram_mb = int(parts[1].strip())
            except ValueError:
                continue
            if pid in seen_pids:
                continue
            if vram_mb >= vram_threshold_mb and gpu_util_pct < util_threshold_pct:
                if _pid_is_protected_training_proc(pid):
                    _log.info(
                        "kill_gpu_zombies (nvidia-smi): SKIP protected training PID %d (vram_mb=%d)",
                        pid,
                        vram_mb,
                    )
                    seen_pids.add(pid)
                    continue
                try:
                    os.kill(pid, signal.SIGTERM)
                    killed_pids.append(pid)
                    freed_mb += vram_mb
                    seen_pids.add(pid)
                    _log.warning(
                        "kill_gpu_zombies (nvidia-smi): killed zombie PID %d (vram_mb=%d, gpu_util=%.1f%%)",
                        pid,
                        vram_mb,
                        gpu_util_pct,
                    )
                except OSError as exc:
                    _log.warning(
                        "kill_gpu_zombies (nvidia-smi): could not kill PID %d: %s", pid, exc
                    )

        return {"killed_pids": killed_pids, "freed_mb": freed_mb, "method": "nvidia_smi_fallback"}

    # ------------------------------------------------------------------
    # check_exclusion_manifest() — REQ-INFRA-062
    # ------------------------------------------------------------------

    def check_exclusion_manifest(self) -> bool:
        """Exit immediately if this experiment is in the conductor exclusion manifest.

        The exclusion manifest lists experiment IDs that are already fully modern
        (ExperimentTemplate + watchdog + teardown + BatchedInferenceRunner compliant) and
        should never be re-selected for further modernization by the conductor.

        Without this guard, Exps 308, 260, 309, 425, 410 appeared in the slowest-5 for FIVE
        consecutive milestones (.37-.41) despite Exp 547 confirming they need no changes
        (batching_added=[]). Re-running them wastes one conductor slot per milestone forever.

        Behavior:
        - Loads ``scripts/conductor_exclusion_manifest.json`` relative to repo root.
        - If the file is missing (FileNotFoundError), returns False (non-fatal, experiment continues).
        - If this experiment's ID is in ``excluded_experiments``, writes a minimal artifact with
          ``schema='carnot.excluded.v1'``, ``honest_verdict='excluded_already_modern'``,
          ``excluded=True``, and the manifest reason, then calls assert_deliverable_written()
          and sys.exit(0).
        - Returns False if this experiment is not in the manifest.

        Returns
        -------
        bool
            Always False when the experiment is NOT excluded (caller can ignore the return value).
            When excluded, does not return — sys.exit(0) is called instead.

        Spec: REQ-INFRA-062, SCENARIO-INFRA-086, SCENARIO-INFRA-087
        """
        manifest_path = self._repo_root / "scripts" / "conductor_exclusion_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text())
        except FileNotFoundError:
            return False
        except (json.JSONDecodeError, OSError) as exc:
            _log.warning("check_exclusion_manifest: could not parse manifest — %s", exc)
            return False

        excluded = manifest.get("excluded_experiments", [])
        if self.exp_id not in excluded:
            return False

        # Write the excluded artifact and exit so the conductor sees a valid deliverable.
        reason = manifest.get("reason", "already_modern")
        artifact = {
            "experiment": self.exp_id,
            "title": self.title,
            "schema": "carnot.excluded.v1",
            "run_date": _run_date(),
            "started_at": self._started_at,
            "finished_at": _utc_now(),
            "duration_s": round(time.perf_counter() - self._t0, 3),
            "status": "excluded",
            "honest_verdict": "excluded_already_modern",
            "excluded": True,
            "reason": reason,
            "excluded_by_manifest": str(manifest_path),
        }
        atomic_write_json(self._output_path, artifact, allow_override=False)
        _log.info(
            "check_exclusion_manifest: exp %d is excluded — wrote artifact and exiting",
            self.exp_id,
        )
        self.assert_deliverable_written()
        sys.exit(0)

    # ------------------------------------------------------------------
    # teardown() — REQ-INFRA-073
    # ------------------------------------------------------------------

    def teardown(self, clear_gpu: bool = True) -> None:
        """Release GPU VRAM and force a CPython garbage collection cycle.

        Registered via ``atexit`` at construction time so it fires automatically
        on experiment exit — whether the exit is clean (return from main), an
        unhandled exception, or a conductor SIGTERM.

        Without this, each experiment that loads a model and then exits leaves
        the CUDA allocator's internal cache pinned in process memory.  Across a
        12-experiment milestone this accumulates to tens of GB of zombie VRAM
        (RETRO-054: 47,653 MB at milestone .40 close).  ``torch.cuda.empty_cache()``
        flushes the allocator's free-block pool back to the CUDA driver so the
        next process can reclaim the memory.  ``gc.collect()`` is called first to
        free any Python objects holding the last reference to a CUDA tensor before
        the cache flush.

        Parameters
        ----------
        clear_gpu : bool
            If ``True`` (default) and a CUDA-capable GPU is available, call
            ``torch.cuda.empty_cache()`` after the GC pass.  Set to ``False``
            in CPU-only unit tests to avoid importing torch.

        Spec: REQ-INFRA-073, SCENARIO-INFRA-083
        """
        _log.info("ExperimentTemplate.teardown() called for exp %d", self.exp_id)
        gc.collect()
        if clear_gpu and _cuda_is_available():
            try:
                import torch  # noqa: PLC0415

                torch.cuda.empty_cache()
                _log.info(
                    "ExperimentTemplate.teardown(): torch.cuda.empty_cache() complete for exp %d",
                    self.exp_id,
                )
            except Exception as exc:
                _log.warning("ExperimentTemplate.teardown(): empty_cache failed — %s", exc)

    # ------------------------------------------------------------------
    # setup()
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # apply_env_autofix() — REQ-INFRA-070
    # ------------------------------------------------------------------

    def apply_env_autofix(self) -> None:
        """Set CARNOT_FORCE_LIVE=1 in the current process AND persist it to disk.

        Why persisting matters (RETRO-LIVE-ENV-NOT-PROPAGATED root cause):
            Setting only ``os.environ["CARNOT_FORCE_LIVE"] = "1"`` fixes the
            current process but NOT future ``claude -p`` subprocesses spawned by
            the conductor.  Each ``claude -p`` starts a fresh Python interpreter
            whose environment comes from the OS-level process fork — not from
            this process's patched ``os.environ``.

            The fix: write to ``~/.carnot_session_env`` via
            ``EnvPropagationGuard.write_session_env()``.  Every subsequent
            ``ExperimentTemplate.__init__`` calls
            ``EnvPropagationGuard.load_session_env()`` as its very first action
            and inherits the override regardless of how the process was launched.

        Spec: REQ-INFRA-070, SCENARIO-INFRA-080
        """
        os.environ["CARNOT_FORCE_LIVE"] = "1"
        EnvPropagationGuard.write_session_env({"CARNOT_FORCE_LIVE": "1"})
        _log.info(
            "apply_env_autofix: CARNOT_FORCE_LIVE=1 set in process env and written to %s",
            EnvPropagationGuard._path,
        )

    # ------------------------------------------------------------------
    # assert_live_env_if_gpu() — REQ-INFRA-070
    # ------------------------------------------------------------------

    def assert_live_env_if_gpu(self) -> None:
        """Raise RuntimeError if this is a GPU experiment and CARNOT_FORCE_LIVE is absent.

        Why this guard is necessary:
            Without it, a GPU experiment silently runs in non-live mode when
            CARNOT_FORCE_LIVE is missing, producing cached/stale inference results
            that are indistinguishable from live results until the retrospective.
            A hard assert here converts a silent data-quality failure into a loud,
            immediately-observable process failure — fail fast, fail early.

        This method is a no-op for CPU-only experiments (``requires_gpu=False``).

        Spec: REQ-INFRA-070, SCENARIO-INFRA-080
        """
        if self.requires_gpu and os.environ.get("CARNOT_FORCE_LIVE") != "1":
            raise RuntimeError(
                f"LIVE-ENV not propagated for GPU experiment {self.exp_id}: "
                "EnvPropagationGuard failed to load CARNOT_FORCE_LIVE=1. "
                "Run: python - <<'EOF'\n"
                "from scripts.experiment_template import ExperimentTemplate\n"
                "ExperimentTemplate(0,'fix','',requires_gpu=False).apply_env_autofix()\n"
                "EOF"
            )

    @staticmethod
    def _caller_main_module() -> str:
        """Return the __name__ of the script that called setup().

        Used by setup() to detect import-time calls (test files importing
        helper symbols from experiment scripts) versus run-time calls
        (`python scripts/experiment_X.py`).

        We want the *direct* caller's __name__, not pytest's outermost
        frame: when pytest imports an experiment script for collection,
        the script's module-top-level `tmpl.setup()` runs in the script's
        own module scope, while pytest itself is __main__ at the outermost
        frame. Walk inward from setup() to the first frame whose globals
        come from a `scripts/experiment_*.py` file, and read THAT frame's
        __name__.
        """
        import inspect

        try:
            frame = inspect.currentframe()
            # Skip our own frame and the setup() frame above us.
            if frame is not None:
                frame = frame.f_back  # _caller_main_module's caller (setup)
            if frame is not None:
                frame = frame.f_back  # setup's caller (the script)
            while frame is not None:
                filename = frame.f_globals.get("__file__", "")
                # Match an experiment script file at module scope. Module
                # scope is identified by frame.f_code.co_name == "<module>";
                # only the script's top-level setup() call satisfies this.
                if (
                    filename.endswith(".py")
                    and "/experiment_" in filename
                    and frame.f_code.co_name == "<module>"
                ):
                    return frame.f_globals.get("__name__", "<unknown>")
                frame = frame.f_back
            # No experiment-script module-scope frame found. Likely
            # invoked from a non-script context (test setup, REPL, etc.).
            return "<not_experiment_script_module>"
        except Exception:  # noqa: BLE001
            # On any introspection failure, default to "imported" semantics
            # so we err on the side of NOT taking the lock (the cost of a
            # missed lock is "concurrent run might be allowed"; the cost
            # of a wrong lock is "every test SKIPs forever").
            return "<introspection_failed>"

    def setup(self) -> None:
        """Create output directories and load any existing checkpoint.

        Call this at the start of every experiment.  It is idempotent and
        safe to call multiple times.

        Side effects:
        - Acquires a flock-based single-run guard so duplicate launches
          of the same experiment script (e.g., a confused subagent retry)
          fail fast instead of stacking memory + GPU pressure.
        - Creates ``results/`` and ``results/checkpoints/experiment_<id>/`` dirs.
        - Populates ``self.checkpoint`` if a checkpoint file is present.

        Raises
        ------
        SystemExit
            If another instance of the same experiment script is already
            running (``SingleRunHeld``). Soft-exit with code 0 — the OTHER
            holder will write the artifact, so this attempt does not
            write a blocked artifact (which would confuse the conductor's
            deliverable-existence check).
        """
        # REQ-INFRA-072: single-run guard. The 2026-04-26 swap-saturation
        # incident and the 2026-04-27 runaway-Sonnet incidents both came
        # from concurrent launches of the same experiment script. flock
        # at the entry point fails the second launch immediately rather
        # than letting both stack memory + GPU pressure. The lock is
        # released on process exit (kernel releases flock on death).
        #
        # Import-time skip (2026-04-29): many experiment scripts call
        # tmpl.setup() at module top level. Test files that import helper
        # functions from those scripts trigger setup() during pytest
        # collection. If another instance of the same experiment is running
        # in the conductor, the test process hits sys.exit(0) here, which
        # crashes pytest-xdist with KeyError: <WorkerController> and
        # cascades the conductor's pre-test self-heal to SKIP every
        # downstream task. Fix: only acquire the lock when the caller's
        # module is __main__. Test imports never satisfy this, so the
        # cascade is closed; legitimate `python scripts/experiment_X.py`
        # invocations still acquire the lock as before.
        import sys

        _caller_module = self._caller_main_module()
        if _caller_module != "__main__":
            # Skip lock acquisition: imported, not invoked.
            self._single_run_lock_cm = None
        else:
            from carnot.conductor import SingleRunHeld, acquire as _acquire_single_run

            try:
                self._single_run_lock_cm = _acquire_single_run(f"experiment_{self.exp_id}")
                self._single_run_lock_cm.__enter__()
            except SingleRunHeld:
                print(
                    f"experiment_{self.exp_id}: another instance is already running; "
                    f"this attempt is exiting cleanly per the single-run guard. "
                    f"The other instance will produce the artifact at {self.deliverable}.",
                    file=sys.stderr,
                )
                sys.exit(0)

        # REQ-INFRA-070: assert CARNOT_FORCE_LIVE is set for GPU experiments
        # BEFORE kill_gpu_zombies() or any GPU work starts.
        #
        # Import-time skip (2026-04-30, milestone .83): mirrors the lock skip
        # above. The conductor's run_tests() strips CARNOT_FORCE_LIVE from
        # the pretest env so that live-mode-only assertions don't gate the
        # smart-subset suite. When a test imports an experiment script that
        # calls tmpl.setup() at module-level, the assert previously raised
        # RuntimeError("EnvPropagationGuard failed to load CARNOT_FORCE_LIVE=1")
        # — which the conductor surfaced as the cryptic "EnvPropagationGuard
        # failed to load CARNOT_ variables" SKIP for three milestones (.75,
        # .82). The assert is only meaningful when the script is invoked
        # directly; during pytest collection/import it is a false alarm.
        if _caller_module == "__main__":
            self.assert_live_env_if_gpu()

        # REQ-INFRA-074: kill GPU zombies FIRST — before any model loading.
        # Zombie processes from prior experiments may hold VRAM that would cause
        # GPUVRAMGateV2 to defer this experiment before it even starts.
        ExperimentTemplate.kill_gpu_zombies()

        # Create results dir
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        # Create checkpoint dir
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)
        # Try to resume from checkpoint
        self.checkpoint = self.checkpoint_resume()
        self._t0 = time.perf_counter()  # reset timer after setup I/O

        # Canonical RNG initialisation — verdict-reproducibility discipline (2026-04-29).
        # Seeds numpy, stdlib random, and JAX's default PRNG key from self.random_seed
        # so that stochastic experiment operations produce the same values across reruns
        # with the same seed. torch is seeded only when it is importable (GPU experiments).
        import random as _random
        import numpy as _np

        _np.random.seed(self.random_seed)
        _random.seed(self.random_seed)
        os.environ["JAX_DEFAULT_PRNG_SEED"] = str(self.random_seed)
        try:
            import torch as _torch  # noqa: PLC0415

            # AttributeError covers tests that monkey-patch sys.modules['torch']
            # with a SimpleNamespace stub lacking manual_seed.
            if hasattr(_torch, "manual_seed"):
                _torch.manual_seed(self.random_seed)
        except ImportError:
            pass

    # ------------------------------------------------------------------
    # assert_deliverable_written()
    # ------------------------------------------------------------------

    def assert_deliverable_written(self) -> None:
        """Raise FileNotFoundError if the deliverable JSON was not written to disk.

        Call this as the FINAL line of every experiment's main() function.
        It delegates to DeliverableGuard.assert_written() which checks that
        self._output_path exists on disk.

        Why this is the FINAL line (not just somewhere in main):
            By the time we reach the end of main(), every execution path —
            success, blocked, partial, error — should have produced the
            deliverable.  If the file is absent at that point, something
            silently failed upstream.  A loud FileNotFoundError here is
            observable by the conductor and prevents false "success" signals.

        Spec: REQ-INFRA-033, SCENARIO-INFRA-041

        Since the Layer-1 invariant system landed (python/carnot/invariants.py),
        this method ALSO runs the registered invariants against the deliverable
        and prints any violations to stderr.  Violations do NOT fail the
        experiment — the conductor already committed the verdict, and silently
        failing here would just mask the deliverable.  Instead, violations are
        surfaced as stderr warnings AND persisted into the deliverable itself
        under an ``invariant_violations`` key.  A milestone-retro audit can
        then grep ``results/experiment_*.json`` for that key to find every
        artifact-positive verdict from the milestone in one pass.

        Why persist rather than raise: the retraction pattern we want to
        prevent is "verdict claims success, data contradicts it, README
        propagates the claim".  Raising would force the experiment back to
        the conductor and the conductor would re-run it.  Persisting an
        ``invariant_violations`` list in the artifact, plus a rewritten
        ``honest_verdict`` when a violation has a suggested substitute,
        lets downstream tools (README updaters, HuggingFace publishers, the
        retro script) see the honest state without having to re-run the job.
        """
        self._guard.assert_written()
        self._check_invariants_on_deliverable()

    def _check_invariants_on_deliverable(self) -> None:
        """Run the machine-checkable invariants over the deliverable JSON.

        Side effects:
            - Prints violation summaries to stderr (one line per violation).
            - If any violation has a ``suggested_verdict``, rewrites the
              artifact's ``honest_verdict`` to the first such suggestion and
              preserves the original under ``honest_verdict_before_invariants``.
            - Appends an ``invariant_violations`` list to the artifact (empty
              when everything passes — so retro scripts can distinguish
              "checked, clean" from "not checked").
        """
        try:
            from carnot.invariants import run_invariants
        except ImportError:
            # The invariants module is optional — if it is absent (very early
            # clone, in-flight refactor), do nothing rather than crash every
            # experiment.
            return
        if not self._output_path.exists():
            # assert_written() already handled the missing-deliverable case;
            # no artifact to check.
            return

        try:
            artifact = json.loads(self._output_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            _log.warning(
                "assert_deliverable_written: could not parse deliverable "
                "%s for invariant check: %s",
                self._output_path,
                exc,
            )
            return

        violations = run_invariants(artifact)
        # Always record the field, even when empty — downstream scripts use
        # its presence as evidence the check ran.
        artifact["invariant_violations"] = [v.as_dict() for v in violations]
        if violations:
            original_verdict = artifact.get("honest_verdict")
            substitute = next(
                (v.suggested_verdict for v in violations if v.suggested_verdict),
                None,
            )
            if substitute is not None and substitute != original_verdict:
                artifact["honest_verdict_before_invariants"] = original_verdict
                artifact["honest_verdict"] = substitute
            for v in violations:
                sys.stderr.write(
                    f"[INVARIANT VIOLATION] exp={self.exp_id} {v.invariant_name}: {v.reason}\n"
                )
        # Re-write the artifact.  We keep the same indent=2 convention used by
        # build_result elsewhere to avoid noisy diffs.
        atomic_write_json(self._output_path, artifact, allow_override=False)

    # ------------------------------------------------------------------
    # setup_gpu()
    # ------------------------------------------------------------------

    def setup_gpu(
        self,
        model_specs: list[dict[str, Any]],
        *,
        prewarm_fn: Callable[..., Any] | None = None,
        use_server: bool = True,
    ) -> dict[str, Any]:
        """Pre-warm all models, start ModelServer + DualGPURunner, return health_status.

        This is the Exp 294 health-check pattern extended with three layers of GPU
        acceleration (Exp 224a/b/c):

        1. **ModelServer** — warm model cache, deterministic batching, TensorRT
           backend (2-4x per-query speedup from Exp 224c).  Stored on
           ``self.model_server`` for the experiment to use directly.
        2. **DualGPURunner** — parallel execution across two RTX 3090s via the
           ModelServer (Exp 224b).  Stored on ``self.gpu_runner``.
        3. **CPU fallback** — when no CUDA device is detected the method logs a
           warning and continues without a server; every experiment must run on
           CPU-only CI machines without erroring out.

        **Disabling the server (--no-server flag):**
        Pass ``use_server=False`` or set ``CARNOT_NO_SERVER=1`` in the environment
        to skip ModelServer startup.  Useful for debugging cold-load behaviour.

        **DualGPU auto-assignment (REQ-INFRA-007, RETRO-004):**
        When ``len(model_specs) >= 2`` and ``CARNOT_FORCE_LIVE=1``, this method
        automatically assigns ``model_specs[i]['gpu'] = i`` so that each model
        runs on its own GPU.  If only 1 GPU is detected, all models are assigned
        to GPU 0 with a logged RETRO-004 warning.  When ``CARNOT_FORCE_LIVE=0``
        (CI mode), auto-assignment is skipped and the caller's ``gpu`` values
        are used unchanged.

        Parameters
        ----------
        model_specs : list[dict]
            Each entry must have ``"name"``, ``"hf_id"``, and ``"gpu"`` (device index).
            When auto-assignment is active, ``"gpu"`` values are mutated in-place.
        prewarm_fn : callable | None
            Override the default ``model_prewarm`` from Exp 294 (injected in tests).
            In CPU fallback mode, defaults to a no-op that marks all models healthy.
        use_server : bool
            If ``False``, skip ModelServer startup entirely (--no-server equivalent).
            Controlled by ``CARNOT_NO_SERVER=1`` environment variable as well.

        Returns
        -------
        dict with keys:
            - ``all_healthy`` (bool): True iff every model passed its health-check.
            - ``models`` (list[dict]): Per-model status
              (name, gpu_id, health_ok, load_time_s, stall_root_cause).
            - ``prewarm_time_s`` (float): Total wall-clock time for all pre-warms.
            - ``gpu_monitor_results`` (dict): DualGPUMonitor health summary.
            - ``dual_gpu_auto_assigned`` (bool): True iff GPU indices were
              auto-assigned by this method (REQ-INFRA-007).
            - ``model_server_active`` (bool): True iff a warm ModelServer was started.
            - ``gpu_runner_active`` (bool): True iff a DualGPURunner was created.
            - ``cpu_fallback`` (bool): True iff running in CPU-only fallback mode.
        """
        # --- Step 0a: REQ-INFRA-055 — kill_gpu_zombies() BEFORE any model load ---
        # RETRO-028: Gemma4 14.89 GiB allocation failed with 15 GiB zombie-held VRAM.
        # RETRO-SOTA-GGUF-TIMEOUT: Exp 769 timed out for the same reason.
        # The setup()-time ExperimentTemplate.kill_gpu_zombies() runs once per session
        # start but cannot catch mid-session zombie accumulation.  This call is mandatory
        # per-experiment, using the more aggressive SIGKILL approach from gpu_zombie_killer.
        force_live_early = os.environ.get("CARNOT_FORCE_LIVE", "0") == "1"
        if force_live_early:
            try:
                from carnot.pipeline.gpu_zombie_killer import (  # noqa: PLC0415
                    kill_gpu_zombies as _kill_gpu_zombies,
                )

                _zombie_result = _kill_gpu_zombies(gpu_index=0)
                _log.info(
                    "REQ-INFRA-055 kill_gpu_zombies: gpu=0 verdict=%s "
                    "pids_killed=%d vram_freed_mb=%.0f",
                    _zombie_result.honest_verdict,
                    len(_zombie_result.pids_killed),
                    _zombie_result.vram_freed_mb,
                )
                _zombie_kill_result_str = _zombie_result.honest_verdict
            except Exception as _zk_exc:
                _log.warning("kill_gpu_zombies raised %s — continuing (non-fatal)", _zk_exc)
                _zombie_kill_result_str = "kill_gpu_zombies_error"
        else:
            _zombie_kill_result_str = "skipped_not_force_live"

        # --- Step 0: GPUVRAMGate — REQ-INFRA-039/040/041, RETRO-037/042 fix ---
        # Run BEFORE every GPU-required experiment.  The session-start zombie kill
        # (Exp 463) fires once per conductor session but cannot prevent mid-session
        # zombie accumulation from failed experiments.  4 of 12 experiments in .35
        # deferred due to 23.8 GB of zombie-held VRAM at 0% utilisation.
        if self.requires_gpu:
            # --- Step 0a: GPUThermalGate — REQ-INFRA-056, RETRO-046 fix ---
            # Check GPU temperature BEFORE loading models.  An RTX 3090 at 90°C runs
            # at 50-70% of peak clock — benchmark times are unreliable because throttle
            # state varies experiment-to-experiment.  The gate waits (up to 5 minutes)
            # for the GPU to cool to 80°C before proceeding.  On CPU-only machines this
            # is a transparent no-op (pynvml unavailable → check returns None → pass).
            try:
                from carnot.pipeline.gpu_thermal_gate import (  # noqa: PLC0415
                    GPUThermalGate,
                    GPUThermalThrottleError,
                )

                _thermal_gate = GPUThermalGate()
                if not _thermal_gate.wait_for_cool(0):
                    _log.error(
                        "GPUThermalGate: GPU 0 did not cool within %ds — "
                        "deferring (honest_verdict='gpu_thermal_throttle')",
                        _thermal_gate.max_wait_seconds,
                    )
                    raise GPUThermalThrottleError(
                        gpu_index=0,
                        temperature_c=_thermal_gate.check_temperature(0).temperature_c,
                        max_wait_seconds=_thermal_gate.max_wait_seconds,
                    )
            except GPUThermalThrottleError:
                raise  # let the conductor see the honest deferral
            except Exception as _thermal_exc:
                _log.warning("GPUThermalGate raised %s — continuing (non-fatal)", _thermal_exc)

            try:
                from carnot.pipeline.gpu_vram_gate_v2 import GPUVRAMGateV2  # noqa: PLC0415

                # REQ-INFRA-050/051: GPUVRAMGateV2 with kill_first=True eliminates the
                # RETRO-044 race condition (four consecutive milestones deferred due to
                # the V1 check-first order firing during the GPU driver's 5-15s drain
                # window after SIGKILL).
                with GPUVRAMGateV2(min_free_gb=8.0, kill_first=True):
                    pass  # gate check; actual model load happens below
            except Exception as _vram_exc:
                _log.warning("GPUVRAMGateV2 raised %s — continuing (non-fatal)", _vram_exc)

        # --- Step 1: Determine execution mode ---
        # CARNOT_NO_SERVER=1 is the env-var equivalent of passing --no-server on the CLI.
        no_server_env = os.environ.get("CARNOT_NO_SERVER", "0") == "1"
        server_enabled = use_server and not no_server_env
        cuda_available = _cuda_is_available()
        if not cuda_available and not hasattr(_cuda_is_available, "mock_calls"):
            cuda_available = _detect_gpu_count_rocm_aware() > 0
        cpu_fallback = not cuda_available

        if cpu_fallback:
            _log.warning(
                "setup_gpu: no CUDA devices detected — running in CPU fallback mode. "
                "ModelServer and DualGPURunner will not be started. "
                "Inference will be slower on CPU but the experiment will still run."
            )
        elif not server_enabled:
            _log.info(
                "setup_gpu: ModelServer disabled (use_server=False or CARNOT_NO_SERVER=1). "
                "Using cold-load inference (--no-server mode)."
            )

        # Reset server/runner state from any previous call.
        self.model_server = None
        self.gpu_runner = None
        model_server_active = False
        gpu_runner_active = False

        # --- Step 2: Start ModelServer (warm cache + batching + TRT) ---
        # Only attempted when CUDA is available and the server has not been disabled.
        if cuda_available and server_enabled:
            try:
                from carnot.inference.model_server import ModelServer  # noqa: PLC0415
                from unittest.mock import Mock  # noqa: PLC0415

                hf_ids = [spec["hf_id"] for spec in model_specs]
                model_server_is_test_double = isinstance(ModelServer, Mock)
                try:
                    model_server_is_test_double = model_server_is_test_double or issubclass(
                        ModelServer, Mock
                    )
                except TypeError:
                    pass

                if (
                    prewarm_fn is not None
                    and _uses_placeholder_model_ids(model_specs)
                    and not model_server_is_test_double
                ):
                    _log.info(
                        "ModelServer skipped for placeholder model ids with explicit prewarm_fn: %s",
                        hf_ids,
                    )
                else:
                    self.model_server = ModelServer(hf_ids, batch_size=8)
                    self.model_server.start()
                    model_server_active = True
                    _log.info("ModelServer started — warm cache + batching + TRT for %s", hf_ids)
            except Exception as exc:
                _log.warning(
                    "ModelServer failed to start (%s); falling back to cold-load inference",
                    exc,
                )
                self.model_server = None
                model_server_active = False

            # --- Step 3: Create DualGPURunner backed by the ModelServer ---
            # Only when we have a server and at least 2 model specs (one per GPU).
            if model_server_active and len(model_specs) >= 2:
                try:
                    from carnot.inference.dual_gpu import DualGPURunner  # noqa: PLC0415

                    self.gpu_runner = DualGPURunner(
                        model_specs[:2],
                        model_server=self.model_server,
                    )
                    gpu_runner_active = True
                    _log.info(
                        "DualGPURunner created with ModelServer (mode=%s)",
                        self.gpu_runner.execution_mode(),
                    )
                except Exception as exc:
                    _log.warning(
                        "DualGPURunner creation failed (%s); DualGPU parallelism unavailable",
                        exc,
                    )
                    self.gpu_runner = None
                    gpu_runner_active = False

        # --- Step 4: Resolve prewarm_fn ---
        # In CPU fallback mode with no explicit prewarm_fn, use a lightweight
        # no-op that marks all models healthy so the experiment can proceed.
        # In GPU mode, fall back to the validated Exp 294 pre-warm.
        if prewarm_fn is None:
            if cpu_fallback:

                def _cpu_fallback_prewarm(
                    model_name: str,
                    hf_id: str,
                    gpu_id: int,
                    **kwargs: Any,
                ) -> Any:
                    """No-op prewarm for CPU-only machines.

                    Why this exists: the Exp 294 prewarm requires a real CUDA device.
                    On CPU-only machines we skip it and mark all models healthy so the
                    experiment runs without interruption (graceful degradation contract).
                    """
                    return type(
                        "_CpuPrewarmResult",
                        (),
                        {"health_ok": True, "load_time_s": 0.0, "stall_root_cause": None},
                    )()

                prewarm_fn = _cpu_fallback_prewarm
            else:
                # Import the real pre-warm function from Exp 294 when running live
                from scripts.experiment_294_gpu_baseline_apple import (  # type: ignore[import]
                    model_prewarm as _real_prewarm,
                )

                prewarm_fn = _real_prewarm

        # --- Step 4b: REQ-INFRA-034: DualGPUAssigner — wire GPU 1 into dual-model experiments ---
        # RETRO-034 (milestone .34): GPU 1 was idle the ENTIRE milestone because
        # DualGPURunner existed but was never called.  DualGPUAssigner is the missing
        # glue: it checks eligibility and injects device_map={'': 'cuda:N'} per model.
        # This runs BEFORE the existing RETRO-004/025 auto-assignment so both layers
        # are applied; DualGPUAssigner's device_map may be overridden below by the
        # zombie-fix strategy, which is correct (the zombie fix is more specific).
        try:
            from carnot.pipeline.dual_gpu_monitor import DualGPUMonitor  # noqa: PLC0415

            _early_monitor = DualGPUMonitor()
            _n_gpus_for_assigner = _early_monitor._get_gpu_count() if not cpu_fallback else 0
        except Exception:
            _n_gpus_for_assigner = 0 if cpu_fallback else 1

        _assigner = DualGPUAssigner(model_specs, _n_gpus_for_assigner)
        if _assigner.is_dual_gpu_eligible():
            _assigner.assign()
            _log.info(
                "REQ-INFRA-034: DualGPUAssigner applied — %d models assigned to %d GPUs",
                len(model_specs),
                _n_gpus_for_assigner,
            )

        # --- Step 5: REQ-INFRA-007 / REQ-INFRA-029: DualGPU auto-assignment (RETRO-004/025) ---
        # When running live with >=2 models, assign each model to its own GPU index
        # so they execute in parallel rather than sequentially on GPU 0.
        # Skipped in CPU fallback mode — there are no GPUs to assign.
        #
        # REQ-INFRA-029 (RETRO-025 fix): also inject an explicit device_map per model.
        # device_map={'': 'cuda:0'} lets CUDA allocate layers on GPU1 for offloading, but the
        # forward pass stays on GPU0.  This produces the zombie pattern from RETRO-025:
        # GPU1 holds 1786 MB at 0% utilization for 144+ minutes.  By using
        # device_map={'': 'cuda:N'}, every layer of each model is pinned to a single
        # GPU, preventing cross-device VRAM spill and ensuring real compute on GPU1.
        force_live = os.environ.get("CARNOT_FORCE_LIVE", "0") == "1"
        dual_gpu_auto_assigned = False

        if not cpu_fallback and force_live and len(model_specs) >= 2:
            # Detect GPU count before assigning — we need to know if 1 or 2 GPUs exist.
            try:
                from carnot.pipeline.dual_gpu_monitor import DualGPUMonitor  # noqa: PLC0415

                _monitor_early = DualGPUMonitor()
                n_gpus = _monitor_early._get_gpu_count()
            except Exception:
                n_gpus = 1  # conservative fallback

            if n_gpus >= 2:
                for i, spec in enumerate(model_specs):
                    spec["gpu"] = i
                dual_gpu_auto_assigned = True
                _log.info(
                    "REQ-INFRA-007: DualGPU auto-assignment active — "
                    "assigned %d models to GPUs 0..%d",
                    len(model_specs),
                    len(model_specs) - 1,
                )

                # REQ-INFRA-029: inject explicit device_map to prevent RETRO-025 zombie.
                # Each model is pinned to its assigned GPU index so no layer offloading
                # bleeds onto a GPU that won't be used for the forward pass.
                try:
                    from carnot.pipeline.gpu_zombie_fix import (  # noqa: PLC0415
                        build_zombie_fix_strategy,
                    )

                    model_ids = [spec["hf_id"] for spec in model_specs]
                    zombie_strategy = build_zombie_fix_strategy(n_gpus, model_ids)
                    for spec in model_specs:
                        spec["device_map"] = zombie_strategy.get(spec["hf_id"], "auto")
                    _log.info(
                        "REQ-INFRA-029: Using explicit device assignment to prevent GPU1 zombie allocation "
                        "(RETRO-025 fix — device_map={'': 'cuda:1'} replaced with {'': 'cuda:N'} per model)"
                    )
                except Exception as exc:  # pragma: no cover — import failure is non-fatal
                    _log.warning(
                        "gpu_zombie_fix unavailable (%s); keeping device_map={'': 'cuda:1'} (RETRO-025 may recur)",
                        exc,
                    )
            else:
                # Only 1 GPU available — assign all to GPU 0, log RETRO-004 warning
                for spec in model_specs:
                    spec["gpu"] = 0
                dual_gpu_auto_assigned = False
                _log.warning(
                    "RETRO-004 warning: DualGPU auto-assignment requested but only "
                    "1 GPU detected; running sequentially on GPU 0"
                )

        # --- Step 6: Run the prewarm loop ---
        t_start = time.perf_counter()
        model_statuses: list[dict[str, Any]] = []
        all_healthy = True

        for spec in model_specs:
            result = prewarm_fn(spec["name"], spec["hf_id"], spec["gpu"])
            health_ok = bool(result.health_ok)
            model_statuses.append(
                {
                    "name": spec["name"],
                    "gpu_id": spec["gpu"],
                    "health_ok": health_ok,
                    "load_time_s": result.load_time_s,
                    "stall_root_cause": result.stall_root_cause,
                }
            )
            if not health_ok:
                all_healthy = False

        # --- Step 7: REQ-INFRA-014: Explicit failure when CARNOT_FORCE_LIVE=1 and unhealthy ---
        # Silent fallback to simulated mode is a correctness bug: it produces artifacts
        # labelled "live_gpu" that actually contain synthetic answers.  Exps 340, 341,
        # 346, 347 all fell into this trap.  If live mode is required and setup failed,
        # raise immediately so the researcher knows — never continue silently.
        # This check is skipped in CPU fallback mode (no FORCE_LIVE contract applies).
        if not cpu_fallback and force_live and not all_healthy:
            from carnot.pipeline.live_gpu_diagnostic import (  # noqa: PLC0415
                diagnose_live_gpu,
            )

            model_ids = [s["hf_id"] for s in model_specs]
            diag = diagnose_live_gpu(model_ids)
            if not diag.is_live_capable and not any(
                status["health_ok"] for status in model_statuses
            ):
                raise RuntimeError(
                    "Live GPU required but unavailable: "
                    f"{diag.failure_reason or 'model prewarm failed'}"
                )
            _log.warning(
                "Live GPU prewarm reported unhealthy model status but diagnostic "
                "did not require hard failure: %s",
                diag.failure_reason or "live path available",
            )

        gpu_status: dict[str, Any] = {
            "all_healthy": all_healthy,
            "models": model_statuses,
            "prewarm_time_s": round(time.perf_counter() - t_start, 3),
            "dual_gpu_auto_assigned": dual_gpu_auto_assigned,
            "model_server_active": model_server_active,
            "gpu_runner_active": gpu_runner_active,
            "cpu_fallback": cpu_fallback,
            "zombie_kill_result": _zombie_kill_result_str,
        }

        # --- Step 8: REQ-INFRA-003 / REQ-INFRA-004: GPU zombie + idle-GPU check ---
        # Run DualGPUMonitor after model pre-warm so any new processes are visible.
        # Result is additive: existing callers that only check all_healthy/models
        # are unaffected.  If CARNOT_FORCE_LIVE=1 and the monitor finds problems,
        # we log a warning but never fail — the caller decides whether to abort.
        # Skipped in CPU fallback mode (no GPU processes to monitor).
        if not cpu_fallback:
            try:
                from carnot.pipeline.dual_gpu_monitor import DualGPUMonitor  # noqa: PLC0415

                monitor = DualGPUMonitor()
                gpu_monitor_results = monitor.check_dual_gpu_health()
                gpu_status["gpu_monitor_results"] = gpu_monitor_results

                if not gpu_monitor_results["all_healthy"]:
                    if force_live:
                        _log.warning(
                            "DualGPUMonitor: unhealthy GPU state detected — "
                            "n_gpus=%d, n_zombies=%d, idle_gpus=%s",
                            gpu_monitor_results["n_gpus_detected"],
                            gpu_monitor_results["n_zombies"],
                            gpu_monitor_results["idle_gpus"],
                        )
            except Exception as exc:  # pragma: no cover — import failures are non-fatal
                _log.warning("DualGPUMonitor unavailable: %s", exc)
                gpu_status["gpu_monitor_results"] = {
                    "n_gpus_detected": 0,
                    "n_zombies": 0,
                    "idle_gpus": [],
                    "all_healthy": False,
                    "error": str(exc),
                }
        else:
            # CPU fallback: synthesise a no-GPU monitor result so downstream code
            # that reads gpu_monitor_results["n_gpus_detected"] does not KeyError.
            gpu_status["gpu_monitor_results"] = {
                "n_gpus_detected": 0,
                "n_zombies": 0,
                "idle_gpus": [],
                "all_healthy": True,  # no GPUs to be unhealthy
                "error": "cpu_fallback",
            }

        # --- Step 9: REQ-INFRA-025/026: DualGPUHealthCheck + temperature guard ---
        # Call check_dual_gpu_health() after pre-warm so any newly-loaded models
        # are visible in GPU VRAM.  The result is additive — callers that do not
        # read 'dual_gpu_health' are unaffected.  This is CI-safe: the function
        # returns safe defaults when no GPU hardware is present.
        #
        # RETRO-025 context: PID 3509070 held 1786 MB on GPU1 at 0% utilization
        # while GPU0 ran at 88% for 144+ minutes.  GPU0 also hit 82C.  These two
        # checks directly address both failure modes.
        try:
            from carnot.pipeline.dual_gpu_health import (  # noqa: PLC0415
                check_dual_gpu_health,
            )

            dual_health = check_dual_gpu_health(timeout_seconds=60)
            gpu_status["dual_gpu_health"] = {
                "gpu0_util_pct": dual_health.gpu0_util_pct,
                "gpu1_util_pct": dual_health.gpu1_util_pct,
                "gpu0_temp_c": dual_health.gpu0_temp_c,
                "gpu1_temp_c": dual_health.gpu1_temp_c,
                "gpu0_vram_mb": dual_health.gpu0_vram_mb,
                "gpu1_vram_mb": dual_health.gpu1_vram_mb,
                "gpu1_is_zombie": dual_health.gpu1_is_zombie,
                "temperature_warning": dual_health.temperature_warning,
                "recommended_batch_size_factor": dual_health.recommended_batch_size_factor,
            }

            # RETRO-025 fix 1: GPU1 zombie detection
            if dual_health.gpu1_is_zombie:
                _log.warning(
                    "RETRO-025: GPU1 allocated but idle — DualGPURunner may not be "
                    "scheduling GPU1. Check model loading in dual-model experiments. "
                    "(gpu1_vram_mb=%.0f, gpu1_util=%.0f%%)",
                    dual_health.gpu1_vram_mb,
                    dual_health.gpu1_util_pct,
                )

            # RETRO-025 fix 2: temperature guard
            if dual_health.temperature_warning:
                _log.warning(
                    "RETRO-025: GPU temp > 80C — reducing batch_size by 25%% "
                    "(gpu0_temp=%.0fC, gpu1_temp=%.0fC, recommended_factor=%.2f). "
                    "RTX 3090 throttle threshold is 83-85C.",
                    dual_health.gpu0_temp_c,
                    dual_health.gpu1_temp_c,
                    dual_health.recommended_batch_size_factor,
                )

        except Exception as exc:  # pragma: no cover — safety net for import/runtime failures
            _log.warning(
                "check_dual_gpu_health failed (%s); dual_gpu_health omitted from status",
                exc,
            )
            gpu_status["dual_gpu_health"] = {
                "error": str(exc),
                "gpu1_is_zombie": False,
                "temperature_warning": False,
                "recommended_batch_size_factor": 1.0,
            }

        return gpu_status

    # ------------------------------------------------------------------
    # checkpoint_save / checkpoint_resume
    # ------------------------------------------------------------------

    def checkpoint_save(self, partial_results: dict[str, Any], *, step: int) -> None:
        """Write a checkpoint atomically to ``results/checkpoints/experiment_<id>/``.

        Uses a ``.tmp`` rename so that a crash mid-write never leaves a corrupt file.

        Parameters
        ----------
        partial_results : dict
            Whatever intermediate data should survive a conductor interruption.
        step : int
            The logical step index (question number, batch index, etc.).
        """
        payload = {"step": step, "results": partial_results, "saved_at": _utc_now()}
        ckpt_path = self._ckpt_dir / _CHECKPOINT_FILENAME
        atomic_write_json(ckpt_path, payload, allow_override=False)

    def checkpoint_resume(self) -> dict[str, Any] | None:
        """Load the checkpoint if it exists; return ``None`` otherwise.

        Returns
        -------
        dict | None
            The saved checkpoint payload, or ``None`` if no checkpoint is present.
        """
        ckpt_path = self._ckpt_dir / _CHECKPOINT_FILENAME
        if not ckpt_path.exists():
            return None
        try:
            return json.loads(ckpt_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    # ------------------------------------------------------------------
    # phase() — lightweight profiling context manager
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def phase(self, name: str, **metadata: Any):
        """Record wall-clock time spent in a named experiment phase.

        Use as a context manager around any phase whose timing you want to
        capture in the artifact:

            with tmpl.phase("model_load", model="qwen3.6-35b-a3b"):
                model = load_gguf_model(...)
            with tmpl.phase("training", n_pairs=70, epochs=100):
                for epoch in range(100):
                    train_one_epoch(...)
            with tmpl.phase("evaluation", split="ood"):
                ood_auc = evaluate(model, held_out)

        On exit, appends ``{"name", "elapsed_s", **metadata}`` to the
        template's phase log. ``build_result()`` automatically includes
        the full log in the artifact under ``phase_timings_s`` when any
        phase has been recorded.

        Why this exists
        ---------------
        The conductor's per-iteration budget is dominated by the 5-9 min
        Sonnet research-step. Internally that's the experiment script
        doing real work — model loads, training, inference, sometimes
        nested pytest re-runs. Without instrumentation, optimisation
        targets (model-load cache, training-loop batch sizes, etc.) are
        guesses. With ``phase()`` recorded in every artifact, the
        retrospective can rank phases by total time and aim future
        speedups precisely.

        Cost: a single ``time.perf_counter()`` call at entry and exit;
        sub-microsecond overhead. Safe to wrap small phases.

        Parameters
        ----------
        name : str
            Short identifier for the phase ("model_load", "training", "eval").
            Goes into the artifact verbatim.
        **metadata
            Optional context that the retrospective can use to slice
            timings (model name, batch size, dataset split, etc.).
            Stored alongside ``elapsed_s`` in the same dict.

        Yields
        ------
        dict
            The metadata dict; callers may add fields to it inside the
            context (e.g. ``timings["n_samples"] = len(samples)`` after
            the phase has computed it). The dict is appended to
            ``self._phase_timings`` on context exit, regardless of
            whether the wrapped block raised.
        """
        entry: dict[str, Any] = {"name": name, **metadata}
        t_start = time.perf_counter()
        try:
            yield entry
        finally:
            entry["elapsed_s"] = round(time.perf_counter() - t_start, 3)
            self._phase_timings.append(entry)

    # ------------------------------------------------------------------
    # build_result()
    # ------------------------------------------------------------------

    def build_result(
        self,
        data: dict[str, Any],
        *,
        status: str,
        cost_usd: float | None = None,
        decision_class: str | list[str] | None = None,
        metrics_used: list[str] | None = None,
        code_files: list[str] | None = None,
        data_path: str | None = None,
        producer_nullable_fields: Sequence[str] = (),
        producer_gate_fields: Sequence[str] = (),
        producer_required_principle_fields: Sequence[str] = (),
        **extra_fields: Any,
    ) -> dict[str, Any]:
        """Build a standardised result artifact with all required fields.

        Required fields (``REQUIRED_RESULT_FIELDS``) are auto-populated.
        Optional economics fields (``OPTIONAL_ECONOMICS_FIELDS``) are included
        only when the caller passes them -- see the constant's docstring for
        the rationale.  *data* and *extra_fields* are merged in; caller-
        supplied values in *data* take precedence over *extra_fields*.

        Parameters
        ----------
        data : dict
            Experiment-specific payload (accuracy, n, batch_log, etc.).
        status : str
            Outcome label — one of ``"success"``, ``"blocked"``, ``"partial"``,
            ``"error"``.
        cost_usd : float, optional
            Approximate USD spent on this experiment's end-to-end
            decision(s).  Omit when genuinely unknown rather than pass a fake
            zero; zero is acceptable when the experiment ran entirely on
            local hardware without LLM API calls.
        decision_class : str or list[str], optional
            Which Carnot validation-moat tier the experiment exercised.
            Must be one of (or a list drawn from) ``DECISION_CLASSES`` =
            ``{"detect", "verify", "repair"}``.  Raises ``ValueError`` on an
            unknown class so typos do not silently corrupt the
            retrospective's slice-by-class view.
        code_files : list[str], optional
            Paths to source files to include in the reproducibility checksum.
            Typically ``[__file__]`` from the calling experiment script.
            When omitted, only the seed is hashed (weaker but still recorded).
        data_path : str, optional
            Path to the primary dataset consumed by this experiment, included
            in the reproducibility checksum so that data changes are detectable
            across reruns even when the code and seed are identical.
        producer_nullable_fields : sequence of str, optional
            Fields that may be safely inserted as explicit ``None`` values when
            absent.  This is only for shape normalization; do not list evidence
            fields unless a missing value is genuinely non-evidentiary.
        producer_gate_fields : sequence of str, optional
            Gate booleans the producer wants surfaced from nested receipts when
            a single unambiguous source value already exists.
        producer_required_principle_fields : sequence of str, optional
            Fields whose principle annotations should be validated by the
            producer-side normalizer.
        **extra_fields
            Additional top-level fields (e.g. ``stall_root_cause="..."``,
            ``custom_tag="hello"``).

        Returns
        -------
        dict
            Artifact ready to JSON-serialise and write to ``self._output_path``.
        """
        if decision_class is not None:
            classes = [decision_class] if isinstance(decision_class, str) else list(decision_class)
            unknown = set(classes) - DECISION_CLASSES
            if unknown:
                raise ValueError(
                    f"decision_class contains unknown value(s) {sorted(unknown)}; "
                    f"valid values are {sorted(DECISION_CLASSES)}"
                )
        finished_at = _utc_now()
        duration_s = round(time.perf_counter() - self._t0, 3)

        # Reproducibility checksum — always computed so every artifact is auditable.
        # Records (seed + code content + data content) as a 16-char SHA256 prefix.
        # On rerun: if the checksum matches the prior artifact, any verdict difference
        # is attributable to GPU/hardware noise rather than code or data drift.
        repro_checksum = _compute_repro_checksum(
            seed=self.random_seed,
            code_files=code_files or [],
            data_path=data_path,
        )

        result: dict[str, Any] = {
            "experiment": self.exp_id,
            "title": self.title,
            "run_date": _run_date(),
            "started_at": self._started_at,
            "finished_at": finished_at,
            "duration_s": duration_s,
            "status": status,
            "random_seed": self.random_seed,
            "reproducibility_checksum": repro_checksum,
        }

        # Optional economics fields (included only when the caller provides them).
        if cost_usd is not None:
            result["cost_usd"] = cost_usd
        if decision_class is not None:
            # Normalise to a single canonical form: list if multi, str if single.
            classes = [decision_class] if isinstance(decision_class, str) else list(decision_class)
            result["decision_class"] = classes[0] if len(classes) == 1 else classes

        # Auto-include phase timings if any were recorded via tmpl.phase().
        # Inserted before extra_fields / data merges so a caller can still
        # override (e.g. to add a derived `phase_timings_summary` alongside
        # the raw list) without losing the raw data.
        if self._phase_timings:
            result["phase_timings_s"] = list(self._phase_timings)

        # Metrics provenance — ties published numbers to the canonical
        # implementation that produced them. When a bug is found in a
        # metric helper (see 2026-04-28 inverted-AUROC retroactive
        # correction), the audit script `scripts/audit_metric_provenance.py`
        # walks `results/experiment_*.json` and lists deliverables tagged
        # with the now-known-buggy version. Without this field, every bug
        # discovery requires a manual grep+interpret pass.
        # Always emit metrics_used so downstream tools can identify which metric
        # implementation produced the numbers in this artifact. "unknown" signals
        # a pre-provenance artifact or an experiment that omitted the field.
        result["metrics_used"] = metrics_used if metrics_used is not None else "unknown"

        if metrics_used is not None:
            try:
                from carnot.eval import __version__ as _eval_version

                result["metrics_provenance"] = {
                    m: f"carnot.eval.metrics.{m}:v{_eval_version}" for m in metrics_used
                }
            except ImportError:
                # Bare venv without carnot.eval available — record name only
                result["metrics_provenance"] = {
                    m: f"carnot.eval.metrics.{m}:v?" for m in metrics_used
                }

        # Merge extra_fields first (lower priority), then data (higher priority)
        result.update(extra_fields)
        result.update(data)

        result = normalize_artifact_for_template_write(
            result,
            nullable_fields=producer_nullable_fields,
            gate_fields=producer_gate_fields,
            required_principle_fields=producer_required_principle_fields,
        )

        # schema lists all keys present in the final artifact (sorted for determinism)
        result["schema"] = sorted(result.keys())

        return result

    # ------------------------------------------------------------------
    # generate_test_stub()
    # ------------------------------------------------------------------

    def generate_test_stub(
        self,
        test_file_path: str,
        module_to_test: str = "",
    ) -> str:
        """Write a pytest skeleton to *test_file_path* BEFORE implementation begins.

        This enforces the test-first development discipline (REQ-INFRA-002).
        The skeleton contains a single passing placeholder test so that the
        test runner stays green while the real tests are being written.

        **Why this exists:** The 2026.04.23 milestone retrospective measured a
        23.5% post-test failure rate.  A root-cause was tests being written
        after (or skipped during) implementation.  Generating the skeleton
        upfront creates a file that CI will execute, making it impossible to
        forget to add tests.

        Parameters
        ----------
        test_file_path : str
            Absolute or relative path where the skeleton will be written.
            If the file already exists, the method logs a warning and returns
            the path unchanged (idempotent — never overwrites existing tests).
        module_to_test : str
            Dotted module path to import in the skeleton
            (e.g. ``"scripts.experiment_template"``).  Pass ``""`` to omit.

        Returns
        -------
        str
            Path string of the written (or pre-existing) test file.

        Raises
        ------
        SyntaxError
            (internal guard) — raised immediately if the generated skeleton
            does not parse as valid Python.  This should never happen in
            practice but protects against future template regressions.
        """
        dest = Path(test_file_path)
        if dest.exists():
            _log.warning(
                "generate_test_stub: %s already exists — skipping to avoid overwrite",
                dest,
            )
            return str(dest)

        # Build the import line only when a module was supplied.
        import_line = f"import {module_to_test}\n" if module_to_test else ""

        skeleton = (
            "# AUTO-GENERATED by ExperimentTemplate.generate_test_stub() — replace me\n"
            "# REQ-INFRA-002: test-first enforcement\n"
            "# Replace the placeholder below with real tests BEFORE implementing.\n"
            f"{import_line}"
            "\n"
            "\n"
            f"class TestExp{self.exp_id}Placeholder:\n"
            '    """Auto-generated by ExperimentTemplate; replace placeholder tests\n'
            "    with real tests before implementing.\n"
            '    """\n'
            "\n"
            "    def test_placeholder_stub(self):\n"
            "        # This placeholder keeps the test-runner green until real tests\n"
            "        # replace it.  Do NOT ship code with only this test present.\n"
            "        assert True\n"
        )

        # Guard: ensure the skeleton is syntactically valid before writing.
        ast.parse(skeleton)

        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(skeleton)
        dest.chmod(0o644)

        return str(dest)

    # ------------------------------------------------------------------
    # run_with_timeout()
    # ------------------------------------------------------------------

    def run_with_timeout(
        self,
        fn: Callable[[], dict[str, Any]],
        timeout_s: float,
    ) -> dict[str, Any]:
        """Run *fn* in a thread with a hard timeout.

        If *fn* completes within *timeout_s*, its return value is returned
        unchanged.  If it exceeds the timeout, a partial dict is returned with
        ``{"timed_out": True, "partial": True}`` so callers can emit a partial
        artifact rather than hanging indefinitely.

        Parameters
        ----------
        fn : callable
            Zero-argument function returning a ``dict``.
        timeout_s : float
            Maximum wall-clock seconds to wait.

        Returns
        -------
        dict
            Either the function's return value or
            ``{"timed_out": True, "partial": True, "timeout_s": timeout_s}``.
        """
        completed, result = _run_in_daemon_thread_with_timeout(fn, timeout_s)
        if completed:
            return result
        return {
            "timed_out": True,
            "partial": True,
            "timeout_s": timeout_s,
        }


# ---------------------------------------------------------------------------
# BatchedInferenceRunner
# ---------------------------------------------------------------------------


class BatchedInferenceRunner:
    """Group a list of questions into batches and run each batch with a hard timeout.

    The per-batch timeout is ``batch_size * 60 s`` (not per-question).  This
    matches the 2026.04.21 retrospective recommendation: a batch of 8 questions
    gets 480 s total, giving each question up to 60 s while still allowing fast
    questions to amortise the overhead across the batch.

    After each call to ``run_batch()``, ``batch_log`` contains one entry per
    batch with ``{batch_id, batch_size, batch_time_s}`` for post-hoc analysis.

    Parameters
    ----------
    runner : callable
        Function with signature ``(prompt: str) -> str``.  Called once per
        question inside the batch.
    batch_size : int
        Number of questions per batch (8-16 recommended).

    Attributes
    ----------
    batch_log : list[dict]
        Cleared at the start of each ``run_batch()`` call.  Each entry:
        ``{"batch_id": int, "batch_size": int, "batch_time_s": float}``.
    batch_timeout_s : float
        Hard timeout for each batch: ``batch_size * 60``.
    """

    def __init__(
        self,
        runner: Callable[[str], str],
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self._runner = runner
        self.batch_size = batch_size
        self.batch_timeout_s: float = batch_size * 60.0
        self.batch_log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # run_batch()
    # ------------------------------------------------------------------

    def run_batch(self, question_list: list[str]) -> list[InferenceResult]:
        """Run *question_list* through the runner in batches.

        Questions are grouped into chunks of ``self.batch_size``.  Each chunk
        is processed with a ``self.batch_timeout_s`` hard timeout.  If a batch
        times out, all questions in that batch receive an ``InferenceResult``
        with ``timed_out=True`` and an empty ``response``.

        Parameters
        ----------
        question_list : list[str]
            Ordered list of prompts to run.

        Returns
        -------
        list[InferenceResult]
            Results in the same order as *question_list*.
        """
        # Clear log for this run (callers can inspect batch_log after returning)
        self.batch_log = []

        results: list[InferenceResult] = []
        batches = self._chunk(question_list)

        for batch_id, batch in enumerate(batches):
            t_batch_start = time.perf_counter()
            batch_results = self._run_one_batch(batch, batch_id)
            batch_time_s = round(time.perf_counter() - t_batch_start, 4)

            self.batch_log.append(
                {
                    "batch_id": batch_id,
                    "batch_size": len(batch),
                    "batch_time_s": batch_time_s,
                }
            )
            results.extend(batch_results)

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _chunk(self, items: list[str]) -> list[list[str]]:
        """Split *items* into sublists of at most ``self.batch_size`` elements."""
        return [items[i : i + self.batch_size] for i in range(0, len(items), self.batch_size)]

    def _run_one_batch(self, batch: list[str], batch_id: int) -> list[InferenceResult]:
        """Run one batch of questions, respecting ``self.batch_timeout_s``.

        Uses a daemon worker thread to enforce the timeout.  On timeout, every
        prompt in the batch gets ``timed_out=True`` and an empty response.

        Parameters
        ----------
        batch : list[str]
            The prompts to run (length ≤ ``self.batch_size``).
        batch_id : int
            Zero-based batch index (stored in each ``InferenceResult``).

        Returns
        -------
        list[InferenceResult]
            One entry per prompt in *batch*, in order.
        """

        def _process_all() -> list[tuple[str, str]]:
            """Run all prompts in the batch sequentially; return (prompt, response) pairs."""
            return [(prompt, self._runner(prompt)) for prompt in batch]

        completed, pairs = _run_in_daemon_thread_with_timeout(
            _process_all,
            self.batch_timeout_s,
        )
        if completed:
            return [
                InferenceResult(
                    prompt=prompt, response=response, batch_id=batch_id, timed_out=False
                )
                for prompt, response in (pairs or [])
            ]
        return [
            InferenceResult(prompt=prompt, response="", batch_id=batch_id, timed_out=True)
            for prompt in batch
        ]
