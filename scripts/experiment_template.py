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

    # 3. (Optional) Pre-warm GPUs using Exp 294 pattern
    MODEL_SPECS = [
        {"name": "Qwen3.5-0.8B", "hf_id": "Qwen/Qwen3.5-0.8B", "gpu": 0},
    ]
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
      SCENARIO-VERIFY-109, SCENARIO-VERIFY-110, SCENARIO-VERIFY-111,
      SCENARIO-VERIFY-112, SCENARIO-VERIFY-113, SCENARIO-VERIFY-114,
      SCENARIO-VERIFY-115, SCENARIO-VERIFY-116,
      SCENARIO-INFRA-011, SCENARIO-INFRA-015
"""

from __future__ import annotations

import ast
import concurrent.futures
import datetime
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

_log = logging.getLogger(__name__)


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
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_date() -> str:
    """Return today's date as an 8-digit string (e.g. ``'20260414'``)."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")


def _get_repo_root() -> Path:
    """Return the repository root, honouring the ``CARNOT_REPO_ROOT`` env override."""
    override = os.environ.get("CARNOT_REPO_ROOT")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[1]


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
    ) -> None:
        self.exp_id = exp_id
        self.title = title
        self.deliverable = deliverable
        self.requires_gpu = requires_gpu
        self._repo_root: Path = repo_root if repo_root is not None else _get_repo_root()
        self.checkpoint: dict[str, Any] | None = None
        self._started_at: str = _utc_now()
        self._t0: float = time.perf_counter()

        # Set by setup_gpu() — warm inference server and dual-GPU runner.
        # None until setup_gpu() is called or when running in CPU fallback mode.
        self.model_server: Any | None = None
        self.gpu_runner: Any | None = None

        # Set by setup()
        self._ckpt_dir: Path = (
            self._repo_root / "results" / "checkpoints" / f"experiment_{exp_id}"
        )
        self._output_path: Path = self._repo_root / deliverable

    # ------------------------------------------------------------------
    # setup()
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Create output directories and load any existing checkpoint.

        Call this at the start of every experiment.  It is idempotent and
        safe to call multiple times.

        Side effects:
        - Creates ``results/`` and ``results/checkpoints/experiment_<id>/`` dirs.
        - Populates ``self.checkpoint`` if a checkpoint file is present.
        """
        # Create results dir
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        # Create checkpoint dir
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)
        # Try to resume from checkpoint
        self.checkpoint = self.checkpoint_resume()
        self._t0 = time.perf_counter()  # reset timer after setup I/O

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
        # --- Step 1: Determine execution mode ---
        # CARNOT_NO_SERVER=1 is the env-var equivalent of passing --no-server on the CLI.
        no_server_env = os.environ.get("CARNOT_NO_SERVER", "0") == "1"
        server_enabled = use_server and not no_server_env
        cuda_available = _cuda_is_available()
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

                hf_ids = [spec["hf_id"] for spec in model_specs]
                self.model_server = ModelServer(hf_ids, batch_size=8)
                self.model_server.start()
                model_server_active = True
                _log.info(
                    "ModelServer started — warm cache + batching + TRT for %s", hf_ids
                )
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

        # --- Step 5: REQ-INFRA-007: DualGPU auto-assignment (RETRO-004) ---
        # When running live with >=2 models, assign each model to its own GPU index
        # so they execute in parallel rather than sequentially on GPU 0.
        # Skipped in CPU fallback mode — there are no GPUs to assign.
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
            model_statuses.append(
                {
                    "name": spec["name"],
                    "gpu_id": spec["gpu"],
                    "health_ok": result.health_ok,
                    "load_time_s": result.load_time_s,
                    "stall_root_cause": result.stall_root_cause,
                }
            )
            if not result.health_ok:
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
            raise RuntimeError(
                f"Live GPU required but unavailable: {diag.failure_reason or 'model prewarm failed'}"
            )

        gpu_status: dict[str, Any] = {
            "all_healthy": all_healthy,
            "models": model_statuses,
            "prewarm_time_s": round(time.perf_counter() - t_start, 3),
            "dual_gpu_auto_assigned": dual_gpu_auto_assigned,
            "model_server_active": model_server_active,
            "gpu_runner_active": gpu_runner_active,
            "cpu_fallback": cpu_fallback,
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
        tmp_path = ckpt_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2))
        tmp_path.rename(ckpt_path)  # atomic on POSIX

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
    # build_result()
    # ------------------------------------------------------------------

    def build_result(
        self,
        data: dict[str, Any],
        *,
        status: str,
        **extra_fields: Any,
    ) -> dict[str, Any]:
        """Build a standardised result artifact with all required fields.

        Required fields (``REQUIRED_RESULT_FIELDS``) are auto-populated.
        *data* and *extra_fields* are merged in; caller-supplied values in
        *data* take precedence over *extra_fields*.

        Parameters
        ----------
        data : dict
            Experiment-specific payload (accuracy, n, batch_log, etc.).
        status : str
            Outcome label — one of ``"success"``, ``"blocked"``, ``"partial"``,
            ``"error"``.
        **extra_fields
            Additional top-level fields (e.g. ``stall_root_cause="..."``,
            ``custom_tag="hello"``).

        Returns
        -------
        dict
            Artifact ready to JSON-serialise and write to ``self._output_path``.
        """
        finished_at = _utc_now()
        duration_s = round(time.perf_counter() - self._t0, 3)

        result: dict[str, Any] = {
            "experiment": self.exp_id,
            "title": self.title,
            "run_date": _run_date(),
            "started_at": self._started_at,
            "finished_at": finished_at,
            "duration_s": duration_s,
            "status": status,
        }

        # Merge extra_fields first (lower priority), then data (higher priority)
        result.update(extra_fields)
        result.update(data)

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
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(fn)
        try:
            result = future.result(timeout=timeout_s)
            executor.shutdown(wait=False)
            return result
        except concurrent.futures.TimeoutError:
            # Do not wait for the thread — it may be blocking on I/O or sleep.
            # wait=False lets the executor clean up the thread lazily.
            executor.shutdown(wait=False)
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
        return [
            items[i : i + self.batch_size] for i in range(0, len(items), self.batch_size)
        ]

    def _run_one_batch(
        self, batch: list[str], batch_id: int
    ) -> list[InferenceResult]:
        """Run one batch of questions, respecting ``self.batch_timeout_s``.

        Uses a ``ThreadPoolExecutor`` to enforce the timeout.  On timeout, every
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

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_process_all)
            try:
                pairs = future.result(timeout=self.batch_timeout_s)
                return [
                    InferenceResult(
                        prompt=prompt, response=response, batch_id=batch_id, timed_out=False
                    )
                    for prompt, response in pairs
                ]
            except concurrent.futures.TimeoutError:
                return [
                    InferenceResult(
                        prompt=prompt, response="", batch_id=batch_id, timed_out=True
                    )
                    for prompt in batch
                ]
