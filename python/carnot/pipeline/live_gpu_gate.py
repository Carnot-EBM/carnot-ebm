"""LiveGPUGate — hard gate that ALL GPU experiments must call at their start.

**Researcher summary (RETRO-015):**
    Four consecutive milestones (2026.04.24, 2026.04.25, 2026.04.26, 2026.04.27)
    produced zero live GPU results because ``CARNOT_FORCE_LIVE=1`` was never exported
    into conductor subprocess environments.  ``scripts/conductor_gpu_env.sh`` was
    created in Exp 365 (RETRO-012 fix) to export the variable, but it was never
    WIRED into the conductor launch sequence — so the variable existed in a script
    that was never sourced.

    This module is the second half of the fix:
    1. ``scripts/session_startup.sh`` now sources ``conductor_gpu_env.sh`` AND exports
       ``CARNOT_FORCE_LIVE=1`` directly (REQ-INFRA-017).
    2. ``LiveGPUGate`` (this module) is a hard gate every GPU experiment MUST call.
       If the env var is missing or the GPU is not live-capable, it raises loud errors
       immediately rather than silently degrading to simulated mode (REQ-INFRA-018).

**Why a hard gate instead of silent fallback?**
    The root cause of RETRO-015 and its predecessors (RETRO-012) was that experiments
    continued silently in simulated mode when live GPU prerequisites were not met.
    The conductor reported "success" but every result was synthetic.  A hard gate that
    raises ``RuntimeError`` forces the issue to the surface on the FIRST experiment of
    a milestone rather than at retro time.

**How to use in an experiment script:**

    ```python
    from python.carnot.pipeline.live_gpu_gate import LiveGPUGate

    # Hard gate — raises RuntimeError if not live (for scripts that should crash loud)
    LiveGPUGate.require_live(model_ids=["Qwen/Qwen3.5-0.8B"])

    # Soft gate — returns blocked artifact if not live (for scripts that prefer blocked)
    result = LiveGPUGate.require_live_or_blocked(tmpl, model_ids=[])
    if result is not None:
        # Gate failed — write the blocked artifact and exit
        import json
        json.dump(result, open(output_path, "w"))
        sys.exit(0)
    ```

Spec: REQ-INFRA-018, SCENARIO-INFRA-019, SCENARIO-INFRA-020, SCENARIO-INFRA-021
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from carnot.pipeline.live_gpu_diagnostic import diagnose_live_gpu

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def build_session_startup_script(project_root: Path) -> str:  # noqa: ARG001
    """Return the shell script content for scripts/session_startup.sh.

    This is the canonical definition of what the startup script must contain.
    Call ``check_session_startup_exists()`` to verify the file is present on disk.

    The script:
    - Uses ``set -euo pipefail`` (strict mode) so errors are not silently swallowed.
    - Sources ``conductor_gpu_env.sh`` if present (belt-and-suspenders).
    - Exports ``CARNOT_FORCE_LIVE=1`` directly as a second layer.
    - Prints a human-readable confirmation line with timestamp.

    Parameters
    ----------
    project_root : Path
        Repository root (unused in the current implementation — included for
        forward compatibility where the script path might be parameterised).

    Returns
    -------
    str
        The shell script content ready to write to ``scripts/session_startup.sh``.

    Spec: REQ-INFRA-017
    """
    return """\
#!/usr/bin/env bash
# Source this at the start of a conductor session to propagate GPU env variables.
# RETRO-015 fix: ensures CARNOT_FORCE_LIVE=1 is inherited by all subprocesses.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/conductor_gpu_env.sh" ]; then
  source "$SCRIPT_DIR/conductor_gpu_env.sh"
fi
export CARNOT_FORCE_LIVE=1
echo "[session_startup] CARNOT_FORCE_LIVE=1 exported at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
"""


def check_session_startup_exists(project_root: Path) -> bool:
    """Return True iff ``scripts/session_startup.sh`` exists under *project_root*.

    Does not verify the file contents — use ``build_session_startup_script()`` to
    generate the expected content and compare if content verification is needed.

    Parameters
    ----------
    project_root : Path
        Repository root.

    Returns
    -------
    bool
        ``True`` iff ``<project_root>/scripts/session_startup.sh`` is present.

    Spec: REQ-INFRA-017
    """
    return (project_root / "scripts" / "session_startup.sh").is_file()


# ---------------------------------------------------------------------------
# LiveGPUGate
# ---------------------------------------------------------------------------


class LiveGPUGate:
    """Hard gate for live GPU experiments.

    All methods are static — this class is a namespace, not an instance.
    Call ``require_live()`` (raises on failure) or ``require_live_or_blocked()``
    (returns blocked artifact on failure) at the start of every GPU experiment.

    Spec: REQ-INFRA-018
    """

    @staticmethod
    def check_env_var() -> bool:
        """Return True iff ``CARNOT_FORCE_LIVE=1`` is set in the environment.

        This is the first gate layer.  If this returns ``False``, the operator
        forgot to source ``scripts/session_startup.sh`` before launching the
        conductor.  Fix: ``source scripts/session_startup.sh``.

        Never raises.

        Spec: REQ-INFRA-018, SCENARIO-INFRA-019
        """
        return os.environ.get("CARNOT_FORCE_LIVE") == "1"

    @staticmethod
    def check_gpu_live(model_ids: list[str] | None = None) -> bool:
        """Return True iff the GPU stack is live-capable (all diagnostic layers pass).

        Delegates to ``diagnose_live_gpu()`` from Exp 352 and returns
        ``result.is_live_capable``.  Passes *model_ids* through to the diagnostic
        so model-loadability is also checked when IDs are supplied.

        Never raises (inherits CI-safe guarantee from ``diagnose_live_gpu()``).

        Parameters
        ----------
        model_ids : list[str] | None
            Model IDs to check loadability for.  Pass ``None`` or ``[]`` to skip
            the model-load layer (GPU driver + torch bindings are still checked).

        Returns
        -------
        bool
            ``True`` iff ``diagnose_live_gpu(model_ids).is_live_capable``.

        Spec: REQ-INFRA-018, SCENARIO-INFRA-020
        """
        if model_ids is None:
            model_ids = []
        result = diagnose_live_gpu(model_ids)
        return result.is_live_capable

    @staticmethod
    def require_live(model_ids: list[str] | None = None) -> None:
        """Raise RuntimeError if env var missing OR GPU not live.

        This is the HARD gate.  Call it at the very start of any GPU experiment
        before any model loading or inference.  It will crash loud and fast if
        prerequisites are not met, rather than letting the experiment run silently
        in simulated mode.

        Parameters
        ----------
        model_ids : list[str] | None
            Passed to ``check_gpu_live()`` for model-loadability validation.

        Raises
        ------
        RuntimeError
            If ``CARNOT_FORCE_LIVE`` is not ``"1"`` — message includes the fix command.
        RuntimeError
            If ``diagnose_live_gpu().is_live_capable`` is ``False`` — message includes
            the failure reason from the diagnostic.

        Spec: REQ-INFRA-018, SCENARIO-INFRA-019, SCENARIO-INFRA-020
        """
        if not LiveGPUGate.check_env_var():
            raise RuntimeError(
                "LiveGPUGate: CARNOT_FORCE_LIVE not set — "
                "source scripts/session_startup.sh before running GPU experiments"
            )
        if not LiveGPUGate.check_gpu_live(model_ids):
            raise RuntimeError(
                "LiveGPUGate: GPU not live — is_live_capable=False "
                "(run diagnose_live_gpu() for the specific failure layer)"
            )

    @staticmethod
    def require_live_or_blocked(
        tmpl: Any,
        model_ids: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """Call require_live(); on failure return a blocked artifact instead of raising.

        This is the SOFT gate for experiments that prefer to write a ``"blocked"``
        result artifact and exit cleanly rather than crashing with an exception.

        Usage pattern::

            result = LiveGPUGate.require_live_or_blocked(tmpl, model_ids)
            if result is not None:
                json.dump(result, open(output_path, "w"))
                sys.exit(0)
            # Proceed with live GPU inference ...

        Parameters
        ----------
        tmpl : ExperimentTemplate
            The experiment template instance; used to call ``build_result()`` to
            construct the blocked artifact.
        model_ids : list[str] | None
            Passed to ``require_live()``.

        Returns
        -------
        dict | None
            A blocked artifact dict (from ``tmpl.build_result()``) if the gate
            failed, or ``None`` if both checks passed and the experiment may proceed.

        Spec: REQ-INFRA-018
        """
        try:
            LiveGPUGate.require_live(model_ids)
            return None
        except RuntimeError as exc:
            return tmpl.build_result({}, status="blocked", blocked_reason=str(exc))

    @staticmethod
    def verify_subprocess_env_propagation(
        env_var: str = "CARNOT_FORCE_LIVE",
    ) -> bool:
        """Spawn a subprocess and verify *env_var* is inherited from current env.

        This is the PROOF that the session_startup.sh fix actually works — not just
        that the script exists.  If this returns ``True``, subprocesses launched by
        the conductor WILL inherit ``CARNOT_FORCE_LIVE=1`` from the parent shell.

        Uses the current ``os.environ`` as the subprocess environment (no override),
        which mirrors exactly what the conductor does when it spawns experiment scripts.

        Parameters
        ----------
        env_var : str
            Environment variable name to check.  Defaults to ``"CARNOT_FORCE_LIVE"``.

        Returns
        -------
        bool
            ``True`` iff the subprocess's stdout contains ``"1"`` (the value of the
            variable when it is set to ``"1"``).  ``False`` if the variable is absent,
            set to a different value, or the subprocess fails.

        Never raises.

        Spec: REQ-INFRA-018, SCENARIO-INFRA-021
        """
        try:
            result = subprocess.run(
                [
                    "python3",
                    "-c",
                    f"import os; print(os.environ.get('{env_var}', ''))",
                ],
                env=os.environ,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return "1" in result.stdout
        except Exception:  # noqa: BLE001 — never raises
            return False
