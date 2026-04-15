"""Conductor GPU environment propagation — RETRO-012 and RETRO-014 fix.

**Why this module exists:**
    Three consecutive milestones (2026.04.29, 2026.05.06, 2026.05.20) observed that
    GPUs reported ``is_live_capable=True`` (Exp 352) yet every experiment ran in
    simulated mode.  The root cause: the research conductor never sets
    ``CARNOT_FORCE_LIVE=1`` in the subprocess environment it spawns for GPU-tagged
    experiments.  The conductor itself (``scripts/research_conductor.py``) is frozen
    and must not be modified.

    Instead, this module:
    1.  Writes ``scripts/conductor_gpu_env.sh`` — a shell script that can be *sourced*
        before any GPU experiment to inject ``CARNOT_FORCE_LIVE=1`` into the shell
        environment.  Any wrapper script, Makefile target, or CI step can source this
        file to unblock live inference without touching the conductor.
    2.  Provides ``RetroJSONEnforcer`` to audit whether experiment result JSONs exist,
        closing RETRO-014 (missing result JSONs for module-primary experiments).

**RETRO-012 fix (this module):** create ``scripts/conductor_gpu_env.sh`` and document
    how to source it before the conductor runs GPU experiments.

**RETRO-014 fix (this module):** ``RetroJSONEnforcer`` detects which experiment IDs
    are missing a result JSON so they can be flagged for human follow-up.

Spec: REQ-INFRA-015, REQ-INFRA-016,
      SCENARIO-INFRA-016, SCENARIO-INFRA-017, SCENARIO-INFRA-018
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# ConductorEnvFix dataclass
# ---------------------------------------------------------------------------


@dataclass
class ConductorEnvFix:
    """Records the outcome of creating the conductor GPU environment script.

    Fields
    ------
    env_script_path : Path
        Absolute path to ``scripts/conductor_gpu_env.sh`` that was written.
    exports : dict[str, str]
        Mapping of shell variable name to value that the script exports.
        Example: ``{"CARNOT_FORCE_LIVE": "1"}``.
    apply_cmd : str
        The shell command a caller should run to activate the exports.
        Always ``"source scripts/conductor_gpu_env.sh"`` for this script.
    is_documented : bool
        ``True`` when the script file was successfully written and contains
        the RETRO-012 comment block, making it self-documenting.
    """

    env_script_path: Path
    exports: dict[str, str]
    apply_cmd: str
    is_documented: bool


# ---------------------------------------------------------------------------
# build_conductor_env_fix
# ---------------------------------------------------------------------------

_ENV_SCRIPT_CONTENT = """\
#!/usr/bin/env bash
# Source this before launching GPU-tagged experiments.
# RETRO-012 fix: propagates CARNOT_FORCE_LIVE=1 into conductor subprocesses.
#
# Usage (in a wrapper script or Makefile):
#   source scripts/conductor_gpu_env.sh
#   python scripts/research_conductor.py
#
# Why: research_conductor.py never sets CARNOT_FORCE_LIVE in the subprocess
# environment it spawns for GPU experiments.  Sourcing this script injects
# the variable into the calling shell so child processes inherit it.
# This closed three consecutive milestones (2026.04.29, 2026.05.06, 2026.05.20)
# of idle GPUs despite hardware being ready (Exp 352: is_live_capable=True).
export CARNOT_FORCE_LIVE=1
"""


def build_conductor_env_fix(project_root: Path) -> ConductorEnvFix:
    """Create ``scripts/conductor_gpu_env.sh`` and return a record of the action.

    Why this function exists
    ------------------------
    The research conductor (``scripts/research_conductor.py``) is frozen — we
    cannot modify it.  Instead we write a shell script that *callers* source
    before running the conductor.  Sourcing the script injects
    ``CARNOT_FORCE_LIVE=1`` into the calling shell, so every child process
    (including the conductor's experiment subprocesses) inherits it.

    Parameters
    ----------
    project_root : Path
        Root of the Carnot repository.  The script is written to
        ``{project_root}/scripts/conductor_gpu_env.sh``.

    Returns
    -------
    ConductorEnvFix
        Dataclass recording the script path, the exported variables, the
        ``source`` command, and whether the script is self-documenting.
    """
    scripts_dir = project_root / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)

    script_path = scripts_dir / "conductor_gpu_env.sh"
    script_path.write_text(_ENV_SCRIPT_CONTENT)

    return ConductorEnvFix(
        env_script_path=script_path,
        exports={"CARNOT_FORCE_LIVE": "1"},
        apply_cmd="source scripts/conductor_gpu_env.sh",
        is_documented=True,
    )


# ---------------------------------------------------------------------------
# verify_env_script_exports
# ---------------------------------------------------------------------------


def verify_env_script_exports(path: Path) -> bool:
    """Return ``True`` iff ``path`` exists and contains ``export CARNOT_FORCE_LIVE=1``.

    Why the exact string ``export CARNOT_FORCE_LIVE=1`` is required
    ----------------------------------------------------------------
    ``CARNOT_FORCE_LIVE=0`` would suppress live mode.  Checking for the exact
    ``=1`` suffix ensures the variable is both exported and set to the enabling
    value.  A value of ``CARNOT_FORCE_LIVE`` without an ``=1`` suffix (e.g.
    ``export CARNOT_FORCE_LIVE``) would propagate whatever the current value is
    in the parent shell — not a reliable guarantee.

    Parameters
    ----------
    path : Path
        Path to the shell script to inspect.

    Returns
    -------
    bool
        ``True`` when the script exists and the correct export line is present.
        ``False`` otherwise (missing file, wrong value, or export absent).
    """
    if not path.exists():
        return False
    content = path.read_text()
    return "export CARNOT_FORCE_LIVE=1" in content


# ---------------------------------------------------------------------------
# RetroJSONEnforcer
# ---------------------------------------------------------------------------


class RetroJSONEnforcer:
    """Audit whether experiment result JSONs exist — RETRO-014 fix.

    **Why this class exists:**
        RETRO-014 identified that module-primary experiments (357, 358, 362)
        wrote Python modules and passed tests but produced no result JSON.
        Each missing JSON requires a partial do-over when downstream code or
        retrospective scripts need the data.

        This class provides two methods:
        - ``check_result_json_exists`` — single-experiment check
        - ``audit_missing_jsons`` — batch audit returning a list of IDs with no JSON

        The enforcer does *not* raise errors; it reports.  Human follow-up or
        a pre-commit hook is responsible for acting on the audit output.

    Spec: REQ-INFRA-016, SCENARIO-INFRA-018
    """

    def check_result_json_exists(self, exp_id: int, results_dir: Path) -> bool:
        """Return ``True`` iff at least one ``experiment_NNN_*.json`` exists.

        Parameters
        ----------
        exp_id : int
            The experiment number to check (e.g. ``357``).
        results_dir : Path
            Directory to search (usually ``{repo_root}/results``).

        Returns
        -------
        bool
            ``True`` when one or more files matching
            ``experiment_{exp_id}_*.json`` exist in ``results_dir``.
        """
        matches = list(results_dir.glob(f"experiment_{exp_id}_*.json"))
        return len(matches) > 0

    def audit_missing_jsons(
        self, exp_ids: list[int], results_dir: Path
    ) -> list[int]:
        """Return the subset of ``exp_ids`` for which no result JSON exists.

        The order of the returned list matches the order of ``exp_ids`` — IDs
        that are present in the directory are simply omitted; the rest are
        returned in their original position.

        Parameters
        ----------
        exp_ids : list[int]
            Experiment IDs to check.
        results_dir : Path
            Directory to search.

        Returns
        -------
        list[int]
            Experiment IDs (in input order) that have no matching result JSON.
        """
        return [
            eid
            for eid in exp_ids
            if not self.check_result_json_exists(eid, results_dir)
        ]
