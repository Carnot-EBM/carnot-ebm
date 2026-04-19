"""DualGPUSweepResult — metrics container for the Exp 505 retroactive harness sweep.

**Why this exists (RETRO-041):**
    GPU 1 contributed 0% forward-pass compute across all milestones despite
    DualGPUHarness existing since Exp 480.  The .37 harness_patch adoption
    covered only newly-written experiments.  All prior dual-model scripts
    continued to route all compute to GPU 0.  Exp 505 performs a retroactive
    sweep to append the DualGPUHarness injection block to every prior script
    that loads two or more models but does not already import DualGPUHarness.

**What this module provides:**
    ``DualGPUSweepResult`` — a dataclass that reports how many scripts were
    found, patched, and skipped during the sweep, plus a manifest of the
    patched script filenames.  ``patch_rate`` is a derived property so the
    conductor can immediately see what fraction of eligible scripts were fixed.

Spec: REQ-INFRA-059, REQ-INFRA-060,
      SCENARIO-INFRA-067, SCENARIO-INFRA-068
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DualGPUSweepResult:
    """Metrics from the Exp 505 retroactive DualGPUHarness sweep.

    All four constructor fields are set by the sweep logic in
    ``experiment_505_dual_gpu_harness_sweep.py``.  They are immutable after
    construction — the sweep runs once, records what happened, and exits.

    Fields
    ------
    n_scripts_found : int
        Total number of ``scripts/experiment_*.py`` files that matched the
        dual-model pattern (hf_id >= 2) AND did not already import
        DualGPUHarness.  This is the universe of scripts that were eligible
        for patching.
    n_scripts_patched : int
        Number of scripts that were successfully modified — i.e., the
        DualGPUHarness injection block was appended.  Must be <= n_scripts_found.
    n_scripts_skipped : int
        Number of found scripts that were NOT modified.  A script is skipped
        when it already contains DualGPUHarness import (means a prior sweep
        or manual edit already covered it) OR when the append write failed.
        n_scripts_patched + n_scripts_skipped == n_scripts_found (invariant).
    patch_manifest : list[str]
        Filenames (basename only, not full path) of the scripts that were
        patched.  Length == n_scripts_patched.

    Spec: REQ-INFRA-059, REQ-INFRA-060, SCENARIO-INFRA-067, SCENARIO-INFRA-068
    """

    n_scripts_found: int
    n_scripts_patched: int
    n_scripts_skipped: int
    patch_manifest: list[str] = field(default_factory=list)

    @property
    def patch_rate(self) -> float:
        """Fraction of found scripts that were successfully patched.

        Returns 0.0 when n_scripts_found == 0 to avoid ZeroDivisionError.
        Returns 1.0 when every found script was patched (n_patched == n_found).

        Why this is a property, not a constructor field:
            It is fully derived from n_scripts_patched and n_scripts_found.
            Storing it redundantly would create an invariant that could be
            violated by callers.  Deriving it on demand keeps the dataclass
            as the single source of truth.

        Spec: REQ-INFRA-060
        """
        if self.n_scripts_found == 0:
            return 0.0
        return self.n_scripts_patched / self.n_scripts_found

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict with all fields including patch_rate.

        All four dataclass fields are included plus the derived ``patch_rate``.
        This is the format written into the experiment artifact under the
        ``sweep_result`` key.

        Returns
        -------
        dict
            Keys: ``n_scripts_found``, ``n_scripts_patched``, ``n_scripts_skipped``,
            ``patch_manifest`` (list of str), ``patch_rate`` (float).

        Spec: REQ-INFRA-060, SCENARIO-INFRA-067, SCENARIO-INFRA-068
        """
        return {
            "n_scripts_found": self.n_scripts_found,
            "n_scripts_patched": self.n_scripts_patched,
            "n_scripts_skipped": self.n_scripts_skipped,
            "patch_manifest": list(self.patch_manifest),
            "patch_rate": self.patch_rate,
        }
