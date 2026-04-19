"""BatchingHookRunner — pre-commit hook integration for batching enforcement.

**Why a pre-commit hook instead of documentation (RETRO-045):**
    Exp 481 documented 77 sequential-loop violations and wrote the BatchedInferenceRunner
    standard.  Despite clear documentation, the conductor continued writing new experiment
    scripts with sequential for-loops, because documentation does not prevent violations —
    it only describes what SHOULD be done.

    The pre-commit hook makes enforcement automatic: git refuses to accept a new script
    with a high-severity batching violation.  The conductor cannot commit without fixing
    the loop first.  This converts a manual review step into an automated gate, closing
    RETRO-045 permanently.

**What this module provides:**
    ``BatchingHookRunner`` — accepts a scripts directory and a list of staged file paths,
    then runs ``BatchingEnforcementAudit`` scoped to only those staged files.  Returns
    violations found in the staged diff only (not pre-existing committed violations).

    ``filter_new_violations`` — given a list of staged file paths, returns only violations
    whose ``script_path`` matches a staged file.  This implements REQ-INFRA-053 (idempotency):
    pre-existing violations in already-committed files do not block new commits.

**Idempotency contract (REQ-INFRA-053):**
    The hook only reports violations in files listed in ``staged_files``.  If a script
    already committed with violations is NOT in ``staged_files`` (meaning it was not
    re-staged for this commit), its violations are ignored.  This prevents the hook from
    blocking all future commits until all 77 pre-existing violations are fixed.

Spec: REQ-INFRA-052, REQ-INFRA-053,
      SCENARIO-INFRA-060, SCENARIO-INFRA-061
"""

from __future__ import annotations

import logging
from pathlib import Path

from carnot.pipeline.batching_audit import BatchingEnforcementAudit, BatchingViolation

_log = logging.getLogger(__name__)


class BatchingHookRunner:
    """Run BatchingEnforcementAudit scoped to a list of staged files.

    This class is the bridge between git's pre-commit hook mechanism and the
    ``BatchingEnforcementAudit`` scanner.  It ensures only violations in STAGED
    files are reported, not violations in previously committed files.

    **Why filter to staged files only:**
        The project already has 77 known violations accumulated over two milestones.
        If the hook flagged every violation on every commit, every future commit would
        fail until all 77 are fixed — infeasible without a dedicated cleanup milestone.
        By filtering to staged files only, the hook enforces the rule going forward
        without holding existing debt hostage.

    Parameters
    ----------
    scripts_dir : str
        Path to the directory containing experiment scripts (typically ``scripts/``).
        ``BatchingEnforcementAudit`` scans this directory for all ``.py`` files.
    staged_files : list[str]
        List of file paths that are staged for the current commit.  Obtained from
        ``git diff --cached --name-only --diff-filter=ACM``.  Only violations in
        files that appear in this list will be reported.

    Spec: REQ-INFRA-052, REQ-INFRA-053
    """

    def __init__(self, scripts_dir: str, staged_files: list[str]) -> None:
        self.scripts_dir = scripts_dir
        self.staged_files = staged_files

    def run(self, raise_on_violation: bool = True) -> list[BatchingViolation]:
        """Run the audit and return only violations in staged files.

        Algorithm:
        1. Run ``BatchingEnforcementAudit(scripts_dir).scan()`` to get all violations.
        2. Filter to only violations whose ``script_path`` matches a staged file.
        3. If ``raise_on_violation=True`` and any high-severity violations remain,
           log each violation and return the list (callers use the list to exit(1)).

        The filtering step is the idempotency guarantee: violations in committed files
        that are NOT being re-staged in this commit are suppressed.

        Parameters
        ----------
        raise_on_violation : bool
            When True (default), violations are logged at ERROR level so the operator
            sees them before the hook exits 1.  When False, violations are returned
            silently (useful for tests that want to inspect the list without log noise).

        Returns
        -------
        list[BatchingViolation]
            High-severity violations found in staged files.  Empty list means the
            commit is clean — the hook should exit 0.

        Spec: REQ-INFRA-052, SCENARIO-INFRA-060, SCENARIO-INFRA-061
        """
        if not self.staged_files:
            _log.debug("BatchingHookRunner: no staged files — skipping audit")
            return []

        audit = BatchingEnforcementAudit(self.scripts_dir)
        all_violations = audit.scan()
        new_violations = self.filter_new_violations(all_violations)

        if raise_on_violation and new_violations:
            for v in new_violations:
                _log.error(
                    "BATCHING VIOLATION [%s] %s:%d — %s",
                    v.severity,
                    v.script_path,
                    v.line_no,
                    v.pattern,
                )

        return new_violations

    def filter_new_violations(
        self,
        violations: list[BatchingViolation] | None = None,
    ) -> list[BatchingViolation]:
        """Return only violations whose script_path is in staged_files.

        This is the idempotency filter for REQ-INFRA-053.  Violations in files
        that were previously committed (and are not being re-staged) are ignored.

        If ``violations`` is None, the audit is run first.

        Parameters
        ----------
        violations : list[BatchingViolation] | None
            Pre-computed violation list.  If None, ``BatchingEnforcementAudit.scan()``
            is called internally.

        Returns
        -------
        list[BatchingViolation]
            Subset of violations whose ``script_path`` resolves to a path in
            ``staged_files``.  Comparison uses resolved absolute paths when possible,
            falling back to string suffix matching.

        Spec: REQ-INFRA-053
        """
        if violations is None:
            audit = BatchingEnforcementAudit(self.scripts_dir)
            violations = audit.scan()

        # Resolve staged file paths to a set of absolute paths for reliable comparison.
        # git returns relative paths from repo root; violation paths may be absolute.
        staged_resolved: set[str] = set()
        for sf in self.staged_files:
            try:
                staged_resolved.add(str(Path(sf).resolve()))
            except OSError:
                staged_resolved.add(sf)

        result: list[BatchingViolation] = []
        for v in violations:
            if not v.is_high_severity:
                continue
            try:
                vpath = str(Path(v.script_path).resolve())
            except OSError:
                vpath = v.script_path
            # Accept if either the resolved path matches or the staged file is a suffix
            # of the violation path (handles relative vs. absolute differences).
            in_staged = vpath in staged_resolved or any(
                vpath.endswith(sf) or sf.endswith(vpath) for sf in self.staged_files
            )
            if in_staged:
                result.append(v)

        return result
