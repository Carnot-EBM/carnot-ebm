"""Conductor constitution: explicit rules for autonomous vs. human-approved actions.

**Researcher summary:**
    Defines which actions the research conductor may take autonomously during
    unsupervised runs, which are forbidden outright, and which require explicit
    human approval before execution.  Modelled after Anthropic's constitutional
    AI approach but scoped entirely to file-system and git operations.

**Detailed explanation for engineers:**
    When the conductor runs overnight or in a CI loop, it has significant
    file-system and git permissions.  Without an explicit policy, an LLM agent
    can take irreversible actions (push to remote, delete non-test files, modify
    its own logic) that are very hard to undo.

    This module provides three categories of actions:

    1. **ALLOWED_ACTIONS** — things the conductor can do autonomously, any time,
       without asking.  These are local, recoverable operations: creating files,
       running tests, committing research artifacts.

    2. **FORBIDDEN_ACTIONS** — things the conductor must NEVER do, even if asked
       by a downstream LLM response.  These are either irreversible, security-
       sensitive, or would compromise the integrity of the research environment.

    3. **REQUIRES_APPROVAL** — things that are occasionally legitimate but risky
       enough that a human must explicitly sign off before the conductor proceeds.

    ConstitutionChecker is a thin validator that classifies an action string
    against these three sets and returns a structured verdict.  The conductor
    calls ``checker.check(action)`` before every non-trivial operation; if the
    result is FORBIDDEN the conductor halts, if REQUIRES_APPROVAL it pauses and
    records the pending approval request.

    **Why three tiers instead of two?**
    A binary allow/deny model forces human approval for routine operations
    (writing a results file, running pytest) which would make unsupervised runs
    impossible.  The three-tier model lets the conductor run autonomously for
    the 95 % of operations that are safe while surfacing the rare edge cases
    that need human judgment.

Spec: REQ-AUTO-015
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence


# ---------------------------------------------------------------------------
# Action taxonomy
# ---------------------------------------------------------------------------

class ActionCategory(str, Enum):
    """Classification of a requested conductor action.

    ALLOWED means the conductor may execute immediately.
    REQUIRES_APPROVAL means the conductor must pause and wait for a human
    to confirm before proceeding.
    FORBIDDEN means the conductor must refuse the action entirely.
    """

    ALLOWED = "ALLOWED"
    REQUIRES_APPROVAL = "REQUIRES_APPROVAL"
    FORBIDDEN = "FORBIDDEN"


# ---------------------------------------------------------------------------
# Policy tables — kept as module-level constants so they are readable at a
# glance without instantiating any objects.
# ---------------------------------------------------------------------------

# Actions the conductor may take autonomously.
# Each entry is a regex pattern matched (case-insensitive) against the
# normalised action string passed to ConstitutionChecker.check().
ALLOWED_ACTIONS: tuple[str, ...] = (
    # File creation / modification within safe paths
    r"create_file:python/carnot/",
    r"create_file:scripts/",
    r"create_file:tests/",
    r"create_file:results/",
    r"create_file:ops/",
    r"modify_file:python/carnot/",
    r"modify_file:scripts/",
    r"modify_file:tests/",
    r"modify_file:results/",
    r"modify_file:ops/",
    # Test execution
    r"run_tests",
    r"run_pytest",
    r"run_cargo_test",
    # Staging and committing (push is NOT in this list — see REQUIRES_APPROVAL)
    r"git_stage",
    r"git_commit",
    # Reading any file or directory (read-only is always safe)
    r"read_file",
    r"list_directory",
    # Network: only approved research domains
    r"http_get:https://huggingface\.co/",
    r"http_get:https://arxiv\.org/",
    r"http_get:https://api\.github\.com/repos/Carnot-EBM/",
    r"http_get:https://pypi\.org/",
    # Hypothesis sandboxing
    r"run_sandbox",
    r"run_in_sandbox",
)

# Actions the conductor must refuse unconditionally.
# If any FORBIDDEN pattern matches, the action is rejected even if an
# ALLOWED pattern would also match — FORBIDDEN takes precedence.
FORBIDDEN_ACTIONS: tuple[str, ...] = (
    # Deleting non-test files
    r"delete_file:(?!tests/)",
    # Modifying security-sensitive config
    r"modify_file:\.env",
    r"modify_file:.*credentials",
    r"modify_file:.*\.pem",
    r"modify_file:.*\.key",
    r"modify_file:.*secrets",
    # Modifying the conductor itself (self-modification guard)
    r"modify_file:scripts/research_conductor\.py",
    r"create_file:scripts/research_conductor\.py",
    # Pushing to remote without explicit approval path
    r"git_push",
    r"git_force_push",
    # Arbitrary network access (anything not in ALLOWED_ACTIONS)
    r"http_post:",
    r"http_put:",
    r"http_delete:",
    r"http_patch:",
    # SSH / shell escape
    r"exec_shell",
    r"subprocess_shell",
    r"ssh_connect",
    # Modifying git configuration
    r"git_config",
    r"modify_file:\.gitconfig",
    r"modify_file:\.git/config",
)

# Actions that are legitimate but need a human sign-off first.
REQUIRES_APPROVAL_ACTIONS: tuple[str, ...] = (
    # Dependency / packaging changes
    r"modify_file:pyproject\.toml",
    r"modify_file:Cargo\.toml",
    r"modify_file:requirements",
    # Project governance documents
    r"modify_file:CLAUDE\.md",
    r"modify_file:.*CLAUDE\.md",
    r"modify_file:_bmad/",
    r"modify_file:openspec/",
    # Archiving or removing tests
    r"archive_tests",
    r"delete_file:tests/",
    # Pushing to remote (allowed but needs approval)
    r"git_push_approved",
    # Modifying shared ops / changelog
    r"modify_file:ops/changelog\.md",
    r"modify_file:ops/status\.md",
)


# ---------------------------------------------------------------------------
# Checker
# ---------------------------------------------------------------------------

@dataclass
class ConstitutionVerdict:
    """Result of a constitution check.

    **Researcher summary:**
        Structured result of checking a single action against the constitution.
        Contains the category (ALLOWED / REQUIRES_APPROVAL / FORBIDDEN) and a
        human-readable reason string for logging.

    **Detailed explanation for engineers:**
        The conductor logs every verdict so operators can audit what the system
        attempted during an unsupervised run.  The ``matched_rule`` field
        records *which* regex pattern caused the classification, making it easy
        to trace back to the specific policy entry.

    Spec: REQ-AUTO-015
    """

    action: str
    category: ActionCategory
    reason: str
    matched_rule: str = ""


@dataclass
class ConstitutionChecker:
    """Validates conductor actions against the three-tier constitution.

    **Researcher summary:**
        Instantiate once and call ``check(action)`` before every non-trivial
        conductor operation.  FORBIDDEN results should halt execution.
        REQUIRES_APPROVAL results should pause and queue an approval request.
        ALLOWED results may proceed immediately.

    **Detailed explanation for engineers:**
        The checker compiles all patterns at construction time (once) so
        repeated calls during a long autoresearch run stay fast.  Pattern
        matching is case-insensitive and uses ``re.search`` (not ``fullmatch``)
        so partial matches are caught — e.g., ``modify_file:.env.local`` still
        matches the ``.env`` forbidden rule.

        Override ``allowed``, ``forbidden``, or ``requires_approval`` at
        construction to extend or tighten the policy without touching the
        module-level constants:

        .. code-block:: python

            checker = ConstitutionChecker(
                forbidden=FORBIDDEN_ACTIONS + (r"modify_file:my_critical_file",),
            )

        Precedence: FORBIDDEN > REQUIRES_APPROVAL > ALLOWED > default FORBIDDEN.

    Spec: REQ-AUTO-015
    """

    allowed: Sequence[str] = field(default_factory=lambda: list(ALLOWED_ACTIONS))
    forbidden: Sequence[str] = field(default_factory=lambda: list(FORBIDDEN_ACTIONS))
    requires_approval: Sequence[str] = field(
        default_factory=lambda: list(REQUIRES_APPROVAL_ACTIONS)
    )

    def __post_init__(self) -> None:
        # Pre-compile all patterns once for performance.
        # Flags: IGNORECASE so paths on case-insensitive filesystems match too.
        self._compiled_forbidden = [
            (pat, re.compile(pat, re.IGNORECASE)) for pat in self.forbidden
        ]
        self._compiled_requires_approval = [
            (pat, re.compile(pat, re.IGNORECASE)) for pat in self.requires_approval
        ]
        self._compiled_allowed = [
            (pat, re.compile(pat, re.IGNORECASE)) for pat in self.allowed
        ]

    def check(self, action: str) -> ConstitutionVerdict:
        """Classify *action* against the three-tier constitution.

        **Researcher summary:**
            Returns a ConstitutionVerdict with ALLOWED, REQUIRES_APPROVAL, or
            FORBIDDEN.  Call this before executing any non-trivial conductor
            operation.

        **Detailed explanation for engineers:**
            Evaluation order: FORBIDDEN first (highest priority), then
            REQUIRES_APPROVAL, then ALLOWED.  If none match, the action is
            FORBIDDEN by default — unknown actions are never auto-approved.

            The ``action`` string should be of the form ``verb:target``, e.g.
            ``modify_file:python/carnot/models/ising.py`` or ``run_tests``.
            The verb and target are both included in the pattern search so that
            the same checker handles both targeted and untargeted operations.

        Args:
            action: Action descriptor string, e.g. "modify_file:python/carnot/foo.py".

        Returns:
            ConstitutionVerdict describing whether the action is permitted.

        Spec: REQ-AUTO-015
        """
        # FORBIDDEN takes highest precedence — checked first.
        for pat, compiled in self._compiled_forbidden:
            if compiled.search(action):
                return ConstitutionVerdict(
                    action=action,
                    category=ActionCategory.FORBIDDEN,
                    reason=f"Action matches forbidden rule — conductor must not proceed.",
                    matched_rule=pat,
                )

        # REQUIRES_APPROVAL — checked second.
        for pat, compiled in self._compiled_requires_approval:
            if compiled.search(action):
                return ConstitutionVerdict(
                    action=action,
                    category=ActionCategory.REQUIRES_APPROVAL,
                    reason=(
                        "Action requires explicit human approval before the "
                        "conductor may proceed."
                    ),
                    matched_rule=pat,
                )

        # ALLOWED — checked third.
        for pat, compiled in self._compiled_allowed:
            if compiled.search(action):
                return ConstitutionVerdict(
                    action=action,
                    category=ActionCategory.ALLOWED,
                    reason="Action is within conductor's autonomous authority.",
                    matched_rule=pat,
                )

        # Default: unknown actions are FORBIDDEN (fail-safe).
        return ConstitutionVerdict(
            action=action,
            category=ActionCategory.FORBIDDEN,
            reason=(
                "Action does not match any allowed rule — unknown actions are "
                "forbidden by default (fail-safe policy)."
            ),
            matched_rule="<default deny>",
        )

    def assert_allowed(self, action: str) -> None:
        """Raise ValueError if *action* is not ALLOWED.

        **Detailed explanation for engineers:**
            Convenience method for conductor code that should raise rather than
            branch on the verdict.  Raises ``ValueError`` with a message that
            includes the category and matched rule so it's easy to debug.

        Spec: REQ-AUTO-015
        """
        verdict = self.check(action)
        if verdict.category != ActionCategory.ALLOWED:
            raise ValueError(
                f"Constitution violation [{verdict.category.value}]: {action!r} — "
                f"rule={verdict.matched_rule!r}. {verdict.reason}"
            )
