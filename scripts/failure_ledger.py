"""Failed-experiment rerun-discipline mechanical enforcement.

The CLAUDE.md "Failed-Experiment Rerun Discipline" rule binds the
planner at policy level: do not re-propose a failed experiment without
explicitly addressing the suspected root cause. Three consecutive
planner runs (.66/.67/.68) confirmed the policy alone is insufficient
— the planner reads the rule as prose and applies it to design
decisions but does NOT populate the structured `prior_failures:` field
on individual tasks.

This module is the mechanical enforcement layer. The conductor calls
into it before spawning the 5-9 min Sonnet research-step. If the new
task's scope matches a previously-failed task and the YAML lacks an
adequate `prior_failures:` entry, the conductor refuses to launch and
writes a `blocked_doomed_rerun_no_root_cause` artifact instead.

Same architectural pattern as `scripts/conductor_gates.py` (the
pre-gate check shipped in commit 7f4e8125).

DESIGN CHOICES
==============

Scope-matching: a "rerun" is detected by extracting a stable
"scope signature" from the task — the slug after the experiment
number, with `-vN` version suffixes stripped. So
`exp870-sota-code-repair-v7` and `exp881-code-repair-v8-gemma4`
both produce a signature containing `code-repair`. Two task
signatures match if either contains the other as a substring,
case-insensitive, of length >= 8.

Conservative bias: false positives (blocking a legitimate
iteration) are more expensive than false negatives (letting a
doomed rerun through). The matcher is intentionally narrow.

Failure detection: a prior task counts as failed if its
honest_verdict, run through the in-process reconciler's
`map_status_label`, returns ⚠/❌ (anything but ✅ Complete).

Validation: a task with a prior-scope match is allowed if its YAML
has a `prior_failures:` entry with all four fields set non-empty:
  experiment_id, verdict, addressed_by, retire_if_same_verdict.

Anything missing → block.

NOT IN SCOPE
============

  - The exclusion-manifest write-back from a triggered
    `retire_if_same_verdict: true` is the conductor's job, not this
    module's. We surface the retirement via `record_outcome` for the
    caller to act on.
  - Learning a smarter scope-matcher from data is future work. The
    initial heuristic is mechanical and auditable.
  - This module does NOT touch closed-source vendors or external
    services. Per CLAUDE.md decentralization rule 1 (local-first),
    everything reads/writes only project-local artifacts.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Reuse the reconciler's verdict-to-label mapping. A failure is anything
# that doesn't classify as ✅ Complete in that mapping.
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from in_process_doc_reconcile import map_status_label  # type: ignore[import-not-found]


# How many characters of slug overlap to declare two scopes the same.
# A short overlap like "v" or "ising" produces too many false positives;
# long substrings like "code-repair" or "live-benchmark" are
# distinctive. Tuned conservatively.
_SCOPE_OVERLAP_MIN = 8


# Structurally-recurring per-milestone scaffolding scopes.  Every milestone
# has a preflight and a milestone retro by design — they are not "reruns"
# of failed experiments, they are scheduled audit tasks.  The .71 first-fire
# of the failure-ledger blocked Exp 917 (preflight v20) because it
# scope-matched 6 prior preflights (404, 740, 754, 767, 731 zombie-kill, plus
# 917 itself).  The matched verdicts (`preflight_v9_clean_manifest_pending`,
# `preflight_v10_patch_applied_gpu_clean`, etc.) are not failures — they are
# clean-but-found-issues preflight reports that the reconciler classified as
# ⚠ Research Finding due to verdict-token rules.  A second-order reconciler
# fix is in scope but not part of this discipline; for now, the failure-
# ledger short-circuits these scaffolding scopes so the discipline does not
# block legitimate every-milestone audit tasks.
#
# Adding a new scope here is intentional and auditable: it must be a task
# that exists *by design* in every milestone, not just one that has happened
# to ship a few times.
_RECURRING_SCAFFOLDING_SCOPES: frozenset[str] = frozenset({
    "preflight",
    "milestone-retro",
    "milestone-retrospective",
    "zombie-kill-preflight",
})


@dataclass
class LedgerEntry:
    """One historical task that ended in a failure verdict."""

    experiment_id: str  # canonicalized (lowercased)
    title: str
    verdict: str
    status_label: str   # ⚠ Blocked / ⚠ Research Finding / ❌ Failed
    scope: str          # extracted slug, version-stripped
    artifact_path: Path | None = None


@dataclass
class ValidationResult:
    """Outcome of validating a task's `prior_failures:` field."""

    valid: bool
    missing_fields: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class LedgerCheck:
    """Outcome of `is_doomed_rerun(task)`."""

    blocked: bool
    reason: str = ""
    matched_priors: list[LedgerEntry] = field(default_factory=list)


def _scope_signature(task_id_or_title: str) -> str:
    """Extract a comparable scope signature from a task id or title.

    Pulls the part after the leading `expNNN-` (or strips a leading
    `Exp NNN: ` from titles), normalizes whitespace to dashes, then
    strips ONLY a clean trailing `-vN` version suffix. Tokens after a
    version (like `-v8-gemma4`) are preserved intentionally — they
    often carry meaningful technical descriptors (the model used, the
    technique applied) that should remain part of the scope signature.

    The conservative-matcher in `_scopes_overlap` is the safety net
    for finding the shared stem ("code-repair") between scopes that
    differ in their trailing tokens.
    """
    s = task_id_or_title.lower().strip()
    # Match either `expNNN-...` or `exp NNN: ...` styles
    m = re.match(r"^exp\s*\d+[-:]\s*(.+)$", s)
    if m:
        s = m.group(1)
    # Convert whitespace and slashes to dashes BEFORE stripping versions
    s = re.sub(r"[\s/]+", "-", s)
    # Collapse repeated dashes
    s = re.sub(r"-+", "-", s)
    # Strip ONLY a clean trailing -vN (no further tokens after the
    # version). `code-repair-v7` → `code-repair`. But
    # `code-repair-v8-gemma4` and `jepa-v25-dg-prm` are preserved
    # because trailing tokens may be meaningful descriptors.
    s = re.sub(r"[-_]v\d+$", "", s)
    return s


def _lcs_length(a: str, b: str) -> int:
    """Longest common contiguous substring length, used for overlap.

    A simple DP table; bounded by `len(a) * len(b)` time. The slugs
    we operate on are short (typically <50 chars), so this is fast
    enough for inline use.

    Returns 0 if either input is empty or if the inputs are
    pathologically large (> 100K cells of DP table).
    """
    if not a or not b:
        return 0
    m, n = len(a), len(b)
    if m * n > 100_000:
        return 0
    prev = [0] * (n + 1)
    best = 0
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        ai = a[i - 1]
        for j in range(1, n + 1):
            if ai == b[j - 1]:
                curr[j] = prev[j - 1] + 1
                if curr[j] > best:
                    best = curr[j]
        prev = curr
    return best


def _scopes_overlap(a: str, b: str, min_chars: int = _SCOPE_OVERLAP_MIN) -> bool:
    """Two scope signatures match if their longest common contiguous
    substring is >= min_chars (case-insensitive).

    Substring (not subsequence) — we want `sota-code-repair` and
    `code-repair-v8-gemma4` to match on `code-repair` (11 chars),
    but we don't want `live-benchmark` and `live-cascade` to match
    on the disconnected `live-` (5 chars).

    Conservative bias: false positives (blocking a legitimate
    iteration) are more expensive than false negatives (letting a
    doomed rerun through). The 8-character minimum is intentional.
    """
    if not a or not b:
        return False
    a_l = a.lower()
    b_l = b.lower()
    if len(a_l) < min_chars or len(b_l) < min_chars:
        return False
    return _lcs_length(a_l, b_l) >= min_chars


class FailureLedger:
    """Indexed ledger of prior failed experiments.

    Loaded once per conductor iteration from project artifacts; queried
    via `is_doomed_rerun(task)` before each Sonnet spawn.
    """

    def __init__(self) -> None:
        self.entries: list[LedgerEntry] = []

    @classmethod
    def load_from_artifacts(cls, repo_root: Path) -> FailureLedger:
        """Walk results/experiment_*.json artifacts and build the ledger.

        Each artifact's `honest_verdict` is mapped via the reconciler's
        rules; anything that isn't ✅ Complete becomes a ledger entry.
        Robust to malformed artifacts (skips with no error).
        """
        ledger = cls()
        results_dir = repo_root / "results"
        if not results_dir.exists():
            return ledger
        for artifact_path in sorted(results_dir.glob("experiment_*.json")):
            try:
                data = json.loads(artifact_path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            # Some early artifacts are top-level lists, not dicts. Skip them.
            if not isinstance(data, dict):
                continue
            # Pull the experiment id and title — be tolerant of
            # different artifact shapes
            exp_num = None
            m = re.match(r"experiment_(\d+)_", artifact_path.stem)
            if m:
                exp_num = m.group(1)
            verdict = data.get("honest_verdict", "")
            # Some early artifacts (Exps 256/257/259/292/293/304/317) have a
            # dict-shaped honest_verdict like {"status": "complete",
            # "explanation": "..."}. Coerce via the inner `status` field when
            # present; otherwise skip — a non-string verdict we can't reduce
            # to a label is not a reliable failure signal.
            if isinstance(verdict, dict):
                verdict = verdict.get("status") or ""
            if not isinstance(verdict, str) or not verdict:
                continue
            label = map_status_label(verdict)
            if label == "✅ Complete":
                continue  # not a failure for ledger purposes
            title = data.get("title") or artifact_path.stem
            # Extract the canonical task id from the artifact filename
            # (e.g., experiment_870_sota_code_repair_v7.json →
            # exp870-sota-code-repair-v7)
            slug = artifact_path.stem
            slug = re.sub(r"^experiment_(\d+)_", lambda m: f"exp{m.group(1)}-", slug)
            slug = slug.replace("_", "-")
            entry = LedgerEntry(
                experiment_id=slug,
                title=title,
                verdict=verdict,
                status_label=label,
                scope=_scope_signature(slug),
                artifact_path=artifact_path,
            )
            ledger.entries.append(entry)
        return ledger

    def matching_priors(self, task: dict) -> list[LedgerEntry]:
        """Return all ledger entries whose scope matches the task's scope.

        Excludes any entry whose experiment_id matches the task's own
        id (an experiment cannot be its own prior failure).

        Short-circuits to an empty list when the target scope is one of
        the structurally-recurring per-milestone scaffolding scopes
        (preflight, milestone retro, etc.).  Those tasks are scheduled
        audit work, not reruns of failed experiments.
        """
        task_id = (task.get("id") or "").lower()
        title = task.get("title") or ""
        target_scope = _scope_signature(task_id) or _scope_signature(title)
        # Prefix match: "milestone-retro-71" starts with "milestone-retro"
        # (the trailing -71 milestone number is not a -vN suffix and
        # survives _scope_signature stripping). Same for "preflight-v20"
        # vs "preflight" — although in that case the suffix-strip already
        # collapses to "preflight", the prefix check is the durable fix.
        if any(target_scope == s or target_scope.startswith(s + "-")
               for s in _RECURRING_SCAFFOLDING_SCOPES):
            return []
        matches: list[LedgerEntry] = []
        for e in self.entries:
            if e.experiment_id.lower() == task_id:
                continue
            if _scopes_overlap(target_scope, e.scope):
                matches.append(e)
        return matches

    def is_doomed_rerun(self, task: dict) -> LedgerCheck:
        """Decide whether a task is a doomed rerun missing root-cause spec.

        Per the CLAUDE.md rule: a task whose scope matches a prior
        failure is allowed only if it carries a `prior_failures:` field
        with the four-part discipline (experiment_id + verdict +
        addressed_by + retire_if_same_verdict) populated.
        """
        priors = self.matching_priors(task)
        if not priors:
            return LedgerCheck(blocked=False, reason="no scope-matching priors")
        validation = validate_prior_failures(task)
        if validation.valid:
            return LedgerCheck(
                blocked=False,
                reason=f"prior_failures field satisfies discipline ({len(priors)} prior(s) matched)",
                matched_priors=priors,
            )
        return LedgerCheck(
            blocked=True,
            reason=(
                f"{len(priors)} prior failure(s) match this task's scope but "
                f"prior_failures field is missing or incomplete: {validation.reason}"
            ),
            matched_priors=priors,
        )


def validate_prior_failures(task: dict) -> ValidationResult:
    """Check that a task's `prior_failures:` field satisfies the four-part
    discipline.

    Returns valid=True only when the field exists, is a non-empty list,
    and every entry has experiment_id + verdict + addressed_by +
    retire_if_same_verdict populated with non-empty values.
    """
    pf = task.get("prior_failures")
    if pf is None:
        return ValidationResult(
            valid=False,
            missing_fields=["prior_failures"],
            reason="no prior_failures field on task",
        )
    if not isinstance(pf, list) or not pf:
        return ValidationResult(
            valid=False,
            missing_fields=["prior_failures (must be non-empty list)"],
            reason="prior_failures must be a non-empty list",
        )
    required = {"experiment_id", "verdict", "addressed_by", "retire_if_same_verdict"}
    for i, entry in enumerate(pf):
        if not isinstance(entry, dict):
            return ValidationResult(
                valid=False,
                missing_fields=[f"prior_failures[{i}] not a dict"],
                reason=f"prior_failures[{i}] is not a dict",
            )
        missing = []
        for fld in required:
            value = entry.get(fld)
            if value in (None, "", [], {}):
                missing.append(fld)
        if missing:
            return ValidationResult(
                valid=False,
                missing_fields=missing,
                reason=f"prior_failures[{i}] missing/empty fields: {missing}",
            )
    return ValidationResult(valid=True)
