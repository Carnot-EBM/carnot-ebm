#!/usr/bin/env python3
"""Exclusion-manifest pre-emit linter for research-roadmap-next.yaml.

Layer 2 of the .208 cascade defense. CLAUDE.md "Exclusion-Manifest
Cross-Check Before Planning" rule already binds the planner at policy
level. This module mechanically enforces the rule at *activation
time*, before the conductor swaps research-roadmap-next.yaml into
research-roadmap.yaml.

Two violation classes (both hard-block by default):

1. EXP_ID_RETIRED — task `id:` parses to an experiment_id that's on
   ops/exclusion_manifest.yaml. The id is "burned" once retired; the
   planner must either pick a different id or add `operator_override:`
   citing the directive that authorizes reuse.

2. SCOPE_MATCHED_PRIOR_FAILURE — task scope matches a previously
   failed task (per FailureLedger semantics) but the YAML lacks an
   adequate `prior_failures:` entry. This is the same check
   FailureLedger.is_doomed_rerun does at task-launch time, hoisted
   to activation time so the cascade aborts earlier.

3. REQUIRES_RETIRED_EXP — task `requires:` chain references a
   retired exp_id. The chain is structurally dead because the
   conductor will GATE_BLOCK the upstream at activation. No override
   path — rewrite the chain.

4. WRONG_MECHANISM_PRECONDITION — task prompt declares a precondition
   pattern that has been retired as wrong-mechanism per a CLAUDE.md
   MANDATORY rule. First wired 2026-05-20 to catch the recurring
   KV260 + `/dev/mmcblk` host-SD-card precondition that escalated five
   consecutive milestones (.254/.256/.257/.258 plus queued-for-.259
   exp2735). Per CLAUDE.md "KV260 SSH-Not-SD-Card Discipline": KV260
   tasks must use `ssh -o ConnectTimeout=5 kria 'true'`, never host
   `/dev/mmcblk*`. HARD-block by default; no operator_override path
   for this class (the precondition produces a meaningless artifact
   regardless of intent).

5. BLOCKED_PATTERN_MATCHED — task title/prompt contains a
   `blocked_patterns:` string from an `ops/exclusion_manifest.yaml`
   `retired_extras` entry. First wired 2026-07-01 after `.469`'s planner
   emitted 3 tasks asserting a same-session-retracted premise as fact —
   neither EXP_ID_RETIRED (brand new task ids) nor
   SCOPE_MATCHED_PRIOR_FAILURE (FailureLedger only matches PAST ARTIFACT
   scope-signatures; these ids had none) would have caught it. Until
   this fix, `blocked_patterns:` entries were pure documentation — this
   is what makes them load-bearing. Same override semantics as
   SCOPE_MATCHED_PRIOR_FAILURE (a valid `prior_failures:` block or
   `operator_override:` downgrades HARD -> WARNING).

Operator-override bypass: a task with `operator_override:` (non-empty
string citing the directive source) downgrades EXP_ID_RETIRED,
SCOPE_MATCHED_PRIOR_FAILURE, and BLOCKED_PATTERN_MATCHED from HARD to
WARNING. REQUIRES_RETIRED_EXP and WRONG_MECHANISM_PRECONDITION have no
override path (structurally dead / produces meaningless artifact
regardless).

Usage (CLI):

    python scripts/exclusion_manifest_lint.py [path/to/research-roadmap-next.yaml]

Exits non-zero on any hard violations. Designed for use in the
conductor's `_activate_next_roadmap()` pre-emit check and as a manual
audit tool.

Same architectural pattern as `scripts/harness_fit_lint.py`.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_EXP_ID_RE = re.compile(r"^exp(\d+)[-_]", re.IGNORECASE)

# Wrong-mechanism precondition patterns: (board_pattern, retired_path_pattern,
# replacement_pattern, claude_md_section).
# When BOTH board_pattern AND retired_path_pattern appear in a task's prompt,
# the task is hard-blocked.
# CLAUDE.md "KV260 SSH-Not-SD-Card Discipline" (2026-05-20) is the first
# entry. Extend this list when a new wrong-mechanism precondition is
# retired by operator directive.
_WRONG_MECHANISM_PATTERNS: list[tuple[re.Pattern[str], re.Pattern[str], str, str]] = [
    (
        re.compile(r"\b(kv260|kria|Kria|KV260)\b"),
        re.compile(r"/dev/mmcblk"),
        "ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'",
        "KV260 SSH-Not-SD-Card Discipline",
    ),
]


@dataclass
class ExclusionRisk:
    """One detected violation on a task in a pre-staged roadmap."""

    task_id: str
    task_title: str
    violation_class: str  # EXP_ID_RETIRED | SCOPE_MATCHED_PRIOR_FAILURE | REQUIRES_RETIRED_EXP
    retired_exp_id: int | None
    retirement_reason: str
    has_operator_override: bool
    severity: str  # HARD | WARNING
    detail: str = ""
    matched_priors: list[str] = field(default_factory=list)


def _load_manifest_exp_ids() -> dict[int, str]:
    """Load retired exp_id -> retirement_reason map.

    Reads ops/exclusion_manifest.yaml as primary source (per the YAML
    file's own comment: "the authoritative source"). Falls back to
    scripts/conductor_exclusion_manifest.json if the YAML is missing
    or unreadable. Returns empty dict on total failure — the linter
    becomes a no-op rather than blocking activation on a manifest
    read error.
    """
    yaml_path = PROJECT_ROOT / "ops" / "exclusion_manifest.yaml"
    json_path = PROJECT_ROOT / "scripts" / "conductor_exclusion_manifest.json"
    out: dict[int, str] = {}

    if yaml_path.exists():
        try:
            with open(yaml_path) as f:
                data = yaml.safe_load(f) or {}
            for entry in data.get("retired", []):
                exp_id = entry.get("experiment_id")
                if isinstance(exp_id, int):
                    out[exp_id] = str(entry.get("reason", ""))[:200]
            if out:
                return out
        except Exception:
            pass  # fall through to JSON

    if json_path.exists():
        try:
            with open(json_path) as f:
                data = json.load(f)
            for entry in data.get("excluded", []):
                exp_id = entry.get("experiment_id")
                if isinstance(exp_id, int):
                    out[exp_id] = str(entry.get("reason", ""))[:200]
        except Exception:
            pass

    return out


def _load_blocked_patterns() -> list[tuple[str, str, str]]:
    """Load (pattern, source_id, reason) triples from
    ops/exclusion_manifest.yaml's ``retired_extras`` entries.

    Origin: 2026-07-01 outer-loop incident. `.469`'s planner ran 8 minutes
    AFTER a same-session retraction landed in known-issues.md (the FoVer
    in-domain candidate-selection-pool premise, proven a construction
    artifact) but still emitted 3 tasks asserting the retracted premise as
    fact. Neither EXP_ID_RETIRED (task ids were brand new, never
    previously retired) nor SCOPE_MATCHED_PRIOR_FAILURE (FailureLedger only
    matches PAST ARTIFACT scope-signatures; these task ids had no prior
    artifact) would have caught it — an outer-loop session had to
    hand-patch the live roadmap after the fact.

    `retired_extras` entries already carry a `blocked_patterns:` field
    (written by past retirement-reporting scripts) that was, until this
    fix, pure documentation — nothing in the live activation path read it.
    This function is what makes it load-bearing: free-text scope
    descriptions curated by a human/AI at retirement time, checked
    against every future draft task's title+prompt regardless of whether
    that task's id or scope-signature was ever seen before.
    """
    yaml_path = PROJECT_ROOT / "ops" / "exclusion_manifest.yaml"
    out: list[tuple[str, str, str]] = []
    if not yaml_path.exists():
        return out
    try:
        with open(yaml_path) as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return out
    for entry in data.get("retired_extras", []) or []:
        source_id = str(entry.get("id", "<unnamed>"))
        reason = str(entry.get("reason", ""))[:200]
        for pattern in entry.get("blocked_patterns", []) or []:
            pattern_str = str(pattern).strip()
            if pattern_str:
                out.append((pattern_str, source_id, reason))
    return out


def _extract_exp_id(task_id_or_str: str) -> int | None:
    """Parse `exp<N>-...` -> N. Returns None on no match."""
    if not task_id_or_str:
        return None
    match = _EXP_ID_RE.match(str(task_id_or_str))
    if match:
        return int(match.group(1))
    return None


def _has_operator_override(task: dict) -> bool:
    """A non-empty `operator_override:` field counts as override.

    The value should be a one-line string citing where the operator
    granted the override (message timestamp, known-issues.md entry,
    etc.). Bare True or empty string does NOT count — operator must
    cite the directive source so it's auditable.
    """
    val = task.get("operator_override")
    if isinstance(val, str) and len(val.strip()) >= 10:
        return True
    return False


def _extract_requires_ids(task: dict) -> list[int]:
    """Parse retired-exp_id references from `requires:` field.

    Supports two shapes:
      requires: exp2091.tier
      requires:
        - exp2091.tier
        - exp1756.bitstream
    Returns the list of integer exp_ids referenced.
    """
    requires = task.get("requires")
    if not requires:
        return []
    if isinstance(requires, str):
        items = [requires]
    elif isinstance(requires, list):
        items = [str(x) for x in requires]
    else:
        return []

    out: list[int] = []
    for item in items:
        match = re.match(r"exp(\d+)\.", item, re.IGNORECASE)
        if match:
            out.append(int(match.group(1)))
    return out


def lint(roadmap_path: Path) -> list[ExclusionRisk]:
    """Scan the pre-staged roadmap YAML for exclusion-manifest violations.

    Returns a list of ExclusionRisk records — empty list means clean.
    A HARD-severity entry in the result means activation should refuse.
    """
    if not roadmap_path.exists():
        return []

    try:
        with open(roadmap_path) as f:
            data = yaml.safe_load(f) or {}
    except Exception as exc:
        return [
            ExclusionRisk(
                task_id="<root>",
                task_title=str(roadmap_path),
                violation_class="YAML_UNREADABLE",
                retired_exp_id=None,
                retirement_reason="",
                has_operator_override=False,
                severity="HARD",
                detail=f"yaml.safe_load failed: {exc}",
            )
        ]

    tasks = data.get("tasks", []) or []
    if not tasks:
        return []

    manifest = _load_manifest_exp_ids()
    blocked_patterns = _load_blocked_patterns()
    if not manifest and not blocked_patterns:
        return []

    risks: list[ExclusionRisk] = []

    # Lazy import — failure_ledger is in scripts/, sibling module.
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from failure_ledger import FailureLedger, validate_prior_failures  # type: ignore[import-not-found]

        ledger = FailureLedger.load_from_artifacts(PROJECT_ROOT)
    except Exception:
        ledger = None  # type: ignore[assignment]
        validate_prior_failures = None  # type: ignore[assignment]

    for task in tasks:
        task_id = str(task.get("id", "<no-id>"))
        task_title = str(task.get("title", ""))[:120]
        override = _has_operator_override(task)

        # CLASS 1: task id parses to retired exp_id.
        task_exp_id = _extract_exp_id(task_id)
        if task_exp_id is not None and task_exp_id in manifest:
            risks.append(
                ExclusionRisk(
                    task_id=task_id,
                    task_title=task_title,
                    violation_class="EXP_ID_RETIRED",
                    retired_exp_id=task_exp_id,
                    retirement_reason=manifest[task_exp_id],
                    has_operator_override=override,
                    severity="WARNING" if override else "HARD",
                    detail=(
                        f"task id reuses retired exp_id {task_exp_id}; "
                        f"manifest reason: {manifest[task_exp_id][:100]}"
                    ),
                )
            )

        # CLASS 2: requires: chain references a retired exp_id.
        # No override path — the chain is structurally dead at activation.
        for req_id in _extract_requires_ids(task):
            if req_id in manifest:
                risks.append(
                    ExclusionRisk(
                        task_id=task_id,
                        task_title=task_title,
                        violation_class="REQUIRES_RETIRED_EXP",
                        retired_exp_id=req_id,
                        retirement_reason=manifest[req_id],
                        has_operator_override=False,  # no override path
                        severity="HARD",
                        detail=(
                            f"requires: references retired exp_id {req_id}; "
                            f"chain is structurally dead because conductor "
                            f"will GATE_BLOCK exp{req_id} at activation"
                        ),
                    )
                )

        # CLASS 5: task title/prompt matches a retired_extras blocked_pattern
        # (2026-07-01). Complements CLASS 2 (SCOPE_MATCHED_PRIOR_FAILURE,
        # which only fires when a PAST ARTIFACT's scope-signature matches —
        # useless for a brand-new task id with no prior artifact, which is
        # exactly what let the `.469` FoVer-in-domain incident through).
        # Same override semantics as CLASS 2: a valid prior_failures: block
        # or operator_override: downgrades HARD -> WARNING.
        haystack = f"{task_title} {task.get('prompt', '')}".lower()
        matched: list[tuple[str, str, str]] = [
            (pattern_str, source_id, reason)
            for pattern_str, source_id, reason in blocked_patterns
            if pattern_str.lower() in haystack
        ]
        if matched:
            valid_priors = False
            if validate_prior_failures is not None:
                try:
                    vr = validate_prior_failures(task)
                    valid_priors = vr.valid
                except Exception:
                    valid_priors = False
            if not valid_priors:
                risks.append(
                    ExclusionRisk(
                        task_id=task_id,
                        task_title=task_title,
                        violation_class="BLOCKED_PATTERN_MATCHED",
                        retired_exp_id=None,
                        retirement_reason="; ".join(
                            dict.fromkeys(reason for _, _, reason in matched)
                        )[:200],
                        has_operator_override=override,
                        severity="WARNING" if override else "HARD",
                        detail=(
                            f"task title/prompt matches retired_extras blocked_pattern(s): "
                            + ", ".join(
                                f"{source_id}:{pattern_str!r}"
                                for pattern_str, source_id, _ in matched[:5]
                            )
                        ),
                    )
                )

        # CLASS 4: wrong-mechanism precondition (2026-05-20).
        # Scans task prompt for joint patterns that have been retired as
        # wrong-mechanism per a CLAUDE.md MANDATORY rule. First wired for
        # KV260 + /dev/mmcblk (host SD card check is meaningless for the
        # board's state — use ssh-reachability instead). No override path:
        # the precondition produces a meaningless artifact regardless of
        # intent.
        prompt_text = str(task.get("prompt", "") or "")
        if prompt_text:
            for (
                board_pat,
                retired_path_pat,
                replacement,
                claude_md_section,
            ) in _WRONG_MECHANISM_PATTERNS:
                board_hit = board_pat.search(prompt_text) or board_pat.search(task_title)
                path_hit = retired_path_pat.search(prompt_text)
                if board_hit and path_hit:
                    risks.append(
                        ExclusionRisk(
                            task_id=task_id,
                            task_title=task_title,
                            violation_class="WRONG_MECHANISM_PRECONDITION",
                            retired_exp_id=None,
                            retirement_reason=(
                                f"CLAUDE.md '{claude_md_section}' retires "
                                f"this precondition; use '{replacement}'"
                            ),
                            has_operator_override=False,  # no override path
                            severity="HARD",
                            detail=(
                                f"task prompt contains retired precondition "
                                f"pattern '{retired_path_pat.pattern}' alongside "
                                f"board reference '{board_pat.pattern}'. "
                                f"Per CLAUDE.md '{claude_md_section}', use "
                                f"'{replacement}' instead. The retired "
                                f"precondition produces a meaningless artifact "
                                f"regardless of intent — no operator_override "
                                f"path."
                            ),
                        )
                    )

        # CLASS 3: scope-matched prior failure without prior_failures: block.
        # Reuses failure_ledger's matcher so semantics match the
        # task-launch-time check.
        if ledger is not None:
            try:
                check = ledger.is_doomed_rerun(task)
                if check.blocked:
                    # Re-check with validate_prior_failures to see if the
                    # task already carries a valid prior_failures: block.
                    valid_priors = False
                    if validate_prior_failures is not None:
                        try:
                            vr = validate_prior_failures(task)
                            valid_priors = vr.valid
                        except Exception:
                            valid_priors = False
                    if not valid_priors:
                        risks.append(
                            ExclusionRisk(
                                task_id=task_id,
                                task_title=task_title,
                                violation_class="SCOPE_MATCHED_PRIOR_FAILURE",
                                retired_exp_id=None,
                                retirement_reason=check.reason[:200],
                                has_operator_override=override,
                                severity="WARNING" if override else "HARD",
                                detail=check.reason[:200],
                                matched_priors=[
                                    str(p.experiment_id) for p in check.matched_priors[:5]
                                ],
                            )
                        )
            except Exception:
                pass  # defensive — never let linter crash kill activation

    return risks


def _format_risks(risks: list[ExclusionRisk]) -> str:
    """Pretty-print risks for log output."""
    lines = []
    hard = [r for r in risks if r.severity == "HARD"]
    warn = [r for r in risks if r.severity == "WARNING"]
    if hard:
        lines.append(f"HARD violations ({len(hard)}):")
        for r in hard:
            lines.append(f"  [{r.violation_class}] {r.task_id} ({r.task_title[:60]})")
            lines.append(f"    detail: {r.detail[:200]}")
            if r.matched_priors:
                lines.append(f"    matched priors: {', '.join(r.matched_priors)}")
    if warn:
        lines.append(f"WARNING violations ({len(warn)}, override present):")
        for r in warn:
            lines.append(f"  [{r.violation_class}] {r.task_id} ({r.task_title[:60]})")
            lines.append(f"    detail: {r.detail[:200]}")
    return "\n".join(lines)


def main() -> int:
    if len(sys.argv) > 1:
        roadmap_path = Path(sys.argv[1])
    else:
        roadmap_path = PROJECT_ROOT / "research-roadmap-next.yaml"

    if not roadmap_path.exists():
        print(f"No roadmap found at {roadmap_path} — nothing to lint")
        return 0

    risks = lint(roadmap_path)
    if not risks:
        print(f"Exclusion-manifest lint clean: {roadmap_path}")
        return 0

    print(f"Exclusion-manifest lint found {len(risks)} violation(s) in {roadmap_path}:")
    print(_format_risks(risks))
    hard_count = sum(1 for r in risks if r.severity == "HARD")
    if hard_count > 0:
        print(f"\nHARD violations: {hard_count} — activation would be REFUSED.")
        return 1
    print("\nAll violations have operator_override — activation would proceed with warnings.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
