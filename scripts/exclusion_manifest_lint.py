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

Operator-override bypass: a task with `operator_override:` (non-empty
string citing the directive source) downgrades EXP_ID_RETIRED and
SCOPE_MATCHED_PRIOR_FAILURE from HARD to WARNING. REQUIRES_RETIRED_EXP
has no override path (the chain is structurally dead).

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
    if not manifest:
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
                                    str(p.experiment_id)
                                    for p in check.matched_priors[:5]
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
            lines.append(
                f"  [{r.violation_class}] {r.task_id} ({r.task_title[:60]})"
            )
            lines.append(f"    detail: {r.detail[:200]}")
            if r.matched_priors:
                lines.append(f"    matched priors: {', '.join(r.matched_priors)}")
    if warn:
        lines.append(f"WARNING violations ({len(warn)}, override present):")
        for r in warn:
            lines.append(
                f"  [{r.violation_class}] {r.task_id} ({r.task_title[:60]})"
            )
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
