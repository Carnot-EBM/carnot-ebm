#!/usr/bin/env python3
"""Pre-experiment dependency audit tool — REQ-INFRA-005.

**Why this exists (NEW-002 from the 2026.04.29 retrospective):**
    Experiments frequently fail mid-run because they try to read result files from
    prior experiments that were never completed (e.g., "load results/experiment_307_
    jepa_real_training.json").  The failure happens *inside* the experiment turn, not
    before it, so the conductor wastes the full experiment turn budget before learning
    the experiment was doomed from the start.  Measurement showed ~5% wall-time overhead
    from these retry loops across a milestone.

    This tool runs *before* an experiment starts.  It parses the "EXISTING CODE TO READ
    FIRST" section of a research prompt, resolves each listed file path, and checks
    whether every file exists on disk.  When any file is missing, the conductor can emit
    a ``blocked`` artifact immediately — spending zero inference tokens — and move on
    to a task whose prerequisites are present.

**Usage:**
    python scripts/experiment_dependency_audit.py --exp-id 328 --prompt-file prompt.txt
    python scripts/experiment_dependency_audit.py --exp-id 328 --yaml-path research-roadmap.yaml

    Exit code 0 → all dependencies found.
    Exit code 1 → one or more dependencies missing (missing paths printed to stdout).

Spec: REQ-INFRA-005, SCENARIO-INFRA-007, SCENARIO-INFRA-008
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# The hardcoded placeholder path used in research-roadmap.yaml prompts.
# When found as a prefix in a listed file path, it is replaced with the
# actual project_root so the tool works correctly regardless of where the
# repo is checked out.
# ---------------------------------------------------------------------------
_ROADMAP_PLACEHOLDER = "/home/ianblenke/github.com/ianblenke/carnot"
_BRACED_PLACEHOLDER = "{project_root}"

# The heading that marks the start of the dependency list in a prompt.
_SECTION_HEADER = "EXISTING CODE TO READ FIRST:"

# A line that marks the end of the dependency section.
_TASK_MARKER = "TASK:"


# ---------------------------------------------------------------------------
# DependencyAudit
# ---------------------------------------------------------------------------


@dataclass
class DependencyAudit:
    """Result of a pre-experiment dependency check.

    Fields
    ------
    experiment_id : str
        The experiment identifier (e.g. "exp327" or "327").  Used for
        traceability in the conductor log and blocked artifacts.
    required_files : list[str]
        Absolute paths of every file listed under "EXISTING CODE TO READ
        FIRST" in the experiment prompt.
    missing_files : list[str]
        Subset of ``required_files`` that do not exist on disk at the time
        the audit ran.
    all_present : bool
        ``True`` iff ``missing_files`` is empty — i.e., the experiment may
        safely proceed (from a file-existence standpoint).
    """

    experiment_id: str
    required_files: list[str] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)
    all_present: bool = True


# ---------------------------------------------------------------------------
# extract_required_files
# ---------------------------------------------------------------------------


def extract_required_files(prompt: str, project_root: str) -> list[str]:
    """Parse the "EXISTING CODE TO READ FIRST:" section of a research prompt.

    **Why bullet parsing, not regex:**
        Research prompts are written by hand and can have slight formatting
        differences.  Parsing line-by-line for the bullet prefix ``- `` is
        more robust than a multi-line regex and easier to read.

    Algorithm
    ---------
    1. Scan for a line that starts with ``EXISTING CODE TO READ FIRST:``.
    2. Collect subsequent lines that start with ``- `` (bullet lines).
    3. Stop when we see a blank line followed by a non-indented, non-bullet
       line (i.e., the next heading or paragraph), or when a line starts
       with ``TASK:``.
    4. For each bullet line:
        a. Strip the ``- `` prefix.
        b. Remove any explanatory comment after `` — `` or `` # ``.
        c. Substitute the hardcoded placeholder path and ``{project_root}``
           with ``project_root``.
        d. If the resulting path is relative, join it to ``project_root``
           to produce an absolute path.

    Parameters
    ----------
    prompt : str
        The full text of the experiment prompt (from the roadmap YAML).
    project_root : str
        Absolute path to the repository root.  Used to resolve relative
        paths and to substitute the hardcoded placeholder.

    Returns
    -------
    list[str]
        Absolute paths of every file listed under the section header.
        Empty list if the section is not present.
    """
    lines = prompt.splitlines()
    in_section = False
    paths: list[str] = []
    pending_blank = False  # True when we just saw a blank line

    for line in lines:
        stripped = line.strip()

        if not in_section:
            # Look for the section header anywhere in the line
            if _SECTION_HEADER in line:
                in_section = True
                pending_blank = False
            continue

        # Inside the section -------------------------------------------------

        if stripped == "":
            # A blank line *might* end the section — wait to see the next line.
            pending_blank = True
            continue

        if pending_blank:
            # We had a blank line; check whether this line is a new heading
            # (non-indented non-bullet line) or the TASK marker.
            if not stripped.startswith("- ") or stripped.startswith(_TASK_MARKER):
                # The section has ended; stop collecting.
                break
            # Otherwise the blank line was just visual spacing — keep going.
            pending_blank = False

        if stripped.startswith(_TASK_MARKER):
            # Explicit TASK: line ends the section immediately.
            break

        if stripped.startswith("- "):
            # It's a bullet — extract the path.
            raw = stripped[2:].strip()  # Remove leading "- "

            # Strip explanatory comments (order matters: em-dash first, then #)
            for separator in (" \u2014 ", " — ", " # "):
                if separator in raw:
                    raw = raw[: raw.index(separator)].strip()

            # Substitute placeholder paths so the tool is repo-location agnostic.
            raw = raw.replace(_BRACED_PLACEHOLDER, project_root)
            raw = raw.replace(_ROADMAP_PLACEHOLDER, project_root)

            # Resolve relative paths to absolute paths.
            if not os.path.isabs(raw):
                raw = str(Path(project_root) / raw)

            paths.append(raw)

    return paths


# ---------------------------------------------------------------------------
# check_dependencies
# ---------------------------------------------------------------------------


def check_dependencies(
    prompt: str,
    project_root: str,
    *,
    experiment_id: str = "unknown",
) -> DependencyAudit:
    """Check that every file required by the prompt exists on disk.

    Calls ``extract_required_files()`` then performs a simple ``os.path.exists()``
    check for each path.  Returns a ``DependencyAudit`` so callers can decide
    how to react (emit a blocked artifact, log a warning, continue, etc.).

    Parameters
    ----------
    prompt : str
        Full text of the experiment prompt.
    project_root : str
        Repository root (used for path resolution and placeholder substitution).
    experiment_id : str
        The experiment identifier to embed in the returned audit.  Defaults
        to ``"unknown"`` so callers that don't know the ID still get a
        well-formed audit.

    Returns
    -------
    DependencyAudit
        Populated audit result.  ``all_present`` is ``True`` iff every
        required file exists (including the empty-list case).
    """
    required = extract_required_files(prompt, project_root)
    missing = [p for p in required if not os.path.exists(p)]
    return DependencyAudit(
        experiment_id=experiment_id,
        required_files=required,
        missing_files=missing,
        all_present=len(missing) == 0,
    )


# ---------------------------------------------------------------------------
# build_blocked_artifact
# ---------------------------------------------------------------------------


def build_blocked_artifact(audit: DependencyAudit) -> dict[str, Any]:
    """Build a conductor-compatible blocked artifact for a failed dependency audit.

    The conductor can write this artifact to the deliverable path immediately
    (without running any inference) to record *why* the experiment was blocked
    and what needs to happen before it can be retried.

    Parameters
    ----------
    audit : DependencyAudit
        The audit result with missing files populated.

    Returns
    -------
    dict
        A dict suitable for JSON serialisation, compatible with the standard
        ``REQUIRED_RESULT_FIELDS`` schema (subset: status, experiment_id,
        missing_files, required_files, next_action).
    """
    next_action = (
        "Run or re-run the experiment(s) that produce the missing result files, "
        "then retry this experiment.  Use the conductor dependency audit pre-hook "
        "(scripts/experiment_dependency_audit.py) to confirm all files are present "
        "before scheduling the retry."
    )
    return {
        "status": "blocked",
        "experiment_id": audit.experiment_id,
        "required_files": audit.required_files,
        "missing_files": audit.missing_files,
        "next_action": next_action,
    }


# ---------------------------------------------------------------------------
# load_experiment_prompt
# ---------------------------------------------------------------------------


def load_experiment_prompt(yaml_path: str, exp_id: str) -> str:
    """Load a research prompt from a roadmap YAML by matching ``exp_id``.

    The roadmap YAML may have a top-level ``tasks`` key (flat list) or a
    ``milestones`` key (list of dicts each with a ``tasks`` list).  This
    function handles both layouts transparently.

    A task matches when its ``id`` field *contains* the string ``exp_id``
    (e.g., ``"327"`` matches ``"exp327-dependency-audit"``).

    Parameters
    ----------
    yaml_path : str
        Path to the roadmap YAML file.
    exp_id : str
        The experiment identifier substring to search for (e.g. ``"327"``).

    Returns
    -------
    str
        The ``prompt`` field of the matched task.

    Raises
    ------
    ValueError
        If no task whose ``id`` contains ``exp_id`` is found.
    """
    # Lazy import so the module can be used without PyYAML in environments
    # that only need the Python API (though YAML is a project dependency).
    import yaml  # noqa: PLC0415

    with open(yaml_path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    # Collect all tasks from both possible layouts.
    all_tasks: list[dict[str, Any]] = []

    if isinstance(data, dict):
        if "tasks" in data:
            all_tasks.extend(data["tasks"])
        if "milestones" in data:
            for milestone in data["milestones"]:
                if isinstance(milestone, dict) and "tasks" in milestone:
                    all_tasks.extend(milestone["tasks"])

    # Match by exp_id substring.
    for task in all_tasks:
        task_id = str(task.get("id", ""))
        if exp_id in task_id:
            return str(task.get("prompt", ""))

    raise ValueError(
        f"No task with id containing {exp_id!r} found in {yaml_path}"
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pre-experiment dependency audit — check that all files listed under "
            "'EXISTING CODE TO READ FIRST' in a research prompt exist on disk."
        )
    )
    parser.add_argument(
        "--exp-id",
        required=True,
        help="Experiment identifier (e.g. '327').  Embedded in output for traceability.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--prompt-file",
        help="Path to a text file containing the experiment prompt.",
    )
    source.add_argument(
        "--yaml-path",
        help=(
            "Path to a roadmap YAML file.  The task whose id contains --exp-id "
            "is located and its prompt is used."
        ),
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help=(
            "Repository root used to resolve relative paths.  Defaults to the "
            "directory two levels above this script (i.e. the repo root)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns
    -------
    int
        Exit code: 0 if all dependencies are present, 1 if any are missing.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Resolve project root.
    if args.project_root is not None:
        project_root = str(Path(args.project_root).resolve())
    else:
        # Default: two directories up from this script (repo root).
        project_root = str(Path(__file__).resolve().parents[1])

    # Load prompt from file or YAML.
    if args.prompt_file is not None:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    else:
        prompt = load_experiment_prompt(args.yaml_path, args.exp_id)

    audit = check_dependencies(prompt, project_root, experiment_id=args.exp_id)

    if audit.all_present:
        n = len(audit.required_files)
        print(f"All ({n}) dependencies found.")
        return 0
    else:
        for path in audit.missing_files:
            print(f"MISSING: {path}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
