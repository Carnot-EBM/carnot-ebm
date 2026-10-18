"""REQ-HARNESS-5920: tell a NEW test regression apart from pre-existing suite debt.

WHY THIS EXISTS (2026-08-29). Eleven experiment modules put
`.venv/bin/pytest tests/python -q` in their verification command list and treated any nonzero
exit as a failed gate. The suite has ~1,726 pre-existing failures that have nothing to do with
any of them, so that exit code can never be zero and those tasks can never qualify. exp6682 --
a live ARC supervisor A/B that ran to completion and produced real rows -- was blocked this way,
by a stale capstone in an unrelated part of the repo.

REQ-HARNESS-5920 already settled the right answer, and nine non-ARC modules already use it:
run the global suite, and require that YOU introduced no NEW failing node id. Quoting the spec:

    readiness may use `global_suite_failure_delta<=0` only when every nonzero node is present
    in the pre-task baseline and no new node id appears. This rule SHALL NOT suppress,
    deselect, relabel, or rewrite unrelated failures.

That last sentence is the point. This is not a way to ignore the suite -- the suite still runs,
in full, and a regression you cause still fails you. It only stops unrelated debt from being
charged to your task.

WHY A SHARED MODULE rather than the per-experiment copies the nine use: eleven more copies of
the same logic is eleven more places for it to drift, and a baseline read from eleven slightly
different helpers is not one baseline.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

#: Derived from __file__, never a hardcoded absolute path -- a baked-in root makes a fresh
#: clone read the operator's checkout (CLAUDE.md "Test-Run Record Integrity" rule 4).
REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_RELATIVE_PATH = Path("ops/global_suite_failure_baseline.json")
GLOBAL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"


def baseline_node_ids(*, root: Path | None = None) -> list[str]:
    """The recorded pre-existing failures, or [] when the baseline is unreadable.

    Returning [] on a missing baseline is deliberate and is the SAFE direction: with no
    baseline every observed failure counts as new, so `delta()` reports not-ready. A guard that
    fails toward "you are clean" when it cannot read its own evidence is the trusted-and-silent
    state this project treats as worse than no guard at all.
    """
    base = Path(root) if root is not None else REPO_ROOT
    try:
        payload = json.loads((base / BASELINE_RELATIVE_PATH).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    nodes = payload.get("baseline_node_ids") if isinstance(payload, Mapping) else None
    return [str(n) for n in nodes] if isinstance(nodes, list) else []


def baseline_error_node_ids(*, root: Path | None = None) -> list[str] | None:
    """The recorded pre-existing ERROR nodes, or None when the baseline does not record them.

    None here means "this ledger predates error tracking", not "zero errors". The 2026-08-29
    baseline records 1,726 FAILED nodes and its run also saw 143 ERROR nodes that were never
    written down. Until a re-baseline records them, error enforcement stays advisory --
    enforcing against an empty set would refuse every honest run, and a check that cries wolf
    trains people to bypass it.
    """
    base = Path(root) if root is not None else REPO_ROOT
    try:
        payload = json.loads((base / BASELINE_RELATIVE_PATH).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    nodes = payload.get("baseline_error_node_ids") if isinstance(payload, Mapping) else None
    return [str(n) for n in nodes] if isinstance(nodes, list) else None


def delta(
    after_node_ids: Sequence[str],
    *,
    after_error_node_ids: Sequence[str] | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Compare an observed failure set against the baseline.

    `ready_allowed` is true only when NO new node id appears AND the total did not grow. Both
    halves are required: a run that fixes one unrelated test and breaks another would have a
    delta of zero while having introduced a regression, and `new_node_ids` is what catches it.
    """
    baseline = baseline_node_ids(root=root)
    before, observed = set(baseline), {str(n) for n in after_node_ids}
    new_nodes = sorted(observed - before)

    # ERROR nodes are a separate ledger from FAILED nodes, because the recorded baseline may
    # predate error tracking. Enforcement activates only when BOTH sides are known: the
    # baseline records its error nodes AND the caller measured this run's. Anything less is
    # reported honestly as not assessed, never silently treated as clean.
    error_ledger = baseline_error_node_ids(root=root)
    errors_assessed = error_ledger is not None and after_error_node_ids is not None
    new_error_nodes: list[str] = []
    if errors_assessed:
        new_error_nodes = sorted(
            {str(n) for n in after_error_node_ids or []} - set(error_ledger or [])
        )

    return {
        "command": GLOBAL_PYTEST_COMMAND,
        "source": BASELINE_RELATIVE_PATH.as_posix(),
        "baseline_node_count": len(before),
        "after_node_count": len(observed),
        "new_node_ids": new_nodes,
        "resolved_node_ids": sorted(before - observed),
        "global_suite_failure_delta": len(observed) - len(before),
        "error_nodes_assessed": errors_assessed,
        "new_error_node_ids": new_error_nodes,
        "ready_allowed": not new_nodes and not new_error_nodes and len(observed) <= len(before),
        "unrelated_debt_preserved_by_exact_node_id": True,
        "global_suite_zero_required": False,
        "principle": (
            "Known unrelated debt is preserved by exact node id and may not increase. The "
            "suite still runs in full; only failures you did not introduce are excused."
        ),
    }


def failure_node_ids_from_pytest_output(text: str) -> list[str]:
    """The failing node ids in a `-q` pytest run's output, deduplicated and sorted."""
    nodes = {
        line[len("FAILED ") :].split(" - ", 1)[0].strip()
        for line in (text or "").splitlines()
        if line.startswith("FAILED ")
    }
    return sorted(n for n in nodes if n)


def error_node_ids_from_pytest_output(text: str) -> list[str]:
    """The ERROR node ids in a `-q` pytest run's output, deduplicated and sorted.

    A setup or collection failure prints `ERROR <node>` in the short summary, not `FAILED`.
    A delta that reads only FAILED lines cannot see a change that turns a passing test into
    a setup error. That exact suppression contaminated two full-suite runs on 2026-08-28.
    """
    nodes = set()
    for line in (text or "").splitlines():
        if not line.startswith("ERROR "):
            continue
        node = line[len("ERROR ") :].split(" - ", 1)[0].strip()
        # The summary can print a bare "ERROR" banner line; a node id always names a path.
        if node and ("/" in node or node.endswith(".py") or "::" in node):
            nodes.add(node)
    return sorted(nodes)


#: Lines that mean pytest DID NOT finish. A truncated run's failure list is incomplete, so a
#: delta computed from it launders unknown breakage into "cleaner than baseline".
_ABORT_MARKERS = (
    "!! Interrupted",
    "Interrupted: ",
    "error during collection",
    "INTERNALERROR",
)


def pytest_run_aborted(text: str) -> bool:
    """True when the output shows pytest stopped before running the whole suite."""
    body = text or ""
    return any(marker in body for marker in _ABORT_MARKERS)


def observed_suite_evidence(stdout: str, exit_code: int) -> dict[str, list[str]] | None:
    """Parse one run's stdout into node-id evidence, or None when it cannot be trusted.

    None is NOT "no failures" and callers must fail closed on it. Two cases return None:

    1. The run ABORTED (collection error, internal error, interrupt). Its failure list is
       incomplete, so any delta over it is meaningless. The 2026-08-27 incident is the shape:
       one stale capstone aborted 57,917 collected tests, zero FAILED lines were printed,
       and a naive delta read that as 1,726 failures fixed.
    2. The run exited nonzero and the output explains NOTHING (no FAILED node, no ERROR
       node). We know it failed and we do not know what failed.

    A zero exit with no parsed nodes is a genuinely clean run and returns empty evidence.
    """
    failures = failure_node_ids_from_pytest_output(stdout)
    errors = error_node_ids_from_pytest_output(stdout)
    if pytest_run_aborted(stdout):
        return None
    if int(exit_code) != 0 and not failures and not errors:
        return None
    return {"failure_node_ids": failures, "error_node_ids": errors}
