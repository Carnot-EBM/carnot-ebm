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


def delta(after_node_ids: Sequence[str], *, root: Path | None = None) -> dict[str, Any]:
    """Compare an observed failure set against the baseline.

    `ready_allowed` is true only when NO new node id appears AND the total did not grow. Both
    halves are required: a run that fixes one unrelated test and breaks another would have a
    delta of zero while having introduced a regression, and `new_node_ids` is what catches it.
    """
    baseline = baseline_node_ids(root=root)
    before, observed = set(baseline), {str(n) for n in after_node_ids}
    new_nodes = sorted(observed - before)
    return {
        "command": GLOBAL_PYTEST_COMMAND,
        "source": BASELINE_RELATIVE_PATH.as_posix(),
        "baseline_node_count": len(before),
        "after_node_count": len(observed),
        "new_node_ids": new_nodes,
        "resolved_node_ids": sorted(before - observed),
        "global_suite_failure_delta": len(observed) - len(before),
        "ready_allowed": not new_nodes and len(observed) <= len(before),
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
