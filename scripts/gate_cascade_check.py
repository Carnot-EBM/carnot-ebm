#!/usr/bin/env python3
"""Report gates in the ACTIVE roadmap that are already doomed (REQ-CONDUCTOR-CASCADE-1).

WHY THIS EXISTS (2026-09-03). exp6942 sat on disk with
`v608_execution_contract_ready_score = 0` while TEN downstream tasks in
milestone 608 gated on that exact field. Nobody joined the two files, so the
milestone burned 13 GATE_BLOCKs and retired 10 of 12 tasks. The artifact was
even validated by hand that afternoon and called benign, because
`summarize_artifact.py` says whether an artifact is honest, not whether
anything depends on it.

THE DISCRIMINATION, hand-tested before this was built (known-issues 2026-09-03
21:35Z). "Report any gate whose upstream field is 0" flags the wrong thing:
seven of 609's twelve tasks gated on upstreams that simply had not run yet —
the NORMAL state of a milestone in progress. The rule is therefore:

- upstream artifact ABSENT            -> silent (not a finding)
- artifact present, field ABSENT,
  upstream not final                  -> silent (bootstrap/in-flight state)
- artifact present, field PRESENT,
  op already fails                    -> PENDING CASCADE (the exp6942 shape)
- artifact present, upstream FINAL,
  field ABSENT                        -> PENDING CASCADE (the field can never
                                         appear; the exp6756 retry-burn shape)

Tested against the incident: fires on exp6942's field (ten dependents), silent
on all seven of 609's absent-upstream gates.

Uses the conductor's OWN gate machinery (`scripts/conductor_gates.py`:
`_find_artifact_by_task_id`, `_eval_op`, `_upstream_is_final`) so this check
cannot drift from what the conductor will actually do at activation time.

Populations are always printed, so "no pending cascades" is distinguishable
from "never looked" (the 2026-09-04 guarded-call lesson). Exit 2 on findings,
0 clean, 1 on an unreadable roadmap (fail closed).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import conductor_gates  # noqa: E402

try:
    import yaml
except ImportError:  # pragma: no cover - yaml ships in the project venv
    yaml = None


def load_tasks(roadmap_path: Path) -> list[dict[str, Any]] | None:
    """The active roadmap's task list, or None when it cannot be read."""
    if yaml is None or not roadmap_path.is_file():
        return None
    try:
        doc = yaml.safe_load(roadmap_path.read_text(encoding="utf-8", errors="replace"))
    except yaml.YAMLError:
        return None
    if not isinstance(doc, dict):
        return None
    tasks = doc.get("tasks")
    return tasks if isinstance(tasks, list) else None


def pending_cascades(
    roadmap_path: Path | None = None,
    results_dir: Path | None = None,
) -> tuple[list[str] | None, list[str]]:
    """Returns (findings, notices). findings is None when the roadmap is unreadable."""
    roadmap_path = roadmap_path or (REPO_ROOT / "research-roadmap.yaml")
    results_dir = results_dir or (REPO_ROOT / "results")
    tasks = load_tasks(roadmap_path)
    notices: list[str] = []
    if tasks is None:
        return None, [f"roadmap unreadable: {roadmap_path}"]

    n_gates = 0
    n_absent = 0
    n_inflight = 0
    # One failing upstream field dooms EVERY task gating on it; group so the
    # report reads as a cascade ("N tasks pending"), not N separate lines.
    doomed: dict[tuple[str, str, str], list[str]] = {}
    for task in tasks:
        if not isinstance(task, dict):
            continue
        task_id = str(task.get("id") or task.get("experiment_id") or "?")
        for gate in task.get("gated_on") or []:
            if not isinstance(gate, dict):
                continue
            n_gates += 1
            upstream = str(gate.get("upstream", ""))
            field = str(gate.get("artifact_field", ""))
            op = str(gate.get("op", "=="))
            expected = gate.get("value")
            path = conductor_gates._find_artifact_by_task_id(upstream, results_dir)
            if path is None:
                n_absent += 1  # upstream has not run: the normal mid-milestone state
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            except (OSError, json.JSONDecodeError):
                n_inflight += 1  # mid-write is indistinguishable from corrupt; not a finding
                continue
            actual = data.get(field)
            final = conductor_gates._upstream_is_final(data)
            if actual is None:
                if final:
                    key = (upstream, field, f"field absent on a FINAL upstream ({path.name})")
                    doomed.setdefault(key, []).append(task_id)
                else:
                    n_inflight += 1
                continue
            passed, _reason = conductor_gates._eval_op(actual, op, expected)
            if not passed:
                key = (
                    upstream,
                    field,
                    f"already {actual!r}, gate wants {op} {expected!r} ({path.name})",
                )
                doomed.setdefault(key, []).append(task_id)

    notices.append(
        f"population: {len(tasks)} task(s), {n_gates} gate(s) evaluated, "
        f"{n_absent} upstream(s) absent (normal), {n_inflight} in-flight/unreadable"
    )

    def _dependent_label(task_id: str) -> str:
        # A dependent that already wrote a blocked artifact burned its attempts;
        # one with no artifact is still PENDING -- the state worth the alarm.
        path = conductor_gates._find_artifact_by_task_id(task_id, results_dir)
        if path is None:
            return task_id
        try:
            verdict = str(json.loads(path.read_text()).get("honest_verdict") or "")
        except (OSError, json.JSONDecodeError):
            return task_id
        return f"{task_id}(already blocked)" if verdict.startswith("blocked") else task_id

    findings = [
        (
            f"PENDING CASCADE: {upstream}.{field} {why} -- "
            f"{len(dependents)} task(s) gate on it: "
            f"{', '.join(_dependent_label(t) for t in dependents)}"
        )
        for (upstream, field, why), dependents in sorted(doomed.items())
    ]
    return findings, notices


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roadmap", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    findings, notices = pending_cascades(args.roadmap, args.results_dir)
    for line in notices:
        print(f"NOTE: {line}")
    if findings is None:
        print("FAIL: roadmap unreadable; the join cannot run")
        return 1
    for line in findings:
        print(line)
    if findings:
        return 2
    print("OK: no gate in the active roadmap already fails against an existing upstream.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
