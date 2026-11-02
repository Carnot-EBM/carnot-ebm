"""REQ-REPORT-8709: bind current plans without promoting historical measurements."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

import yaml

from carnot.reporting import v746_contract_authority as previous
from carnot.reporting import v746_contract_runner as retained_runner
from carnot.reporting import v746_retained_evidence as e
from carnot.reporting.roadmap_contract import compare_contract, parse_design

Json = dict[str, Any]
ROOT = e.ROOT
NAME, TASK, MILESTONE = (
    "experiment_8709_v747_contract_methods",
    "exp8709-contract-methods",
    "2026.11.747",
)
DESIGN, ACTIVE, STAGED = previous.DESIGN, previous.ACTIVE, previous.STAGED
PIN = "d0fb6404987d9825a83e7f3b910e1d9a5eaa6a9caabee9aef693faa3050e7c83"
PRESERVED = "openspec/change-proposals/research-roadmap-v746-preserved-20261102.md"
RETAINED = dict(
    path=str(ROOT / "results" / (previous.NAME + ".json")),
    sha256="sha256:6aabdc1d57654175f8ec01af407c16855d3c24214665bcbdc2d0016428ea4b7d",
)
TESTS = ["tests/python/test_v747_contract_8709.py"]
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    f"python/carnot/reporting/v747_contract_{s}.py" for s in ["methods", "report", "runner"]
] + [CLI]
original = previous.original


def gate(path: Path, field: str, expected: Any, observed: Any, upstream: str = TASK) -> Json:
    """Attach current invocation ownership while preserving missing, zero and changed states."""
    return dict(previous.gate(path, field, expected, observed, upstream))


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Show real boundaries so a supervisor can distinguish progress from a stall."""
    print(f"[exp8709] phase={phase} completed={completed} pending={pending}", flush=True)


def authority(root: Path) -> Json:
    """Require full equality and an independent digest, even if every mutable copy agrees."""
    rows: list[Json] = []
    refs, failures = [], []
    try:
        text = (root / DESIGN).read_text()
        chosen = root / STAGED if (root / STAGED).is_file() else root / ACTIVE
        planned = yaml.safe_load(chosen.read_bytes())
        active = yaml.safe_load((root / ACTIVE).read_bytes())
        actual = (
            planned if chosen == root / STAGED and active["milestone"] == "2026.11.746" else active
        )
        result = compare_contract(
            text, planned, actual, milestone=MILESTONE, first_id=8709, count=14
        )
        tasks = parse_design(text, milestone=MILESTONE)[1]
        envelope = json.loads(
            text.split("<!-- V747_TASK_CONTRACT_START -->")[1].split("```json")[1].split("```")[0]
        )
        for i, row in enumerate(result["rows"]):
            full = (
                i < min(len(tasks), len(planned["tasks"]), len(actual["tasks"]))
                and tasks[i] == planned["tasks"][i] == actual["tasks"][i]
            )
            row["checks"]["full_object"] = full
            row["task_object"] = deepcopy(tasks[i]) if i < len(tasks) else None
            matched = all(row["checks"].values())
            row.update(
                matched=matched,
                censored=not matched,
                excluded=False,
                effective_independent_groups=0,
                absolute_metric=int(matched),
                status="completed" if matched else "censored",
            )
        rows = result["rows"]
        ready = (
            all(r["matched"] for r in rows)
            and not set(result["errors"]) - {"row_mismatch"}
            and previous.original.tasks_digest(tasks) == PIN
            and envelope == dict(milestone=MILESTONE, tasks=tasks)
            and f"Canonical full-task SHA-256: `{PIN}`" in text
            and "<!-- V747_TASK_CONTRACT_END -->" in text
        )
        refs = [e.reference(root / DESIGN), e.reference(chosen), e.reference(root / ACTIVE)]
        if not ready:
            failures.append(
                gate(root / DESIGN, "fourteen_complete_sealed_objects", True, False, TASK)
            )
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        ready = False
        failures.append(gate(root / DESIGN, "v747_authority", MILESTONE, str(error), TASK))
    if not rows:
        rows = [
            dict(
                unit_id=f"exp{8709 + i}",
                order=i + 1,
                matched=False,
                absolute_metric=0,
                censored=True,
                excluded=False,
                status="censored",
                effective_independent_groups=0,
            )
            for i in range(14)
        ]
    return dict(ready=bool(ready), rows=rows, references=refs, gate_check_summary=failures)


def mutations() -> list[Json]:
    """Change each trust boundary privately so historical design bytes stay untouched."""
    text = (ROOT / DESIGN).read_text()
    tasks = parse_design(text, milestone=MILESTONE)[1]
    modes = [
        "activated",
        "staged",
        "count",
        "order",
        "id",
        "title",
        "phase",
        "deliverable",
        "track",
        "priority",
        "model",
        "MODEL_SPECS",
        "inference_substrate_class",
        "prompt",
        "gated_on",
        "prior_failures",
        "start",
        "end",
        "table",
        "digest",
        "rehashed",
        "foreign_active",
    ]
    rows = []
    with TemporaryDirectory(prefix="exp8709-authority-", dir="/var/tmp") as directory:
        root = Path(directory)
        design = root / DESIGN
        design.parent.mkdir(parents=True)
        for i, mode in enumerate(modes):
            value, changed = dict(milestone=MILESTONE, tasks=deepcopy(tasks)), text
            if mode == "count":
                value["tasks"].pop()
            elif mode == "order":
                value["tasks"].reverse()
            elif mode in {
                "id",
                "title",
                "phase",
                "deliverable",
                "track",
                "priority",
                "model",
                "MODEL_SPECS",
                "inference_substrate_class",
                "prompt",
            }:
                value["tasks"][0][mode] = "forged"
            elif mode == "gated_on":
                value["tasks"][1]["gated_on"][0]["value"] = 99
            elif mode == "prior_failures":
                value["tasks"][0]["prior_failures"][0]["retire_if_same_verdict"] = False
            elif mode in {"start", "end"}:
                changed = changed.replace(f"<!-- V747_TASK_CONTRACT_{mode.upper()} -->", "")
            elif mode == "table":
                changed = changed.replace("| 1 |", "| 99 |", 1)
            elif mode == "digest":
                changed = changed.replace(PIN, "0" * 64)
            elif mode == "rehashed":
                value["tasks"][0]["prompt"] += "forged"
                start, tail = text.split("```json", 1)
                _, end = tail.split("```", 1)
                changed = (start + "```json\n" + json.dumps(value) + "\n```" + end).replace(
                    PIN, previous.original.tasks_digest(value["tasks"])
                )
            design.write_text(changed)
            (root / ACTIVE).write_text(yaml.safe_dump(value))
            stage = root / STAGED
            if mode in {"staged", "foreign_active"}:
                stage.write_text(yaml.safe_dump(value))
                (root / ACTIVE).write_text(
                    "milestone: "
                    + ("2026.11.746" if mode == "staged" else "2026.11.745")
                    + "\ntasks: []\n"
                )
            else:
                stage.unlink(missing_ok=True)
            observed = authority(root)["ready"]
            rows.append(
                dict(
                    mutation=mode,
                    expected=mode in {"activated", "staged"},
                    observed=observed,
                    passed=observed == (mode in {"activated", "staged"}),
                )
            )
            progress("mutation_" + mode, i + 1, len(modes) - i - 1)
    return rows


def retained() -> Json:
    """Authenticate original commands, tests and completed databases under the pinned primary."""
    refs, failures = [dict(RETAINED)], []
    history: Json = dict(references=[])
    qualification: Json = dict(
        passed=False, original_node_count=0, original_rejection_line_executed=False
    )
    try:
        value = e.load(RETAINED)
        history = value["historical_failure_reproduction"]
        terminal = previous.terminal(RETAINED, "contract_ready_score", 8695)
        refs += terminal["references"]
        failures += terminal["gate_check_summary"]
        work, validation = (
            e.load(value["primitive_reference"]),
            e.load(value["validation_reference"]),
        )
        refs += [value["primitive_reference"], value["validation_reference"]]
        qualification = retained_runner.qualify_producer(
            work["plan"], validation["retained_child_rows"], work["historical_failure"]
        )
        original_rows = qualification["aggregate"]["retained_child_rows"]
        for row in original_rows:
            child = e.load(row)
            refs += (
                [row]
                + child["source_hashes"]
                + child["sqlite_references"]
                + [child["junit_reference"]]
            )
            for source in child["source_hashes"]:
                if e.sha256_file(Path(source["original_path"])) != source["original_sha256"]:
                    raise ValueError("original_execution_source_drift")
        if not qualification["passed"] or qualification != value["producer_coverage_union"]:
            raise ValueError("retained_original_execution_identity")
        seal = Path(value["terminal_validation_sidecar_path"]).parent / "independent_seal.json"
        refs.append(e.reference(seal))
        if e.load(refs[-1]) != {
            k: value[k] for k in ["primitive_reference", "validation_reference"]
        }:
            raise ValueError("retained_independent_seal")
    except (OSError, ValueError, KeyError, TypeError) as error:
        failures.append(
            dict(
                gate(
                    Path(RETAINED["path"]),
                    "retained_producer_authority",
                    RETAINED["sha256"],
                    e.sha256_file(Path(RETAINED["path"]))
                    if Path(RETAINED["path"]).is_file()
                    else None,
                    "exp8695-retained-contract-authority",
                ),
                diagnostic=str(error),
            )
        )
    refs = [
        r
        for r in refs
        if Path(r["path"]).is_file() and e.sha256_file(Path(r["path"])) == r["sha256"]
    ]
    return dict(
        passed=not failures,
        qualification=qualification,
        historical_failure=history,
        references=refs,
        gate_check_summary=failures,
    )


def _methods() -> Json:
    """Keep original numerical objectives while registering distinct future branch methods."""
    with patch.object(previous, "DESIGN", PRESERVED):
        result = previous.methods()
    result["references"].append(
        dict(
            path=str(ROOT / PRESERVED),
            sha256="sha256:cf1d0255715e08cf3793bbf69635485dfea5885f0173c1c71e949be59c1ca24d",
        )
    )
    diagnostics = []
    for name, pin, diagnosis in [
        (
            "experiment_8697_v746_learning_union_qualification",
            "1e9e81e361ffdbe70fae3c8e61460776abb74834b313fe575dd1ae20e6f39525",
            "retained start497771539033492 with current end232478218800900 produced duration-265293.320232592; retained clocks belong to their source invocation",
        ),
        (
            "experiment_8702_v746_installed_batch_prototype",
            "879b1c1ebe5aa1996cd34ba4b635b6d30da6dbd4ee58e5fe4cbf8ef68b790e36",
            "original legacy-export teardown guard failed with +553MB retained memory",
        ),
    ]:
        ref = dict(path=str(ROOT / "results" / (name + ".json")), sha256="sha256:" + pin)
        value = e.load(ref)
        diagnostics.append(
            dict(
                reference=ref,
                honest_verdict=value["honest_verdict"],
                disqualified=value["verdict_class"] == "disqualified",
                diagnosis=diagnosis,
            )
        )
        result["references"].append(ref)
    partition = e.load(result["cohort_partition_reference"])
    result.update(
        original_partition=partition,
        immutable_diagnostics=diagnostics,
        registered_v747_changes={
            "exp8710": "current invocation clocks; retained measurement clocks remain provenance",
            "exp8712": "model-free eight-feature relational diagnostic; absent ninth probabilities remain absent",
            "exp8715": "valid-alternative and independently labeled perturbation controls",
            "exp8716": "isolated legacy-import execution preserving original memory assertions",
            "exp8719": "compiled signed32/24 kernel; CPU execution is not physical FPGA execution",
        },
        future_dependencies=[
            dict(
                task_id=t["id"],
                path=t["deliverable"],
                status="future_dependency_not_observation",
                eligibility="branch_owned_operands_only",
            )
            for t in parse_design((ROOT / DESIGN).read_text(), milestone=MILESTONE)[1][1:]
        ],
    )
    return result


def methods() -> Json:
    """External absence stays blocked so one missing historical file cannot crash publication."""
    try:
        return _methods()
    except (OSError, ValueError, KeyError, TypeError) as error:
        path = ROOT / original.PROTOCOL
        return dict(
            learning_authority_inputs_ready_score=0,
            references=[],
            head_checkpoint_reference=None,
            historical_quarantines={},
            future_dependencies=[],
            immutable_diagnostics=[],
            gate_check_summary=[
                dict(
                    gate(path, "original_methods", original.PROTOCOL_PIN, str(error), TASK),
                    diagnostic=str(error),
                )
            ],
        )
