"""REQ-REPORT-8695: independently bind fourteen V746 objects and original methods.

This branch reads original scientific authority directly. It never treats a
historical failed report as permission to run learning or as evidence of benefit.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from typing import Any

import yaml

from carnot.reporting import v744_contract_methods as original
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.roadmap_contract import compare_contract, parse_design
from carnot.reporting import v746_retained_evidence as e

Json = dict[str, Any]
ROOT = e.ROOT
NAME = "experiment_8695_v746_retained_contract_authority"
TASK, MILESTONE = "exp8695-retained-contract-authority", "2026.11.746"
DESIGN, STAGED, ACTIVE = original.DESIGN, original.STAGED, original.ACTIVE
PIN = "5c635232bc3e986e64593a6f42606d90e73491aa82c813541bbb84c5e88067ab"
CLI = f"scripts/experiments/{NAME}.py"
TESTS = [f"tests/python/test_v746_{s}_8695.py" for s in ["retained_evidence", "contract_authority"]]
OWNED = [
    f"python/carnot/reporting/v746_{s}.py"
    for s in ["retained_evidence", "contract_authority", "contract_runner", "contract_report"]
] + [CLI]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual phase boundaries so supervised CPU work stays observable."""
    print(f"[exp8695] phase={phase} completed={completed} pending={pending}", flush=True)


def gate(path: Path, field: str, expected: Any, observed: Any, upstream: str = TASK) -> Json:
    """Keep absence, missing fields, zero and altered bytes distinct for reopening decisions."""
    row = original.gate(path, field, expected, observed, upstream)
    row["check"] = field
    row["path_hash"] = e.sha256_file(path) if path.is_file() else None
    return dict(row)


def authority(root: Path) -> Json:
    """Read the visible table separately and reject coordinated replacements by a fixed digest."""
    refs, failures = [], []
    rows: list[Json] = []
    try:
        text = (root / DESIGN).read_text()
        chosen = root / STAGED if (root / STAGED).is_file() else root / ACTIVE
        planned = yaml.safe_load(chosen.read_bytes())
        actual = (
            yaml.safe_load((root / ACTIVE).read_bytes()) if (root / ACTIVE).is_file() else planned
        )
        if chosen == root / STAGED and actual.get("milestone") == "2026.11.745":
            actual = planned
        result = compare_contract(
            text, planned, actual, milestone=MILESTONE, first_id=8695, count=14
        )
        machine = parse_design(text, milestone=MILESTONE)[1]
        envelope = json.loads(
            text.split("<!-- V746_TASK_CONTRACT_START -->")[1].split("```json")[1].split("```")[0]
        )
        for i, row in enumerate(result["rows"]):
            row["checks"]["full_object"] = (
                i < min(len(machine), len(planned["tasks"]), len(actual["tasks"]))
                and machine[i] == planned["tasks"][i] == actual["tasks"][i]
            )
            row.update(
                matched=all(row["checks"].values()), excluded=False, effective_independent_groups=0
            )
            row.update(
                censored=not row["matched"],
                absolute_metric=int(row["matched"]),
                status="completed" if row["matched"] else "censored",
            )
        rows = result["rows"]
        ready = bool(
            all(r["matched"] for r in rows)
            and not set(result["errors"]) - {"row_mismatch"}
            and original.tasks_digest(machine) == original.tasks_digest(planned["tasks"]) == PIN
            and envelope == dict(milestone=MILESTONE, tasks=machine)
            and f"Canonical full-task SHA-256: `{PIN}`" in text
            and "<!-- V746_TASK_CONTRACT_END -->" in text
        )
        refs = [e.reference(root / DESIGN), e.reference(chosen)]
        if (root / ACTIVE).is_file():
            refs.append(e.reference(root / ACTIVE))
        if not ready:
            failures.append(gate(root / DESIGN, "fourteen_complete_sealed_objects", True, False))
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        ready = False
        failures.append(gate(root / DESIGN, "v746_authority", MILESTONE, str(error)))
    if not rows:
        rows = [
            dict(
                unit_id=f"exp{8695 + i}",
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
    return dict(ready=ready, rows=rows, references=refs, gate_check_summary=failures)


def terminal(ref: Json, field: str, number: int) -> Json:
    """Require the original producer's identity, readiness and hash-bound terminal acceptance."""
    refs, failures = [ref], []
    path = Path(ref["path"])
    try:
        if not path.is_file() or e.sha256_file(path) != ref["sha256"]:
            row = gate(
                path,
                "producer_sha256",
                ref["sha256"],
                e.sha256_file(path) if path.is_file() else None,
                f"exp{number}",
            )
            row["operand_state"] = "changed_bytes" if path.is_file() else "missing_file"
            failures.append(row)
        else:
            value = e.load(ref)
            for key, expected in [
                (field, 1),
                ("experiment_id", number),
                ("required_checks_passed", True),
                ("flagged_adversarial", False),
            ]:
                if key not in value or value[key] != expected:
                    failures.append(gate(path, key, expected, value.get(key), f"exp{number}"))
            side = Path(value["terminal_validation_sidecar_path"])
            refs.append(e.reference(side))
            document = e.load(refs[-1])
            publication = document.get("publication", document)
            bound = publication.get(
                "primary_sha256", document.get("candidate_reference", {}).get("sha256")
            )
            if publication.get("sidecar_path"):
                refs.append(e.reference(Path(publication["sidecar_path"])))
                accepted = e.load(refs[-1])["report"]["passed"] is True
            else:
                accepted = document["report"]["passed"] is True
            if bound != ref["sha256"] or not accepted:
                failures.append(
                    gate(
                        side,
                        "producer_terminal_hash_and_acceptance",
                        ref["sha256"],
                        bound,
                        f"exp{number}",
                    )
                )
    except (OSError, ValueError, KeyError, TypeError) as error:
        failures.append(gate(path, "producer_terminal", True, str(error), f"exp{number}"))
    return dict(passed=not failures, references=refs, gate_check_summary=failures)


def methods() -> Json:
    """Preserve registered scientific operands and qualify learning authority independently."""
    try:
        p = original.protocol()
    except (OSError, ValueError, KeyError, TypeError) as error:
        return dict(
            learning_authority_inputs_ready_score=0,
            references=[],
            head_checkpoint_reference=None,
            historical_quarantines={},
            future_dependencies=[],
            gate_check_summary=[
                gate(
                    ROOT / original.PROTOCOL,
                    "original_protocol_sha256",
                    original.PROTOCOL_PIN,
                    str(error),
                )
            ],
        )
    history = original.historical(p)
    producers = [
        (8431, "head_ready_score"),
        (8580, "current_validator_ready_score"),
        (8581, "online_adapter_ready_score"),
    ]
    terminals = [terminal(p["current_producers"][str(n)], field, n) for n, field in producers]
    keys = [
        "objective_updates",
        "objective_audit",
        "cohort_partition_reference",
        "learning_registration",
        "head_checkpoint_reference",
        "learning_data_reference",
        "learning_label_references",
        "test_inventory_reference",
        "relational_evidence_protocol",
        "installed_batching_protocol",
        "historical_quarantines",
    ]
    result = {k: deepcopy(p[k]) for k in keys}
    refs = [
        e.reference(ROOT / original.PROTOCOL),
        dict(
            path=str(
                ROOT / "openspec/change-proposals/research-roadmap-v745-preserved-20261101.md"
            ),
            sha256="sha256:01fb99526e816d6ff017196bcd17aad215562ef217bcdf113746fcbfc8e38a63",
        ),
    ]
    refs += [
        p[k]
        for k in [
            "head_checkpoint_reference",
            "learning_data_reference",
            "cohort_partition_reference",
            "test_inventory_reference",
        ]
    ]
    refs += p["learning_label_references"]
    refs += [ref for row in terminals for ref in row["references"]]
    failures = [r for row in terminals for r in row["gate_check_summary"]]
    authenticated = []
    for ref in refs:
        if not original.checked(ref):
            path = Path(ref["path"])
            row = gate(
                path,
                "original_method_operand_sha256",
                ref["sha256"],
                e.sha256_file(path) if path.is_file() else None,
            )
            row["operand_state"] = "changed_bytes" if path.is_file() else "missing_file"
            failures.append(row)
        else:
            authenticated.append(ref)
    result.update(
        learning_authority_inputs_ready_score=int(
            history["learning_operands_ready_score"] == 1 and not failures
        ),
        producer_terminal_rows=terminals,
        references=authenticated,
        attempted_references=refs,
        gate_check_summary=failures,
        independently_registered_branch_scores={
            k: v for k, v in history.items() if k.endswith("_inputs_ready_score")
        },
        future_dependencies=[
            dict(task_id=t["id"], path=t["deliverable"], status="future_dependency_not_observation")
            for t in parse_design((ROOT / DESIGN).read_text(), milestone=MILESTONE)[1][1:]
        ],
    )
    return result


def mutations() -> list[Json]:
    """Exercise each contract boundary, including consistent changes to all mutable copies."""
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
        "model",
        "MODEL_SPECS",
        "inference_substrate_class",
        "prompt",
        "gate",
        "prior_failures",
        "start_marker",
        "end_marker",
        "table",
        "digest",
        "envelope_digest",
        "rehashed",
        "foreign_active",
    ]
    rows = []
    with TemporaryDirectory(prefix="exp8695-authority-", dir="/var/tmp") as directory:
        root = Path(directory)
        design = root / DESIGN
        design.parent.mkdir(parents=True)
        for index, mode in enumerate(modes):
            value, changed = dict(milestone=MILESTONE, tasks=deepcopy(tasks)), text
            if mode == "count":
                value["tasks"].pop()
            if mode == "order":
                value["tasks"].reverse()
            if mode in {
                "id",
                "title",
                "phase",
                "deliverable",
                "model",
                "inference_substrate_class",
                "prompt",
            }:
                value["tasks"][0][mode] = "forged"
            if mode == "MODEL_SPECS":
                value["tasks"][0][mode] = ["forged"]
            if mode == "gate":
                value["tasks"][1]["gated_on"][0]["value"] = 99
            if mode == "prior_failures":
                value["tasks"][0][mode][0]["retire_if_same_verdict"] = False
            if mode in {"start_marker", "end_marker"}:
                changed = changed.replace(
                    f"<!-- V746_TASK_CONTRACT_{mode.split('_')[0].upper()} -->", ""
                )
            if mode == "table":
                changed = changed.replace("| 1 |", "| 99 |", 1)
            if mode == "digest":
                changed = changed.replace(PIN, "0" * 64)
            if mode == "envelope_digest":
                changed = changed.replace(
                    '"milestone": "2026.11.746",',
                    '"canonical_tasks_sha256": "' + "0" * 64 + '", "milestone": "2026.11.746",',
                    1,
                )
            if mode == "rehashed":
                value["tasks"][0]["prompt"] += "forged"
                digest = original.tasks_digest(value["tasks"])
                changed = re.sub(
                    r"(```json\s*)(.*?)(\s*```)",
                    lambda m: m[1] + json.dumps(dict(value, canonical_tasks_sha256=digest)) + m[3],
                    changed,
                    count=1,
                    flags=re.S,
                ).replace(PIN, digest)
            design.write_text(changed)
            (root / ACTIVE).write_text(yaml.safe_dump(value))
            stage = root / STAGED
            if mode in {"staged", "foreign_active"}:
                stage.write_text(yaml.safe_dump(value))
                (root / ACTIVE).write_text(
                    "milestone: "
                    + ("2026.11.745" if mode == "staged" else "2026.11.744")
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
            progress("mutation_" + mode, index + 1, len(modes) - index - 1)
    return rows
