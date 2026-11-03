"""REQ-REPORT-8653: freeze current authority without changing scientific operands.

Registration establishes which work may proceed. It cannot establish learning
benefit, and optional resources cannot suppress independent CPU evidence.
"""

from __future__ import annotations

from pathlib import Path
from copy import deepcopy
from tempfile import TemporaryDirectory
import re
import time
from typing import Any
from unittest.mock import patch

import yaml

from carnot.reporting import v742_contract_methods as previous
from carnot.reporting import v736_contract_methods as base
from carnot.reporting import v736_historical_inputs as readers
from carnot.reporting.current_work_receipt import canonical_hash as canonical_hash
from carnot.reporting.roadmap_contract import compare_contract, parse_design


Json = dict[str, Any]
ROOT = base.ROOT
NAME, TASK, MILESTONE = (
    "experiment_8653_v743_contract_methods",
    "exp8653-contract-methods",
    "2026.10.743",
)
DESIGN, ACTIVE, STAGED = base.DESIGN, base.ACTIVE, base.STAGED
PROTOCOL = "openspec/change-proposals/v743-methods-protocol.json"
PROTOCOL_PIN = "sha256:82b4edee47cc7e369ef74c4e1fd319909ed27d02a3cadd039ee96d2b210d25d7"
TASKS_PIN = "69d4edc2aec6bc1ba6b2028d1a9d3b06400be2501f8c40ed093a09f737acd1e0"
INHERITED = [
    "objective_updates",
    "objective_audit",
    "streaming_extraction_protocol",
    "installed_entry_2x2",
    "evidence_canary",
]
CLI = f"scripts/experiments/{NAME}.py"
TESTS = [f"tests/python/test_v743_contract_{s}_8653.py" for s in ["methods", "runner"]]
TEST = TESTS[0]
OWNED = [
    f"python/carnot/reporting/v743_contract_{s}.py" for s in ["methods", "report", "runner"]
] + [CLI]
MODEL_SPECS: list[Json] = []
reference, read, tasks_digest = base.reference, base.read, base.tasks_digest
checksum, atomic_json, sha256_file = base.checksum, base.atomic_json, base.sha256_file
checked = readers.checked


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real boundaries so the parent can distinguish progress from a stall."""
    if phase in {
        "preconditions_before",
        "authority_before",
        "authority_after",
        "measurement_frozen",
    }:
        completed = min(completed, 14)
        pending = 14 - completed
    print(f"[exp8653] phase={phase} completed={completed} pending={pending}", flush=True)


def protocol() -> Json:
    """Authenticate original scientific operands and the independently sealed task map."""
    p = read(dict(path=str(ROOT / PROTOCOL), sha256=PROTOCOL_PIN))
    inherited = read(p["inherited_v742_protocol_reference"])
    checks = {
        k: dict(source="V742", value_equal=p[k] == inherited[k], sha256=canonical_hash(p[k]))
        for k in INHERITED
    }
    refs = [
        p[k]
        for k in [
            "inherited_v742_protocol_reference",
            "v743_activation_reference",
            "v743_design_reference",
            "literature_reference",
            "preserved_v742_design_reference",
        ]
    ]
    refs += p["historical_readers"] + [r["source_reference"] for r in p["literature_method_rows"]]
    refs += [
        p["cohort_partition_reference"],
        p["reset_callback_payload_protocol"]["compact_fixture_reference"],
    ] + p["reset_callback_payload_protocol"]["actual_public_frame_references"]
    if (
        p["inherited_v742_protocol_reference"]["sha256"] != previous.PROTOCOL_PIN
        or not all(r["value_equal"] for r in checks.values())
        or checks != p["inherited_operand_checks"]
        or tasks_digest(p["task_map_v743"]) != TASKS_PIN
        or not all(checked(r) for r in refs)
    ):
        raise ValueError("sealed_scientific_or_registration_operand")
    read(p["reset_callback_payload_protocol"]["compact_fixture_reference"])
    partition = read(p["cohort_partition_reference"])
    nodes = [n for row in partition["cohorts"] for n in row["node_ids"]]
    if (
        len(nodes) != len(set(nodes))
        or len(nodes) != 64
        or set(nodes) != set(partition["original_node_ids"])
        or len(partition["original_runner_node_ids"]) != 27
        or not set(partition["original_runner_node_ids"]) <= set(nodes)
        or not all(checked(r) for r in partition["source_references"])
    ):
        raise ValueError("original_cohort_partition")
    return dict(p)


def authority(root: Path) -> Json:
    """Use staging or real V743 activation; an older live queue has no current authority."""
    p = protocol()
    refs, failures = [], []
    try:
        text = (root / DESIGN).read_text()
        stage, active = root / STAGED, root / ACTIVE
        chosen = stage if stage.is_file() else active
        planned = yaml.safe_load(chosen.read_bytes())
        actual = (
            yaml.safe_load(active.read_bytes()) if stage.is_file() and active.is_file() else planned
        )
        if actual.get("milestone") != MILESTONE and stage.is_file():
            actual = planned
        result = compare_contract(
            text, planned, actual, milestone=MILESTONE, first_id=8653, count=14
        )
        machine = parse_design(text, milestone=MILESTONE)[1]
        for i, row in enumerate(result["rows"]):
            row["checks"]["full_dictionary"] = (
                i < len(planned["tasks"])
                and planned["tasks"][i] == p["task_map_v743"][i] == machine[i] == actual["tasks"][i]
            )
            row["checks"]["prior"] = (
                not p["task_map_v743"][i].get("prior_failures") or row["checks"]["prior"]
            )
            row["matched"] = all(row["checks"].values())
            row["excluded"] = not row["matched"]
            row["raw_numerator"] = sum(row["checks"].values())
            row["raw_denominator"] = len(row["checks"])
        ready = bool(
            not set(result["errors"]) - {"row_mismatch"}
            and all(r["matched"] for r in result["rows"])
            and re.search(
                r"(?:canonical_tasks_sha256:|Canonical task SHA256:)\s*`?(?:sha256:)?" + TASKS_PIN,
                text,
            )
            and "<!-- V743_TASK_CONTRACT_END -->" in text
            and tasks_digest(planned["tasks"]) == TASKS_PIN
        )
        refs = [reference(root / DESIGN), reference(chosen)]
        rows = result["rows"]
        if not ready:
            failures.append(gate(root / DESIGN, "fourteen_complete_objects", True, False, TASK))
    except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
        ready = False
        failures.append(
            gate(root / DESIGN, "v743_design_and_authority", MILESTONE, str(error), TASK)
        )
        rows = [
            dict(
                unit_id=t["id"],
                order=i + 1,
                matched=False,
                status="unstarted",
                checks={},
                censored=True,
                excluded=False,
            )
            for i, t in enumerate(p["task_map_v743"])
        ]
    for row in rows:
        row.update(
            absolute_metric=int(row["matched"]),
            effective_independent_groups=0,
            missing_reason=None if row["matched"] else "authority_absent_or_mismatched",
        )
    return dict(
        ready=ready,
        rows=rows,
        gate_check_summary=failures,
        references=refs,
        staging_present=(root / STAGED).is_file(),
    )


def historical(p: Json) -> Json:
    """Authenticate independent current branches without clearing any historical failure."""
    from carnot.reporting.v742_contract_inputs import historical as current

    value = dict(current(p))
    value["circular_gate_regression_rows"][0]["actual_training_requires"] = [
        "learning_operands_ready_score",
        "Exp8655.learning_execution_ready_score",
    ]
    return value


def runtime(root: Path, raw: Path | None = None) -> Json:
    """Read the existing CUDA evidence once without opening a device or model."""
    return dict(previous.runtime(root, raw))


def measure(root: Path, raw: Path, run_date: str) -> Json:
    """Reuse resource checks, then freeze this invocation and its optional revision controls."""
    snapshot = raw / "authority"
    snapshot.mkdir(parents=True, exist_ok=True)
    for name in [DESIGN, ACTIVE, STAGED]:
        source, target = root / name, snapshot / name
        if source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
    with (
        patch.multiple(
            base,
            NAME=NAME,
            TASK=TASK,
            MILESTONE=MILESTONE,
            PROTOCOL=PROTOCOL,
            OWNED=OWNED,
            TESTS=TESTS,
            protocol=protocol,
            authority=authority,
            progress=progress,
        ),
        patch.object(readers, "historical", historical),
        patch.object(readers, "runtime", lambda ignored, logs: runtime(root, logs)),
    ):
        work = dict(base.measure(snapshot, raw))
    work["run_date"] = run_date
    work["numerical"] = numerical(work["protocol"])
    work["authority_mutation_rows"] = mutation_controls(snapshot)
    work["end_ns"] = time.monotonic_ns()
    work["duration_s"] = (work["end_ns"] - work["start_ns"]) / 1e9
    work["phase_spans"][0].update(end_monotonic_ns=work["end_ns"], duration_s=work["duration_s"])
    atomic_json(raw / "measurement.json", work)
    atomic_json(raw / "measurement_seal.json", work)
    progress("fourteen_task_measurement_frozen", 14, 0)
    return work


def mutation_controls(root: Path) -> list[Json]:
    """Exercise staging, activation and independent mutations without touching live files."""
    p = protocol()
    rows = []
    with TemporaryDirectory(prefix="exp8653-authority-", dir="/var/tmp") as directory:
        private = Path(directory)
        design = private / DESIGN
        design.parent.mkdir(parents=True)
        text = (root / DESIGN).read_text() if (root / DESIGN).is_file() else ""
        # Controls use the sealed complete design even when a private caller omits authority.
        if not text:
            text = Path(p["v743_design_reference"]["path"]).read_text()
        modes = [
            "staged",
            "activated",
            "count",
            "order",
            "id",
            "title",
            "phase",
            "deliverable",
            "MODEL_SPECS",
            "model",
            "inference_substrate_class",
            "gate",
            "prompt",
            "retirement",
            "marker",
            "table",
            "digest",
            "rehashed",
        ]
        for index, mode in enumerate(modes):
            value = dict(milestone=MILESTONE, tasks=deepcopy(p["task_map_v743"]))
            changed = text
            if mode == "count":
                value["tasks"].pop()
            if mode == "order":
                value["tasks"].reverse()
            if mode in {
                "id",
                "title",
                "phase",
                "deliverable",
                "prompt",
                "model",
                "inference_substrate_class",
            }:
                value["tasks"][0][mode] = "forged"
            if mode == "MODEL_SPECS":
                value["tasks"][0][mode] = [{"name": "forged"}]
            if mode == "gate":
                value["tasks"][1]["gated_on"][0]["value"] = 99
            if mode == "retirement":
                value["tasks"][0]["prior_failures"][0]["retire_if_same_verdict"] = False
            if mode == "marker":
                changed = changed.replace("<!-- V743_TASK_CONTRACT_END -->", "")
            if mode == "table":
                changed = changed.replace("| 1 |", "| 99 |", 1)
            if mode == "digest":
                changed = changed.replace(TASKS_PIN, "0" * 64)
            if mode == "rehashed":
                value["tasks"][0]["prompt"] += "forged"
                changed = re.sub(
                    r"(```json\s*)(.*?)(\s*```)",
                    lambda match: match[1] + __import__("json").dumps(value) + match[3],
                    changed,
                    count=1,
                    flags=re.S,
                )
                changed = changed.replace(TASKS_PIN, tasks_digest(value["tasks"]))
            design.write_text(changed)
            (private / ACTIVE).write_text(yaml.safe_dump(value))
            stage = private / STAGED
            if mode == "staged":
                stage.write_text(yaml.safe_dump(value))
                (private / ACTIVE).write_text("milestone: 2026.10.742\ntasks: []\n")
            else:
                stage.unlink(missing_ok=True)
            observed = authority(private)["ready"]
            rows.append(
                dict(
                    mutation=mode,
                    expected=mode in {"staged", "activated"},
                    observed=observed,
                    passed=observed == (mode in {"staged", "activated"}),
                )
            )
            progress("authority_control_" + mode, index + 1, len(modes) - index - 1)
    return rows


def numerical(p: Json) -> Json:
    """Evaluate one original feature row so checkpoint custody includes real arithmetic."""
    import numpy as np
    from carnot.verify.nine_input_8533 import probabilities

    progress("small_head_inference_before", 0, 1)
    head = read(p["head_checkpoint_reference"])["heads"]["spline9"][0]
    data = read(p["learning_data_reference"])["view"]["rows"][0]
    probability = float(probabilities(head, np.asarray([data["x"]], dtype=float))[0][0])
    progress("small_head_inference_after", 1, 0)
    return dict(
        evaluations=1,
        checkpoint=p["head_checkpoint_reference"],
        unit_id=data["unit_id"],
        probability=probability,
        parameter_count=len(head["parameters"]),
        finite=bool(np.isfinite(probability)),
        language_model_calls=0,
    )


def gate(path: Path, field: str, expected: Any, observed: Any, upstream: str = TASK) -> Json:
    """Keep absent files, absent fields and observed zero distinct for reopening decisions."""
    row = dict(base.gate(path, field, expected, observed, upstream))
    row["operand_state"] = (
        "missing_file"
        if not path.is_file()
        else "missing_field"
        if observed is None
        else "observed_zero"
        if observed == 0
        else "observed_value"
    )
    return row
