"""REQ-REPORT-8640: freeze current authority without changing scientific operands.

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

from carnot.reporting import v741_contract_methods as previous
from carnot.reporting import v736_contract_methods as base
from carnot.reporting import v736_historical_inputs as readers
from carnot.reporting.current_work_receipt import canonical_hash as canonical_hash
from carnot.reporting.roadmap_contract import compare_contract, parse_design


Json = dict[str, Any]
ROOT = base.ROOT
NAME, TASK, MILESTONE = (
    "experiment_8640_v742_contract_methods",
    "exp8640-contract-methods",
    "2026.10.742",
)
DESIGN, ACTIVE, STAGED = base.DESIGN, base.ACTIVE, base.STAGED
PROTOCOL = "openspec/change-proposals/v742-methods-protocol.json"
PROTOCOL_PIN = "sha256:3676d3b10d9c769990185931bd844fd2bcd6cf51a5a2f552d3e659049ce4b8bb"
TASKS_PIN = "37b4a48b1d0649006b646d10749f79badecb1c9dbbf84869a92f591f7fdbe12a"
INHERITED = [
    "objective_updates",
    "objective_audit",
    "streaming_extraction_protocol",
    "installed_entry_2x2",
    "evidence_canary",
]
CLI = f"scripts/experiments/{NAME}.py"
TESTS = [f"tests/python/test_v742_contract_{s}_8640.py" for s in ["methods", "runner"]]
TEST = TESTS[0]
OWNED = [
    f"python/carnot/reporting/v742_contract_{s}.py" for s in ["methods", "report", "runner"]
] + ["python/carnot/reporting/v742_contract_inputs.py", CLI]
MODEL_SPECS: list[Json] = []
reference, read, tasks_digest = base.reference, base.read, base.tasks_digest
checksum, atomic_json, sha256_file = base.checksum, base.atomic_json, base.sha256_file
checked, gate = readers.checked, base.gate


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real boundaries so the parent can distinguish progress from a stall."""
    if phase in {
        "preconditions_before",
        "authority_before",
        "authority_after",
        "measurement_frozen",
    }:
        completed = min(completed, 13)
        pending = 13 - completed
    print(f"[exp8640] phase={phase} completed={completed} pending={pending}", flush=True)


def protocol() -> Json:
    """Authenticate original scientific operands and the independently sealed task map."""
    p = read(dict(path=str(ROOT / PROTOCOL), sha256=PROTOCOL_PIN))
    inherited = read(p["inherited_v741_protocol_reference"])
    checks = {
        k: dict(source="V741", value_equal=p[k] == inherited[k], sha256=canonical_hash(p[k]))
        for k in INHERITED
    }
    refs = [
        p[k]
        for k in [
            "inherited_v741_protocol_reference",
            "v742_activation_reference",
            "v742_design_reference",
            "literature_reference",
            "preserved_v741_design_reference",
        ]
    ]
    refs += p["historical_readers"] + [r["source_reference"] for r in p["literature_method_rows"]]
    if (
        p["inherited_v741_protocol_reference"]["sha256"] != previous.PROTOCOL_PIN
        or not all(r["value_equal"] for r in checks.values())
        or checks != p["inherited_operand_checks"]
        or tasks_digest(p["task_map_v742"]) != TASKS_PIN
        or not all(checked(r) for r in refs)
    ):
        raise ValueError("sealed_scientific_or_registration_operand")
    return dict(p)


def authority(root: Path) -> Json:
    """Use staging or real V742 activation; an older live queue has no current authority."""
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
            text, planned, actual, milestone=MILESTONE, first_id=8640, count=13
        )
        machine = parse_design(text, milestone=MILESTONE)[1]
        for i, row in enumerate(result["rows"]):
            row["checks"]["full_dictionary"] = (
                i < len(planned["tasks"])
                and planned["tasks"][i] == p["task_map_v742"][i] == machine[i] == actual["tasks"][i]
            )
            row["checks"]["prior"] = (
                not p["task_map_v742"][i].get("prior_failures") or row["checks"]["prior"]
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
            and "<!-- V742_TASK_CONTRACT_END -->" in text
            and tasks_digest(planned["tasks"]) == TASKS_PIN
        )
        refs = [reference(root / DESIGN), reference(chosen)]
        rows = result["rows"]
        if not ready:
            failures.append(gate(root / DESIGN, "thirteen_complete_objects", True, False, TASK))
    except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
        ready = False
        failures.append(
            gate(root / DESIGN, "v742_design_and_authority", MILESTONE, str(error), TASK)
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
            for i, t in enumerate(p["task_map_v742"])
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

    return dict(current(p))


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
    progress("thirteen_task_measurement_frozen", 13, 0)
    return work


def mutation_controls(root: Path) -> list[Json]:
    """Exercise staging, activation and independent mutations without touching live files."""
    p = protocol()
    rows = []
    with TemporaryDirectory(prefix="exp8640-authority-", dir="/var/tmp") as directory:
        private = Path(directory)
        design = private / DESIGN
        design.parent.mkdir(parents=True)
        text = (root / DESIGN).read_text() if (root / DESIGN).is_file() else ""
        # Controls use the sealed complete design even when a private caller omits authority.
        if not text:
            text = Path(p["v742_design_reference"]["path"]).read_text()
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
            value = dict(milestone=MILESTONE, tasks=deepcopy(p["task_map_v742"]))
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
                changed = changed.replace("<!-- V742_TASK_CONTRACT_END -->", "")
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
                (private / ACTIVE).write_text("milestone: 2026.10.741\ntasks: []\n")
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
