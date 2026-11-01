"""REQ-REPORT-8615: freeze current authority without changing scientific operands.

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

from carnot.reporting import v739_contract_methods as previous
from carnot.reporting import v738_contract_methods as scientific
from carnot.reporting import v736_contract_methods as base
from carnot.reporting import v736_historical_inputs as readers
from carnot.reporting.current_work_receipt import canonical_hash as canonical_hash
from carnot.reporting.roadmap_contract import compare_contract, parse_design


Json = dict[str, Any]
ROOT = base.ROOT
NAME, TASK, MILESTONE = (
    "experiment_8615_v740_contract_methods",
    "exp8615-contract-methods",
    "2026.10.740",
)
DESIGN, ACTIVE, STAGED = base.DESIGN, base.ACTIVE, base.STAGED
PROTOCOL = "openspec/change-proposals/v740-methods-protocol.json"
PROTOCOL_PIN = "sha256:e498f2c621b7c85bdc1262181a734ab94db73082ad860df0ed36e76f8bf415de"
TASKS_PIN = "38c2808bdfa97a61b2312d5482f0a30b20f0ab45ac0f23fab67186b1bb5b30c3"
INHERITED = ["objective_updates", "objective_audit", "evidence_canary", "resident_session"]
CLI = f"scripts/experiments/{NAME}.py"
TESTS = [f"tests/python/test_v740_contract_{s}_8615.py" for s in ["methods", "runner"]]
TEST = TESTS[0]
OWNED = [
    f"python/carnot/reporting/v740_contract_{s}.py" for s in ["methods", "report", "runner"]
] + [CLI]
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
        completed = min(completed, 12)
        pending = 12 - completed
    print(f"[exp8615] phase={phase} completed={completed} pending={pending}", flush=True)


def protocol() -> Json:
    """Independent byte pins prevent a repaired envelope from changing inherited science."""
    p = read(dict(path=str(ROOT / PROTOCOL), sha256=PROTOCOL_PIN))
    inherited = read(p["inherited_v738_protocol_reference"])
    checks = {
        k: dict(source="V738", value_equal=p[k] == inherited[k], sha256=canonical_hash(p[k]))
        for k in INHERITED
    }
    refs = [
        p[k]
        for k in [
            "inherited_v738_protocol_reference",
            "inherited_v739_protocol_reference",
            "v740_activation_reference",
            "v740_design_reference",
            "literature_reference",
            "methods_reference",
            "conductor_reference",
        ]
    ]
    refs += [r["source_reference"] for r in p["literature_method_rows"]]
    if (
        p["inherited_v738_protocol_reference"]["sha256"] != scientific.PROTOCOL_PIN
        or not all(r["value_equal"] for r in checks.values())
        or checks != p["inherited_operand_checks"]
        or tasks_digest(p["task_map_v740"]) != TASKS_PIN
        or not all(checked(r) for r in refs)
    ):
        raise ValueError("sealed_scientific_or_registration_operand")
    return dict(p)


def authority(root: Path) -> Json:
    """Use staging or real V740 activation; an older live queue has no current authority."""
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
            text, planned, actual, milestone=MILESTONE, first_id=8615, count=12
        )
        machine = parse_design(text, milestone=MILESTONE)[1]
        for i, row in enumerate(result["rows"]):
            row["checks"]["full_dictionary"] = (
                i < len(planned["tasks"])
                and planned["tasks"][i] == p["task_map_v740"][i] == machine[i] == actual["tasks"][i]
            )
            row["checks"]["prior"] = (
                not p["task_map_v740"][i].get("prior_failures") or row["checks"]["prior"]
            )
            row["matched"] = all(row["checks"].values())
            row["excluded"] = not row["matched"]
            row["raw_numerator"] = sum(row["checks"].values())
            row["raw_denominator"] = len(row["checks"])
        ready = bool(
            not set(result["errors"]) - {"row_mismatch"}
            and all(r["matched"] for r in result["rows"])
            and re.search(
                r"(?:canonical_tasks_sha256:|Canonical task SHA256:)\s*`?" + TASKS_PIN, text
            )
            and "<!-- V740_TASK_CONTRACT_END -->" in text
            and tasks_digest(planned["tasks"]) == TASKS_PIN
        )
        refs = [reference(root / DESIGN), reference(chosen)]
        rows = result["rows"]
        if not ready:
            failures.append(gate(root / DESIGN, "twelve_complete_objects", True, False, TASK))
    except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
        ready = False
        failures.append(
            gate(root / DESIGN, "v740_design_and_authority", MILESTONE, str(error), TASK)
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
            for i, t in enumerate(p["task_map_v740"])
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
    """Resolve exact V738 paths to sealed original bytes; old verdicts remain unchanged."""
    original_checked = readers.checked
    replacements = {
        str(ROOT / ACTIVE): p["v738_activation_reference"],
        str(ROOT / DESIGN): p["v738_design_reference"],
    }

    def preserved(ref: Json) -> bool:
        replacement = replacements.get(ref.get("path", ""))
        if replacement is not None and ref["sha256"] == replacement["sha256"]:
            return bool(original_checked(replacement))
        return bool(original_checked(ref))

    with patch.object(readers, "checked", preserved):
        h = dict(previous.historical(p))
        rows = [readers.family(p["original_producers"][str(n)], n) for n in [8608, 8609]]
    for row in rows:
        row["historical_readiness_promoted"] = False
    h["families"] += rows
    h["references"] += [r for row in rows for r in row["references"]]
    h["gate_check_summary"] += [g for row in rows for g in row["gate_check_summary"]]
    for number, field in [
        (8593, "resident_measurement_ready_score"),
        (8596, "resident_cost_ready_score"),
        (8608, "lean_prototype_ready_score"),
        (8609, "required_checks_passed"),
    ]:
        ref = p["original_producers"][str(number)]
        try:
            observed = read(ref).get(field)
        except (OSError, ValueError, KeyError, TypeError):
            observed = None
        if observed != 1:
            h["resident_inputs_ready_score"] = 0
            h["gate_check_summary"].append(
                gate(Path(ref["path"]), field, 1, observed, f"exp{number}")
            )
    h["resident_inputs_ready_score"] = int(
        h["resident_inputs_ready_score"] and all(r["qualified"] for r in rows)
    )
    return h


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
    work["authority_mutation_rows"] = mutation_controls(snapshot)
    work["end_ns"] = time.monotonic_ns()
    work["duration_s"] = (work["end_ns"] - work["start_ns"]) / 1e9
    work["phase_spans"][0].update(end_monotonic_ns=work["end_ns"], duration_s=work["duration_s"])
    atomic_json(raw / "measurement.json", work)
    progress("twelve_task_measurement_frozen", 12, 0)
    return work


def mutation_controls(root: Path) -> list[Json]:
    """Exercise staging, activation and independent mutations without touching live files."""
    p = protocol()
    rows = []
    with TemporaryDirectory(prefix="exp8615-authority-", dir="/var/tmp") as directory:
        private = Path(directory)
        design = private / DESIGN
        design.parent.mkdir(parents=True)
        text = (root / DESIGN).read_text() if (root / DESIGN).is_file() else ""
        # Controls use the sealed complete design even when a private caller omits authority.
        if not text:
            text = Path(p["v740_design_reference"]["path"]).read_text()
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
            value = dict(milestone=MILESTONE, tasks=deepcopy(p["task_map_v740"]))
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
                changed = changed.replace("<!-- V740_TASK_CONTRACT_END -->", "")
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
                (private / ACTIVE).write_text("milestone: 2026.10.739\ntasks: []\n")
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
