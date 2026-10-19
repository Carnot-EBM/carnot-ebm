"""REQ-REPORT-8579: bind current tasks without changing inherited science.

Original evidence qualifies each branch independently. Missing optional devices
cannot suppress a CPU registration or turn historical nulls into benefits.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
from typing import Any
from unittest.mock import patch

import yaml

from carnot.reporting import v736_contract_methods as base
from carnot.reporting import v736_historical_inputs as history
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.roadmap_contract import compare_contract, parse_design
from carnot.reporting.v709_execution import child as child

Json = dict[str, Any]
ROOT = base.ROOT
NAME, TASK, MILESTONE = (
    "experiment_8579_v737_contract_methods",
    "exp8579-contract-methods",
    "2026.10.737",
)
DESIGN, ACTIVE, STAGED = base.DESIGN, base.ACTIVE, base.STAGED
PROTOCOL = "openspec/change-proposals/v737-methods-protocol.json"
PROTOCOL_PIN = "sha256:d3c46c6e94b66198f1b2bf049152edf382bc4e181af95b5b89cdca8dad4dd475"
TASKS_PIN = "2ea815d961d6c8c5a9063608219869c72944c46278617125da74ddb48a8e9c75"
INHERITED = base.INHERITED
CLI = f"scripts/experiments/{NAME}.py"
TESTS = [f"tests/python/test_v737_contract_{part}_8579.py" for part in ["methods", "runner"]]
TEST = TESTS[0]
OWNED = [
    f"python/carnot/reporting/v737_contract_{part}.py" for part in ["methods", "report", "runner"]
] + [CLI]
MODEL_SPECS: list[Json] = []
reference, read, tasks_digest = base.reference, base.read, base.tasks_digest
checksum, atomic_json, sha256_file = base.checksum, base.atomic_json, base.sha256_file
checked, family = history.checked, history.family
_MEASURE, _RUNTIME = base.measure, history.runtime


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush counts so a supervisor sees both work and pending obligations."""
    if phase in {
        "preconditions_before",
        "authority_before",
        "authority_after",
        "measurement_frozen",
    }:
        completed = min(completed, 12)
        pending = 12 - completed
    print(f"[exp8579] phase={phase} completed={completed} pending={pending}", flush=True)


def gate(path: Path, field: str, expected: Any, observed: Any, upstream: str = TASK) -> Json:
    """Attribute missing and changed operands to this invocation or their original producer."""
    return dict(base.gate(path, field, expected, observed, upstream))


def protocol() -> Json:
    """Separate byte pins prevent repaired envelope hashes from changing scientific rules."""
    p = read(dict(path=str(ROOT / PROTOCOL), sha256=PROTOCOL_PIN))
    old = read(p["inherited_v735_protocol_reference"])
    resident = read(p["inherited_v736_protocol_reference"])
    checks = {
        k: dict(
            source="V736" if k == "resident_session" else "V735",
            value_equal=p[k] == (resident if k == "resident_session" else old)[k],
            sha256=canonical_hash(p[k]),
        )
        for k in [*INHERITED, "resident_session"]
    }
    if (
        not all(r["value_equal"] for r in checks.values())
        or checks != p["inherited_operand_checks"]
        or tasks_digest(p["task_map_v737"]) != TASKS_PIN
    ):
        raise ValueError("scientific_or_task_operand")
    ref = p["preserved_v735_design_reference"]
    if (
        ref["sha256"] != "sha256:f198170ac62aeb1074482f9b35dd71995f9c8f6ada4235b831958124a7dbfd15"
        or sha256_file(Path(ref["path"])) != ref["sha256"]
    ):
        raise ValueError("original_design")
    for key in [
        "literature_reference",
        "conductor_reference",
        "v737_activation_reference",
        "v737_design_reference",
        "methods_reference",
    ]:
        if not checked(p[key]):
            raise ValueError("immutable_registration_input")
    return dict(p)


def authority(root: Path) -> Json:
    """Compare full objects to genuine activated bytes when staging was consumed."""
    p = protocol()
    paths = [root / DESIGN, root / ACTIVE]
    refs = [reference(path) for path in paths if path.is_file()]
    failures = []
    try:
        text = paths[0].read_text()
        active = yaml.safe_load(paths[1].read_bytes())
        stage = root / STAGED
        staged = (
            yaml.safe_load(stage.read_bytes())
            if stage.is_file()
            else yaml.safe_load(Path(p["v737_activation_reference"]["path"]).read_bytes())
        )
        if stage.is_file():
            refs.append(reference(stage))
        result = compare_contract(
            text, staged, active, milestone=MILESTONE, first_id=8579, count=12
        )
        machine = parse_design(text, milestone=MILESTONE)[1]
        for i, row in enumerate(result["rows"]):
            row["checks"]["full_dictionary"] = (
                i < len(active["tasks"])
                and active["tasks"][i] == p["task_map_v737"][i] == staged["tasks"][i] == machine[i]
            )
            row["matched"] = all(row["checks"].values())
        ready = bool(
            result["passed"]
            and all(r["matched"] for r in result["rows"])
            and re.search(
                r"(?:canonical_tasks_sha256:|Canonical task SHA256:)\s*`?" + TASKS_PIN, text
            )
            and "<!-- V737_TASK_CONTRACT_END -->" in text
            and tasks_digest(active["tasks"]) == TASKS_PIN
        )
        rows = result["rows"]
        if not ready:
            failures.append(gate(paths[0], "twelve_complete_objects", True, False))
    except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
        ready = False
        failures.append(gate(paths[0], "v737_design_and_activation", MILESTONE, str(error)))
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
            for i, t in enumerate(p["task_map_v737"])
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
    """A source change invalidates its consumers rather than unrelated original families."""
    numbers = [8431, 8529, 8552, 8554, 8571, 8572]
    rows = []
    for i, n in enumerate(numbers):
        progress(f"family_{n}_before", i, len(numbers) - i)
        rows.append(family(p["original_producers"][str(n)], n))
        progress(f"family_{n}_after", i + 1, len(numbers) - i - 1)
    by = {r["experiment_id"]: r for r in rows}
    failures = [g for row in rows for g in row["gate_check_summary"]]
    head_ok, union_ok = by[8431]["qualified"], by[8554]["qualified"]
    try:
        head = read(p["original_producers"]["8431"])
        ref = dict(path=head["frozen_heads_path"], sha256=head["frozen_heads_sha256"])
        if not checked(ref):
            raise ValueError("original_head_scaler_temperature_hash")
        by[8431]["references"].append(ref)
    except (OSError, ValueError, KeyError, TypeError) as error:
        head_ok = False
        failures.append(
            gate(
                Path(p["original_producers"]["8431"]["path"]),
                "original_head_scaler_temperature",
                True,
                str(error),
                "exp8431",
            )
        )
    try:
        receipt = read(p["original_producers"]["8554"])
        primitive = read(receipt["primitive_reference"])
        from carnot.reporting.historical_learning_8554 import exact_union

        union = exact_union(primitive["nodes"], primitive["receipts"])
        union_ok = bool(
            union_ok
            and len(primitive["nodes"]) == 158
            and union == receipt["exact_union_receipt"]
            and union["passed"]
        )
        if not union_ok:
            raise ValueError("exact_158_node_original_union")
    except (OSError, ValueError, KeyError, TypeError) as error:
        union_ok = False
        failures.append(
            gate(
                Path(p["original_producers"]["8554"]["path"]),
                "exact_158_node_original_union",
                True,
                str(error),
                "exp8554",
            )
        )
    resident_ok = all(by[n]["qualified"] for n in [8571, 8572])
    for n, field in [(8571, "prototype_ready_score"), (8572, "resident_inference_ready_score")]:
        try:
            observed = read(p["original_producers"][str(n)]).get(field)
        except (OSError, ValueError, KeyError, TypeError):
            observed = None
        if observed != 1:
            resident_ok = False
            failures.append(
                gate(Path(p["original_producers"][str(n)]["path"]), field, 1, observed, f"exp{n}")
            )
    return dict(
        families=rows,
        gate_check_summary=failures,
        learning_inputs_ready_score=int(head_ok and union_ok),
        resident_inputs_ready_score=int(head_ok and resident_ok),
        canary_inputs_ready_score=int(head_ok and by[8529]["qualified"] and by[8552]["qualified"]),
        references=[r for row in rows for r in row["references"]],
        historical_model_provenance=[
            dict(experiment_id=r["experiment_id"], provenance=r["historical_model_provenance"])
            for r in rows
        ],
    )


def runtime(root: Path, raw: Path | None = None) -> Json:
    """Use only the supplied receipt reader in a bounded child, without a device probe."""
    with patch.object(history, "child", child):
        return dict(_RUNTIME(root, raw))


def measure(root: Path, raw: Path) -> Json:
    """Reuse resource checks while freezing current authority for later downstream readers."""
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
        patch.object(history, "historical", historical),
        patch.object(history, "runtime", lambda ignored, logs: runtime(root, logs)),
    ):
        work = dict(_MEASURE(snapshot, raw))
    progress("twelve_task_measurement_complete", len(work["contract"]["rows"]), 0)
    return work
