"""REQ-REPORT-8602: freeze current authority without changing scientific operands.

Registration establishes which work may proceed. It cannot establish learning
benefit, and optional resources cannot suppress independent CPU evidence.
"""

from __future__ import annotations

from pathlib import Path
import re
import time
from typing import Any
from unittest.mock import patch

import yaml

from carnot.reporting import v738_contract_methods as previous
from carnot.reporting import v736_contract_methods as base
from carnot.reporting import v736_historical_inputs as readers
from carnot.reporting.current_work_receipt import canonical_hash as canonical_hash
from carnot.reporting.roadmap_contract import compare_contract, parse_design


Json = dict[str, Any]
ROOT = base.ROOT
NAME, TASK, MILESTONE = (
    "experiment_8602_v739_contract_methods",
    "exp8602-contract-methods",
    "2026.10.739",
)
DESIGN, ACTIVE, STAGED = base.DESIGN, base.ACTIVE, base.STAGED
PROTOCOL = "openspec/change-proposals/v739-methods-protocol.json"
PROTOCOL_PIN = "sha256:08706ec3730aac2f7dc27f2c28277d93c93cbe11644f4d7385988aeeab8d4088"
TASKS_PIN = "042e6ad2f6348ba50220c1b107338df99b5f7a61a05ff11ddc92e11203dfe9ad"
INHERITED = [*base.INHERITED, "resident_session"]
CLI = f"scripts/experiments/{NAME}.py"
TESTS = [f"tests/python/test_v739_contract_{s}_8602.py" for s in ["methods", "runner"]]
TEST = TESTS[0]
OWNED = [
    f"python/carnot/reporting/v739_contract_{s}.py" for s in ["methods", "report", "runner"]
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
        completed = min(completed, 13)
        pending = 13 - completed
    print(f"[exp8602] phase={phase} completed={completed} pending={pending}", flush=True)


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
            "v739_activation_reference",
            "v739_design_reference",
            "literature_reference",
            "methods_reference",
            "conductor_reference",
        ]
    ]
    refs += [r["source_reference"] for r in p["literature_method_rows"]]
    if (
        p["inherited_v738_protocol_reference"]["sha256"] != previous.PROTOCOL_PIN
        or not all(r["value_equal"] for r in checks.values())
        or checks != p["inherited_operand_checks"]
        or tasks_digest(p["task_map_v739"]) != TASKS_PIN
        or not all(checked(r) for r in refs)
    ):
        raise ValueError("sealed_scientific_or_registration_operand")
    return dict(p)


def authority(root: Path) -> Json:
    """Use staging or real V739 activation; an older live queue has no current authority."""
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
            text, planned, actual, milestone=MILESTONE, first_id=8602, count=13
        )
        machine = parse_design(text, milestone=MILESTONE)[1]
        for i, row in enumerate(result["rows"]):
            row["checks"]["full_dictionary"] = (
                i < len(planned["tasks"])
                and planned["tasks"][i] == p["task_map_v739"][i] == machine[i] == actual["tasks"][i]
            )
            row["checks"]["prior"] = (
                not p["task_map_v739"][i].get("prior_failures") or row["checks"]["prior"]
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
            and "<!-- V739_TASK_CONTRACT_END -->" in text
            and tasks_digest(planned["tasks"]) == TASKS_PIN
        )
        refs = [reference(root / DESIGN), reference(chosen)]
        rows = result["rows"]
        if not ready:
            failures.append(gate(root / DESIGN, "thirteen_complete_objects", True, False, TASK))
    except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
        ready = False
        failures.append(
            gate(root / DESIGN, "v739_design_and_authority", MILESTONE, str(error), TASK)
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
            for i, t in enumerate(p["task_map_v739"])
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
    """Current resident provenance and costs are independent of learning qualification."""
    h = dict(previous.historical(p))
    rows = []
    for n, field in [
        (8593, "resident_measurement_ready_score"),
        (8596, "resident_cost_ready_score"),
    ]:
        progress(f"family_{n}_before")
        original_checked = readers.checked
        preserved = p["v738_activation_reference"]

        def original_authority_checked(ref: Json) -> bool:
            if ref.get("path") == str(ROOT / ACTIVE) and ref["sha256"] == preserved["sha256"]:
                return bool(original_checked(preserved))
            return bool(original_checked(ref))

        with patch.object(readers, "checked", original_authority_checked):
            row = readers.family(p["original_producers"][str(n)], n)
        row["affected_dependency_revalidation"] = dict(
            original_path=str(ROOT / ACTIVE),
            preserved_reference=preserved,
            rule="Only exact original V738 authority bytes may replace its later scheduling path",
        )
        try:
            observed = read(p["original_producers"][str(n)]).get(field)
        except (OSError, ValueError, KeyError, TypeError):
            observed = None
        if observed != 1:
            row["qualified"] = False
            row["gate_check_summary"].append(
                gate(Path(p["original_producers"][str(n)]["path"]), field, 1, observed, f"exp{n}")
            )
        rows.append(row)
        progress(f"family_{n}_after", 1, 0)
    head = next(r for r in h["families"] if r["experiment_id"] == 8431)
    h["resident_inputs_ready_score"] = int(head["qualified"] and all(r["qualified"] for r in rows))
    from carnot.reporting import v738_contract_inputs as canary

    try:
        prior = read(p["original_producers"]["8591"])
        binding = prior["canary_revision_binding"]
        descriptor = Path(binding["private_cli_receipts"][0]["argv"][-1])
        validator = next(r for r in h["families"] if r["experiment_id"] == 8580)
        h["canary_inputs_ready_score"] = int(
            head["qualified"]
            and validator["qualified"]
            and binding["ready"]
            and canary.accept_canary(descriptor)
            and all(
                checked(dict(path=r[stream + "_path"], sha256=r[stream + "_sha256"]))
                for r in binding["private_cli_receipts"]
                for stream in ["stdout", "stderr"]
            )
        )
        h["references"] += [p["original_producers"]["8591"], reference(descriptor)]
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        h["canary_inputs_ready_score"] = 0
    h["families"] += rows
    h["references"] += [r for row in rows for r in row["references"]]
    h["gate_check_summary"] += [g for row in rows for g in row["gate_check_summary"]]
    return h


def runtime(root: Path, raw: Path | None = None) -> Json:
    """Read cached CUDA evidence once; metadata cannot authorize a device probe."""
    from carnot.verify import cuda_reader_8410 as reader

    directory = raw or root / "runtime"
    progress("optional_cuda_reader_before", 0, 1)
    cached_refs: list[Json] = []
    inventory_ref = None
    failures = []
    try:
        cached = reader.cached(ROOT / reader.CACHED, directory / "cached")
        current = reader.inventory(cached["previous"])
        atomic_json(directory / "inventory.json", current)
        cached_refs = cached["refs"]
        inventory_ref = reference(directory / "inventory.json")
    except (OSError, ValueError, KeyError, TypeError) as error:
        failures.append(
            gate(ROOT / reader.CACHED, "optional_cached_cuda_reader", True, str(error), "exp8410")
        )
    receipt = root / "ops/cuda-runtime-repair-receipt.json"
    failures.append(
        gate(
            receipt,
            "supplied_causal_repair_and_current_validator_custody",
            True,
            False if receipt.is_file() else None,
            "exp8410",
        )
    )
    progress("optional_cuda_reader_after", 1, 0)
    return dict(
        ready=False,
        probe_count=0,
        context_opened=False,
        gate_check_summary=failures,
        cached_references=cached_refs,
        inventory_reference=inventory_ref,
        repair_supplied=receipt.is_file(),
        duration_s=0.0,
    )


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
    work["end_ns"] = time.monotonic_ns()
    work["duration_s"] = (work["end_ns"] - work["start_ns"]) / 1e9
    work["phase_spans"][0].update(end_monotonic_ns=work["end_ns"], duration_s=work["duration_s"])
    atomic_json(raw / "measurement.json", work)
    progress("thirteen_task_measurement_frozen", 13, 0)
    return work
