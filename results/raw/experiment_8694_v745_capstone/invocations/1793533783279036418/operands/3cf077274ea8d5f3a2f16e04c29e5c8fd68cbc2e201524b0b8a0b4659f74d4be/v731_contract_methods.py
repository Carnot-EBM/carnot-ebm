"""REQ-REPORT-8499: bind thirteen tasks; missing independent design remains blocked."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import time
from typing import Any, Iterator
from unittest.mock import patch

import yaml

from carnot.reporting import v729_contract_methods as shipped
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.roadmap_contract import parse_design

Json = dict[str, Any]
base, history = shipped.base, shipped.history
ROOT, DESIGN, ACTIVE, STAGED = shipped.ROOT, shipped.DESIGN, shipped.ACTIVE, shipped.STAGED
NAME, TASK, MILESTONE = (
    "experiment_8499_v731_contract_methods",
    "exp8499-contract-methods",
    "2026.10.731",
)
PROTOCOL = "openspec/change-proposals/v731-methods-protocol.json"
METHODS = "docs/research-notes/v731-contract-methods.md"
PROTOCOL_PIN = "sha256:c723698169abcf5ff01e086ca898b7685ae5f4e01e926bd8fc16cf9e8e31270b"
METHODS_PIN = "sha256:f286cfac9807e22204bfb91ab4286c529886b53e4880d86e909c4705c0e5a70d"
TASKS_PIN = "473142c672c68b16605f548ee9c6b1903b36754f9b0fc36e9619d3df3e937f24"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v731_contract_methods_8499.py"
OWNED = [
    f"python/carnot/reporting/v731_{n}.py" for n in ["contract_methods", "contract_runner"]
] + [CLI]
MODEL_SPECS: list[Json] = []
FIELDS = shipped.FIELDS
INHERITED = (
    *shipped.INHERITED,
    "append_protocol",
    "append_cost",
    "evidence_canary",
    "validation_repairs",
)
read, reference, checksum, sha256_file, tasks_digest = (
    shipped.read,
    shipped.reference,
    shipped.checksum,
    shipped.sha256_file,
    shipped.tasks_digest,
)
_MEASURE, _BUILD, _REPLAY = (
    shipped.measure,
    shipped.build,
    shipped.replay,
)
_ASSESS = shipped._ASSESS
_HISTORICAL_CHILD = history.child
measurement_clock = time.monotonic_ns
SECTION_PINS = {
    "identity_brier": "sha256:edf685764ceb558076972d23265778b06433732c58ac8f7cc618a27c7fe50911",
    "primary_contrasts": "sha256:109c4a8402201a8271c37c1ccad479b918edb466f5574e704c502505c7e61b3c",
    "publication_qualification": "sha256:4542606cc64b95edc77e5215fd8314794636a47708165f0e1a225e51d06a8cf5",
    "strict_import_repair": "sha256:a82502dafd735db0f1411d2a0b8f83984660f86da906bc8e7321605482306bb7",
    "protocol_delta": "sha256:5b91403b45ee2370a5989fd85ad347e71ca6746daf83737ae3ca457851407542",
    "ledger_lag": "sha256:fe1f40640be6fcc52de81626354988925f66cba3f1afe8c3cf51d8301338fccd",
    "preserved_v730_design_reference": "sha256:ef32218c161c56d16327fd027b9d110b72d6101b135583acd0220eaf0813815b",
    "v730_authority_references": "sha256:4a578b5d8431e01966d6a3045c74d3cb2b26fb85d3a69ce8ee2fc74fbc72f635",
    "v730_fact_references": "sha256:742a64c0bb83d503b0074cfea547db8a018deabd69f73ac2f6dc6aaec008fb41",
    "failed_v730_qualification": "sha256:6ba9b39a53c86d4014b2b309a7aace5cf29a8413755e52b80f4ba93e7920b67f",
    "exposure_ledger": "sha256:81496cf90a82da931d5b3c115b8f497f7883e6f9e8d2aa3892094120d045b1fa",
    "retirement_ledger": "sha256:db3c2bafb693f77c1e6652e75502e78233e44297bbab2a751c4f05f0462cbe19",
    "task_methods_v731": "sha256:3de0b023355041cb3972ae020c42625ce73222bfc1693f167b0422cced26b17a",
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual counts so the supervisor can see bounded work advancing."""
    if phase == "preconditions_before":
        pending = 13
    if phase == "methods_frozen_after":
        completed = 13
    if phase == "authority_after":
        pending = max(0, 13 - completed)
    print(f"[exp8499] phase={phase} completed={completed} pending={pending}", flush=True)


def protocol() -> Json:
    """Independent seals preserve scientific operands even after a candidate is rehashed."""
    if sha256_file(ROOT / METHODS) != METHODS_PIN:
        raise ValueError("methods_hash")
    p = dict(read(dict(path=str(ROOT / PROTOCOL), sha256=PROTOCOL_PIN)))
    old = read(p["inherited_protocol_reference"])
    if any(p[k] != old[k] for k in INHERITED):
        raise ValueError("inherited_scientific_operand")
    ref = p["scalar_basis_reference"]
    if sha256_file(Path(ref["path"])) != ref["sha256"]:
        raise ValueError("shipped_scalar_basis_hash")
    if any(canonical_hash(p[k]) != pin for k, pin in SECTION_PINS.items()):
        raise ValueError("separate_validation_or_view_operand")
    return p


@contextmanager
def binding() -> Iterator[None]:
    """Narrow dated adapters retain shipped assertions and the original historical readers."""
    constants: Json = dict(
        NAME=NAME,
        TASK=TASK,
        MILESTONE=MILESTONE,
        PROTOCOL=PROTOCOL,
        METHODS=METHODS,
        PROTOCOL_PIN=PROTOCOL_PIN,
        METHODS_PIN=METHODS_PIN,
        TASKS_PIN=TASKS_PIN,
        OWNED=OWNED,
        CLI=CLI,
        TEST=TEST,
    )
    with (
        patch.multiple(
            base,
            **constants,
            protocol=protocol,
            authority=authority,
            build=build,
            progress=progress,
            assess_authorities=lambda *a, **kw: _ASSESS(*a, **dict(kw, first_id=8499, count=13)),
        ),
        patch.multiple(base.legacy, TASK=TASK, MILESTONE=MILESTONE),
        patch.multiple(
            shipped,
            **constants,
            binding=binding,
            protocol=protocol,
            historical=historical,
            measurement_clock=measurement_clock,
            build=build,
        ),
    ):
        yield


def authority(root: Path, raw: Path) -> Json:
    """The shipped reader checks order; full equality also binds unlisted task fields."""
    paths = [root / n for n in [DESIGN, STAGED, ACTIVE]]
    snapshots = {}
    for role, path in zip(["design", "staged", "active"], paths):
        if path.is_file():
            snapshots[role] = base.snapshot(path, raw, role)
        else:
            snapshots[role] = dict(source_path=str(path), exists=False, sha256=None)
    try:
        table, machine = parse_design(paths[0].read_text(), milestone=MILESTONE)
        result = base.assess_authorities(
            *paths, raw / "typed", milestone=MILESTONE, first_id=8499, count=13
        )
        active = yaml.safe_load(paths[2].read_bytes()) if paths[2].is_file() else {}
        tasks = active.get("tasks", [])
        for i, row in enumerate(result["contract_rows"]):
            row["checks"]["full_object"] = (
                i < len(tasks) and i < len(machine) and tasks[i] == machine[i]
            )
            row["matched"] = all(row["checks"].values())
            row.update(
                absolute_metric=int(row["matched"]),
                raw_numerator=sum(row["checks"].values()),
                raw_denominator=len(row["checks"]),
                excluded=not row["matched"],
            )
        failures = result["gate_check_summary"]
        if tasks_digest(machine) != TASKS_PIN or result["canonical_tasks_sha256"] != TASKS_PIN:
            failures.append(
                base.failure(
                    paths[0], "pinned_complete_task_digest", TASKS_PIN, tasks_digest(machine)
                )
            )
        if not all(row["matched"] for row in result["contract_rows"]):
            failures.append(base.failure(paths[2], "full_task_objects", True, False))
        for marker in ["START", "END"]:
            if f"<!-- V731_TASK_CONTRACT_{marker} -->" not in paths[0].read_text():
                failures.append(
                    base.failure(paths[0], "complete_json_" + marker.lower(), True, None)
                )
        if paths[1].is_file():
            staged = yaml.safe_load(paths[1].read_bytes())
            if staged.get("tasks") != machine:
                failures.append(base.failure(paths[1], "staged_full_task_objects", True, False))
        result.update(
            observed_count=len(tasks),
            tasks=tasks,
            observed_active_tasks_sha256=tasks_digest(tasks),
            activated=not failures,
            gate_check_summary=failures,
        )
    except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
        rows = [
            dict(
                unit_id=f"exp{8499 + i}",
                family=f"exp{8499 + i}",
                order=i + 1,
                arm="active_contract",
                seed=None,
                status="unstarted",
                checks=dict(independent_authority=False),
                matched=False,
                absolute_metric=0,
                raw_numerator=0,
                raw_denominator=1,
                censored=True,
                excluded=False,
                effective_independent_groups=0,
                missing_reason="required_authority_unavailable_or_invalid",
            )
            for i in range(13)
        ]
        result = dict(
            activated=False,
            canonical_tasks_sha256=TASKS_PIN,
            contract_rows=rows,
            observed_count=0,
            tasks=[],
            observed_active_tasks_sha256=None,
            gate_check_summary=[
                base.failure(
                    paths[0],
                    "required_authority",
                    True,
                    None if isinstance(error, FileNotFoundError) else str(error),
                )
            ],
        )
    plan = yaml.safe_load(paths[2].read_bytes()) if paths[2].is_file() else {}
    tasks = plan.get("tasks", [])
    result.update(
        observed_count=len(tasks), tasks=tasks, observed_active_tasks_sha256=tasks_digest(tasks)
    )
    if any(g["artifact_field"] == "required_authority" for g in result["gate_check_summary"]):
        result["canonical_tasks_sha256"] = None
        for field, expected in [
            ("independent_visible_task_table", 13),
            ("embedded_complete_task_objects", 13),
            ("printed_canonical_task_digest", TASKS_PIN),
            ("complete_json_end", True),
        ]:
            result["gate_check_summary"].append(base.failure(paths[0], field, expected, None))
    for i, row in enumerate(result["contract_rows"]):
        present = i < len(tasks)
        row.update(
            unit_id=tasks[i]["id"] if present else f"exp{8499 + i}",
            family=f"exp{8499 + i}",
            status="completed" if present else "unstarted",
            censored=not present,
            effective_independent_groups=0,
            missing_reason=None if row["matched"] else "authority_absent_or_mismatched",
        )
    for gate in result["gate_check_summary"]:
        gate.update(check=gate["artifact_field"], upstream_id=TASK, passed=False)
    result["authority_snapshots"] = snapshots
    result["staging_disposition"] = "existing" if paths[1].is_file() else "consumed_or_absent"
    return dict(result)


def historical(root: Path, raw: Path) -> Json:
    """Separate immutable roots replay qualified inputs without current model execution."""
    completed = 0

    def isolated(name: str, argv: list[str], logs: Path, **kwargs: Any) -> Json:
        """Report aggregate progress while each original reader keeps its bounded receipt."""
        nonlocal completed
        progress("historical_family_before", completed, len(FIELDS) - completed)
        receipt = dict(_HISTORICAL_CHILD(name, argv, logs, **kwargs))
        completed += 1
        progress("historical_family_after", completed, len(FIELDS) - completed)
        return receipt

    with patch.object(history, "child", isolated):
        return dict(history.historical(root, raw, CLI, protocol()))


def measure(root: Path, raw: Path) -> Json:
    """Seal current authority before independently bounded historical qualification."""
    with binding():
        work = dict(_MEASURE(root, raw))
    work["code_refs"] += [
        reference(ROOT / f"python/carnot/reporting/{name}.py")
        for name in [
            "v726_contract_methods",
            "v727_historical_inputs",
            "v729_contract_methods",
            "v729_historical_inputs",
            "v729_contract_runner",
        ]
    ]
    base.atomic_json(raw / "measurement.json", work)
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Reuse the complete schema while keeping methods and independent input gates distinct."""
    with binding():
        value = dict(_BUILD(work, receipts, raw, output))
    value.update(
        experiment_id=8499,
        run_date="20261014",
        intended_count=13,
        task_id=TASK,
        milestone=MILESTONE,
        honest_verdict=f"complete_{value['verdict_class']}_v731_contract_methods",
        random_seed=7318499,
        future_dependencies=[
            dict(experiment_id=n, status="future_dependency_not_observation")
            for n in range(8500, 8512)
        ],
        protocol_path=str(ROOT / PROTOCOL),
        protocol_sha256=PROTOCOL_PIN,
        contract_rows=value["rows"],
        authority_snapshots=work["contract"]["authority_snapshots"],
        historical_authority_limits=dict(
            value["historical_authority_limits"],
            V730="fourteen scheduled slots; flagged Exp8489 sidecar mismatch; Exp8494 observed speed disqualified; zero resolved ARC outcomes",
            V729="truncated design preserved byte-identically; ten cascade-blocked tasks; source/objective/append science unmeasured; Exp8482 original ARC line350 untested",
        ),
        methodology_note="Administrative V731 table/JSON/digest/activated authority; missing design remains blocked with full dictionary equality, and separately qualified original V726/V728 inputs. No model loads, generations, CUDA or board calls. Methods readiness is not scientific benefit; exposed evidence keeps both generalization scores zero.",
    )
    value["sample_size_budget"]["intended_task_slots"] = 13
    value["acceptance_gates"].pop("complete_fourteen_objects")
    value["acceptance_gates"]["complete_thirteen_objects"] = bool(
        value["current_contract_ready_score"]
    )
    value.update(
        exposure_ledger=work["protocol"]["exposure_ledger"],
        retirement_ledger=work["protocol"]["retirement_ledger"],
        ledger_lag=work["protocol"]["ledger_lag"],
        qualification_registrations={
            k: work["protocol"][k] for k in ["publication_qualification", "strict_import_repair"]
        },
    )
    value["field_principles"].update(
        {
            k: "Bind to sealed invocation operands; administrative qualification does not establish scientific benefit."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Fresh shipped reducers authenticate primitive rows, receipts and original family operands."""
    with binding():
        return bool(_REPLAY(path))
