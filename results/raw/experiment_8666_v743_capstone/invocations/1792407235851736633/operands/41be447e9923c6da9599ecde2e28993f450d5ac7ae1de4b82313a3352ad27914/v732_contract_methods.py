"""REQ-REPORT-8512: current authority cannot repair failed historical authority."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

import yaml

from carnot.reporting import v729_contract_methods as shipped
from carnot.reporting import v731_contract_methods as previous
from carnot.reporting import v732_historical_inputs as history
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.roadmap_contract import parse_design

Json = dict[str, Any]
base = shipped.base
ROOT, DESIGN, ACTIVE, STAGED = shipped.ROOT, shipped.DESIGN, shipped.ACTIVE, shipped.STAGED
NAME, TASK, MILESTONE = (
    "experiment_8512_v732_contract_methods",
    "exp8512-contract-methods",
    "2026.10.732",
)
PROTOCOL = "openspec/change-proposals/v732-methods-protocol.json"
METHODS = "docs/research-notes/v732-contract-methods.md"
PROTOCOL_PIN = "sha256:66b761adbfee7c7671d66a9b93ec78afda3508097d95a206ac7d83f1faba2c94"
METHODS_PIN = "sha256:916b9c07ccfd5600cac55cebc99581808c4ce8263857b48266004347f420b231"
TASKS_PIN = "7554f32e4312fe5da9ab7ddb2fe0ecaf760ee243e2aab15e6d7c58e53eef77b2"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v732_contract_methods_8512.py"
OWNED = [
    f"python/carnot/reporting/v732_{n}.py"
    for n in ["contract_methods", "contract_runner", "historical_inputs"]
] + [CLI]
MODEL_SPECS: list[Json] = []
FIELDS = history.FIELDS
INHERITED = (*previous.INHERITED, "identity_brier", "primary_contrasts")
read, reference, checksum, sha256_file, tasks_digest = (
    shipped.read,
    shipped.reference,
    shipped.checksum,
    shipped.sha256_file,
    shipped.tasks_digest,
)
_MEASURE, _BUILD, _REPLAY, _AUTHORITY, _ASSESS = (
    shipped.measure,
    shipped.build,
    shipped.replay,
    base.authority,
    shipped._ASSESS,
)
measurement_clock = shipped.measurement_clock

SECTION_PINS = {
    "optimization_diagnostics": "sha256:b069483fc7bab696318de649cb9e7ad3400e3a1d924789e528b81ed39eccbb4e",
    "arc_mutation_receipts": "sha256:c9557b33dcb2ad7b1d7126ce38110462778372bfd570e9f0f5840d828cf5deab",
    "protocol_delta": "sha256:799166975f733ddc0d6d1b487fd6e13aebf117bfb3dcd0e1bf57c810bae9735b",
    "ledger_lag_v731": "sha256:55c532e03ac3160743ec9a8c94266119f1cfe2889733b694446b535706b2ba74",
    "v731_scheduling_dispositions": "sha256:eb839492776d8891cbf23fa9730791c2bbbb233c6051fe5014c6c437e50fe215",
    "literature_mapping_v732": "sha256:1d13119d13ff86132271494733761a9b21983635f3a45081958ba55331fa98c4",
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Actual counts keep bounded readers visible without adding artificial work."""
    print(f"[exp8512] phase={phase} completed={completed} pending={pending}", flush=True)


def protocol() -> Json:
    """The original reader keeps method validity separate from its failed task gate."""
    old = previous.protocol()
    if sha256_file(ROOT / METHODS) != METHODS_PIN:
        raise ValueError("methods_hash")
    p = dict(read(dict(path=str(ROOT / PROTOCOL), sha256=PROTOCOL_PIN)))
    if any(p[k] != old[k] for k in INHERITED):
        raise ValueError("inherited_scientific_operand")
    if (
        sha256_file(Path(p["scalar_basis_reference"]["path"]))
        != p["scalar_basis_reference"]["sha256"]
    ):
        raise ValueError("shipped_scalar_basis_hash")
    ref = p["preserved_v731_design_reference"]
    if (
        ref["sha256"] != "sha256:0958abd881d34263ccb8d967d8cfbdfb8c7ede315e5f97812ed4b7c5f6c517e6"
        or sha256_file(Path(ref["source_path"])) != ref["sha256"]
    ):
        raise ValueError("failed_history_bytes")
    if tasks_digest(p["task_map_v732"]) != TASKS_PIN:
        raise ValueError("sealed_task_map")
    if any(canonical_hash(p[k]) != pin for k, pin in SECTION_PINS.items()):
        raise ValueError("separate_registration_operand")
    return p


@contextmanager
def binding() -> Iterator[None]:
    """Change only identity and narrow adapters; shipped checks retain their assertions."""
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
            assess_authorities=lambda *a, **kw: _ASSESS(*a, **dict(kw, first_id=8512, count=14)),
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
    """Bind complete objects and end markers while recording consumed staging honestly."""
    with binding():
        result = dict(_AUTHORITY(root, raw))
    design = root / DESIGN
    text = design.read_text() if design.is_file() else ""
    failures = result["gate_check_summary"]
    for marker in ["START", "END"]:
        if f"<!-- V732_TASK_CONTRACT_{marker} -->" not in text:
            failures.append(base.failure(design, "complete_json_" + marker.lower(), True, None))
    if (root / STAGED).is_file() and text:
        machine = parse_design(text, milestone=MILESTONE)[1]
        if yaml.safe_load((root / STAGED).read_bytes()).get("tasks") != machine:
            failures.append(base.failure(root / STAGED, "staged_full_task_objects", True, False))
    activation = protocol()["v732_activation_reference"]
    if (
        sha256_file(Path(activation["path"])) != activation["sha256"]
        or tasks_digest(yaml.safe_load(Path(activation["path"]).read_bytes())["tasks"]) != TASKS_PIN
    ):
        failures.append(
            base.failure(Path(activation["path"]), "immutable_activation_receipt", TASKS_PIN, None)
        )
    result["activation_receipt"] = activation
    result["activated"] = not failures
    for i, row in enumerate(result["contract_rows"]):
        present = i < len(result["tasks"])
        row.update(
            unit_id=result["tasks"][i]["id"] if present else f"exp{8512 + i}",
            family=f"exp{8512 + i}",
            status="completed" if present else "unstarted",
            censored=not present,
            effective_independent_groups=0,
            missing_reason=None if row["matched"] else "authority_absent_or_mismatched",
        )
    return result


def historical(root: Path, raw: Path) -> Json:
    """Each family replays under its own historical authority and original invocation date."""
    return dict(history.historical(root, raw, CLI, protocol()))


def measure(root: Path, raw: Path) -> Json:
    """Seal current authority before bounded, model-free historical qualification."""
    with binding():
        work = dict(_MEASURE(root, raw))
    work["preconditions"].update(task_cap_s=4800, measurement_cap_s=3600, validation_reserve_s=900)
    atomic_json(raw / "measurement.json", work)
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Readiness qualifies methods and inputs without claiming scientific benefit."""
    receipts = receipts + [
        r
        for r in work["history"]["receipts"]
        if not r.get("normal_exit", True) and r not in receipts
    ]
    with binding():
        value = dict(_BUILD(work, receipts, raw, output))
    qualified = value["required_checks_passed"]
    families = {f["experiment_id"]: f["qualified"] for f in work["history"]["families"]}
    value.update(
        experiment_id=8512,
        run_date="20261014",
        task_id=TASK,
        milestone=MILESTONE,
        honest_verdict=f"complete_{value['verdict_class']}_v732_contract_methods",
        random_seed=7328512,
        future_dependencies=[
            dict(experiment_id=n, status="future_dependency_not_observation")
            for n in range(8513, 8526)
        ],
        protocol_path=str(ROOT / PROTOCOL),
        protocol_sha256=PROTOCOL_PIN,
        contract_rows=value["rows"],
        authority_snapshots=work["contract"]["authority_snapshots"],
        activation_receipt=work["contract"]["activation_receipt"],
        exposure_ledger=work["protocol"]["exposure_ledger"],
        retirement_ledger=work["protocol"]["retirement_ledger"],
        ledger_lag=work["protocol"]["ledger_lag_v731"],
        v731_scheduling_dispositions=work["protocol"]["v731_scheduling_dispositions"],
        qualification_registrations={
            k: work["protocol"][k] for k in ["optimization_diagnostics", "arc_mutation_receipts"]
        },
        literature_mapping=work["protocol"]["literature_mapping_v732"],
        historical_authority_limits=dict(
            value["historical_authority_limits"],
            V731="failed incomplete authority; three producers, three pre-gates, six cascade skips and a capstone; source/objective/append science unmeasured",
            V730="source exposed-history; objective publication flagged and append speed disqualified",
        ),
        methodology_note="Administrative full-object V732 custody and unchanged independently sealed V731 methods. Original V726 and V730 family consumers qualify inputs separately. Zero model loads, generations, device probes or board calls. Exposed history grants no generalization or new scientific benefit.",
    )
    value["service_inputs_ready_score"] = int(
        qualified and all(families.get(n, False) for n in [8407, 8492, 8493])
    )
    value["acceptance_gates"]["service_inputs"] = bool(value["service_inputs_ready_score"])
    value["field_principles"].update(
        {
            k: "Bind to sealed invocation operands; qualification does not establish scientific benefit."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["field_principles"]["service_inputs_ready_score"] = (
        "Original Exp8407 five-input workload and qualified V730 Exp8492 profile/Exp8493 append store; disqualified Exp8494 benefit stays excluded."
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Shipped cold reducers distrust aggregates, repaired hashes and altered original operands."""
    with binding():
        return bool(_REPLAY(path))
