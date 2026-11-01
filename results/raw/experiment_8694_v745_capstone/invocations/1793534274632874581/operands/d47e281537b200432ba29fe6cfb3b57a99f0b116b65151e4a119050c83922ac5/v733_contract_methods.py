"""REQ-REPORT-8526: bind complete authority without changing historical science."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

import yaml

from carnot.reporting import v729_contract_methods as shipped
from carnot.reporting import v732_contract_methods as previous
from carnot.reporting import v733_historical_inputs as history
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.roadmap_contract import parse_design

Json = dict[str, Any]
base = shipped.base
ROOT, DESIGN, ACTIVE, STAGED = shipped.ROOT, shipped.DESIGN, shipped.ACTIVE, shipped.STAGED
NAME, TASK, MILESTONE = (
    "experiment_8526_v733_contract_methods",
    "exp8526-contract-methods",
    "2026.10.733",
)
PROTOCOL = "openspec/change-proposals/v733-methods-protocol.json"
METHODS = "docs/research-notes/v733-contract-methods.md"
PROTOCOL_PIN = "sha256:3ac4be7f48816ceb4eb4183fe6640402824a1205e5d0c84d8f3d76d0c2d84db8"
METHODS_PIN = "sha256:1660c9016e528e5a112c57bfeadf097e0b628ab3d19d9ef2ced95dc82daf95da"
TASKS_PIN = "2ef10dc73811b0b01d16722133a0ba5b5d36ee703bbc72a521f513396582ade3"
CLI = f"scripts/experiments/{NAME}.py"
TESTS = [
    f"tests/python/test_v733_{s}_8526.py"
    for s in ["contract_methods", "contract_replay", "historical_inputs"]
]
TEST = TESTS[0]
OWNED = [
    f"python/carnot/reporting/v733_{s}.py"
    for s in ["contract_methods", "contract_runner", "historical_inputs"]
] + [CLI]
MODEL_SPECS: list[Json] = []
FIELDS = history.FIELDS
INHERITED = (
    "identity_brier",
    "primary_contrasts",
    "optimization_diagnostics",
    "objective_updates",
    "objective_audit",
    "evidence_canary",
)
read, reference, checksum, sha256_file, tasks_digest = (
    shipped.read,
    shipped.reference,
    shipped.checksum,
    shipped.sha256_file,
    shipped.tasks_digest,
)
measurement_clock = shipped.measurement_clock
_BUILD, _REPLAY = shipped.build, shipped.replay
SECTION_PINS = {
    "nine_input_native_inference": "sha256:7e7323ab9571d63f89ff5220fec50f73ded29f1915e10f5dbcfdc6606ec3fb03",
    "arc_capability_mask": "sha256:ca3c0388ea6e34297e21261a7ea1001227f909a6d47e82e67de5bc09ff29b60b",
    "protocol_delta": "sha256:7fc94a9042c9bf0718eede661df86eeaeabe8a8be2a1ab064d7a9e4e256f17cb",
    "ledger_lag_v732": "sha256:31b6edef76f9bc086a26660b9528498f7a607cd50ac08d47e37ccf955ebc13e9",
    "v732_scheduling_dispositions": "sha256:b2c53f457a05c8146f128c4ff03577469af87cc444d405314cfc0d04a5329c9c",
    "literature_mapping_v733": "sha256:aed2478456f9ea2b7b086e2d2cf298a92f056e1051de13f313dc8ff5d9d8df5a",
    "gate_fields_v733": "sha256:cee94f49c935aa8d0b2c6e1b2608c685e6f6a7085b782d72e3ec2f8369f58c81",
    "task_methods_v733": "sha256:8204212267f8a56a765b8dd9d66070fe025626f49617f927f43f08e9984373b8",
    "execution_budget_v733": "sha256:1be59b47c1469d43880ace6efc9ec6cecc177fd7844a170764425f12d5f86261",
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real counts so a supervised child cannot appear silently stalled."""
    completed = 14 if phase == "finished" else completed
    print(f"[exp8526] phase={phase} completed={completed} pending={pending}", flush=True)


def protocol() -> Json:
    """Independent seals prevent a repaired envelope from changing scientific rules."""
    old = previous.protocol()
    if sha256_file(ROOT / METHODS) != METHODS_PIN:
        raise ValueError("methods_hash")
    p = dict(read(dict(path=str(ROOT / PROTOCOL), sha256=PROTOCOL_PIN)))
    if any(p[k] != old[k] for k in INHERITED):
        raise ValueError("inherited_scientific_operand")
    if tasks_digest(p["task_map_v733"]) != TASKS_PIN:
        raise ValueError("sealed_task_map")
    for key in ["preserved_v732_design_reference", "v733_activation_reference"]:
        ref = p[key]
        if (
            sha256_file(Path(ref["path"])) != ref["sha256"]
            or sha256_file(Path(ref["source_path"])) != ref["sha256"]
        ):
            raise ValueError("original_authority_hash")
    if any(canonical_hash(p[k]) != pin for k, pin in SECTION_PINS.items()):
        raise ValueError("new_method_operand")
    return p


@contextmanager
def binding() -> Iterator[None]:
    """Reuse shipped assertions with only current identity and explicit reader adapters."""
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
            assess_authorities=lambda *a, **kw: shipped._ASSESS(
                *a, **dict(kw, first_id=8526, count=14)
            ),
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
    """Full dictionaries bind prompts and routing; consumed staging is never invented."""
    with binding():
        result = dict(shipped._AUTHORITY(root, raw))
    design = root / DESIGN
    text = design.read_text() if design.is_file() else ""
    failures = result["gate_check_summary"]
    for marker in ["START", "END"]:
        if f"<!-- V733_TASK_CONTRACT_{marker} -->" not in text:
            failures.append(base.failure(design, "complete_json_" + marker.lower(), True, None))
    if (root / STAGED).is_file() and text:
        machine = parse_design(text, milestone=MILESTONE)[1]
        if yaml.safe_load((root / STAGED).read_bytes()).get("tasks") != machine:
            failures.append(base.failure(root / STAGED, "staged_full_task_objects", True, False))
    result["activation_receipt"] = protocol()["v733_activation_reference"]
    for gate in failures:
        gate.update(upstream_id=TASK, check=gate["artifact_field"], passed=False)
    result["activated"] = not failures
    for i, row in enumerate(result["contract_rows"]):
        present = i < len(result["tasks"])
        row.update(
            unit_id=result["tasks"][i]["id"] if present else f"exp{8526 + i}",
            family=f"exp{8526 + i}",
            status="completed" if present else "unstarted",
            censored=not present,
            effective_independent_groups=0,
            missing_reason=None if row["matched"] else "authority_absent_or_mismatched",
        )
    return result


def historical(root: Path, raw: Path) -> Json:
    """Original family consumers preserve their dates, flags and distinct producer fields."""
    return dict(history.historical(root, raw, CLI, protocol()))


def measure(root: Path, raw: Path) -> Json:
    """Seal current authority before bounded model-free historical qualification."""
    with binding():
        work = dict(shipped.measure(root, raw))
    work["preconditions"].update(task_cap_s=4800, measurement_cap_s=3600, validation_reserve_s=900)
    work["code_refs"] += [reference(ROOT / n) for n in TESTS[1:]]
    atomic_json(raw / "measurement.json", work)
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Independent readiness gates never turn method qualification into scientific benefit."""
    receipts = receipts + [
        r
        for r in work["history"]["receipts"]
        if not r.get("normal_exit", True) and r not in receipts
    ]
    with binding():
        value = dict(_BUILD(work, receipts, raw, output))
    p = work["protocol"]
    qualified = value["required_checks_passed"]
    families = {f["experiment_id"]: f["qualified"] for f in work["history"]["families"]}
    groups = dict(
        source=[8430, 8431, 8486, 8487],
        learning=[8430, 8431, 8433, 8434, 8516],
        nine_input=[8430, 8431],
        service=[8407, 8492, 8493, 8520],
    )
    for group, numbers in groups.items():
        score = int(qualified and all(families.get(n, False) for n in numbers))
        value[group + "_inputs_ready_score"] = score
        value["acceptance_gates"][group + "_inputs"] = bool(score)
    value.update(
        experiment_id=8526,
        run_date="20261015",
        task_id=TASK,
        milestone=MILESTONE,
        honest_verdict=f"complete_{value['verdict_class']}_v733_contract_methods",
        random_seed=7338526,
        future_dependencies=[
            dict(experiment_id=n, status="future_dependency_not_observation")
            for n in range(8527, 8540)
        ],
        contract_rows=value["rows"],
        authority_snapshots=work["contract"]["authority_snapshots"],
        activation_receipt=work["contract"]["activation_receipt"],
        exposure_ledger=p["exposure_ledger"],
        retirement_ledger=p["retirement_ledger"],
        ledger_lag=p["ledger_lag_v732"],
        v732_scheduling_dispositions=p["v732_scheduling_dispositions"],
        qualification_registrations={
            k: p[k] for k in ["nine_input_native_inference", "arc_capability_mask"]
        },
        literature_mapping=p["literature_mapping_v733"],
        service_head_coefficients=34,
        historical_authority_limits=dict(
            value["historical_authority_limits"],
            V732="fourteen scheduling slots; source diagnostic failed; learning validation/accounting failed; five-input service qualified separately; 32 ARC redirects unresolved",
        ),
        methodology_note="Administrative complete-object V733 authority and independently sealed inherited methods only. Original family consumers preserve historical dates and quarantines. Zero model calls or device probes. All 900 answers and 300 questions are exposed history; independently measured numerical work is distinct from factual generalization.",
    )
    value["field_principles"].update(
        {
            k: "Bind to sealed invocation operands; readiness never establishes scientific benefit."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["field_principles"].update(
        source_inputs_ready_score="Original V726 panel/head and V730 source-control consumers; independent of new source fitting.",
        learning_inputs_ready_score="Original V726 initial head/store/stream plus Exp8516 qualified objective publication; never Exp8517 benefit.",
        nine_input_inputs_ready_score="Original V726 nine-input panel and 109-coefficient head only; historical service speed is not a prerequisite.",
        service_inputs_ready_score="Original five-input workload and qualified append/service consumers only; no native inference certification.",
        methods_ready_score="Full current authority and independent method seals plus owned validation; optional runtime and scientific outcomes are separate.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Shipped cold reducers reconstruct families and authority rather than trusting summaries."""
    with binding():
        return bool(_REPLAY(path))
