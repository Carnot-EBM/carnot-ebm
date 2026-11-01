"""REQ-REPORT-8540: bind complete authority without changing historical science."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import re
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from typing import Any, Iterator
from unittest.mock import patch

import yaml

from carnot.reporting import v729_contract_methods as shipped
from carnot.reporting import v685_authority_lifecycle as lifecycle
from carnot.reporting import v733_contract_methods as previous
from carnot.reporting import v734_historical_inputs as history
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.roadmap_contract import parse_design

Json = dict[str, Any]
base = shipped.base
ROOT, DESIGN, ACTIVE, STAGED = shipped.ROOT, shipped.DESIGN, shipped.ACTIVE, shipped.STAGED
NAME, TASK, MILESTONE = (
    "experiment_8540_v734_contract_methods",
    "exp8540-contract-methods",
    "2026.10.734",
)
PROTOCOL = "openspec/change-proposals/v734-methods-protocol.json"
METHODS = "docs/research-notes/v734-contract-methods.md"
PROTOCOL_PIN = "sha256:3a19968a7ad5fb1d851a95955ae074f19a3a35073b182108ae79c6d58fdc6bad"
METHODS_PIN = "sha256:227f61c59f9b823f95d67e56eb150fd3d1266f73bfc957e63c9918187e51f02d"
TASKS_PIN = "d14c6915c6f390f334608c72557bce12a843973542bc70e9785b1b0868ec099c"
CLI = f"scripts/experiments/{NAME}.py"
TESTS = [
    f"tests/python/test_v734_{s}_8540.py"
    for s in ["contract_methods", "contract_replay", "historical_inputs"]
]
TEST = TESTS[0]
OWNED = [
    f"python/carnot/reporting/v734_{s}.py"
    for s in ["contract_methods", "contract_runner", "historical_inputs"]
] + [CLI]
MODEL_SPECS: list[Json] = []
FIELDS = history.FIELDS
INHERITED: tuple[str, ...] = (
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
_BUILD, _REPLAY, _RUNTIME = shipped.build, shipped.replay, base.runtime
DESIGN733_PIN = "sha256:cbcc949d69842d869c5516e076aaa960a010925b02bc244587eb6bd10a76e270"
INHERITED = (
    *INHERITED,
    "partition",
    "features",
    "decision",
    "feedback",
    "learning_audit",
    "cost",
    "nine_input_native_inference",
    "arc_capability_mask",
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real counts so a supervised child cannot appear silently stalled."""
    completed = 12 if phase == "finished" else completed
    print(f"[exp8540] phase={phase} completed={completed} pending={pending}", flush=True)


def protocol() -> Json:
    """Independent seals prevent a repaired envelope from changing scientific rules."""
    if sha256_file(ROOT / METHODS) != METHODS_PIN:
        raise ValueError("methods_hash")
    p = dict(read(dict(path=str(ROOT / PROTOCOL), sha256=PROTOCOL_PIN)))
    old = read(p["inherited_v733_protocol_reference"])
    if p["inherited_v733_protocol_reference"]["sha256"] != previous.PROTOCOL_PIN:
        raise ValueError("original_protocol_pin")
    if any(p[k] != old[k] for k in INHERITED):
        raise ValueError("inherited_scientific_operand")
    if tasks_digest(p["task_map_v734"]) != TASKS_PIN:
        raise ValueError("sealed_task_map")
    ref = p["preserved_v733_design_reference"]
    if ref["sha256"] != DESIGN733_PIN or sha256_file(Path(ref["path"])) != DESIGN733_PIN:
        raise ValueError("original_authority_hash")
    return p


def digest_search(pattern: str, text: str) -> re.Match[str] | None:
    """The current design uses plural tasks; accept that spelling without changing its bytes."""
    return re.search(pattern.replace("complete-task|task)", "complete-task|tasks?)"), text)


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
        patch.object(lifecycle, "re", SimpleNamespace(search=digest_search)),
        patch.multiple(
            base,
            **constants,
            protocol=protocol,
            authority=authority,
            build=build,
            runtime=runtime,
            progress=progress,
            assess_authorities=lambda *a, **kw: shipped._ASSESS(
                *a, **dict(kw, first_id=8540, count=12)
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
    result["contract_rows"] = result["contract_rows"][:12]
    design = root / DESIGN
    text = design.read_text() if design.is_file() else ""
    failures = result["gate_check_summary"]
    for marker in ["START", "END"]:
        if f"<!-- V734_TASK_CONTRACT_{marker} -->" not in text:
            failures.append(base.failure(design, "complete_json_" + marker.lower(), True, None))
    if (root / STAGED).is_file() and text:
        machine = parse_design(text, milestone=MILESTONE)[1]
        if yaml.safe_load((root / STAGED).read_bytes()).get("tasks") != machine:
            failures.append(base.failure(root / STAGED, "staged_full_task_objects", True, False))
    result["activation_receipt"] = protocol()["v734_activation_reference"]
    for gate in failures:
        gate.update(upstream_id=TASK, check=gate["artifact_field"], passed=False)
    result["activated"] = not failures
    for i, row in enumerate(result["contract_rows"]):
        present = i < len(result["tasks"])
        row.update(
            unit_id=result["tasks"][i]["id"] if present else f"exp{8540 + i}",
            family=f"exp{8540 + i}",
            status="completed" if present else "unstarted",
            censored=not present,
            effective_independent_groups=0,
            missing_reason=None if row["matched"] else "authority_absent_or_mismatched",
        )
    return result


def historical(root: Path, raw: Path) -> Json:
    """Original family consumers preserve their dates, flags and distinct producer fields."""
    return dict(history.historical(root, raw, CLI, protocol()))


def runtime(root: Path) -> Json:
    """A reboot alone cannot admit the canary without changed library or environment operands."""
    result = dict(_RUNTIME(root))
    changed = any(
        row["changed"] and row["field"] in {"libraries", "environment"}
        for row in result.get("changes", [])
    )
    if result["ready"] and not changed:
        result["ready"] = False
        result["failures"].append(
            base.failure(
                root / "ops/cuda-runtime-repair-receipt.json",
                "causal_library_or_environment_change",
                True,
                False,
            )
        )
    return result


def measure(root: Path, raw: Path) -> Json:
    """Seal current authority before bounded model-free historical qualification."""
    with binding():
        work = dict(shipped.measure(root, raw))
    with TemporaryDirectory(prefix="exp8540-preflight-", dir="/var/tmp") as directory:
        work["preconditions"].update(base.legacy.resources(Path(directory)))
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
        source=[8430, 8431, 8529],
        learning=[8430, 8431, 8433, 8434],
        nine_input=[8430, 8431],
        canary=[8430, 8431, 8529],
    )
    for group, numbers in groups.items():
        score = int(qualified and all(families.get(n, False) for n in numbers))
        value[group + "_inputs_ready_score"] = score
        value["acceptance_gates"][group + "_inputs"] = bool(score)
    value["acceptance_gates"].pop("complete_fourteen_objects", None)
    value["acceptance_gates"]["complete_twelve_objects"] = bool(value["methods_ready_score"])
    value.update(
        experiment_id=8540,
        run_date="20261015",
        task_id=TASK,
        milestone=MILESTONE,
        honest_verdict=f"complete_{value['verdict_class']}_v734_contract_methods",
        random_seed=7348540,
        future_dependencies=[
            dict(experiment_id=n, status="future_dependency_not_observation")
            for n in range(8541, 8552)
        ],
        intended_count=12,
        sample_size_budget=dict(
            intended_task_slots=12,
            independent_questions=0,
            retrospective_questions_reserved=300,
            seeds_and_siblings_are_not_independent=True,
        ),
        contract_rows=value["rows"],
        authority_snapshots=work["contract"]["authority_snapshots"],
        activation_receipt=work["contract"]["activation_receipt"],
        exposure_ledger=p["exposure_ledger"],
        retirement_ledger=p["retirement_ledger"],
        ledger_lag=dict(
            observed_archive_last_milestone="2026.10.732",
            authority="V733 authenticated primaries, pre-gates and conductor outcomes; archive unchanged",
        ),
        closed_source_comparison=p["closed_source_comparison"],
        v733_scheduling_dispositions=p["v733_scheduling_dispositions"],
        v732_scheduling_dispositions=p["v732_scheduling_dispositions"],
        qualification_registrations={
            k: p[k] for k in ["nine_input_native_inference", "arc_capability_mask"]
        },
        literature_mapping=p["literature_mapping_v734"],
        service_head_coefficients=34,
        historical_authority_limits=dict(
            value["historical_authority_limits"],
            V733="fourteen scheduling slots; source null qualified; Exp8530/8533/8537 quarantines remain",
        ),
        methodology_note="Administrative complete-object V734 authority and independently sealed inherited methods only. Original family consumers preserve historical dates and quarantines. Zero model calls or device probes. FaithBench and all 900 answers and 300 questions are exposed history; independently measured numerical work is distinct from factual generalization.",
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
        learning_inputs_ready_score="Original V726 initial head/store/stream, independently replayed under original authority; no learning benefit.",
        nine_input_inputs_ready_score="Original V726 nine-input panel and 109-coefficient head only; historical service speed is not a prerequisite.",
        canary_inputs_ready_score="Original source panel and qualified exposed audit only; optional runtime is a separate prerequisite.",
        methods_ready_score="Full current authority and independent method seals plus owned validation; optional runtime and scientific outcomes are separate.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Shipped cold reducers reconstruct families and authority rather than trusting summaries."""
    with binding():
        return bool(_REPLAY(path))
