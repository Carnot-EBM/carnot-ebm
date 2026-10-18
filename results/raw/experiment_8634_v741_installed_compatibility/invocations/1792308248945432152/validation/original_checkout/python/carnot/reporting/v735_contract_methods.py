"""REQ-REPORT-8552: reuse original consumers while keeping scientific rules fixed."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, Iterator
from unittest.mock import patch

import yaml

from carnot.reporting import v734_contract_methods as previous
from carnot.reporting import v735_historical_inputs as history
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import reader_receipt
from carnot.reporting.roadmap_contract import parse_design

Json = dict[str, Any]
shipped, base = previous.shipped, previous.base
ROOT, DESIGN, ACTIVE, STAGED = previous.ROOT, previous.DESIGN, previous.ACTIVE, previous.STAGED
NAME, TASK, MILESTONE = (
    "experiment_8552_v735_contract_methods",
    "exp8552-contract-methods",
    "2026.10.735",
)
PROTOCOL = "openspec/change-proposals/v735-methods-protocol.json"
METHODS = "docs/research-notes/v735-contract-methods.md"
PROTOCOL_PIN = "sha256:cdd83a34032ff1977091dcc423e3466bd21adff242b9f275b99a9c9d8e72c3eb"
METHODS_PIN = "sha256:190daa38c93f1b5206b9fecf79fee0417631befdf18aedcf96b059b440df264c"
TASKS_PIN = "9696b266453ac3dd98bd694f81ccbd6e85e1065ebf4173c291d35d33db7cb10d"
DESIGN734_PIN = "sha256:9a213619fcc67cbaa26e0b8fcdc8492917b6b874e22d332b1038ded73d2231a8"
CLI = f"scripts/experiments/{NAME}.py"
TESTS = [
    f"tests/python/test_v735_{s}_8552.py"
    for s in ["contract_methods", "contract_replay", "historical_inputs"]
]
TEST = TESTS[0]
OWNED = [
    f"python/carnot/reporting/v735_{s}.py"
    for s in ["contract_methods", "contract_runner", "historical_inputs"]
] + [CLI]
MODEL_SPECS: list[Json] = []
FIELDS, INHERITED = history.FIELDS, previous.INHERITED
read, reference, checksum, sha256_file, tasks_digest = (
    previous.read,
    previous.reference,
    previous.checksum,
    previous.sha256_file,
    previous.tasks_digest,
)
measurement_clock = previous.measurement_clock
_BUILD, _REPLAY, _ASSESS = shipped.build, shipped.replay, shipped._ASSESS


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual counts so slow original readers remain observable to the parent."""
    completed = min(completed, 13)
    if phase in {"preconditions_before", "authority_after", "methods_frozen_after"}:
        pending = 13 - completed
    print(f"[exp8552] phase={phase} completed={completed} pending={pending}", flush=True)


def protocol() -> Json:
    """Independent original pins prevent rehashed envelopes from changing numerical rules."""
    if sha256_file(ROOT / METHODS) != METHODS_PIN:
        raise ValueError("methods_hash")
    p = dict(read(dict(path=str(ROOT / PROTOCOL), sha256=PROTOCOL_PIN)))
    old = read(p["inherited_v734_protocol_reference"])
    if p["inherited_v734_protocol_reference"]["sha256"] != previous.PROTOCOL_PIN:
        raise ValueError("original_protocol_pin")
    if any(p[k] != old[k] for k in INHERITED):
        raise ValueError("inherited_scientific_operand")
    if tasks_digest(p["task_map_v735"]) != TASKS_PIN:
        raise ValueError("sealed_task_map")
    ref = p["preserved_v734_design_reference"]
    if ref["sha256"] != DESIGN734_PIN or sha256_file(Path(ref["path"])) != DESIGN734_PIN:
        raise ValueError("original_authority_hash")
    return p


def digest_search(pattern: str, text: str) -> Any:
    """Accept the current printed digest label without rewriting original authority bytes."""
    return previous.digest_search(
        pattern, text.replace("canonical_tasks_sha256:", "Canonical task SHA256:")
    )


@contextmanager
def binding() -> Iterator[None]:
    """Change current identity through small adapters; original readers retain their assertions."""
    constants: Json = {
        k: globals()[k]
        for k in [
            "NAME",
            "TASK",
            "MILESTONE",
            "PROTOCOL",
            "METHODS",
            "PROTOCOL_PIN",
            "METHODS_PIN",
            "TASKS_PIN",
            "OWNED",
            "CLI",
            "TEST",
        ]
    }
    with (
        patch.object(previous.lifecycle, "re", SimpleNamespace(search=digest_search)),
        patch.multiple(
            base,
            **constants,
            protocol=protocol,
            authority=authority,
            build=build,
            runtime=runtime,
            progress=progress,
            assess_authorities=lambda *a, **kw: _ASSESS(*a, **dict(kw, first_id=8552, count=13)),
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
    """Full dictionaries bind prompts and routing while consumed staging stays absent."""
    with binding():
        result = dict(shipped._AUTHORITY(root, raw))
    result["contract_rows"] = result["contract_rows"][:13]
    path = root / DESIGN
    text = path.read_text() if path.is_file() else ""
    failures = result["gate_check_summary"]
    for marker in ["START", "END"]:
        if f"<!-- V735_TASK_CONTRACT_{marker} -->" not in text:
            failures.append(base.failure(path, "complete_json_" + marker.lower(), True, None))
    if (root / STAGED).is_file() and text:
        machine = parse_design(text, milestone=MILESTONE)[1]
        if yaml.safe_load((root / STAGED).read_bytes()).get("tasks") != machine:
            failures.append(base.failure(root / STAGED, "staged_full_task_objects", True, False))
    result["activation_receipt"] = protocol()["v735_activation_reference"]
    for gate in failures:
        gate.update(upstream_id=TASK, check=gate["artifact_field"], passed=False)
    result["activated"] = not failures
    for row in result["contract_rows"]:
        row.update(
            effective_independent_groups=0,
            missing_reason=None if row["matched"] else "authority_absent_or_mismatched",
        )
    return result


def historical(root: Path, raw: Path) -> Json:
    """Only original panel, head, store, stream and qualified source audit gate input readiness."""
    return dict(history.historical(root, raw, CLI, protocol()))


def runtime(root: Path) -> Json:
    """The shipped Exp8410 receipt reader requires causal library changes and real copy evidence."""
    return dict(previous.runtime(root))


def measure(root: Path, raw: Path) -> Json:
    """Freeze primitive bytes before validation; optional resources never disable CPU branches."""
    with TemporaryDirectory(prefix="exp8552-preflight-", dir="/var/tmp") as directory:
        probe = Path(directory) / "write_probe"
        probe.write_bytes(b"exp8552 disk scratch")
        resources = dict(base.legacy.resources(Path(directory)))
        resources["scratch_write_read_passed"] = probe.read_bytes() == b"exp8552 disk scratch"
    with binding():
        work = dict(shipped.measure(root, raw))
    work["preconditions"].update(resources)
    work["preconditions"].update(task_cap_s=4800, measurement_cap_s=3600, validation_reserve_s=900)
    work["code_refs"] += [reference(ROOT / n) for n in TESTS[1:]]
    p = work["protocol"]
    work["v734_reader_receipts"] = [
        reader_receipt(t["id"], root / "results", field="required_checks_passed")
        for t in p["task_map_v734"]
    ]
    work["v734_primary_references"] = p["v734_primary_references"]
    atomic_json(raw / "measurement.json", work)
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Administrative qualification cannot clear historical flags or establish scientific benefit."""
    receipts = receipts + [
        r
        for r in work["history"]["receipts"]
        if not r.get("normal_exit", True) and r not in receipts
    ]
    with binding():
        value = dict(_BUILD(work, receipts, raw, output))
    p, qualified = work["protocol"], value["required_checks_passed"]
    families = {f["experiment_id"]: f["qualified"] for f in work["history"]["families"]}
    for group, numbers in dict(
        source=[8430, 8431, 8529],
        learning=[8430, 8431, 8433, 8434],
        nine_input=[8430, 8431],
        canary=[8430, 8431, 8529],
    ).items():
        score = int(qualified and all(families.get(n, False) for n in numbers))
        value[group + "_inputs_ready_score"] = score
        value["acceptance_gates"][group + "_inputs"] = bool(score)
    value["acceptance_gates"].pop("complete_fourteen_objects", None)
    value["acceptance_gates"]["complete_thirteen_objects"] = bool(value["methods_ready_score"])
    value.update(
        experiment_id=8552,
        run_date="20261015",
        task_id=TASK,
        milestone=MILESTONE,
        honest_verdict=f"complete_{value['verdict_class']}_v735_contract_methods",
        random_seed=7358552,
        future_dependencies=[
            dict(experiment_id=n, status="future_dependency_not_observation")
            for n in range(8553, 8565)
        ],
        intended_count=13,
        contract_rows=value["rows"],
        authority_snapshots=work["contract"]["authority_snapshots"],
        activation_receipt=work["contract"]["activation_receipt"],
        closed_source_comparison=p["closed_source_comparison"],
        literature_mapping=p["literature_mapping_v735"],
        hallmark_release_admissibility=p["hallmark_release_admissibility"],
        ledger_lag=p["archive_authority"],
        v734_reader_receipts=work["v734_reader_receipts"],
        v734_primary_references=work["v734_primary_references"],
        preserved_quarantines=dict(
            V734_learning="Exp8541: 152/158 original nodes passed; family4 killed at 480 seconds",
            V734_native="Exp8545: numerical and complete-request parity passed; preliminary cold replay failed",
            V734_arc="Exp8548: stored replay 134 seconds; four exact conditional anchors failed",
            scientific_benefit="Learning, native speed and capability-mask benefit remain unmeasured",
        ),
        sample_size_budget=dict(
            intended_task_slots=13,
            independent_questions=0,
            retrospective_questions_reserved=300,
            seeds_and_siblings_are_not_independent=True,
        ),
        methodology_note="Administrative full-object V735 authority and separately sealed inherited methods. Original readers preserve dates, input roles and quarantines. FaithBench and all 300 questions remain exposed. Zero model calls or device probes; no new source fits, benefit or speed measurement.",
    )
    value["source_artifact_hashes"] += work["v734_primary_references"]
    value["source_artifact_hashes"] += [
        dict(path=r["gate_path"], sha256=r["gate_sha256"])
        for r in work["v734_reader_receipts"]
        if r["gate_path"] is not None
    ]
    value["field_principles"].update(
        {
            k: "Bind exact invocation operands; readiness never establishes independent scientific benefit."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["field_principles"].update(
        learning_inputs_ready_score="Original V726 head/store/stream consumer qualification; independent of HALLMARK, CUDA and failed unrelated families.",
        nine_input_inputs_ready_score="Original nine-input panel and 109-coefficient head; no native speed qualification is implied.",
        canary_inputs_ready_score="Original panel/head and Exp8529 audit; runtime evidence is a separate gate.",
        v734_reader_receipts="Actual live consumer paths and hashes include pre-gates under their produced basenames.",
        preserved_quarantines="Original owned failures remain failures; this administrative artifact cannot repair them.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Fresh original consumers reconstruct families and authority rather than trusting summaries."""
    try:
        value = json.loads(path.read_bytes())
        work = read(value["primitive_reference"])
        actual = [
            reader_receipt(t["id"], Path(work["root"]) / "results", field="required_checks_passed")
            for t in protocol()["task_map_v734"]
        ]
        if actual != work["v734_reader_receipts"]:
            return False
        with binding():
            return bool(_REPLAY(path))
    except (OSError, ValueError, KeyError, TypeError):
        return False
