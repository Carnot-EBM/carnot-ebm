"""REQ-REPORT-8443: separate source, learning and service admission boundaries."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sys
import time
from tempfile import TemporaryDirectory
from typing import Any, Iterator
from unittest.mock import patch

import yaml

from carnot.reporting import v726_contract_methods as legacy
from carnot.reporting import v726_capstone_evidence as cap
from carnot.reporting import v727_historical_inputs as h
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child

Json = dict[str, Any]
ROOT, DESIGN, ACTIVE, STAGED = legacy.ROOT, legacy.DESIGN, legacy.ACTIVE, legacy.STAGED
NAME, TASK, MILESTONE = (
    "experiment_8443_v727_contract_methods",
    "exp8443-contract-methods",
    "2026.10.727",
)
PROTOCOL = "openspec/change-proposals/v727-methods-protocol.json"
METHODS = "docs/research-notes/v727-contract-methods.md"
PROTOCOL_PIN = "sha256:c08fc9978bf7898bd9c6f1a1e41e905cfd454c2d6782fab0de3209a8b6e0560c"
METHODS_PIN = "sha256:2b59139f9628567cc10ae200aaf81987b3b3abb74bd2c6b76f929fba5f4576a9"
TASKS_PIN = "026228ee20a3160ad55ba35fe8286fab6db3a14d62a89b8d3d3d171e1acb1aa8"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v727_contract_methods_8443.py"
OWNED = [
    "python/carnot/reporting/v727_contract_methods.py",
    "python/carnot/reporting/v727_historical_inputs.py",
    "python/carnot/reporting/v727_contract_runner.py",
    CLI,
]
EXTRA_TEST = "tests/python/test_v727_historical_failures_8443.py"
MODEL_SPECS: list[Json] = []
INHERITED = ("partition", "features", "decision", "feedback", "learning_audit", "cost")
read, reference, checksum, sha256_file, tasks_digest = (
    legacy.read,
    legacy.reference,
    legacy.checksum,
    legacy.sha256_file,
    legacy.tasks_digest,
)
_AUTHORITY, _MEASURE, _REPLAY = legacy.authority, legacy.measure, legacy.replay
_ASSESS = legacy.assess_authorities


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed boundaries expose real work while a child keeps its detailed logs."""
    print(f"[exp8443] phase={phase} completed={completed} pending={pending}", flush=True)


def protocol() -> Json:
    """Independent seals prevent a rehashed candidate from moving methods or roles."""
    if sha256_file(ROOT / METHODS) != METHODS_PIN:
        raise ValueError("methods_hash")
    p = dict(read(dict(path=str(ROOT / PROTOCOL), sha256=PROTOCOL_PIN)))
    old = read(p["inherited_protocol_reference"])
    if any(p[k] != old[k] for k in INHERITED):
        raise ValueError("inherited_scientific_operand")
    return p


@contextmanager
def binding() -> Iterator[None]:
    """Only current constants and adapters change; shipped assertions remain intact."""
    with patch.multiple(
        legacy,
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
        protocol=protocol,
        authority=authority,
        runtime=runtime,
        build=build,
        progress=progress,
        assess_authorities=lambda *args, **kwargs: _ASSESS(*args, **dict(kwargs, first_id=8443)),
    ):
        yield


def authority(root: Path, raw: Path) -> Json:
    """Missing independent design cannot erase the actually observed activated slots."""
    with binding():
        result = _AUTHORITY(root, raw)
    plan = yaml.safe_load((root / ACTIVE).read_bytes()) if (root / ACTIVE).is_file() else {}
    tasks = plan.get("tasks", [])
    result.update(
        observed_count=len(tasks), tasks=tasks, observed_active_tasks_sha256=tasks_digest(tasks)
    )
    if any(g["artifact_field"] == "required_authority" for g in result["gate_check_summary"]):
        result["canonical_tasks_sha256"] = None
        for field, expected in [
            ("independent_visible_task_table", 14),
            ("embedded_complete_task_objects", 14),
            ("printed_canonical_task_digest", TASKS_PIN),
        ]:
            result["gate_check_summary"].append(
                dict(legacy.legacy.failure(root / DESIGN, field, expected, None), upstream_id=TASK)
            )
    for i, row in enumerate(result["contract_rows"]):
        present = i < len(tasks)
        row.update(
            unit_id=tasks[i]["id"] if present else f"exp{8443 + i}",
            family=f"exp{8443 + i}",
            status="completed" if present else "unstarted",
            censored=not present,
            effective_independent_groups=0,
            missing_reason=None if row["matched"] else "independent_authority_absent_or_mismatched",
        )
    return dict(result)


def runtime(root: Path) -> Json:
    """The qualified supplied-receipt reader performs no CUDA or device probe."""
    result = dict(legacy.legacy.runtime(root))
    for gate in result["failures"]:
        gate["upstream_id"] = TASK
    return result


def historical(root: Path, raw: Path) -> Json:
    """Each original family has its own bounded child and does not gate unrelated work."""
    families, refs, receipts = [], [], []
    try:
        refs = h.custody(root, raw / "authority")
    except (OSError, ValueError) as error:
        progress("historical_authority_blocked_" + str(error))
    for i, number in enumerate(h.FIELDS):
        progress("historical_before_" + str(number), i, len(h.FIELDS) - i)
        path = next(
            (root / "results").glob(f"experiment_{number}_*.json"),
            root / "results" / f"experiment_{number}_missing.json",
        )
        primary = cap.freeze(path, raw / "primaries")
        refs.append(primary)
        request, output = raw / f"request-{number}.json", raw / f"summary-{number}.json"
        atomic_json(
            request,
            dict(
                number=number,
                primary=primary,
                authority=refs[:2] if len(refs) >= 2 and refs[0].get("source_path") else [],
            ),
        )
        receipt = child(
            f"historical_{number}",
            [
                sys.executable,
                "-u",
                str(ROOT / CLI),
                "--history-request",
                str(request),
                "--history-output",
                str(output),
            ],
            raw / "logs",
            deadline=120,
            scope="upstream",
        )
        receipts.append(receipt)
        family = (
            json.loads(output.read_bytes())
            if receipt["passed"]
            else dict(
                experiment_id=number,
                qualified=False,
                gate_check_summary=[
                    legacy.legacy.failure(
                        path, "bounded_historical_reader", True, receipt["actual_exit"]
                    )
                ],
            )
        )
        families.append(family)
        progress("historical_after_" + str(number), i + 1, len(h.FIELDS) - i - 1)
    return dict(families=families, references=refs, receipts=receipts)


def measure(root: Path, raw: Path) -> Json:
    """Methods and resources are sealed before independent historical family reductions."""
    with binding():
        work = dict(_MEASURE(root, raw))
    work["preconditions"].update(measurement_cap_s=3600, task_cap_s=4800, validation_reserve_s=900)
    work["code_refs"].append(reference(ROOT / EXTRA_TEST))
    work["history"] = historical(root, raw / "historical")
    work["phase_spans"] += [
        dict(name=r["name"], duration_s=r["duration_s"]) for r in work["history"]["receipts"]
    ]
    work["ended_monotonic_ns"] = time.monotonic_ns()
    work["invocation_argv"] = list(sys.argv)
    atomic_json(raw / "measurement.json", work)
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Each gate has one evidence boundary; absence never becomes scientific failure."""
    value = dict(legacy.legacy.build(work, receipts, raw, output))
    qualified = value["required_checks_passed"] and not work.get("adversarial_findings")
    families = work["history"]["families"]
    learning = (
        qualified
        and all(f["qualified"] for f in families if f["experiment_id"] != 8407)
        and len(families) == 5
    )
    service = qualified and any(f["qualified"] for f in families if f["experiment_id"] == 8407)
    ready = qualified and work["contract"]["activated"] and not work["failures"]
    verdict = "disqualified" if not qualified else "circular_positive" if ready else "blocked"
    historical_models = []
    for ref in work["history"]["references"]:
        if ref.get("exists") and Path(ref["path"]).name.startswith("experiment_"):
            v = cap.load(ref)
            historical_models.append(
                dict(
                    producer=v["experiment_id"],
                    producer_hash=ref["sha256"],
                    imported_fields=["historical_model_provenance", h.FIELDS[v["experiment_id"]]],
                    historical_model_provenance=v.get("historical_model_provenance", []),
                )
            )
    value.update(
        experiment_id=8443,
        task_id=TASK,
        milestone=MILESTONE,
        run_date="20261012",
        honest_verdict=f"complete_{verdict}_v727_contract_methods",
        verdict_class=verdict,
        intended_count=14,
        random_seed=7278443,
        required_checks_passed=bool(qualified),
        flagged_adversarial=not qualified,
        methods_ready_score=int(ready),
        contract_ready_score=int(ready),
        current_contract_ready_score=int(ready),
        learning_inputs_ready_score=int(learning),
        service_inputs_ready_score=int(service),
        runtime_repair_ready_score=int(qualified and work["runtime"]["ready"]),
        contract_digest=work["contract"]["canonical_tasks_sha256"],
        historical_family_receipts=families,
        historical_validation_receipts=work["history"]["receipts"],
        historical_model_provenance=historical_models,
        methods_protocol_reference=reference(ROOT / PROTOCOL),
        methods_protocol_path=str(ROOT / PROTOCOL),
        methods_protocol_sha256=PROTOCOL_PIN,
        methods_note_path=str(ROOT / METHODS),
        methods_note_sha256=METHODS_PIN,
        current_llm_invocations=0,
        numeric_learning_substrate="verifier_ensemble_against_cached_candidates",
        invocation_argv=work["invocation_argv"],
        future_dependencies=[
            dict(experiment_id=i, status="future_dependency_not_observation")
            for i in range(8444, 8457)
        ],
        methodology_note="Administrative contract and separate historical prerequisite qualification only. Public labels remain retrospective, with older overlap unknown. Independent factuality, current Qwen skill and hidden-game transfer remain open.",
    )
    value["source_artifact_hashes"] = [
        *value["source_artifact_hashes"],
        *[
            dict(path=ref["snapshot_path"], sha256=ref["sha256"], source_path=ref["path"])
            for ref in work["history"]["references"]
            if ref.get("exists")
        ],
    ]
    value["sample_size_budget"]["intended_task_slots"] = 14
    value["cited_upstream_artifacts"] += historical_models
    value["gate_check_summary"] += [g for f in families for g in f["gate_check_summary"]]
    value["raw_shard_hashes"] += [
        dict(path=ref["snapshot_path"], sha256=ref["sha256"])
        for ref in work["history"]["references"]
        if ref.get("exists")
    ]
    value["acceptance_gates"].update(
        current_contract=bool(ready),
        learning_inputs=bool(learning),
        service_inputs=bool(service),
        sealed_methods=True,
    )
    value["field_principles"].update(
        {
            k: "Bind this field to exact invocation evidence; prerequisite readiness does not establish scientific benefit."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["field_principles"].update(
        learning_inputs_ready_score="All original panel/head/store/stream producers qualify independently of current methods and service.",
        service_inputs_ready_score="Original Exp8407 complete five-input workload qualifies independently of learning and CUDA.",
        contract_digest="Independent design digest; null when table, full objects or digest are absent.",
        historical_family_receipts="Original producer hash and exact readiness field, replayed using sealed original authority.",
        methods_protocol_reference="Fixed seal covers inherited operands and every activated task's method clauses; missing independent design remains blocked.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Fresh reduction distrusts rehashed family summaries as well as contract rows."""
    try:
        value = json.loads(path.read_bytes())
        work = read(value["primitive_reference"])
        if value != build(
            work,
            value["validation_receipts"],
            Path(value["primitive_reference"]["path"]).parent,
            Path(value["publication_output"]),
        ):
            return False
        with TemporaryDirectory(prefix="exp8443-replay-", dir="/var/tmp") as directory:
            actual = historical(Path(work["root"]), Path(directory) / "history")
            if actual["families"] != work["history"]["families"]:
                return False
        with binding():
            return bool(_REPLAY(path))
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False
