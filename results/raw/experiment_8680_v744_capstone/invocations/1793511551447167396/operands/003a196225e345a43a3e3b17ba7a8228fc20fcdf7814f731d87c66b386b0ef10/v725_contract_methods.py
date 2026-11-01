"""REQ-REPORT-8416: freeze methods without inventing missing executable authority."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
from typing import Any

import yaml

from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.evidence_features_custody_7980 import reference as reference
from carnot.reporting.v685_authority_lifecycle import assess_authorities, tasks_digest
from carnot.reporting.roadmap_contract import parse_design
from carnot.reporting.v710_contract_replay import snapshot
from carnot.verify import cuda_reader_8410 as cuda
from carnot.verify.runtime_evidence_execution_8382 import resources

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME, TASK, MILESTONE = (
    "experiment_8416_v725_contract_methods",
    "exp8416-contract-methods",
    "2026.10.725",
)
DESIGN, ACTIVE, STAGED = (
    "openspec/change-proposals/research-roadmap-vNEXT.md",
    "research-roadmap.yaml",
    "research-roadmap-next.yaml",
)
PROTOCOL = "openspec/change-proposals/v725-methods-protocol.json"
METHODS = "docs/research-notes/v725-methods.md"
PROTOCOL_PIN = "sha256:9bf6669dde2b92a3aea196b036f59f7bfbb5b72d4bd53c1e013f42c5899f6a48"
METHODS_PIN = "sha256:bb37ae02133e8a3f71ca7cbf231f6907083007a2890e3a65db09d1db13731808"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v725_contract_methods_8416.py"
OWNED = [
    "python/carnot/reporting/v725_contract_methods.py",
    "python/carnot/reporting/v725_contract_runner.py",
    CLI,
]
MODEL_SPECS: list[Json] = []


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Actual boundaries and counts let the supervisor distinguish progress from silence."""
    print(f"[exp8416] phase={phase} completed={completed} pending={pending}", flush=True)


def read(ref: Json) -> Any:
    """Exact named bytes prevent repaired summary hashes from replacing primitive evidence."""
    path = Path(ref["path"])
    if sha256_file(path) != ref["sha256"]:
        raise ValueError("source_hash")
    return json.loads(path.read_bytes())


def protocol() -> Json:
    """Independent fixed hashes keep public methods and their note immutable during replay."""
    if sha256_file(ROOT / METHODS) != METHODS_PIN:
        raise ValueError("methods_hash")
    return dict(read(dict(path=str(ROOT / PROTOCOL), sha256=PROTOCOL_PIN)))


def failure(path: Path, field: str, expected: Any, observed: Any) -> Json:
    """Missing evidence stays null so it cannot be mistaken for a measured zero."""
    return dict(
        check=field,
        upstream_id=TASK,
        artifact_path=str(path),
        artifact_hash=sha256_file(path) if path.is_file() else None,
        artifact_field=field,
        op="==",
        expected=expected,
        observed=observed,
        passed=False,
    )


def authority(root: Path, raw: Path) -> Json:
    """Reuse the shipped full-object reader; absence cannot manufacture its independent table."""
    active, design, staged = root / ACTIVE, root / DESIGN, root / STAGED
    plan = yaml.safe_load(active.read_bytes()) if active.is_file() else {}
    tasks = plan.get("tasks", []) if isinstance(plan, dict) else []
    failures = []
    try:
        machine = parse_design(design.read_text(), milestone=MILESTONE)[1]
        result = assess_authorities(
            design, staged, active, raw, milestone=MILESTONE, first_id=8416, count=13
        )
        if tasks_digest(machine) != result["canonical_tasks_sha256"]:
            raise ValueError("independent_machine_digest")
        for i, row in enumerate(result["contract_rows"]):
            row["checks"]["full_object"] = i < len(tasks) and tasks[i] == machine[i]
            row["matched"] = all(row["checks"].values())
            row.update(
                absolute_metric=int(row["matched"]),
                raw_numerator=sum(row["checks"].values()),
                raw_denominator=len(row["checks"]),
                excluded=not row["matched"],
            )
        failures = result["gate_check_summary"]
    except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
        failures.append(
            failure(
                design,
                "independent_visible_table_complete_tasks_digest",
                True,
                None if isinstance(error, (FileNotFoundError, IndexError)) else str(error),
            )
        )
        rows = []
        for i in range(13):
            task = tasks[i] if i < len(tasks) else None
            checks = dict(
                independent_design_contract=False,
                activated_task_present=task is not None,
                sequence=bool(task and str(task["id"]).startswith(f"exp{8416 + i}-")),
            )
            rows.append(
                dict(
                    unit_id=task["id"] if task else f"exp{8416 + i}",
                    family=f"exp{8416 + i}",
                    order=i + 1,
                    arm="active_contract",
                    seed=None,
                    status="completed" if task else "unstarted",
                    checks=checks,
                    matched=False,
                    absolute_metric=0,
                    raw_numerator=sum(checks.values()),
                    raw_denominator=len(checks),
                    censored=task is None,
                    excluded=False,
                    effective_independent_groups=0,
                    missing_reason="independent_design_binding_absent"
                    if task
                    else "complete_task_object_absent",
                )
            )
        result = dict(activated=False, canonical_tasks_sha256=None, contract_rows=rows)
    if len(tasks) != 13:
        failures.append(
            failure(active, "complete_task_count", 13, len(tasks) if active.is_file() else None)
        )
    result.update(
        activated=not failures and all(r["matched"] for r in result["contract_rows"]),
        gate_check_summary=failures,
        observed_count=len(tasks),
        tasks=tasks,
        observed_active_tasks_sha256=tasks_digest(tasks),
        staging_disposition="existing" if staged.is_file() else "consumed_or_absent",
    )
    return dict(result)


def runtime(root: Path) -> Json:
    """A supplied causal transcript is the only admission route; never inventory devices."""
    path = root / "ops/cuda-runtime-repair-receipt.json"
    start = time.monotonic()
    result: Json = dict(ready=False, receipt=None, failures=[], current_device_probe_count=0)
    try:
        document = json.loads(path.read_bytes())
        current, previous = (
            read(document["current_reference"]),
            read(document["previous_reference"]),
        )
        receipt = cuda.repair(
            path, sha256_file(path), current, time.time_ns(), document["frontier_wall_ns"]
        )
        changes = cuda.compare(previous, current)
        transcript = read(document["transcript_reference"])
        context = dict(
            boot_identity=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            namespaces={p.name: str(p.readlink()) for p in Path("/proc/self/ns").iterdir()},
        )
        causal = any(
            row["changed"]
            for row in changes
            if row["field"]
            in {"kernel_driver", "libraries", "namespaces", "boot_identity", "environment"}
        )
        api = transcript["api_returns"]
        success = all(
            api.get(k) == 0
            for k in [
                "cuInit",
                "cuCtxCreate",
                "cuMemAlloc",
                "cuMemcpyHtoD",
                "cuMemcpyDtoH",
                "cuMemFree",
                "cuCtxDestroy",
            ]
        )
        ready = (
            causal
            and document["execution_context"] == context
            and transcript["execution_context"] == context
        )
        ready = (
            ready
            and success
            and transcript["byte_copy_parity"] is True
            and transcript["cleanup_passed"] is True
        )
        ready = ready and 0 < transcript["duration_s"] <= 30 and time.monotonic() - start <= 30
        if not ready:
            raise ValueError("causal_same_context_bounded_copy_free_transcript")
        result.update(ready=True, receipt=receipt, changes=changes, transcript=transcript)
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["failures"] = [
            failure(
                path,
                "supplied_causal_runtime_repair",
                True,
                None if not path.is_file() else str(error),
            )
        ]
    result["inspection_duration_s"] = time.monotonic() - start
    return result


def measure(root: Path, raw: Path) -> Json:
    """Seal actual input bytes before deriving administrative readiness or absent external gates."""
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    progress("preconditions_before", 0, 13)
    p = protocol()
    preconditions = resources(raw)
    preconditions.update(
        free_disk_bytes=shutil.disk_usage(raw).free,
        tool_paths={
            name: shutil.which(str(ROOT / ".venv/bin" / name))
            for name in ["python", "pytest", "coverage", "ruff", "mypy"]
        },
        validation_reserve_s=1200,
        measurement_cap_s=3000,
    )
    failures = []
    if not (
        preconditions["private_disk_backed_scratch"]
        and preconditions["available_memory_bytes"] >= preconditions["minimum_memory_bytes"]
        and preconditions["free_disk_bytes"] >= 104857600
        and all(preconditions["tool_paths"].values())
    ):
        failures.append(failure(raw, "private_disk_memory_tools_budget", True, preconditions))
    refs = []
    for i, name in enumerate([ACTIVE, DESIGN, STAGED]):
        path = root / name
        if path.is_file():
            saved = snapshot(path, raw / "authority_inputs", str(i))
            refs.append(
                dict(
                    path=saved["snapshot_path"],
                    sha256=saved["sha256"],
                    source_path=str(path),
                    role=name,
                )
            )
    c = authority(root, raw / "authority")
    progress("authority_after", c["observed_count"], 13 - c["observed_count"])
    repair = runtime(root)
    for ref in p["source_references"]:
        if sha256_file(Path(ref["path"])) != ref["sha256"]:
            raise ValueError("frozen_source_hash")
    code_refs = [
        reference(ROOT / name)
        for name in [
            *OWNED,
            TEST,
            "python/carnot/reporting/roadmap_contract.py",
            "python/carnot/reporting/primary_publication.py",
            "python/carnot/reporting/v685_authority_lifecycle.py",
            "python/carnot/reporting/v709_execution.py",
            "scripts/experiment_template.py",
            "scripts/adversarial_verify.py",
            "scripts/verdict_row_consistency_lint.py",
            "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
        ]
    ]
    work = dict(
        root=str(root),
        started_monotonic_ns=start,
        ended_monotonic_ns=time.monotonic_ns(),
        preconditions=preconditions,
        contract=c,
        runtime=repair,
        failures=failures,
        protocol=p,
        authority_inputs=refs,
        source_refs=p["source_references"],
        code_refs=code_refs,
        phase_spans=[
            dict(
                name="preconditions_authority_methods",
                duration_s=(time.monotonic_ns() - start) / 1e9,
            )
        ],
    )
    atomic_json(raw / "measurement.json", work)
    progress("methods_after", 1, 0)
    return work


def checksum(value: Json) -> str:
    """Bind every claimed field while keeping the checksum outside its own operands."""
    return str(canonical_hash({k: v for k, v in value.items() if k != "reproducibility_checksum"}))


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Validation qualifies classification separately from contract admission and optional CUDA."""
    qualified = bool(receipts) and all(r["passed"] for r in receipts)
    c, p = work["contract"], work["protocol"]
    ready = qualified and c["activated"] and not work["failures"]
    verdict = "disqualified" if not qualified else "circular_positive" if ready else "blocked"
    failures = [*work["failures"], *c["gate_check_summary"], *work["runtime"]["failures"]]
    failures += [
        failure(
            Path(r["stdout_path"]), "required_owned_check", r["expected_exit"], r["actual_exit"]
        )
        for r in receipts
        if not r["passed"]
    ]
    rows = c["contract_rows"]
    historical = [
        dict(
            producer=int(Path(ref["source_path"]).name.split("_")[1]),
            producer_hash=ref["sha256"],
            imported_fields=["honest_verdict", "historical_model_provenance"],
            historical_model_provenance=read(ref).get("historical_model_provenance", []),
        )
        for ref in work["source_refs"]
        if Path(ref["source_path"]).name.startswith("experiment_")
    ]
    value: Json = dict(
        experiment_id=8416,
        task_id=TASK,
        milestone=MILESTONE,
        run_date="20261011",
        honest_verdict=f"complete_{verdict}_v725_contract_methods",
        verdict_class=verdict,
        gate_check_summary=failures,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        historical_model_provenance=historical,
        rows=rows,
        intended_count=13,
        completed_count=sum(r["status"] == "completed" for r in rows),
        failed_count=0,
        censored_count=sum(r["censored"] for r in rows),
        excluded_count=sum(r["excluded"] for r in rows),
        independent_count=0,
        sample_size_budget=dict(
            intended_task_slots=13,
            independent_questions=0,
            retrospective_questions_reserved=300,
            seeds_and_siblings_are_not_independent=True,
        ),
        verifier_is_oracle=True,
        exposure_scope="administrative contract; released human labels exposed; older overlap unknown",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=qualified,
        flagged_adversarial=not qualified,
        acceptance_gates=dict(
            owned_classification=qualified,
            current_contract=bool(ready),
            methods_frozen=True,
            optional_cuda=work["runtime"]["ready"],
            scientific_benefit=False,
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(
            output.parent / "raw" / output.stem / "terminal_validation.json"
        ),
        adversarial_findings=work.get("adversarial_findings", []),
        preconditions_checked=work["preconditions"],
        duration_s=(work["ended_monotonic_ns"] - work["started_monotonic_ns"]) / 1e9,
        phase_spans=work["phase_spans"],
        random_seed=7258416,
        source_artifact_hashes=work["source_refs"],
        code_config_hashes=work["code_refs"],
        raw_shard_hashes=[reference(raw / "measurement.json"), *work.get("coverage_shards", [])],
        cited_upstream_artifacts=historical,
        current_contract_ready_score=int(ready),
        methods_ready_score=int(ready),
        canonical_tasks_sha256=c["canonical_tasks_sha256"],
        observed_active_tasks_sha256=c["observed_active_tasks_sha256"],
        methods_protocol_path=str(ROOT / PROTOCOL),
        methods_protocol_sha256=PROTOCOL_PIN,
        methods_note_path=str(ROOT / METHODS),
        methods_note_sha256=METHODS_PIN,
        methods_frozen_score=1,
        runtime_repair_ready_score=int(qualified and work["runtime"]["ready"]),
        runtime_repair_receipt=work["runtime"]["receipt"],
        current_device_probe_count=0,
        independent_eval_ready_score=0,
        exposure_obligations=p["independent_evaluation"],
        scope_reduction_compliance=p["scope_reduction_compliance"],
        literature_mapping=p["literature_mapping"],
        primitive_reference=reference(raw / "measurement.json"),
        publication_output=str(output),
        repository_health=dict(status="not_run", reason="scoped task; global health separate"),
        future_dependencies=[
            dict(experiment_id=i, status="future_dependency_not_observation")
            for i in range(8417, 8429)
        ],
        methodology_note=f"Oracle contract custody only; {c['observed_count']} actual task objects, {13 - c['observed_count']} absent. Sealed retrospective methods grant no semantic or independent benefit.",
    )
    value["field_principles"] = {
        k: "Bind this field to sealed invocation operands; qualification cannot establish scientific benefit."
        for k in [*value, "field_principles", "reproducibility_checksum"]
    }
    value["field_principles"].update(
        {
            "canonical_tasks_sha256": "Independent complete-task design digest; null when its binding is absent.",
            "observed_active_tasks_sha256": "Hash only the actually observed full objects; never substitute this for the absent complete design digest.",
            "methods_frozen_score": "Sealed methods exist; admission still requires the complete current contract and owned qualification.",
            "methods_note_path": "Name the explanatory note published before reduction.",
            "methods_note_sha256": "Fix exact note bytes independently of candidate summary hashes.",
            "future_dependencies": "Name planned producer IDs without reading absent future files or claiming they ran.",
            "primitive_reference": "Bind the invocation measurement and preserved authority bytes used by fresh cold replay.",
            "repository_health": "Global health is explicitly unmeasured and cannot be inferred from scoped qualification.",
            "raw_shard_hashes": "Retain actual invocation primitives and genuine Coverage.py child database bytes.",
            "code_config_hashes": "Pin newly owned files, test assertions and unchanged shipped readers and validators.",
            "methodology_note": "Limit oracle administrative qualification to observed task objects and exposed methods.",
            "publication_output": "Name the one reader-visible primary that receives only checked terminal bytes.",
        }
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Cold readers rederive contract rows from source bytes rather than a self-rehashed summary."""
    try:
        value = json.loads(path.read_bytes())
        work = read(value["primitive_reference"])
        if work["protocol"] != protocol() or work["source_refs"] != protocol()["source_references"]:
            return False
        for ref in [
            *value["source_artifact_hashes"],
            *value["code_config_hashes"],
            *value["raw_shard_hashes"],
            *work["authority_inputs"],
        ]:
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                return False
        with TemporaryDirectory(prefix="exp8416-cold-", dir="/var/tmp") as directory:
            root = Path(directory)
            for ref in work["authority_inputs"]:
                target = root / ref["role"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ref["path"], target)
            actual = authority(root, root / "checks")
            actual.pop("authority_snapshots", None)
            expected = {k: v for k, v in work["contract"].items() if k != "authority_snapshots"}
            normalized = json.loads(json.dumps(actual).replace(str(root), work["root"]))
            if normalized != expected:
                return False
        for receipt in value["validation_receipts"]:
            saved = json.loads(
                Path(receipt["stdout_path"]).with_suffix(".receipt.json").read_bytes()
            )
            if receipt != saved:
                return False
            for stream in ["stdout", "stderr"]:
                if sha256_file(Path(receipt[stream + "_path"])) != receipt[stream + "_sha256"]:
                    return False
        runtime_actual = runtime(Path(work["root"]))
        if {k: v for k, v in runtime_actual.items() if k != "inspection_duration_s"} != {
            k: v for k, v in work["runtime"].items() if k != "inspection_duration_s"
        }:
            return False
        return bool(
            value
            == build(
                work,
                value["validation_receipts"],
                Path(value["primitive_reference"]["path"]).parent,
                Path(value["publication_output"]),
            )
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False
