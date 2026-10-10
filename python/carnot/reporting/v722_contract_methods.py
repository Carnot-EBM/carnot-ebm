"""REQ-REPORT-8374: direct input custody never reopens closed utility science."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from typing import Any

import yaml

from carnot.reporting import v721_contract_methods as previous
from carnot.reporting import v720_frozen_inputs as inputs
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.reporting.roadmap_contract import parse_design
from carnot.reporting.v685_authority_lifecycle import assess_authorities, tasks_digest
from carnot.reporting.v710_contract_replay import require_reference, snapshot

Json = dict[str, Any]
ROOT, DESIGN, ACTIVE, STAGED = previous.ROOT, previous.DESIGN, previous.ACTIVE, previous.STAGED
NAME, TASK, MILESTONE = (
    "experiment_8374_v722_contract_methods",
    "exp8374-contract-methods",
    "2026.10.722",
)
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v722_contract_methods_8374.py"
PROTOCOL = "openspec/change-proposals/v722-direct-service-protocol.json"
METHODS = "openspec/change-proposals/v722-methods-manifest.json"
PROTOCOL_PIN = "sha256:ab877c98112dc1a1497bb9aa1f2c28751e1b78671de743a6e4c036eeeddcc47e"
METHODS_PIN = "sha256:2aed5b0f7e7e7bf0a26b84170be1de4bd871909c3bb58ea7174e1d70c98a2f90"
OWNED = [
    "python/carnot/reporting/v722_contract_methods.py",
    "python/carnot/reporting/v722_contract_runner.py",
    CLI,
]
MODEL_SPECS: list[Json] = []


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real counts so evidence reading remains visible to the operator."""
    print(f"[exp8374] phase={phase} completed={completed} pending={pending}", flush=True)


def failure(path: Path, field: str, expected: Any, observed: Any) -> Json:
    """Name both missing and wrong operands; absence cannot become numeric zero."""
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
    """An activated task list cannot stand in for missing independent design bytes."""
    staged = root / STAGED
    try:
        tasks = parse_design((root / DESIGN).read_text(), milestone=MILESTONE)[1]
        checked = assess_authorities(
            root / DESIGN,
            staged if staged.exists() else root / ACTIVE,
            root / ACTIVE,
            raw,
            milestone=MILESTONE,
            first_id=8374,
            count=14,
        )
        if tasks_digest(tasks) != checked["canonical_tasks_sha256"]:
            raise ValueError("complete_task_digest")
        for row, task in zip(checked["contract_rows"], tasks, strict=True):
            if not task.get("prior_failures"):
                row["checks"]["prior"] = True
            row["matched"] = all(row["checks"].values())
            row.update(
                absolute_metric=int(row["matched"]),
                raw_numerator=sum(row["checks"].values()),
                excluded=not row["matched"],
                full_task_sha256=canonical_hash(task),
            )
        checked["gate_check_summary"] = [
            g
            for g in checked["gate_check_summary"]
            if g["artifact_field"] != "contract_rows.matched"
            or not all(r["matched"] for r in checked["contract_rows"])
        ]
        checked["activated"] = not checked["gate_check_summary"]
        checked["tasks"] = tasks
    except (OSError, ValueError, KeyError, TypeError, IndexError, yaml.YAMLError):
        plan = yaml.safe_load((root / ACTIVE).read_bytes()) if (root / ACTIVE).is_file() else {}
        tasks = plan.get("tasks", [])
        rows = []
        for index in range(14):
            task = tasks[index] if index < len(tasks) else {}
            checks = dict(
                independent_design_contract=False,
                activated_task_present=bool(task),
                sequence=str(task.get("id", "")).startswith(f"exp{8374 + index}-"),
            )
            rows.append(
                dict(
                    family=task.get("id", f"exp{8374 + index}"),
                    unit_id=task.get("id", f"exp{8374 + index}"),
                    arm="active_contract",
                    seed=None,
                    order=index + 1,
                    status="completed" if task else "unstarted",
                    matched=False,
                    checks=checks,
                    absolute_metric=0,
                    raw_numerator=sum(checks.values()),
                    raw_denominator=len(checks),
                    censored=not bool(task),
                    excluded=True,
                    effective_independent_groups=0,
                    full_task_sha256=canonical_hash(task),
                    missing_reason="independent_design_contract_absent",
                )
            )
        checked = dict(
            activated=False,
            contract_rows=rows,
            tasks=tasks,
            canonical_tasks_sha256=None,
            active_tasks_sha256=tasks_digest(tasks),
            planning_matched=False,
            gate_check_summary=[
                failure(root / DESIGN, "independent_design_contract_available", True, None)
            ],
        )
    checked["staging_disposition"] = "existing" if staged.exists() else "consumed_by_activation"
    return dict(checked)


def measure(root: Path, raw: Path) -> Json:
    """Authenticate original producers separately without running any closed procedure."""
    began = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    memory = (
        int(
            next(
                line.split()[1]
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:")
            )
        )
        * 1024
    )
    mounts = [line.split() for line in Path("/proc/self/mountinfo").read_text().splitlines()]
    mounted = max(
        (m for m in mounts if raw.resolve().is_relative_to(Path(m[4]))), key=lambda m: len(m[4])
    )
    filesystem = mounted[mounted.index("-") + 1]
    resources: Json = dict(
        private_disk_backed_scratch=filesystem not in ["tmpfs", "ramfs"],
        filesystem=filesystem,
        scratch_path=str(raw),
        available_memory_bytes=memory,
        minimum_memory_bytes=536870912,
        task_cap_s=4800,
        model_loads=0,
        llm_calls=0,
    )
    progress("resource_preconditions_checked")
    progress("authority_before")
    contract = authority(root, raw / "authority")
    refs = [
        snapshot(root / name, raw / "authority", str(i))
        for i, name in enumerate([DESIGN, STAGED, ACTIVE])
    ]
    failures = list(contract["gate_check_summary"])
    if not resources["private_disk_backed_scratch"] or memory < resources["minimum_memory_bytes"]:
        failures.append(failure(raw, "private_disk_memory_budget", True, resources))
    progress("authority_after", 14, 0)
    frozen = inputs.load(root, raw)
    refs.extend(frozen["refs"])
    failures.extend(frozen["failures"])
    methods: Json = {}
    deployment: Json = {}
    reuse: Json = dict(kernel={}, trajectory={})
    historical: Json = {}
    try:
        methods = inputs.bind(dict(path=str(root / METHODS), sha256=METHODS_PIN), raw, refs)
        deployment = inputs.bind(dict(path=str(root / PROTOCOL), sha256=PROTOCOL_PIN), raw, refs)
        for operand in [
            *methods["papers"],
            *methods["references"],
            methods["reference_scan"],
            deployment["panels"]["vectors"],
            deployment["panels"]["metadata"],
            deployment["panels"]["natural_predictor"],
        ]:
            inputs.bind(operand, raw, refs, parse=False)
        for phase, operand in methods["reusable"].items():
            progress(phase + "_authentication_before")
            inputs.bind(operand, raw, refs, parse=False)
            reuse[phase] = previous.reusable(operand, operand["readiness_field"], raw, refs)
            progress(phase + "_authentication_after", 1, 0)
        values = [inputs.bind(operand, raw, refs) for operand in methods["historical"]]
        utility, guard, runtime, capstone = values
        previous.reusable(methods["historical"][0], "static_audit_ready_score", raw, refs)
        previous.reusable(methods["historical"][0], "learning_audit_ready_score", raw, refs)
        historical = dict(
            executed=capstone["actual_executed_task_count"],
            pre_gate=capstone["pre_gate_count"],
            absent=capstone["missing_output_count"],
            task_dispositions=capstone["task_dispositions"],
            utility=dict(
                H1_gain=utility["H1"]["bootstrap_summary"]["all_intended"]["mean_gain"],
                H2_gain=utility["H2"]["block_bootstrap_summary"]["mean_gain"],
                scopes=utility["qualified_scope_decision"],
                closed=True,
                rerun=False,
            ),
            table=dict(
                fast_path_fraction=guard["fast_path_fraction"],
                guard_ready_score=guard["guard_ready_score"],
                required_for_direct=False,
            ),
            runtime=dict(
                honest_verdict=runtime["honest_verdict"],
                current_calls=runtime["model_invocation_counts"],
            ),
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        failures.append(
            error.gate
            if isinstance(error, inputs.CustodyError)
            else failure(root / METHODS, "authenticated_direct_methods", True, str(error))
        )
    progress("inputs_sealed", len(refs), 0)
    return dict(
        root=str(root),
        contract=contract,
        inputs=frozen,
        protocol=frozen["protocol"],
        protocol_sha256=frozen["protocol_sha256"],
        deployment=deployment,
        methods=methods,
        support=dict(ready=bool(frozen["support"]), counts=frozen["support"].get("counts", {})),
        historical_model_provenance=frozen["historical"],
        refs=refs,
        failures=failures,
        input_failures=list(failures),
        historical_dispositions=historical,
        resource_checks=resources,
        history={},
        numeric={},
        **reuse,
        started_monotonic_ns=began,
        ended_monotonic_ns=time.monotonic_ns(),
    )


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Readiness is engineering custody; closed utility stays descriptive and zero."""
    owned_receipts = [r for r in receipts if r.get("scope") != "global"]
    value = dict(previous.legacy.build(work, owned_receipts, raw, output))
    owned = bool(owned_receipts) and all(r["passed"] for r in owned_receipts)
    direct = (
        owned
        and bool(work["inputs"]["heads"])
        and work["kernel"].get("ready", False)
        and work["trajectory"].get("ready", False)
        and bool(work["deployment"])
    )
    kind = "disqualified" if not owned else "blocked" if work["failures"] else "circular_positive"
    value.update(
        experiment_id=8374,
        task_id=TASK,
        milestone=MILESTONE,
        run_date="20261010",
        honest_verdict="complete_" + kind + "_direct_contract_methods",
        verdict_class=kind,
        required_checks_passed=owned,
        flagged_adversarial=not owned,
        current_contract_ready_score=int(owned and work["contract"]["activated"]),
        direct_inputs_ready_score=int(direct),
        frozen_heads_ready_score=int(owned and bool(work["inputs"]["heads"])),
        local_kernel_ready_score=int(owned and work["kernel"].get("ready", False)),
        frozen_trajectory_ready_score=int(owned and work["trajectory"].get("ready", False)),
        protocol_path=str(Path(work["root"]) / PROTOCOL),
        protocol_sha256=PROTOCOL_PIN if work["deployment"] else None,
        direct_service_protocol=work["deployment"],
        methods_manifest=work["methods"],
        historical_dispositions=work["historical_dispositions"],
        random_seed=7228374,
        validation_receipts=receipts,
        repository_health=[r for r in receipts if r.get("scope") == "global"],
        methodology_note="Administrative full-object oracle agreement and authenticated original direct inputs only. Closed exposed H1/H2 are never rerun. Future producer paths are dependencies, not inputs. Execution correctness and numerical parity grant no semantic benefit. Zero current LLM calls; no generator updates or vendor speedup adoption.",
    )
    value["preconditions_checked"].update(work["resource_checks"])
    adapter = value["invocation_argv"]
    value["invocation_adapter_argv"] = adapter
    value["invocation_argv"] = [
        "20261010" if item == "20261008" and index and adapter[index - 1] == "--date" else item
        for index, item in enumerate(adapter)
    ]
    value["acceptance_gates"].update(
        direct_inputs=bool(direct),
        frozen_protocol=bool(work["deployment"]),
        scientific_benefit=False,
    )
    value["gate_check_summary"] = [
        *work["failures"],
        *[
            failure(
                Path(r.get("stdout_path", str(raw))), r.get("name", "owned_validation"), True, False
            )
            for r in owned_receipts
            if not r["passed"]
        ],
    ]
    value["field_principles"].update(
        {
            k: "Bind administrative custody to this invocation; never import historical or future science credit."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["field_principles"].update(
        direct_inputs_ready_score="Authenticate original qualified heads, direct kernel and delayed trajectory separately from activated authority.",
        direct_service_protocol="Freeze unchanged direct SciPy probability actions and a separately versioned dyadic-logit prototype before measurements.",
        historical_dispositions="Preserve eight executed producers, two pre-gate receipts, four absent primaries and closed exposed utility findings.",
        repository_health="Keep the one bounded full repository test run separate from owned qualification.",
        invocation_adapter_argv="Preserve the internal legacy date adapter separately from the exact current invocation.",
    )
    value["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in value.items() if k != "reproducibility_checksum"}
    )
    return value


def replay(path: Path) -> bool:
    """Recompute operand meaning so a rehashed forged summary still fails."""
    try:
        value = json.loads(path.read_bytes())
        for ref in [
            value["work_reference"],
            *value["source_artifact_hashes"],
            *value["code_config_hashes"],
        ]:
            require_reference(ref)
        work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
        for field in ["execution_manifest_reference", "owned_coverage_reference"]:
            if work.get(field):
                require_reference(work[field])
        for receipt in value["validation_receipts"]:
            for stream in ["stdout", "stderr"]:
                require_reference(
                    dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"])
                )
        with TemporaryDirectory(prefix="exp8374-cold-") as directory:
            actual = measure(Path(work["root"]), Path(directory))
            for key in [
                "contract",
                "inputs",
                "deployment",
                "methods",
                "support",
                "historical_dispositions",
                "kernel",
                "trajectory",
                "protocol",
                "protocol_sha256",
                "historical_model_provenance",
                "input_failures",
            ]:
                left, right = actual[key], work[key]
                if key == "inputs":
                    left, right = (
                        {k: v for k, v in obj.items() if k != "refs"} for obj in [left, right]
                    )
                if key == "contract":
                    left, right = (
                        {k: v for k, v in obj.items() if k != "authority_snapshots"}
                        for obj in [left, right]
                    )
                if left != right:
                    return False
        if value["source_artifact_hashes"] != work["refs"]:
            return False
        return bool(
            build(
                work,
                value["validation_receipts"],
                Path(value["work_reference"]["path"]).parent,
                Path(value["publication_output"]),
            )
            == value
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False
