"""REQ-REPORT-8437: immutable upstream operands bind conditional speed ceilings."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import time
from typing import Any

import yaml

from carnot.reporting import transaction_attribution_8436 as upstream
from carnot.reporting import v726_contract_methods as authority
from carnot.reporting.current_work_receipt import atomic_json, sha256_file, ZERO_INVOCATION_COUNTS
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.verify import acceleration_ceiling_8437 as k
from carnot.verify.transaction_attribution_8436 import attribute

Json = dict[str, Any]
ROOT = authority.ROOT
NAME, TASK = "experiment_8437_v726_acceleration_ceiling", "exp8437-acceleration-ceiling"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_acceleration_ceiling_8437.py"
OWNED = [
    "python/carnot/verify/acceleration_ceiling_8437.py",
    "python/carnot/reporting/acceleration_ceiling_8437.py",
    "python/carnot/reporting/acceleration_ceiling_runner_8437.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
UPSTREAM = "results/experiment_8436_v726_transaction_attribution.json"
PIN = "sha256:4cf40daca5cf3dd1bc27d5e3159e52d8887d2fa6f65e7b2ae144e039df14ac7b"
REQUIRED = upstream.REQUIRED
reference, read, checksum = authority.reference, upstream.read, upstream.checksum


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed counts expose genuine work without artificial runtime padding."""
    print(f"[exp8437] phase={phase} completed={completed} pending={pending}", flush=True)


def gate(path: Path, field: str, expected: Any, observed: Any) -> Json:
    """A missing input is null rather than a failed numerical observation."""
    return dict(upstream.gate(path, field, expected, observed), upstream=upstream.TASK)


def producer(root: Path) -> Json:
    """A separately pinned terminal defeats substitutions that recompute local hashes."""
    path = root / UPSTREAM
    value = read(dict(path=str(path), sha256=PIN))
    terminal = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
    sidecar = read_bound_sidecar(path, Path(terminal["publication"]["sidecar_path"]))
    if not (
        value["clock_attribution_ready_score"] == 1
        and value["required_checks_passed"]
        and not value["flagged_adversarial"]
        and not value["adversarial_findings"]
        and sidecar["report"]["passed"]
        and upstream.replay(path)
    ):
        raise ValueError("unqualified_8436")
    return dict(value)


def reduce(root: Path, operands: list[Json]) -> list[Json]:
    """Recompute every measured repeat against the original producer's independent seal."""
    original = producer(root)["rows"]
    if len(operands) != 135 or len(original) != 135:
        raise ValueError("repeat_roster")
    rows = []
    for index, (ref, expected) in enumerate(zip(operands, original, strict=True)):
        if index % 15 == 0:
            progress("primitive_reduction", index, 135 - index)
        if ref["sha256"] != expected["primitive_reference"]["sha256"]:
            raise ValueError("producer_row_substitution")
        row = attribute(read(ref))
        if row != {key: item for key, item in expected.items() if key != "primitive_reference"}:
            raise ValueError("producer_attribution_drift")
        rows.append(dict(row, primitive_reference=ref))
    progress("primitive_reduction_complete", 135, 0)
    return rows


def measure(root: Path, raw: Path, private: Path) -> Json:
    """Check current authority and host resources before admitting cached primitives."""
    progress("preconditions_before")
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    probe = private / "disk_probe"
    probe.write_bytes(b"private disk scratch")
    mounts = [line.split(" - ") for line in Path("/proc/self/mountinfo").read_text().splitlines()]
    mount = max(
        (m for m in mounts if private.resolve().is_relative_to(Path(m[0].split()[4]))),
        key=lambda m: len(m[0].split()[4]),
    )
    memory = int(Path("/proc/meminfo").read_text().split("MemAvailable:")[1].split()[0]) * 1024
    resources: Json = dict(
        disk_backed=mount[1].split()[0] not in ("tmpfs", "ramfs"),
        available_memory_bytes=memory,
        memory_available=memory > 256_000_000,
        disk_free=shutil.disk_usage(private).free > 256_000_000,
        private_permissions=private.stat().st_mode & 0o077 == 0,
        scratch_readback=probe.read_bytes() == b"private disk scratch",
        scratch_path=str(private),
        task_cap_s=4800,
        validation_reserve_s=900,
        tools={
            n: (ROOT / ".venv/bin" / n).is_file()
            for n in ("python", "pytest", "coverage", "ruff", "mypy")
        },
        optional_cuda_board_V722="not required; no device commands",
    )
    active = authority.authority(root, raw / "authority")
    gates = (
        []
        if active["activated"]
        else [gate(root / authority.ACTIVE, "activated_complete_V726_authority", True, False)]
    )
    for name in (
        "disk_backed",
        "memory_available",
        "disk_free",
        "private_permissions",
        "scratch_readback",
    ):
        if not resources[name]:
            gates.append(gate(private, name, True, resources[name]))
    for name, present in resources["tools"].items():
        if not present:
            gates.append(gate(ROOT / ".venv/bin" / name, "tool_available", True, None))
    refs = [
        reference(Path(s["snapshot_path"]))
        for s in active["authority_snapshots"].values()
        if s.get("exists")
    ]
    refs.extend(
        reference(ROOT / p)
        for p in (
            "ops/exclusion_manifest.yaml",
            "ops/north-star.md",
            "_bmad/prd.md",
            "research-hardware-wishlist.md",
        )
    )
    operands, rows, provenance = [], [], []
    observed = sha256_file(root / UPSTREAM) if (root / UPSTREAM).is_file() else None
    if observed != PIN:
        gates.append(gate(root / UPSTREAM, "qualified_8436_sha256", PIN, observed))
    else:
        try:
            value = producer(root)
            refs.append(reference(root / UPSTREAM))
            provenance = value["historical_model_provenance"]
            for index, row in enumerate(value["rows"]):
                if index % 15 == 0:
                    progress("snapshot_operands", index, 135 - index)
                source = row["primitive_reference"]
                read(source)
                path = raw / "operands" / f"{index:03}.json"
                path.parent.mkdir(exist_ok=True)
                shutil.copyfile(source["path"], path)
                operands.append(reference(path))
            rows = reduce(root, operands)
        except (OSError, ValueError, KeyError, TypeError) as error:
            gates.append(
                gate(root / UPSTREAM, "qualified_terminal_and_primitives", True, str(error))
            )
            rows = []
    if not rows:
        rows = [
            dict(
                cell=cell,
                repeat=repeat,
                status="censored",
                missing_reason="required_qualified_operands_unavailable",
            )
            for cell in upstream.source.CELLS
            for repeat in range(1, 6)
        ]
    work = dict(
        root=str(root),
        raw=str(raw),
        rows=rows,
        operands=operands,
        refs=refs,
        gate_check_summary=gates,
        authority=active,
        preconditions_checked=resources,
        historical_model_provenance=provenance,
        code_config_hashes=[reference(ROOT / p) for p in [*OWNED, TEST]],
        unchanged_guard_hashes=[reference(ROOT / p) for p in upstream.GUARDS],
        started_monotonic_ns=start,
        duration_s=(time.monotonic_ns() - start) / 1e9,
        phase_spans=[
            dict(phase="authenticate_and_reduce", start_ns=start, end_ns=time.monotonic_ns())
        ],
    )
    atomic_json(raw / "measurement.json", work)
    progress(
        "preconditions_and_reduction_after",
        sum(row["status"] == "completed" for row in rows),
        sum(row["status"] != "completed" for row in rows),
    )
    return work


def build(work: Json, receipts: list[Json]) -> Json:
    """Readiness requires qualified operands and every owned check, independent of gain."""
    valid = [row for row in work["rows"] if row["status"] == "completed"]
    complete = len(valid) == 135
    passed = bool(receipts) and all(row["passed"] for row in receipts)
    required = REQUIRED <= {row["name"] for row in receipts if row["passed"]}
    verdict = (
        "disqualified"
        if not passed or work.get("adversarial_findings", [])
        else "blocked"
        if work["gate_check_summary"]
        else "null"
        if complete and required
        else "disqualified"
    )
    analysis = k.analyze(valid)
    primary = [row for row in valid if row["cell"] == k.PRIMARY]
    stages = {
        name: sum(row["stage_ns"].get(name, 0) for row in primary) / 1e9
        for name in sorted({name for row in primary for name in row["stage_ns"]})
    }
    value: Json = dict(
        experiment_id=8437,
        task_id=TASK,
        milestone="2026.10.726",
        run_date="20261012",
        honest_verdict="complete_" + verdict + "_acceleration_ceiling",
        verdict_class=verdict,
        gate_check_summary=work["gate_check_summary"],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS, current_llm_invocations=0),
        historical_model_provenance=work["historical_model_provenance"],
        rows=work["rows"],
        intended_count=135,
        completed_count=len(valid),
        failed_count=0,
        censored_count=135 - len(valid),
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            cells=27,
            measured_repeats_per_cell=5,
            primary_repeats=5,
            independent_deployment_samples=0,
            repeats_add_independent_questions=0,
        ),
        verifier_is_oracle=False,
        exposure_scope="retrospective_exposed_cached_cost_operands",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=passed and (required or verdict == "blocked"),
        flagged_adversarial=verdict == "disqualified",
        acceptance_gates=dict(
            authentic_inputs=not work["gate_check_summary"],
            complete_135_repeats=complete,
            owned_validation=passed,
            scoped_validation=required,
            findings_clear=not work.get("adversarial_findings", []),
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(Path(work["raw"]) / "terminal_validation.json"),
        adversarial_findings=work.get("adversarial_findings", []),
        preconditions_checked=work["preconditions_checked"],
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        random_seed=k.SEED,
        source_artifact_hashes=work["refs"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes=work["operands"] + work.get("coverage_shards", []),
        cited_upstream_artifacts=[
            dict(
                ref,
                imported_fields="qualified disjoint primitive clocks, cell/repeat identities and historical model provenance",
            )
            for ref in work["refs"]
        ],
        acceleration_analysis_ready_score=int(verdict == "null"),
        actual_native_speedup=None,
        native_parity_ready_score=0,
        measured_primary_composite_stages_s=stages,
        owned_coverage=work.get("owned_coverage", {}),
        unchanged_guard_hashes=work["unchanged_guard_hashes"],
        work_reference=reference(Path(work["raw"]) / "measurement.json"),
        bound_assumptions=dict(
            unchanged_service=True,
            identical_work_and_durability=True,
            disjoint_K="direct_scoring+coefficient_update",
            optimistic_Kplus="K+python_boundary_conversion",
            unaccelerated="initialization,persistence,gaps,response,tail; tail overlaps complete wall and is not added twice",
            hardware_impossibility_theorem=False,
            native_speed_measured=False,
        ),
        hardware_acquisition_implications="Arithmetic hardware alone cannot justify a 10x service claim when every primary K/T is below0.9. No native or board purchase qualification. Measure composite persistence and snapshot stages before acquisition. Durability/batching redesign requires a new hypothesis and equivalence proof.",
        target_comparison=dict(
            kernel_counterfactual_100x=analysis["descriptive_interval"]["arithmetic"]["100"][
                "pooled_point"
            ]
            if primary
            else None,
            research_program_aspiration="Parity at10-100x cheaper versus an LLM judge (ops/north-star.md); comparator differs from equivalent Python service NFR-01",
            unchanged_NFR_01="minimum10x throughput over equivalent Python inference workloads",
            unchanged_service_target_met=bool(
                primary
                and analysis["descriptive_interval"]["arithmetic"]["100"]["pooled_point"] >= 10
            ),
        ),
        **analysis,
    )
    value["field_principles"] = {
        key: "Bind exact primitive or invocation evidence; qualification establishes no native speed or generalization benefit."
        for key in value
    }
    value["field_principles"].update(
        descriptive_interval="Identical seeded indices pool T,K,Kplus within dependent host repeats; no deployment inference.",
        acceptance_gates="Authentic complete operands and passing owned checks qualify an analytical ceiling, not scientific benefit.",
        lower_bound_nonkernel_removal="Fraction of original total outside arithmetic that must additionally disappear for10x at infinite arithmetic acceleration.",
        measured_primary_composite_stages_s="Measured composite totals only; no allocation to unmeasured internal operations.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Cold reconstruction binds both local hashes and independently sealed producer bytes."""
    try:
        value = json.loads(path.read_bytes())
        if value["reproducibility_checksum"] != checksum(value):
            return False
        work = read(value["work_reference"])
        if work["authority"]["activated"]:
            snapshot = work["authority"]["authority_snapshots"]["active"]
            active = yaml.safe_load(Path(snapshot["snapshot_path"]).read_bytes())
            if authority.tasks_digest(active["tasks"]) != authority.TASKS_PIN:
                return False
        for ref in [
            *value["source_artifact_hashes"],
            *value["code_config_hashes"],
            *value["unchanged_guard_hashes"],
            *value["raw_shard_hashes"],
        ]:
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                return False
        if (
            any(row["status"] == "completed" for row in work["rows"])
            and reduce(Path(work["root"]), work["operands"]) != work["rows"]
        ):
            return False
        return bool(value == build(work, value["validation_receipts"]))
    except (OSError, ValueError, KeyError, TypeError):
        return False
