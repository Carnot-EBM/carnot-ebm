"""REQ-REPORT-8776: freeze independent methods without reopening old learning work."""

from copy import deepcopy
from datetime import datetime, UTC
from pathlib import Path
import json
import shutil
import time
from typing import Any

import yaml

from carnot.reporting import v744_contract_methods as runtime_reader
from carnot.reporting import v751_branch_contract as archive
from carnot.reporting import v751_branch_evidence as arc
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.roadmap_contract import compare_contract, parse_design
from carnot.reporting.v746_retained_evidence import load, reference
from carnot.reporting.v749_contract_methods import digest, encode

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME, TASK, MILESTONE = (
    "experiment_8776_v752_branch_methods",
    "exp8776-branch-methods",
    "2026.11.752",
)
DESIGN = "openspec/change-proposals/research-roadmap-vNEXT.md"
ACTIVE, STAGED = "research-roadmap.yaml", "research-roadmap-next.yaml"
PIN = "96a277806aa53175f8661b2e09da69a3c62ddab8f88d0eaaf21e6f25aadc945d"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [f"python/carnot/reporting/v752_branch_{s}.py" for s in ["methods", "report", "runner"]] + [
    CLI
]
TESTS = ["tests/python/test_v752_branch_methods_8776.py"]
CONSUMERS = [
    "tests/python/test_primary_publication_7928.py",
    "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
    "tests/python/test_restricted_decision_audit_8210.py",
]
MODEL_SPECS: list[Json] = []
HISTORICAL_PINS = {
    8763: "sha256:8d3018a69d01e05a8162f85d4c23a0b2b891aa89e586a40fac2d3f8a71fa5c19",
    8764: "sha256:bfe690e41a7f14940741fa8863956834aecfa494a14b08e8093e96fb46206662",
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush phase boundaries so a parent can distinguish bounded work from a stall."""
    print(f"[exp8776] phase={phase} completed={completed} pending={pending}", flush=True)


def authority(root: Path) -> Json:
    """Compare complete objects; an independent pin rejects coordinated source edits."""
    try:
        text = (root / DESIGN).read_text()
        table, tasks = parse_design(text, milestone=MILESTONE)
        active = yaml.safe_load((root / ACTIVE).read_bytes())
        staged = (
            yaml.safe_load((root / STAGED).read_bytes())
            if (root / STAGED).is_file()
            else deepcopy(active)
        )
        for roadmap in [active, staged]:
            for supplied, expected in zip(roadmap["tasks"], tasks):
                if supplied == dict(expected, prompt=encode(expected["prompt"])):
                    supplied["prompt"] = expected["prompt"]
        result = compare_contract(
            text, staged, active, milestone=MILESTONE, first_id=8776, count=13
        )
        for i, row in enumerate(result["rows"]):
            row["checks"]["full_object"] = (
                i < min(len(tasks), len(active["tasks"]), len(staged["tasks"]))
                and tasks[i] == active["tasks"][i] == staged["tasks"][i]
            )
            row.update(
                task_object=deepcopy(tasks[i]) if i < len(tasks) else None,
                effective_independent_groups=0,
            )
            row.update(
                matched=all(row["checks"].values()),
                raw_numerator=sum(row["checks"].values()),
                raw_denominator=len(row["checks"]),
            )
            row.update(
                absolute_metric=int(row["matched"]),
                censored=not row["matched"],
                excluded=False,
                missing_reason=None if row["matched"] else "authority_mismatch",
                cost_s=None,
            )
        ready = (
            result["passed"]
            and all(r["matched"] for r in result["rows"])
            and digest(tasks) == PIN
            and f"Canonical tasks SHA-256: `{PIN}`" in text
            and "<!-- V752_TASK_CONTRACT_END -->" in text
        )
        return dict(
            ready=bool(ready),
            rows=result["rows"],
            visible_table=table,
            references=[
                reference(root / p) for p in [DESIGN, ACTIVE, STAGED] if (root / p).is_file()
            ],
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError):
        return dict(ready=False, rows=[], visible_table=[], references=[])


def mutations(source: Path, root: Path) -> list[Json]:
    """Private mutations prove every field binds without changing the execution queue."""
    design = root / DESIGN
    design.parent.mkdir(parents=True)
    text = (source / DESIGN).read_text()
    tasks = parse_design(text, milestone=MILESTONE)[1]
    rows = []
    for i, task in enumerate(tasks):
        for field in task:
            changed = deepcopy(tasks)
            changed[i][field] = "forged"
            design.write_text(text)
            (root / ACTIVE).write_text(yaml.safe_dump(dict(milestone=MILESTONE, tasks=changed)))
            rows.append(dict(task_id=task["id"], field=field, passed=not authority(root)["ready"]))
        progress("full_object_controls", i + 1, 12 - i)
    for mode in [
        "valid",
        "encoded",
        "count",
        "order",
        "table",
        "digest",
        "rehashed",
        "start",
        "end",
        "staged",
    ]:
        changed, body = deepcopy(tasks), text
        if mode == "encoded":
            for task in changed:
                task["prompt"] = encode(task["prompt"])
        elif mode == "count":
            changed.pop()
        elif mode == "order":
            changed.reverse()
        elif mode == "table":
            body = body.replace("| 1 |", "| 99 |", 1)
        elif mode == "digest":
            body = body.replace(PIN, "0" * 64)
        elif mode == "rehashed":
            changed[0]["prompt"] += "forged"
            head, tail = text.split("```json", 1)
            _, end = tail.split("```", 1)
            body = (
                head
                + "```json\n"
                + json.dumps(dict(milestone=MILESTONE, tasks=changed))
                + "\n```"
                + end
            )
            body = body.replace(PIN, digest(changed))
        elif mode in {"start", "end"}:
            body = body.replace(f"<!-- V752_TASK_CONTRACT_{mode.upper()} -->", "")
        design.write_text(body)
        (root / ACTIVE).write_text(yaml.safe_dump(dict(milestone=MILESTONE, tasks=changed)))
        if mode == "staged":
            changed[0]["routing"] = "forged"
            (root / STAGED).write_text(yaml.safe_dump(dict(milestone=MILESTONE, tasks=changed)))
        rows.append(
            dict(mutation=mode, passed=authority(root)["ready"] == (mode in {"valid", "encoded"}))
        )
    return rows


def protocols(tasks: list[Json]) -> Json:
    """Seal planned inputs and causal rules before any branch evaluates new observations."""
    common = dict(
        canonical_tasks_sha256=PIN, implementation_certified=False, evaluation_performed=False
    )
    values = dict(
        evidence=dict(
            task_ids=[8777, 8779, 8782],
            arms=["local_record", "joint_symbolic", "evidence_set"],
            inputs=["signed_facts", "query", "partial_checks"],
            truth=["support", "refute", "insufficient"],
            exposed_replication_worlds=120,
            new_six_edge_worlds=120,
            retention_seven_edge_worlds=120,
            canary_worlds=12,
            check_budget=5,
            bootstrap_draws=10000,
            primary_contrasts=2,
            cluster="canonical_world",
            exposure_scope="constructed; old 120 worlds exposed",
            required_controls=[
                "missing_hop",
                "conflict",
                "valid_alternative",
                "surface_twins",
                "independent_boolean_truth",
            ],
        ),
        memory=dict(
            task_ids=[8778, 8780, 8785],
            inputs=[
                "visible_signed_facts",
                "query",
                "partial_symbolic_checks",
                "released_verified_feedback",
            ],
            observed_features=[
                "checked_predicates",
                "satisfied_local_predicates",
                "missing_predicates",
                "shared_variables",
                "observed_sign_conflicts",
                "current_successful_bindings",
            ],
            decision_indicators=["support", "refute", "insufficient"],
            typed_decisions=["accept", "reject", "escalate"],
            architecture=dict(input_dim=9, hidden_dims=[4], energy_normalization="exp(-energy)"),
            model="opt_in_GibbsModel_small_energy_policy",
            loss="proper_three_class_Brier",
            splits=dict(fit=40, tune=20, stream=120, retention=120, canary=12),
            split_isolation=[
                "canonical_graph_disjoint",
                "all_surface_siblings_same_split",
                "fit_and_tune_only_development",
                "no_retention_updates",
                "no_stream_tuning",
            ],
            causal_gates=[
                "prediction_sealed_before_label_release",
                "fixed_read_state_within_query",
                "no_truth_bearing_features",
                "delayed_verified_template_addition",
                "atomic_template_and_coefficient_commit",
                "duplicate_acknowledgment_rejected",
                "restart_equals_uninterrupted",
                "dropped_feedback_never_trains",
            ],
            arms=[
                "frozen",
                "read_only",
                "additions_only",
                "calibration_only",
                "combined",
                "shuffled_feedback",
            ],
            template_limit=32,
            check_budget=5,
            delay_queries=8,
            pending_capacity=32,
            seeds=[11, 23, 37, 53, 71],
            template_schema="alpha_normalized_join_or_conflict_constraint; no stored world answers",
            architecture_tuning="freeze architecture, clipping, epoch and learning-rate grid before development-only tuning",
            required_manifests=[
                "exp8777:40_fit_20_tune_development",
                "exp8777:120_six_edge_stream",
                "exp8777:120_seven_edge_retention",
                "exp8777:12_disjoint_canary",
            ],
            bootstrap_draws=10000,
            simultaneous_interval_percent=98.75,
            gain_lower_bound_gt=0,
            harm_upper_bound_le=0.01,
            primary_checks=[
                "combined_vs_frozen_Brier",
                "combined_vs_read_only_Brier",
                "false_acceptance_vs_frozen",
                "max_initial_middle_final_retention_harm",
            ],
            old_learning_union_authorized=False,
            oracle_feedback=True,
        ),
        capacity=dict(
            task_ids=[8781],
            capacities=[8, 32],
            policies=["FIFO", "uniform_pending_plus_arriving_discard"],
            admission_probability="1/(C+1) per discarded record when full",
            updates="unweighted",
            template_limit=32,
            check_budget=5,
            paired_seeds=[11, 23, 37, 53, 71],
            bootstrap_draws=10000,
            primary_contrasts=4,
            required_accounting=["tracked", "dropped", "released", "duplicate", "pending"],
            theorem_claim=False,
        ),
        cost=dict(
            task_ids=[8783, 8784, 8787],
            ordinary_starts=3,
            paired_process_repeats=2,
            updates_per_window=32,
            arms=["Python", "JSON_native", "resident_native"],
            workload_count=1,
            components=[
                "startup",
                "feature_constraint",
                "update",
                "transfer",
                "serialization",
                "file_sync",
                "directory_sync",
                "acknowledgment",
                "restart_verification",
                "shutdown",
                "unassigned",
            ],
            prefix_preparation_separate=True,
            per_feedback_durability=True,
            observer_overhead_max=0.05,
            clock_reconciliation_max=0.02,
            timing_repeats_are_independent=False,
            exact_producer_identity_required=True,
        ),
    )
    for value in values.values():
        value.update(common)
        value["task_operands"] = [
            deepcopy(t) for t in tasks if int(t["id"].split("-")[0][3:]) in value["task_ids"]
        ]
    return values


def memory_ready(value: Json) -> bool:
    """Protocol completeness grants planning custody while implementation stays separately gated."""
    return bool(
        value.get("inputs")
        and value.get("split_isolation")
        and value.get("causal_gates")
        and value == protocols([t for t in value["task_operands"]])["memory"]
    )


def seal_protocols(raw: Path, values: Json) -> Json:
    """Each branch gets its own immutable operand file for later independent readers."""
    refs = {}
    for name, value in values.items():
        atomic_json(raw / f"{name}_protocol.json", value)
        refs[name] = reference(raw / f"{name}_protocol.json")
    return refs


def gate(path: Path, field: str, observed: Any, expected: Any = True, upstream: str = TASK) -> Json:
    """Keep missing files, missing fields and measured zero distinct for downstream gates."""
    return dict(
        upstream=upstream,
        path=str(path.absolute()),
        sha256=reference(path)["sha256"] if path.is_file() else None,
        check=field,
        field=field,
        operator="==",
        expected=expected,
        observed=observed,
        passed=observed == expected,
    )


def runtime_input(root: Path, raw: Path) -> Json:
    """Use one existing authority read; visibility metadata alone cannot qualify a model."""
    value = dict(runtime_reader.runtime(root, raw))
    path = root / "ops/cuda-runtime-repair-receipt.json"
    changed = any(
        r.get("changed") and r.get("field") in {"libraries", "environment"}
        for r in value.get("changes", [])
    )
    provenance = value.get("mandated_model_provenance", {})
    model_path = Path(provenance.get("path", "/nonexistent"))
    usable = (
        provenance.get("hf_id") == "unsloth/Qwen3.8-27B-GGUF"
        and model_path.is_file()
        and reference(model_path)["sha256"] == provenance.get("sha256")
    )
    ready = bool(value.get("ready") and changed and usable and path.is_file())
    failures = (
        [
            gate(
                path,
                "authenticated_changed_runtime_and_mandated_model",
                "missing_file" if not path.is_file() else "rejected_producer",
                upstream="exp8410",
            )
        ]
        if not ready
        else []
    )
    failures += value.get("gate_check_summary", value.get("failures", []))
    refs = value.get("cached_references", []) + (
        [value["inventory_reference"]] if value.get("inventory_reference") else []
    )
    if path.is_file():
        refs.append(reference(path))
    if usable:
        refs.append(reference(model_path))
    return dict(
        ready=ready,
        authority=value,
        gate_check_summary=failures,
        references=refs,
        reader_count=1,
        model_loads=0,
        device_probes=0,
    )


def freeze(root: Path, raw: Path, run_date: str) -> Json:
    """Freeze current clocks, independent operands and historical verdicts before validation."""
    start = datetime.now(UTC).isoformat()
    raw.mkdir(parents=True, exist_ok=True)
    snapshot = raw / "authority"
    snapshot.mkdir()
    for name in [DESIGN, ACTIVE, STAGED]:
        source, target = root / name, snapshot / name
        if source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
    contract = authority(snapshot)
    tasks = [r["task_object"] for r in contract["rows"]]
    values = protocols(tasks)
    refs = seal_protocols(raw, values)
    progress("protocols_frozen_before_evaluation", 4, 0)
    probe = raw / "scratch_probe"
    probe.write_bytes(b"private writable scratch")
    tools = {
        name: (ROOT / ".venv/bin" / name).is_file() for name in ["python", "pytest", "ruff", "mypy"]
    }
    resources = dict(
        private_scratch_writable=probe.read_bytes() == b"private writable scratch",
        tools=tools,
        free_bytes=shutil.disk_usage(raw).free,
        required_free_bytes=128 * 1024 * 1024,
        projected_validation_s=1800,
        validation_reserve_s=900,
        task_cap_s=4800,
        old_learning_union_scheduled=False,
        no_model_needed=True,
    )
    resources["projection_fits"] = (
        resources["free_bytes"] >= resources["required_free_bytes"]
        and resources["projected_validation_s"] + resources["validation_reserve_s"]
        <= resources["task_cap_s"]
    )
    progress("runtime_authority_once")
    runtime = runtime_input(ROOT, raw / "runtime")
    progress("arc_frontier_once")
    frontier = arc.freeze(raw / "arc")
    historical = []
    references = (
        contract["references"]
        + list(refs.values())
        + runtime["references"]
        + frontier["references"]
    )
    for number, name in [(8763, "v751_branch_contract"), (8764, "v751_publication_cost_diagnosis")]:
        path = ROOT / f"results/experiment_{number}_{name}.json"
        ref = dict(path=str(path), sha256=HISTORICAL_PINS[number])
        value = load(ref)
        historical.append(
            dict(
                reference=ref,
                experiment_id=number,
                honest_verdict=value["honest_verdict"],
                verdict_class=value["verdict_class"],
                promoted=False,
            )
        )
        references.append(ref)
    manifest = ROOT / "ops/exclusion_manifest.yaml"
    excluded = yaml.safe_load(manifest.read_bytes())
    retired = {
        str(r.get("experiment_id"))
        for r in excluded.get("retired", []) + excluded.get("retired_experiments", [])
        if isinstance(r, dict)
    }
    references.append(reference(manifest))
    health = ROOT / f"results/raw/{NAME}/startup/full_python_suite.receipt.json"
    health_ref = reference(health) if health.is_file() else None
    if health_ref:
        references.append(health_ref)
    return dict(
        root=str(snapshot),
        raw=str(raw.absolute()),
        invocation_id=str(time.time_ns()),
        run_date=run_date,
        started_at_utc=start,
        contract=contract,
        protocols=values,
        branch_protocol_refs=refs,
        runtime=runtime,
        arc=frontier,
        preserved_primary_rows=historical,
        references=references,
        sources=[
            reference(ROOT / p)
            for p in OWNED
            + TESTS
            + CONSUMERS
            + archive.prior.HISTORICAL
            + archive.prior.TESTS
            + [
                "tests/python/test_v752_archive_8776.py",
                "tests/python/test_v751_branch_contract_8763.py",
                "python/carnot/reporting/v750_archive_bound.py",
                "python/carnot/reporting/v750_archive_runner.py",
                "python/carnot/reporting/v751_branch_contract.py",
                "python/carnot/reporting/v751_branch_evidence.py",
                "python/carnot/reporting/v744_contract_methods.py",
                "python/carnot/reporting/primary_publication.py",
                "python/carnot/reporting/roadmap_contract.py",
                "scripts/experiment_template.py",
                "scripts/adversarial_verify.py",
                "scripts/verdict_row_consistency_lint.py",
                "pyproject.toml",
            ]
        ],
        preconditions=resources,
        exclusion_passed="8776" not in retired,
        memory_protocol_ready=memory_ready(values["memory"]) and digest(tasks) == PIN,
        repository_health_reference=dict(
            reference=health_ref,
            full_suite_pass=bool(health_ref and load(health_ref)["passed"]),
            repository_health_is_dependency=False,
        ),
    )
