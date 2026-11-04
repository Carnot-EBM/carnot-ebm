"""REQ-REPORT-8789: qualify custody without claiming a learning result."""

from copy import deepcopy
from datetime import datetime, UTC
import importlib
import json
from pathlib import Path
import resource
import shutil
import time
from typing import Any

import yaml

from carnot.reporting import v752_branch_methods as old
from carnot.reporting import v752_branch_report as old_report
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.roadmap_contract import compare_contract, parse_design
from carnot.reporting.v746_retained_evidence import load, reference
from carnot.reporting.v748_collection_contract import namespace
from carnot.reporting.v749_contract_methods import digest, encode

Json = dict[str, Any]
ROOT, DESIGN, ACTIVE, STAGED = old.ROOT, old.DESIGN, old.ACTIVE, old.STAGED
NAME, TASK, MILESTONE = (
    "experiment_8789_v753_branch_qualification",
    "exp8789-branch-qualification",
    "2026.11.753",
)
PIN = "472ded8826d2026a662403a8908a9c6fb1a9ef90c3d52b5d3547a65b05b1432b"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/v753_branch_qualification.py",
    "python/carnot/reporting/v753_qualification_runner.py",
    CLI,
]
TESTS = ["tests/python/test_v753_branch_qualification_8789.py"]
CONSUMERS = old.CONSUMERS
MODEL_SPECS: list[Json] = []
ARCHIVE = "openspec/change-proposals/research-roadmap-v752-preserved-20261103.md"
UPSTREAM_PINS = {
    "results/experiment_8776_v752_branch_methods.json": "sha256:08dada784731b1e95717b7bcb42f719dc003a3e414c1a57aa83a45fbc4e0256e",
    "results/experiment_8788_v752_capstone.json": "sha256:63832832ee7263aaa2c46a9cd6a3641c0f5cd6d625544bdfd96fa552623d4f67",
    ARCHIVE: "sha256:9643522778637bc54e622a7a5dc62faf1279944464c6df2a5028d590478d9872",
}


def gate(path: Path, field: str, observed: Any, expected: Any = True, upstream: str = TASK) -> Json:
    """Blocked gates name this producer rather than inheriting an older task alias."""
    return dict(old.gate(path, field, observed, expected, upstream))


def now() -> str:
    """Each invocation owns its clocks, including terminal failure publications."""
    return datetime.now(UTC).isoformat()


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """A flushed boundary keeps bounded CPU children observable to their parent."""
    print(f"[exp8789] phase={phase} completed={completed} pending={pending}", flush=True)


def snapshot(source: Path, target: Path) -> Path:
    """Copy authority bytes before mutation tests can touch their private trees."""
    for name in [DESIGN, ACTIVE, STAGED]:
        path = source / name
        if path.is_file():
            out = target / name
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(path.read_bytes())
    return target


def authority(root: Path) -> Json:
    """Every complete task field binds to an independent digest and visible table."""
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
            text, staged, active, milestone=MILESTONE, first_id=8789, count=12
        )
        for i, row in enumerate(result["rows"]):
            row["checks"]["full_object"] = (
                i < min(len(tasks), len(active["tasks"]), len(staged["tasks"]))
                and tasks[i] == active["tasks"][i] == staged["tasks"][i]
            )
            row.update(
                task_object=deepcopy(tasks[i]) if i < len(tasks) else None,
                matched=all(row["checks"].values()),
                effective_independent_groups=0,
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
            and "<!-- V753_TASK_CONTRACT_END -->" in text
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
    """Change every field independently, then attack counts, encoding and shared hashes."""
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
            (root / ACTIVE).write_text(json.dumps(dict(milestone=MILESTONE, tasks=changed)))
            rows.append(dict(task_id=task["id"], field=field, passed=not authority(root)["ready"]))
        progress("authority_controls", i + 1, 11 - i)
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
            ).replace(PIN, digest(changed))
        elif mode in {"start", "end"}:
            body = body.replace(f"<!-- V753_TASK_CONTRACT_{mode.upper()} -->", "")
        design.write_text(body)
        (root / ACTIVE).write_text(json.dumps(dict(milestone=MILESTONE, tasks=changed)))
        if mode == "staged":
            changed[0]["routing"] = "forged"
            (root / STAGED).write_text(json.dumps(dict(milestone=MILESTONE, tasks=changed)))
        rows.append(
            dict(mutation=mode, passed=authority(root)["ready"] == (mode in {"valid", "encoded"}))
        )
    return rows


def protocols(tasks: list[Json]) -> Json:
    """Freeze the stream estimand before any learner sees new worlds or labels."""
    memory = deepcopy(old.protocols([])["memory"])
    memory.update(
        task_ids=[8790, 8792, 8797],
        independent_streams=30,
        worlds_per_stream=32,
        seeds=[11, 23, 37],
        arms=["read_only", "additions_only", "calibration_only", "combined", "shuffled_feedback"],
        retention_worlds_per_stream=12,
        retention_checkpoints=[0, 16, 32],
        splits=dict(fit=40, tune=20),
        reset_learned_state_between_streams=True,
        structural_hashes_disjoint=True,
        labels_accessible_only_after_prediction_seal=True,
        development_exposure="exposed material only",
        independent_unit="reset_stream",
        planned_online_rows=14400,
        bootstrap_draws=10000,
        simultaneous_interval_percent=99.5,
        comparison_family_size=10,
        primary_contrasts=["combined_minus_read_only", "combined_minus_calibration_only"],
        primary_Brier_upper_bound_lt=-0.01,
        retention_and_error_upper_bound_le=0.02,
        retention_contrasts="three adaptive arms minus read_only at steps16/32",
        error_contrasts=[
            "combined_minus_read_only_false_accept",
            "combined_minus_read_only_false_reject",
        ],
        required_positive_controls=[
            "actual_changed_checks",
            "later_decision_change",
            "oracle_headroom",
        ],
        canonical_tasks_sha256=PIN,
        required_manifests=[
            "40_fit_20_tune_exposed",
            "30x32_new_worlds",
            "30x12_disjoint_retention",
        ],
        task_operands=[
            deepcopy(t) for t in tasks if int(t["id"].split("-")[0][3:]) in {8790, 8792, 8797}
        ],
    )
    for stale in ["gain_lower_bound_gt", "harm_upper_bound_le", "primary_checks"]:
        memory.pop(stale)
    return dict(memory=memory)


def memory_ready(value: Json) -> bool:
    """Protocol custody requires exact frozen operands and certifies no implementation."""
    return value == protocols(value.get("task_operands", []))["memory"]


def preflight(raw: Path) -> Json:
    """Measure bounded parsing and memory before allocating the qualification work."""
    raw.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    probe = raw / "scratch_probe"
    probe.write_bytes(b"private writable scratch")
    imports = True
    try:
        for name in ["coverage", "pytest", "yaml"]:
            importlib.import_module(name)
    except ImportError:
        imports = False
    upstream_checks = []
    for name, expected in UPSTREAM_PINS.items():
        path = ROOT / name
        observed = reference(path)["sha256"] if path.is_file() else "missing_file"
        upstream_checks.append(gate(path, "exact_upstream_bytes", observed, expected))
    original = load(reference(ROOT / "results" / (old.NAME + ".json")))
    original_source_checks = [
        gate(
            Path(ref["path"]),
            "unchanged_original_source",
            reference(Path(ref["path"]))["sha256"],
            ref["sha256"],
        )
        for ref in original["code_config_hashes"]
        if ref["path"] in {str(ROOT / p) for p in old.OWNED + old.TESTS}
    ]
    probe_start = time.monotonic()
    authority(ROOT)
    elapsed = time.monotonic() - probe_start
    setup_s = probe_start - start
    projected_validation_s = setup_s + elapsed * 600 + 600
    tools = {
        name: shutil.which(str(ROOT / ".venv/bin" / name)) is not None
        for name in ["python", "pytest", "ruff", "mypy"]
    }
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    available = (
        int(
            next(
                line.split()[1]
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:")
            )
        )
        * 1024
    )
    return dict(
        private_scratch_writable=probe.read_bytes() == b"private writable scratch",
        tools=tools,
        dependency_imports_passed=imports,
        upstream_checks=upstream_checks,
        original_source_checks=original_source_checks,
        measured_setup_s=setup_s,
        measured_probe_s=elapsed,
        peak_rss_bytes=rss,
        available_memory_bytes=available,
        projected_peak_memory_bytes=max(rss * 4, 1024**3),
        free_bytes=shutil.disk_usage(raw).free,
        projected_validation_s=projected_validation_s,
        validation_reserve_s=900,
        task_cap_s=4800,
        projection_fits=imports
        and all(r["passed"] for r in original_source_checks)
        and all(r["passed"] for r in upstream_checks)
        and all(tools.values())
        and projected_validation_s + 900 < 4800
        and available > max(rss * 4, 1024**3)
        and shutil.disk_usage(raw).free > 128 * 1024**2,
    )


def freeze(root: Path, raw: Path, run_date: str) -> Json:
    """Seal current authority and immutable historical bytes before evaluation starts."""
    started = now()
    preconditions = preflight(raw)
    authority_root = snapshot(root, raw / "authority")
    contract = authority(authority_root)
    values = protocols([r["task_object"] for r in contract["rows"]])
    refs = old.seal_protocols(raw, values)
    progress("memory_protocol_frozen", 1, 0)
    runtime = old.runtime_input(ROOT, raw / "runtime")
    progress("runtime_authority_read_once")
    arc = old.arc.freeze(raw / "arc")
    progress("arc_frontier_read_once")
    historical, references = (
        [],
        contract["references"] + list(refs.values()) + runtime["references"] + arc["references"],
    )
    for number in [8776, 8788]:
        path = next((ROOT / "results").glob(f"experiment_{number}_*.json"))
        ref = reference(path)
        value = load(ref)
        expected_task = old.TASK if number == 8776 else "exp8788-capstone"
        preconditions["upstream_checks"].append(
            gate(path, "exact_task_id", value.get("task_id"), expected_task)
        )
        saved = raw / "retained_inputs" / path.name
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_bytes(path.read_bytes())
        references += [ref, reference(saved)]
        historical.append(
            dict(
                reference=ref,
                experiment_id=number,
                honest_verdict=value["honest_verdict"],
                verdict_class=value["verdict_class"],
                promoted=False,
            )
        )
    references += [reference(ROOT / ARCHIVE), reference(ROOT / "ops/exclusion_manifest.yaml")]
    health = ROOT / f"results/raw/{NAME}/startup/full_python_suite.receipt.json"
    preconditions["projection_fits"] = preconditions["projection_fits"] and all(
        r["passed"] for r in preconditions["upstream_checks"]
    )
    return dict(
        root=str(authority_root),
        raw=str(raw.absolute()),
        invocation_id=raw.name,
        run_date=run_date,
        started_at_utc=started,
        contract=contract,
        protocols=values,
        branch_protocol_refs=refs,
        runtime=runtime,
        arc=arc,
        references=references,
        sources=[
            reference(ROOT / p)
            for p in OWNED
            + TESTS
            + old.OWNED
            + old.TESTS
            + [
                "python/carnot/reporting/primary_publication.py",
                "scripts/adversarial_verify.py",
                "scripts/verdict_row_consistency_lint.py",
                "scripts/experiment_template.py",
            ]
        ],
        preconditions=preconditions,
        exclusion_passed=not any(
            str(r.get("experiment_id")) in {"8789", TASK}
            for r in yaml.safe_load((ROOT / "ops/exclusion_manifest.yaml").read_bytes()).get(
                "retired_experiments", []
            )
            if isinstance(r, dict)
        ),
        memory_protocol_ready=memory_ready(values["memory"]),
        preserved_primary_rows=historical,
        repository_health_reference=dict(
            reference=reference(health) if health.is_file() else None,
            full_suite_pass=load(reference(health))["passed"] if health.is_file() else False,
            repository_health_is_dependency=False,
        ),
    )


reducer = namespace(old_report, "v753_reducer")
base_build = reducer.build


def build(work: Json, validation: Json, output: Path) -> Json:
    """Readiness requires both original measured failure and current owned validation."""
    checked = deepcopy(validation)
    original = work.get("original_failure", {})
    qualified = original.get("passed", False) and validation.get("original_coverage", {}).get(
        "complete", False
    )
    checked["receipts"] = validation["receipts"] + [
        dict(name="original_failure_qualification", passed=qualified)
    ]
    value = dict(base_build(work, checked, output))
    verdict = (
        "circular_positive" if value["verdict_class"] == "positive" else value["verdict_class"]
    )
    if not work["preconditions"]["projection_fits"]:
        verdict = "blocked"
    value.update(
        experiment_id=8789,
        honest_verdict=f"complete_{verdict}_v753_branch_qualification",
        verdict_class=verdict,
        intended_count=12,
        n_samples=0,
        random_seed=7538789,
        original_missing_statement_rows=[
            dict(
                path=old.OWNED[2],
                line=n,
                executed=n in original.get("missing_statement_lines_executed", []),
                mechanism="real original --private-control child under Coverage.py",
            )
            for n in [88, 89, 90, 107]
        ],
        original_node_inventory=validation.get("original_node_inventory", []),
        original_owned_statement_coverage=validation.get("original_coverage", {}),
        original_failure_publication=original,
        qualification_continuation_closed=not qualified,
        changed_coverage_mechanism="original runner child is traced before import; unit and subprocess SQLite are combined without exclusions",
        methodology_note="Administrative and protocol qualification only. Thirty independent reset streams are planned; seeds are nested. Original numerical learning remains closed. Optional runtime and ARC evidence gate only their branch.",
    )
    value["acceptance_gates"]["original_failure_publication_and_coverage"] = qualified
    value["field_principles"] = dict.fromkeys(
        [*value, "field_principles"],
        "Bind exact current identity, immutable operands and measured execution. Protocol readiness certifies no scientific benefit or independent generalization.",
    )
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


reducer.a = __import__(__name__, fromlist=["*"])
reducer.build = build
replay = reducer.replay
