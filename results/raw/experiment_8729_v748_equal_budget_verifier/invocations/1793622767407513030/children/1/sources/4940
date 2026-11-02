"""REQ-REPORT-8696: qualify the original core against sealed producer authority.

Original fixture paths remain canonical. A private checkout preserves their exact
source bytes, checked before and after execution; historical passes are never reused.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone, UTC
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import mkdtemp
import time
from typing import Any

from carnot.reporting import learning_core_8654 as core
from carnot.reporting import v746_contract_authority as authority
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash

Json = dict[str, Any]
c, ROOT = core.c, core.ROOT
NAME, TASK = "experiment_8696_v746_frozen_learning_core", "exp8696-frozen-learning-core"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [f"python/carnot/reporting/learning_core{s}_8696.py" for s in ("", "_runner")] + [CLI]
TESTS = ["tests/python/test_learning_core_8696.py"]
MODEL_SPECS: list[Json] = []
UPSTREAM = ROOT / "results/experiment_8695_v746_retained_contract_authority.json"
UPSTREAM_PIN = "sha256:6aabdc1d57654175f8ec01af407c16855d3c24214665bcbdc2d0016428ea4b7d"
REQUIRED = {
    "owned_tests",
    "ruff_check",
    "ruff_format",
    "mypy",
    "spec",
    "private_e2e018",
    "private_e2e021",
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush measured boundaries and counts for the supervising parent."""
    print(f"[exp8696] phase={phase} completed={completed} pending={pending}", flush=True)


def freeze_sources(sources: list[Json], checkout: Path) -> Json:
    """Copy bytes to a private checkout without linking cleanup to shared results."""
    rows = []
    for index, source in enumerate(sources):
        path = Path(source["path"])
        relative = (
            path.relative_to(ROOT) if path.is_relative_to(ROOT) else Path("external") / str(index)
        )
        target = checkout / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        target.chmod(0o444)
        rows.append(dict(original=source, frozen=c.ref(target)))
    atomic_json(checkout / "source-seal.json", dict(sources=rows))
    (checkout / "source-seal.json").chmod(0o444)
    return dict(c.ref(checkout / "source-seal.json"))


def check_bundle(reference: Json) -> bool:
    """Changed original or copied bytes invalidate the whole affected invocation."""
    try:
        return all(
            c.ref(Path(row["original"]["path"])) == row["original"]
            and c.ref(Path(row["frozen"]["path"])) == row["frozen"]
            and row["original"]["sha256"] == row["frozen"]["sha256"]
            for row in c.read(reference)["sources"]
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def fresh_fixture(reference: Json, destination: Path) -> Json:
    """Keep scientific bytes while giving a mechanics replay its own real clock epoch."""
    value = c.read(reference)
    destination.mkdir(parents=True, exist_ok=True)
    for name in (
        "stream_rows.json",
        "released_feedback.json",
        "retention_rows.json",
        "coefficient_transitions.json",
    ):
        target = destination / name
        shutil.copyfile(Path(value["raw"]) / name, target)
        target.chmod(0o444)
    now = time.monotonic_ns()
    value.update(raw=str(destination), started_monotonic_ns=now, ended_monotonic_ns=now)
    path = destination / "measurement.json"
    atomic_json(path, value)
    path.chmod(0o444)
    return dict(c.ref(path))


def prepare(raw: Path, run_date: str, upstream: Path = UPSTREAM) -> Json:
    """Authenticate the branch directly; optional hardware never gates CPU controls."""
    progress("preconditions_no_model_load_MODEL_SPECS=[]_current_llm_calls=0")
    started = time.monotonic_ns()
    started_at = datetime.now(UTC).isoformat()
    raw.mkdir(parents=True, exist_ok=True)
    scratch = Path(mkdtemp(prefix="core8696-", dir="/var/tmp"))
    probe = scratch / "writable"
    with probe.open("wb") as stream:
        stream.write(b"disk-backed")
        stream.flush()
        os.fsync(stream.fileno())
    fs = c.supervise(
        "scratch_filesystem",
        ["stat", "-f", "-c", "%T", str(scratch)],
        raw / "preconditions",
        deadline=10,
    )
    filesystem = Path(fs["stdout_path"]).read_text().strip()
    preconditions = dict(
        available_memory_bytes=int(
            Path("/proc/meminfo").read_text().split("MemAvailable:")[1].split()[0]
        )
        * 1024,
        free_disk_bytes=shutil.disk_usage(scratch).free,
        scratch_path=str(scratch),
        filesystem=filesystem,
        private_disk_backed_scratch=fs["passed"]
        and filesystem not in {"tmpfs", "ramfs"}
        and probe.read_bytes() == b"disk-backed",
        tools={
            n: c.ref(ROOT / ".venv/bin" / n) if (ROOT / ".venv/bin" / n).is_file() else None
            for n in ("python", "pytest", "coverage", "ruff", "mypy")
        },
        optional_hardware_required=False,
        current_llm_calls=0,
        measurement_cap_s=3600,
        validation_reserve_s=900,
        task_cap_s=4800,
    )
    p = c.read(
        dict(path=str(ROOT / authority.original.PROTOCOL), sha256=authority.original.PROTOCOL_PIN)
    )
    inherited = c.read({k: p["inherited_v743_protocol_reference"][k] for k in ("path", "sha256")})
    for key in ("objective_updates", "objective_audit"):
        if (
            p[key] != inherited[key]
            or canonical_hash(p[key]) != p["inherited_operand_checks"][key]["sha256"]
        ):
            raise ValueError("original_registered_method:" + key)
    partition = c.read(p["cohort_partition_reference"])
    groups = core.groups(partition, partition["original_node_ids"])
    if [len(r["node_ids"]) for r in partition["cohorts"]] != [39, 25] or len(
        partition["original_runner_node_ids"]
    ) != 27:
        raise ValueError("original_partition")
    contract = authority.authority(ROOT)
    value = json.loads(upstream.read_bytes()) if upstream.is_file() else {}
    failures = list(contract["gate_check_summary"])
    if value.get("learning_authority_ready_score") != 1:
        failures.append(
            authority.gate(
                upstream,
                "learning_authority_ready_score",
                1,
                value.get("learning_authority_ready_score"),
                authority.TASK,
            )
        )
    terminal = authority.terminal(
        dict(path=str(upstream), sha256=UPSTREAM_PIN), "learning_authority_ready_score", 8695
    )
    failures += terminal["gate_check_summary"]
    terminals = [
        authority.terminal(p["current_producers"][str(n)], field, n)
        for n, field in [
            (8431, "head_ready_score"),
            (8580, "current_validator_ready_score"),
            (8581, "online_adapter_ready_score"),
        ]
    ]
    failures += [row for t in terminals for row in t["gate_check_summary"]]
    if (
        not preconditions["private_disk_backed_scratch"]
        or preconditions["available_memory_bytes"] < 536870912
        or preconditions["free_disk_bytes"] < 1073741824
        or not all(preconditions["tools"].values())
    ):
        failures.append(authority.gate(raw, "cpu_resources", True, False, TASK))
    historical = c.read(partition["historical_primitive_reference"])
    original_fixture = historical["mechanics_fixture"]
    fixture = fresh_fixture(original_fixture, scratch / "fixture")
    refs = [
        c.ref(ROOT / authority.original.PROTOCOL),
        p["inherited_v743_protocol_reference"],
        p["cohort_partition_reference"],
        p["head_checkpoint_reference"],
        p["learning_data_reference"],
        partition["historical_primitive_reference"],
        original_fixture,
        fixture,
    ]
    refs += p["learning_label_references"] + p["operand_references"]["learning"]
    refs += [
        p["v743_producers"]["8654"],
        dict(
            path=str(
                ROOT / "openspec/change-proposals/research-roadmap-v745-preserved-20261101.md"
            ),
            sha256="sha256:01fb99526e816d6ff017196bcd17aad215562ef217bcdf113746fcbfc8e38a63",
        ),
    ]
    refs += [
        c.ref(Path(fixture["path"]).parent / name)
        for name in (
            "stream_rows.json",
            "released_feedback.json",
            "retention_rows.json",
            "coefficient_transitions.json",
        )
    ]
    refs += contract["references"] + [r for t in terminals for r in t["references"]]
    refs += [
        r
        for r in terminal["references"]
        if Path(r["path"]).is_file() and c.ref(Path(r["path"]))["sha256"] == r["sha256"]
    ]
    refs += core.operand_references([c.read(fixture)["cells"]])
    prior_attempts = [
        c.ref(p)
        for p in (ROOT / "results/raw" / NAME / "invocations").glob("*/failure_summary.json")
    ]
    refs += prior_attempts
    for reference in refs + partition["source_references"]:
        if c.ref(Path(reference["path"]))["sha256"] != reference["sha256"]:
            raise ValueError("original_operand_identity")
    paths = [
        ROOT / s
        for s in OWNED
        + TESTS
        + [
            core.old_report.RUNNER,
            "tests/python/conftest.py",
            "pyproject.toml",
            "ops/exclusion_manifest.yaml",
        ]
    ]
    sources = core.source_closure(paths + [Path(r["path"]) for r in partition["source_references"]])
    bundle = freeze_sources(sources, scratch / "checkout")
    return dict(
        raw=str(raw),
        run_date=run_date,
        invocation_id=raw.name,
        started_at_utc=started_at,
        started_monotonic_ns=started,
        preconditions=preconditions,
        input_failures=failures,
        upstream=value,
        partition=partition,
        grouping_rows=groups,
        input_references=refs,
        sources=sources,
        frozen_source_bundle=bundle,
        mechanics_fixture=fixture,
        original_mechanics_fixture=original_fixture,
        prior_failed_invocation_references=prior_attempts,
        protocol=p,
        contract=contract,
        receipts=[],
        children=[],
        launch_budget_rows=[],
        collected=[],
        outcomes=[],
        phase_spans=[],
        duration_s=0.0,
        source_reuse_refusals=[],
    )


def collect(work: Json, targets: list[str]) -> list[str]:
    """Collect exact nodes once, retaining command, exit and unmodified inventory."""
    receipt = c.supervise(
        "collection",
        [sys.executable, "-m", "pytest", *core.OPTIONS, "--collect-only", *targets],
        Path(work["raw"]) / "collection",
        deadline=90,
    )
    work["receipts"].append(receipt)
    nodes = [
        n
        for n in Path(receipt["stdout_path"]).read_text().splitlines()
        if n.startswith("tests/") and "::" in n
    ]
    atomic_json(Path(work["raw"]) / "collection.json", dict(nodes=nodes, receipt=receipt))
    return nodes


def instrument(
    work: Json, name: str, nodes: list[str], sources: list[Json], budget: float, kind: str
) -> Json:
    """Check the private seal on both sides of an actual retained coverage child."""
    if not check_bundle(work["frozen_source_bundle"]):
        raise ValueError("source_drift_before_child")
    receipt = dict(core.instrument(work, name, nodes, sources, budget, kind))
    if not check_bundle(work["frozen_source_bundle"]):
        work["source_reuse_refusals"].append(
            authority.gate(Path(work["raw"]), "source_drift_after_child", True, False, TASK)
        )
    return receipt


def execute(work: Json) -> None:
    """Run only admitted original module groups; a killed child cannot qualify."""
    work["collected"] = collect(work, work["partition"]["original_node_ids"])
    plan = core.groups(work["partition"], work["collected"])
    for index, group in enumerate(plan):
        progress("original_core_module", index, len(plan) - index)
        row = core.admission(group, work["started_monotonic_ns"], time.monotonic_ns())
        work["launch_budget_rows"].append(row)
        if not row["admitted"]:
            break
        paths = [ROOT / group["module"], *map(Path, group["coverage_sources"])]
        if "objective_checkpoint" in group["module"]:
            paths += [
                ROOT / "python/carnot/verify" / n
                for n in ("objective_checkpoint_8461.py", "objective_derivatives_8461.py")
            ]
        instrument(
            work,
            f"core{index}",
            group["node_ids"],
            [c.ref(p) for p in paths],
            group["child_budget_s"],
            "A",
        )
        progress("original_core_module_finished", index + 1, len(plan) - index - 1)


def recompute(work: Json, scratch: Path) -> None:
    """Read actual retained evidence; consumer data cannot qualify producer statements."""
    if not check_bundle(work["frozen_source_bundle"]):
        raise ValueError("frozen_source_identity")
    for reference in work["input_references"]:
        if c.ref(Path(reference["path"]))["sha256"] != reference["sha256"]:
            raise ValueError("original_operand_identity")
    work["outcomes"] = []
    for manifest in work["children"]:
        receipt = manifest["receipt"]
        for key in ("coverage_config_reference", "junit_reference"):
            if c.ref(Path(receipt[key]["path"])) != receipt[key]:
                raise ValueError("child_config_or_junit_identity")
        outcomes = core.terminal_nodes(Path(receipt["junit_reference"]["path"]))
        if sorted(r["node_id"] for r in outcomes) != sorted(manifest["node_ids"]):
            raise ValueError("child_node_assignment")
        if manifest["kind"] == "A":
            work["outcomes"] += outcomes
    for kind, key in [("A", "original_coverage"), ("owned", "owned_coverage")]:
        children = [m for m in work["children"] if m["kind"] == kind]
        work[key] = dict(complete=False, files={})
        if children:
            sources = list({s["path"]: s for m in children for s in m["sources"]}.values())
            lines = core.attest(children, sources)
            selected = [
                s
                for s in sources
                if s["path"]
                in (
                    [str(ROOT / p) for p in OWNED]
                    if kind == "owned"
                    else [str(ROOT / core.old_report.RUNNER)]
                )
            ]
            work[key] = core.old_report.statement_report(lines, selected)
            if not c.combine_copies(children, scratch / kind)["second_combine_invariance"][
                "passed"
            ]:
                raise ValueError("combine_changed_originals")


def build(work: Json, output: Path) -> Json:
    """Reduce fresh core evidence without granting training or scientific benefit."""
    partition, outcomes = work["partition"], work["outcomes"]
    intended = partition["cohorts"][0]["node_ids"]
    seen = [r["node_id"] for r in outcomes]
    original = work.get("original_coverage", {}).get("files", {})
    runner = original.get(str(ROOT / core.old_report.RUNNER), {})
    subset = set(partition["original_runner_node_ids"]) & set(intended)
    checks = dict(
        inputs=not work["input_failures"],
        exact_original_core=sorted(seen) == sorted(intended)
        and all(r["outcome"] == "passed" for r in outcomes),
        original_runner_subset=subset <= set(seen) and bool(runner.get("executed")),
        source_identity=not work["source_reuse_refusals"],
        retained_producer=bool([m for m in work["children"] if m["kind"] == "A"])
        and not work.get("child_errors"),
        owned_statements=work.get("owned_coverage", {}).get("complete", False),
        launch_budgets=bool(work["launch_budget_rows"])
        and all(r["admitted"] for r in work["launch_budget_rows"]),
        validation=REQUIRED <= {r["name"] for r in work["receipts"] if r["passed"]}
        and all(r["passed"] for r in work["receipts"]),
    )
    ready = all(checks.values())
    owned_failed = (
        any(not r["passed"] for r in work["receipts"])
        or bool(work.get("child_errors"))
        or bool(work["source_reuse_refusals"])
    )
    verdict = (
        "disqualified"
        if owned_failed
        else "blocked"
        if work["input_failures"]
        else "circular_positive"
        if ready
        else "disqualified"
    )
    rows = [
        dict(
            node_id=n,
            unit_id=n,
            status="completed" if n in seen else "censored",
            outcome=next((r["outcome"] for r in outcomes if r["node_id"] == n), "unstarted"),
            censored=n not in seen,
            excluded=False,
            effective_independent_groups=0,
        )
        for n in intended
    ]
    failures = deepcopy(work["input_failures"] + work["source_reuse_refusals"])
    if not work["input_failures"] or owned_failed:
        failures += [
            authority.gate(Path(work["raw"]) / "work.json", k, True, v, TASK)
            for k, v in checks.items()
            if not v
        ]
    value = dict(
        experiment_id=8696,
        task_id=TASK,
        milestone="2026.11.746",
        run_date=work["run_date"],
        invocation_id=work["invocation_id"],
        honest_verdict="complete_" + verdict + "_v746_frozen_learning_core",
        verdict_class=verdict,
        gate_check_summary=failures,
        inference_substrate="deterministic_verifier",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(current_llm_calls=0, model_loads=0, generations=0),
        target_model=dict(
            kind="authenticated_numerical_checkpoint",
            reference=work["protocol"]["head_checkpoint_reference"],
            inference_scope="original numerical assertion controls",
        ),
        historical_model_provenance=work["upstream"].get("historical_model_provenance", []),
        rows=rows,
        original_core_node_rows=rows,
        intended_count=len(intended),
        completed_count=len(outcomes),
        failed_count=sum(r["outcome"] != "passed" for r in outcomes),
        censored_count=sum(n not in seen for n in intended),
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            unit="original assertion node", intended=39, independent_questions=0
        ),
        verifier_is_oracle=True,
        exposure_scope="exposed original deterministic controls; cohort A only",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=ready,
        flagged_adversarial=bool(work.get("adversarial_findings")),
        acceptance_gates={
            k: dict(
                passed=v,
                principle="Fresh exact producer execution differs from scientific benefit and complete learning qualification.",
            )
            for k, v in checks.items()
        },
        validation_receipts=work["receipts"],
        terminal_validation_sidecar_path=str(Path(work["raw"]) / "terminal.json"),
        adversarial_findings=work.get("adversarial_findings", []),
        preconditions_checked=work["preconditions"],
        duration_s=work["duration_s"],
        started_at_utc=work["started_at_utc"],
        ended_at_utc=work.get("ended_at_utc"),
        phase_spans=work["phase_spans"],
        random_seed=7468696,
        source_artifact_hashes=work["input_references"],
        code_config_hashes=work["sources"],
        raw_shard_hashes=[r["reference"] for m in work["children"] for r in m["files"]],
        cited_upstream_artifacts=work["input_references"],
        branch_gate_values=dict(learning_core_ready_score=int(ready)),
        learning_core_ready_score=int(ready),
        frozen_source_bundle=work["frozen_source_bundle"],
        mechanics_fixture_reference=work["mechanics_fixture"],
        original_mechanics_fixture_reference=work["original_mechanics_fixture"],
        prior_failed_invocation_references=work["prior_failed_invocation_references"],
        remainder_node_ids=partition["cohorts"][1]["node_ids"],
        cohort_partition_reference=work["protocol"]["cohort_partition_reference"],
        original_runner_node_ids=partition["original_runner_node_ids"],
        original_statement_coverage=work.get("original_coverage", {}),
        owned_statement_coverage=work.get("owned_coverage", {}),
        child_shard_manifest=work["children"],
        launch_budget_rows=work["launch_budget_rows"],
        historical_quarantines=work["protocol"]["historical_quarantines"],
        preserved_historical_limits=[
            "Exp8654 source-drift rejection",
            "Exp8681 disqualified",
            "Exp8691 historical flag",
            "V733 source-information null",
            "V735 cold-cost null",
            "V737 supervisor null",
            "V743 batch-one null",
            "Exp8692 constructed CPU32/24 only",
        ],
        training_eligible=False,
        complete_learning_qualification=False,
        physical_fpga_execution=False,
        scope="core_only",
        contract_reference=work["contract"],
        primitive_reference=c.ref(Path(work["raw"]) / "work.json"),
        invocation_seal_reference=work["invocation_seal_reference"],
        publication_output=str(output),
        methodology_note="Original assertions and canonical fixture paths execute once per module against a sealed private transitive checkout. Immutable SQLite and JUnit are retained before combining copies. Exp8697 must authenticate this same identity and execute the exact remainder; core readiness cannot authorize training.",
    )
    value["field_principles"] = {
        k: "Bind actual invocation evidence to immutable original operands; execution readiness makes no generalization or hardware claim."
        for k in value
    }
    value["field_principles"].update(
        field_principles="Explain each emitted field without changing its evidence scope.",
        reproducibility_checksum="Hash canonical report fields before adding this checksum.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value
