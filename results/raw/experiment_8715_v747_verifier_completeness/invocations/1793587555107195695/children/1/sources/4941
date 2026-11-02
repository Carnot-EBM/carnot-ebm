"""REQ-REPORT-8697, REQ-VERIFY-8697: qualify original evidence without model loads."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import shutil
import signal
import sys
from tempfile import TemporaryDirectory, mkdtemp
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import learning_core_8696 as base
from carnot.reporting import v746_contract_authority as authority
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.primary_publication import publish_primary
from scripts.adversarial_verify import verify_artifact

Json = dict[str, Any]
core, c, ROOT = base.core, base.c, base.ROOT
NAME, TASK = (
    "experiment_8697_v746_learning_union_qualification",
    "exp8697-learning-union-qualification",
)
CLI = f"scripts/experiments/{NAME}.py"
OWNED = ["python/carnot/reporting/learning_union_8697.py", CLI]
TESTS = ["tests/python/test_learning_union_8697.py"]
UPSTREAM = ROOT / "results/experiment_8696_v746_frozen_learning_core.json"
UPSTREAM_PIN = "sha256:fb110af2636a044d398028c1a63fe003c9b95e02e4962580e6d42228e9ccb515"
MODEL_SPECS: list[Json] = []
REQUIRED = {"owned_tests", "ruff_check", "ruff_format", "mypy", "spec", "private_e2e020"}
E2E020 = "tests/python/test_hard_exit_learning_qualification_8206.py"


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush boundaries so the parent can distinguish slow work from a stalled child."""
    print(f"[exp8697] phase={phase} completed={completed} pending={pending}", flush=True)


def prepare(raw: Path, run_date: str, upstream: Path = UPSTREAM) -> Json:
    """Authenticate original branch authority before admitting any CPU measurement."""
    progress("preconditions_no_model_load_MODEL_SPECS=[]_llm_calls=0")
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True)
    scratch = Path(mkdtemp(prefix="union8697-", dir="/var/tmp"))
    probe = scratch / "writable"
    with probe.open("wb") as stream:
        stream.write(b"disk-backed")
        stream.flush()
        os.fsync(stream.fileno())
    fs = c.supervise(
        "scratch_filesystem", ["stat", "-f", "-c", "%T", str(scratch)], raw / "preconditions"
    )
    filesystem = Path(fs["stdout_path"]).read_text().strip()
    preconditions = dict(
        available_memory_bytes=int(
            Path("/proc/meminfo").read_text().split("MemAvailable:")[1].split()[0]
        )
        * 1024,
        free_disk_bytes=shutil.disk_usage(scratch).free,
        private_disk_backed_scratch=fs["passed"]
        and filesystem not in {"tmpfs", "ramfs"}
        and probe.read_bytes() == b"disk-backed",
        scratch_path=str(scratch),
        filesystem=filesystem,
        tools={
            n: c.ref(ROOT / ".venv/bin" / n) if (ROOT / ".venv/bin" / n).is_file() else None
            for n in ("python", "pytest", "coverage", "ruff", "mypy")
        },
        optional_hardware_required=False,
        measurement_cap_s=3600,
        validation_reserve_s=900,
        task_cap_s=4800,
    )
    contract, methods = authority.authority(ROOT), authority.methods()
    failures = contract["gate_check_summary"] + methods["gate_check_summary"]
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
    core.groups(partition, partition["original_node_ids"])
    terminal = authority.terminal(
        dict(path=str(upstream), sha256=UPSTREAM_PIN), "learning_core_ready_score", 8696
    )
    failures += terminal["gate_check_summary"]
    value = json.loads(upstream.read_bytes()) if upstream.is_file() else {}
    for key, expected in [("learning_core_ready_score", 1), ("flagged_adversarial", False)]:
        if key not in value or value[key] != expected:
            failures.append(
                authority.gate(
                    upstream, key, expected, value.get(key), "exp8696-frozen-learning-core"
                )
            )
    old: Json = {}
    if terminal["passed"]:
        old = c.read(value["primitive_reference"])
        if c.read(old["invocation_seal_reference"]) != {
            k: v for k, v in old.items() if k != "invocation_seal_reference"
        } or not base.check_bundle(old["frozen_source_bundle"]):
            failures.append(
                authority.gate(
                    upstream,
                    "original_shared_source_seal",
                    True,
                    False,
                    "exp8696-frozen-learning-core",
                )
            )
            old = {}
    if (
        not preconditions["private_disk_backed_scratch"]
        or preconditions["available_memory_bytes"] < 536870912
        or preconditions["free_disk_bytes"] < 1073741824
        or not all(preconditions["tools"].values())
    ):
        failures.append(authority.gate(raw, "cpu_resources", True, False, TASK))
    refs = (
        methods["references"]
        + contract["references"]
        + [c.ref(ROOT / authority.original.PROTOCOL), p["cohort_partition_reference"]]
    )
    refs += [
        r
        for r in terminal["references"]
        if Path(r["path"]).is_file() and c.ref(Path(r["path"])) == r
    ]
    if old:
        refs += old["input_references"] + [
            value["primitive_reference"],
            old["invocation_seal_reference"],
        ]
        for child in old["children"]:
            refs += [
                child["seal_reference"],
                child["receipt"]["junit_reference"],
                child["receipt"]["coverage_config_reference"],
            ]
            refs += [r["reference"] for r in child["files"]]
    authenticated = []
    for reference in refs:
        path = Path(reference["path"])
        observed = c.ref(path)["sha256"] if path.is_file() else None
        if observed != reference["sha256"]:
            row = authority.gate(
                path, "original_operand_sha256", reference["sha256"], observed, TASK
            )
            row["operand_state"] = "changed_bytes" if path.is_file() else "missing_file"
            failures.append(row)
        else:
            authenticated.append(reference)
    sources = core.source_closure(
        [
            ROOT / path
            for path in OWNED
            + TESTS
            + [E2E020, "pyproject.toml", "tests/python/conftest.py", "ops/exclusion_manifest.yaml"]
        ]
        + [Path(r["path"]) for r in partition["source_references"]]
    )
    return dict(
        raw=str(raw),
        run_date=run_date,
        invocation_id=raw.name,
        started_monotonic_ns=start,
        started_at_utc=datetime.now(UTC).isoformat(),
        preconditions=preconditions,
        input_failures=failures,
        protocol=p,
        partition=partition,
        methods=methods,
        contract=contract,
        input_references=authenticated,
        sources=sources,
        frozen_source_bundle=base.freeze_sources(sources, scratch / "checkout"),
        upstream_reference=dict(path=str(upstream), sha256=UPSTREAM_PIN),
        shared_source_bundle=old.get("frozen_source_bundle"),
        mechanics_fixture=old.get("mechanics_fixture"),
        core_children=[m for m in old.get("children", []) if m["kind"] == "A"],
        children=[],
        receipts=[],
        launch_budget_rows=[],
        source_reuse_refusals=[],
        outcomes=[],
        statement_rows=[],
        phase_spans=[],
        duration_s=0.0,
    )


def shared_identity(work: Json) -> bool:
    """One original source identity is required for both halves of the union."""
    return base.check_bundle(work["frozen_source_bundle"]) and (
        not work["shared_source_bundle"] or base.check_bundle(work["shared_source_bundle"])
    )


def instrument(
    work: Json, name: str, nodes: list[str], sources: list[Json], budget: float, kind: str
) -> Json:
    """Retain actual children with the same original fixture and both source seals."""
    if not shared_identity(work):
        raise ValueError("source_drift_before_child")
    with patch.dict(
        os.environ,
        CARNOT_8603_CASE_ROOT=str(Path(work["raw"]) / "deadline_cases"),
        CARNOT_8616_CASE_ROOT=str(Path(work["raw"]) / "deadline_cases"),
    ):
        receipt = dict(core.instrument(work, name, nodes, sources, budget, kind))
    if not shared_identity(work):
        work["source_reuse_refusals"].append(
            authority.gate(Path(work["raw"]), "source_drift_after_child", True, False, TASK)
        )
    return receipt


def execute(work: Json) -> None:
    """Collect once and keep the registered module fixtures and child budgets intact."""
    receipt = c.supervise(
        "collection",
        [
            sys.executable,
            "-m",
            "pytest",
            *core.OPTIONS,
            "--collect-only",
            *work["partition"]["original_node_ids"],
            E2E020,
        ],
        Path(work["raw"]) / "collection",
        deadline=90,
    )
    work["receipts"].append(receipt)
    collected = [
        n
        for n in Path(receipt["stdout_path"]).read_text().splitlines()
        if n.startswith("tests/") and "::" in n
    ]
    originals = [n for n in collected if not n.startswith(E2E020)]
    core.groups(work["partition"], originals)
    atomic_json(Path(work["raw"]) / "collection.json", dict(nodes=collected, receipt=receipt))
    groups = deepcopy(work["partition"]["cohorts"][1]["groups"])
    groups.append(
        dict(
            module=E2E020,
            node_ids=[n for n in collected if n.startswith(E2E020)],
            child_budget_s=600,
            consumer_only=True,
        )
    )
    for index, group in enumerate(groups):
        progress("original_remainder", index, len(groups) - index)
        admission = core.admission(group, work["started_monotonic_ns"], time.monotonic_ns())
        work["launch_budget_rows"].append(admission)
        if not admission["admitted"]:
            break
        kind = "e2e" if group["module"] == E2E020 else "consumer" if group["consumer_only"] else "B"
        sources = [c.ref(ROOT / group["module"])]
        if kind == "B":
            sources += [c.ref(ROOT / core.old_report.RUNNER)]
        name = "private_e2e020" if kind == "e2e" else f"remainder{index}"
        instrument(work, name, group["node_ids"], sources, group["child_budget_s"], kind)
        progress("original_remainder_finished", index + 1, len(groups) - index - 1)


def recompute(work: Json, scratch: Path) -> None:
    """Re-read producer SQLite and JUnit; a count never establishes a covered statement."""
    if not shared_identity(work):
        raise ValueError("shared_source_identity")
    for reference in work["input_references"]:
        if c.ref(Path(reference["path"]))["sha256"] != reference["sha256"]:
            raise ValueError("original_operand_identity")
    manifests = work["core_children"] + work["children"]
    work["outcomes"] = []
    producer = [m for m in manifests if m["kind"] in {"A", "B"}]
    for manifest in manifests:
        receipt = manifest["receipt"]
        for key in ("coverage_config_reference", "junit_reference"):
            if c.ref(Path(receipt[key]["path"])) != receipt[key]:
                raise ValueError("child_config_or_junit_identity")
        outcomes = core.terminal_nodes(Path(receipt["junit_reference"]["path"]))
        if sorted(r["node_id"] for r in outcomes) != sorted(manifest["node_ids"]):
            raise ValueError("child_node_assignment")
        core.attest([manifest], manifest["sources"])
        if manifest["kind"] in {"A", "B", "consumer"}:
            work["outcomes"] += outcomes
    nodes = [r["node_id"] for r in work["outcomes"]]
    if len(nodes) != len(set(nodes)):
        raise ValueError("duplicate_original_nodes")
    work["statement_rows"] = []
    work["owned_coverage"] = dict(complete=False, files={})
    for kind, selected in [
        ("producer", producer),
        ("owned", [m for m in manifests if m["kind"] == "owned"]),
    ]:
        if selected:
            sources = list({s["path"]: s for m in selected for s in m["sources"]}.values())
            lines = core.attest(selected, sources)
            targets = [
                c.ref(ROOT / path)
                for path in (OWNED if kind == "owned" else [core.old_report.RUNNER])
            ]
            report = core.old_report.statement_report(lines, targets)
            if not c.combine_copies(selected, scratch / kind)["second_combine_invariance"][
                "passed"
            ]:
                raise ValueError("combine_changed_originals")
            if kind == "owned":
                work["owned_coverage"] = report
            else:
                runner = report["files"][str(ROOT / core.old_report.RUNNER)]
                work["statement_rows"] = [
                    dict(
                        source_reference=targets[0], line=line, executed=line in runner["executed"]
                    )
                    for line in runner["statements"]
                ]


def build(work: Json, output: Path) -> Json:
    """Publish execution readiness separately from learning benefit and device claims."""
    intended, outcomes = work["partition"]["original_node_ids"], work["outcomes"]
    seen = [r["node_id"] for r in outcomes]
    exact = (
        len(intended) == 64
        and len(seen) == len(set(seen))
        and sorted(seen) == sorted(intended)
        and all(r["outcome"] == "passed" for r in outcomes)
    )
    checks = dict(
        original_authority=not work["input_failures"],
        exact_union=exact,
        original_runner_nodes=len(work["partition"]["original_runner_node_ids"]) == 27
        and set(work["partition"]["original_runner_node_ids"]) <= set(seen),
        original_runner_statements=len(work["statement_rows"]) == 83
        and all(r["executed"] for r in work["statement_rows"]),
        source_identity=not work["source_reuse_refusals"],
        owned_statements=work.get("owned_coverage", {}).get("complete", False),
        children=bool(work["core_children"]) and not work.get("child_errors"),
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
    failures = deepcopy(work["input_failures"] + work["source_reuse_refusals"])
    if not work["input_failures"] or owned_failed:
        failures += [
            authority.gate(Path(work["raw"]) / "work.json", k, True, v, TASK)
            for k, v in checks.items()
            if not v
        ]
    rows = [
        dict(
            unit_id=n,
            node_id=n,
            status="completed" if n in seen else "censored",
            outcome=next((r["outcome"] for r in outcomes if r["node_id"] == n), "unstarted"),
            censored=n not in seen,
            excluded=False,
            effective_independent_groups=0,
        )
        for n in intended
    ]
    value = dict(
        experiment_id=8697,
        task_id=TASK,
        milestone="2026.11.746",
        run_date=work["run_date"],
        invocation_id=work["invocation_id"],
        honest_verdict="complete_" + verdict + "_v746_learning_union_qualification",
        verdict_class=verdict,
        gate_check_summary=failures,
        inference_substrate="deterministic_verifier",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(current_llm_calls=0, model_loads=0, generations=0),
        target_model=dict(
            kind="authenticated_numerical_checkpoint",
            reference=work["protocol"]["head_checkpoint_reference"],
        ),
        historical_model_provenance=work["protocol"].get("historical_model_provenance", []),
        rows=rows,
        original_node_rows=rows,
        statement_rows=work["statement_rows"],
        intended_count=64,
        completed_count=len(outcomes),
        failed_count=sum(r["outcome"] != "passed" for r in outcomes),
        censored_count=sum(n not in seen for n in intended),
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            unit="original assertion node", intended=64, independent_questions=0
        ),
        verifier_is_oracle=True,
        exposure_scope="exposed original assertion controls; exact39+25 union",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=ready,
        flagged_adversarial=bool(work.get("adversarial_findings")),
        acceptance_gates={
            k: dict(
                passed=v,
                principle="Original producer execution qualifies readiness, not scientific benefit.",
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
        random_seed=7468697,
        source_artifact_hashes=work["input_references"],
        code_config_hashes=work["sources"],
        raw_shard_hashes=[
            r["reference"] for m in work["core_children"] + work["children"] for r in m["files"]
        ],
        cited_upstream_artifacts=work["input_references"],
        branch_gate_values=dict(learning_execution_ready_score=int(ready)),
        learning_execution_ready_score=int(ready),
        exact_union_receipt=dict(
            passed=exact,
            expected=64,
            observed=len(seen),
            cohort_sizes=[39, 25],
            original_runner_count=27,
        ),
        learning_operand_references={
            k: deepcopy(work["protocol"][k])
            for k in [
                "head_checkpoint_reference",
                "learning_data_reference",
                "learning_label_references",
                "learning_registration",
                "objective_updates",
                "objective_audit",
                "cohort_partition_reference",
            ]
        },
        frozen_source_bundle=work["frozen_source_bundle"],
        shared_source_bundle=work["shared_source_bundle"],
        upstream_reference=work["upstream_reference"],
        child_shard_manifest=work["core_children"] + work["children"],
        owned_statement_coverage=work.get("owned_coverage", {}),
        launch_budget_rows=work["launch_budget_rows"],
        contract_reference=work["contract"],
        historical_quarantines=work["protocol"]["historical_quarantines"],
        preserved_historical_limits=[
            "Exp8681 disqualified",
            "Exp8691 historical flag",
            "V733 source-information null",
            "V735 cold-cost null",
            "V737 supervisor null",
            "V743 batch-one null",
            "Exp8692 constructed CPU32/24 only",
            "Exp8693/8694 external evidence blocked",
        ],
        physical_fpga_execution=False,
        training_eligible=ready,
        complete_learning_qualification=ready,
        primitive_reference=c.ref(Path(work["raw"]) / "work.json"),
        invocation_seal_reference=work["invocation_seal_reference"],
        publication_output=str(output),
        methodology_note="Authenticate the original frozen core and independently registered methods. Execute the disjoint remainder unchanged against identical sources and private inputs. Join retained SQLite and JUnit, with consumer evidence separate. Execution readiness does not establish learning benefit or hardware acceleration.",
    )
    value["field_principles"] = {
        k: "Bind this invocation to exact original evidence; readiness is separate from scientific benefit."
        for k in value
    }
    value["field_principles"].update(
        field_principles="Explain every field.",
        reproducibility_checksum="Hash canonical report fields before adding the checksum.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def freeze(work: Json) -> None:
    """Keep the primitive seal separate so changing a report checksum cannot add readiness."""
    seal = Path(work["raw"]) / "invocation_seal.json"
    atomic_json(seal, {k: v for k, v in work.items() if k != "invocation_seal_reference"})
    seal.chmod(0o444)
    work["invocation_seal_reference"] = c.ref(seal)
    atomic_json(Path(work["raw"]) / "work.json", work)


def replay(path: Path) -> bool:
    """Reconstruct final bytes from sealed operands rather than trusting report assertions."""
    try:
        value = json.loads(path.read_bytes())
        work = c.read(value["primitive_reference"])
        if c.read(work["invocation_seal_reference"]) != {
            k: v for k, v in work.items() if k != "invocation_seal_reference"
        }:
            return False
        for receipt in work["receipts"]:
            for stream in ("stdout", "stderr"):
                if c.ref(Path(receipt[stream + "_path"]))["sha256"] != receipt[stream + "_sha256"]:
                    return False
        with TemporaryDirectory(prefix="cold8697-", dir="/var/tmp") as scratch:
            recompute(work, Path(scratch))
        return build(work, Path(value["publication_output"])) == value
    except (OSError, ValueError, KeyError, TypeError):
        return False


def terminal(path: Path, raw: Path) -> Json:
    """Preserve actual scanner severity and execute bounded fresh-process cold replay."""
    findings = json.loads(path.read_bytes())["adversarial_findings"]
    receipts = []
    for name, args, expected in [
        (
            "adversarial",
            ["scripts/adversarial_verify.py", "--json", str(path)],
            int(bool(findings)),
        ),
        ("rows", ["scripts/verdict_row_consistency_lint.py", "--strict", str(path)], 0),
        ("cold", [CLI, "--cold-replay", str(path)], 0),
    ]:
        receipts.append(
            c.supervise(
                name,
                [sys.executable, "-u", *args],
                raw / "terminal",
                expected=expected,
                deadline=180,
            )
        )
    scan = json.loads(Path(receipts[0]["stdout_path"]).read_bytes())["reports"][0]
    return dict(
        passed=all(r["passed"] for r in receipts)
        and scan["flags"] == findings
        and not any(f["severity"] == "critical" for f in findings),
        adversarial=scan,
        receipts=receipts,
    )


def validation(work: Json) -> None:
    """Measure only new code and run the applicable checks without historical controllers."""
    progress("validation_900_seconds_reserved")
    instrument(work, "owned_tests", TESTS, [c.ref(ROOT / p) for p in OWNED], 600, "owned")
    for name, args in [
        ("ruff_check", ["-m", "ruff", "check", *OWNED, *TESTS]),
        ("ruff_format", ["-m", "ruff", "format", "--check", *OWNED, *TESTS]),
        ("mypy", ["-m", "mypy", "--strict", "--follow-imports=silent", *OWNED]),
        ("spec", ["scripts/check_spec_coverage.py", "--files", *TESTS]),
    ]:
        work["receipts"].append(
            c.supervise(name, [sys.executable, *args], Path(work["raw"]) / "checks", deadline=180)
        )


def run(
    output: Path, run_date: str, upstream: Path = UPSTREAM, *, private_control: bool = False
) -> int:
    """Measure once, reserve validation time, and atomically publish only checked bytes."""
    raw = output.parent / "raw" / NAME / "invocations" / str(time.time_ns())
    work = prepare(raw, run_date, upstream)
    if not work["input_failures"] and not private_control:
        execute(work)
    if not private_control:
        validation(work)
    with TemporaryDirectory(prefix="combine8697-", dir="/var/tmp") as scratch:
        recompute(work, Path(scratch))
    work["duration_s"] = (time.monotonic_ns() - work["started_monotonic_ns"]) / 1e9
    work["ended_at_utc"] = datetime.now(UTC).isoformat()
    work["phase_spans"] = [
        dict(
            phase="execution_and_validation",
            start_monotonic_ns=work["started_monotonic_ns"],
            end_monotonic_ns=work["started_monotonic_ns"] + round(work["duration_s"] * 1e9),
            duration_s=work["duration_s"],
        )
    ]
    freeze(work)
    candidate = raw / "preterminal.json"
    atomic_json(candidate, build(work, output))
    work["adversarial_findings"] = verify_artifact(candidate)["flags"]
    freeze(work)
    value = build(work, output)
    publication = publish_primary(output, value, lambda path: terminal(path, raw))
    atomic_json(Path(value["terminal_validation_sidecar_path"]), dict(publication=publication))
    progress("normal_exit_" + value["verdict_class"], 1, 0)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Reject unsafe private destinations before any evidence or shared result can change."""
    progress("start_no_model_load_MODEL_SPECS=[]_llm_calls=0")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261101")
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--upstream", type=Path, default=UPSTREAM)
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    args = parser.parse_args(argv)
    try:
        datetime.strptime(args.date, "%Y%m%d")
    except ValueError:
        return 2
    if (
        args.date != "20261101"
        or (
            args.private_output
            and (
                args.private_output.absolute().is_relative_to(ROOT)
                or args.private_output.resolve().is_relative_to(ROOT)
            )
        )
        or (args.private_control and not args.private_output)
        or (args.upstream != UPSTREAM and not args.private_output)
    ):
        return 2
    if args.deliberate_error:
        return 7
    if args.cold_replay:
        return int(not replay(args.cold_replay))
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        return run(
            args.private_output or ROOT / "results" / (NAME + ".json"),
            args.date,
            args.upstream,
            private_control=args.private_control,
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)
