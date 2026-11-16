"""REQ-REPORT-9139-V779: qualify schedule guards without qualifying model work.

Only a small child executes here. The complete original scope stays visible and
its conservative budget is reported before any dependent scientific execution.
"""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import time
from typing import Any

from coverage import Coverage

from carnot.reporting import validation_schedule_9139 as s
from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
)
from carnot.reporting.primary_publication import publish_primary

Json = dict[str, Any]
ROOT = s.ROOT
OWNED = [
    "python/carnot/reporting/validation_schedule_9139.py",
    "python/carnot/reporting/validation_schedule_runner_9139.py",
    s.CLI,
]
SEAM = "python/carnot/reporting/forward_evidence_runner_9111.py"


def foreign_argv(script: str, *args: str) -> list[str]:
    """Private CLI commands must work outside the checkout with PYTHONPATH removed."""
    return [
        "env",
        "-u",
        "PYTHONPATH",
        "-C",
        "/tmp",
        str(ROOT / ".venv/bin/python"),
        str(ROOT / script),
        *args,
    ]


def commands(raw: Path) -> list[tuple[str, list[str]]]:
    """Freeze owned checks once; the original 621-node run belongs to the next experiment."""
    py = str(ROOT / ".venv/bin/python")
    return [
        (
            "owned_tests",
            [
                py,
                "-m",
                "coverage",
                "run",
                "--rcfile=" + str(raw / "coverage.ini"),
                "-m",
                "pytest",
                "-n0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                s.TEST,
            ],
        ),
        (
            "e2e018_021",
            [
                py,
                "-m",
                "pytest",
                "-n0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
                "tests/python/test_restricted_decision_audit_8210.py",
            ],
        ),
        ("ruff_check", [str(ROOT / ".venv/bin/ruff"), "check", *OWNED, SEAM, s.TEST]),
        ("ruff_format", [str(ROOT / ".venv/bin/ruff"), "format", "--check", *OWNED, SEAM, s.TEST]),
        (
            "strict_mypy",
            [str(ROOT / ".venv/bin/mypy"), "--strict", "--follow-imports=silent", *OWNED, SEAM],
        ),
        ("spec_coverage", [py, "scripts/check_spec_coverage.py", s.TEST]),
    ]


def measure(work: Json, raw: Path) -> None:
    """Run one unchanged small module, then consume its evidence in the real normal runner."""
    plan = work["schedule"]
    atomic_json(raw / "complete_schedule.json", plan)
    work["references"].append(s.ref(raw / "complete_schedule.json"))
    s.progress("budget_" + ("feasible" if plan["budget_equation"]["feasible"] else "infeasible"))
    chosen = plan["original_node_manifest"][0].split("::")[0]
    subset = [n for n in plan["original_node_manifest"] if n.split("::")[0] == chosen]
    probe = s.schedule(
        dict(
            nodes=subset,
            events=[],
            required_command_union=[],
            original_assertion_references=[
                r for r in plan["original_assertion_hashes"] if r["path"].endswith(chosen)
            ],
        )
    )
    invocation = raw.parent.name
    bundle = s.execute(probe, raw / "partition_probe", invocation, work["code_config_hashes"])
    work["bundle"] = bundle
    work["rows"] = probe_rows(bundle)
    work["receipts"].extend(p["receipt"] for p in bundle["partitions"])
    path = raw / "partition_probe/bundle.json"
    work["bundle_reference"] = s.ref(path)
    work["references"].append(work["bundle_reference"])
    try:
        s.consume(bundle, invocation)
    except ValueError as error:
        work["measurement_error"] = str(error)
        return
    argv = foreign_argv(
        s.prior.old.CLI, "--schedule-v779", str(path), "--schedule-invocation", invocation
    )
    receipt = s.prior.run_check("normal_orchestration", argv, raw / "normal", 180)
    work["receipts"].append(receipt)
    work["isolated_child_receipts"] = [receipt]
    work["consumed_reference"] = s.ref(path.parent / "consumed.json")
    work["references"].append(work["consumed_reference"])
    if not receipt["passed"]:
        work["measurement_error"] = "normal_orchestration_failed"


def probe_rows(bundle: Json) -> list[Json]:
    """Retain failed and unfinished probe nodes without accepting their reuse."""
    events = [
        json.loads(line)
        for part in bundle["partitions"]
        for line in s.prior.old.bound(part["events_reference"]).read_text().splitlines()
    ]
    return s.reduce_nodes(bundle["schedule"]["original_node_manifest"], events)


def coverage_rows(tracer: Coverage) -> list[Json]:
    """Count all new statements; only the small changed seam is owned in the old runner."""
    import difflib

    files = []
    for name in [*OWNED, SEAM]:
        _, statements, excluded, missing, _ = tracer.analysis2(str(ROOT / name))
        selected = set(statements)
        if name == SEAM:
            before = s.HISTORICAL_READER.read_text().splitlines()
            after = (ROOT / name).read_text().splitlines()
            added = {
                i + 1
                for tag, _, _, start, end in difflib.SequenceMatcher(
                    a=before, b=after
                ).get_opcodes()
                if tag in {"insert", "replace"}
                for i in range(start, end)
            }
            selected &= added
        files.append(
            dict(
                path=name,
                statements=sorted(selected),
                excluded=sorted(selected & set(excluded)),
                missing=sorted(selected & set(missing)),
            )
        )
    return files


def verification(work: Json, raw: Path) -> None:
    """Wait for each child to exit before combining its actual coverage data."""
    stop = time.monotonic() + 1200
    for name, argv in commands(raw):
        if time.monotonic() >= stop:
            break
        s.progress(name + "_before")
        work["receipts"].append(
            s.prior.run_check(name, argv, raw / "verification", min(240, stop - time.monotonic()))
        )
        s.progress(name + "_after")
    tracer = Coverage(config_file=str(raw / "coverage.ini"))
    tracer.combine(strict=True, keep=True)
    tracer.save()
    files = coverage_rows(tracer)
    total = sum(len(r["statements"]) for r in files)
    work["coverage"] = dict(
        files=files,
        statement_percent=100 * (total - sum(len(r["missing"]) for r in files)) / total,
        excluded_statements=sum(len(r["excluded"]) for r in files),
        data_reference=s.ref(raw / ".coverage"),
    )
    work["verification_root"] = str(raw)


def qualified(work: Json) -> bool:
    """Readiness requires executed checks, unchanged logs and actual owned coverage."""
    try:
        raw = Path(work["verification_root"])
        actual = work["receipts"][-len(commands(raw)) :]
        if len(actual) != len(commands(raw)):
            return False
        for receipt, (name, argv) in zip(actual, commands(raw), strict=True):
            if (
                receipt["name"] != name
                or receipt["argv"] != argv
                or not receipt["passed"]
                or not receipt["normal_exit"]
                or receipt["timed_out"]
            ):
                return False
            for stream in ["stdout", "stderr"]:
                s.prior.old.bound(
                    dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"])
                )
        tracer = Coverage(
            config_file=False, data_file=str(s.prior.old.bound(work["coverage"]["data_reference"]))
        )
        tracer.set_option("report:exclude_lines", [])
        tracer.load()
        rows = coverage_rows(tracer)
        if rows != work["coverage"]["files"] or any(r["missing"] or r["excluded"] for r in rows):
            return False
        result = s.consume(work["bundle"], work["bundle"]["invocation_id"])
        actual_child = json.loads(s.prior.old.bound(work["consumed_reference"]).read_bytes())
        return result == actual_child and all(r["passed"] for r in work["isolated_child_receipts"])
    except (KeyError, OSError, ValueError):
        return False


def build(work: Json, date: str, start: int, end: int) -> Json:
    """Reduce current probe rows without promoting historical execution or human benefit."""
    blocked = any(not c["passed"] for c in work["checks"])
    ready = not blocked and qualified(work)
    plan = work.get("schedule", {})
    nodes = plan.get("original_node_manifest", [])
    probe_rows = {r["unit_id"]: r for r in work.get("rows", [])}
    rows = [
        probe_rows.get(
            n,
            dict(
                unit_id=n,
                qid=None,
                group_id=n,
                sibling=None,
                arm="original_assertions",
                condition="deferred_to_exp9140",
                seed=s.SEED,
                absolute_metric=0,
                numerator=0,
                denominator=1,
                status="censored",
                exclusion_reason=None,
                censor_reason="full_qualification_owned_by_exp9140",
            ),
        )
        for n in nodes
    ]
    counts = dict(
        intended=len(rows),
        attempted=len(probe_rows),
        completed=sum(r["status"] == "completed" for r in rows),
        failed=sum(r["status"] == "failed" for r in rows),
        censored=sum(r["status"] == "censored" for r in rows),
        effective_independent_groups=0,
    )
    receipt = build_current_work_receipt(
        run_id="exp9139",
        owner_pid=work.get("owner_pid", os.getpid()),
        events=[],
        inference_substrate="deterministic_no_model_load",
        inference_substrate_details=dict(work="CPU schedule and receipt validation"),
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=start,
        ended_monotonic_ns=end,
    )
    union = [n for p in plan.get("execution_graph", []) for n in p["node_ids"]]
    value = dict(
        experiment_id=9139,
        task_id="exp9139-validation-schedule",
        milestone="2026.11.779",
        schema="carnot.validation_schedule.v2",
        run_date=date,
        honest_verdict="complete_blocked_upstream"
        if blocked
        else "complete_null_schedule_qualified"
        if ready
        else "complete_disqualified_schedule_checks",
        verdict_class="blocked" if blocked else "null" if ready else "disqualified",
        schedule_ready_score=int(ready),
        required_checks_passed=bool(ready),
        gate_check_summary=work["checks"],
        rows=rows,
        sample_size_budget=counts,
        intended_count=counts["intended"],
        attempted_count=counts["attempted"],
        completed_count=counts["completed"],
        failed_count=counts["failed"],
        censored_count=counts["censored"],
        effective_independent_groups=0,
        verifier_is_oracle=True,
        independent_generalization_score=0,
        claim_scope=dict(
            exposure_scope="Exposed human development; administrative schedule only",
            independent_generalization_score=0,
            scientific_benefit_claimed=False,
            arc_solve_claimed=False,
            open_gaps=["GAP-ORACLE-DISTINCT", "GAP-DETECTOR-AUROC-4208"],
        ),
        MODEL_SPECS=[],
        requested_inference_substrate_class="no_model_load",
        inference_substrate_class="blocked_no_run" if blocked else "no_model_load",
        inference_substrate="deterministic_no_model_load",
        current_llm_invocations=0,
        generated_tokens=0,
        model_invocation_counts=dict(
            receipt["invocation_counts"],
            current_llm_invocations=0,
            forward_calls=0,
            generated_tokens=0,
        ),
        duration_s=(end - start) / 1e9,
        random_seed=s.SEED,
        current_work_receipts=[receipt],
        source_references=work["references"],
        code_config_hashes=work["code_config_hashes"],
        validation_receipts=work["receipts"],
        owned_statement_coverage=work.get("coverage", {}),
        schedule_reference=plan,
        preserved_semantic_controls=work.get("semantic_controls", []),
        preserved_count_controls=work.get("count_controls", []),
        measurement_error=work.get("measurement_error"),
        computational_root_cause="unknown; retained stack shows a subprocess wait only",
        original_source_pins=work.get("original_source_pins", []),
        scope_equality=dict(
            obligation_union_equality=Counter(union) == Counter(nodes),
            no_duplicate_execution_controls="Reject duplicate partition IDs, nodes and phase events",
        ),
        current_revision_reference=dict(
            authority="carnot.validation_schedule.v2",
            historical_reader_reference=work.get("historical_reader_reference"),
            isolated_child_receipts=work.get("isolated_child_receipts", []),
        ),
        full_621_node_qualification_score=0,
        model_work_authorized=False,
        projected_execution_verdict="infeasible_with_current_bounds"
        if plan and not plan["budget_equation"]["feasible"]
        else "unmeasured"
        if not plan
        else "bounded_plan",
        repository_health=work.get("repository_health", {}),
        acceptance_gates=dict(
            authenticated_inputs=not blocked,
            original_scope_equality=bool(nodes and Counter(union) == Counter(nodes)),
            owned_validation=bool(ready),
            full_qualification_deferred=True,
        ),
        preconditions_checked=work.get("resources", {}),
    )
    value["field_principles"] = {
        **s.prior.old.principles(),
        "schedule_ready_score": "A tested schedule prototype never substitutes for full forward qualification.",
        "schedule_reference": "Every original node, assertion and command stays visible and budgeted.",
        "scope_equality": "Changing execution order cannot shrink the original obligation union.",
        "current_revision_reference": "New scheduling authority cannot reinterpret historical failures.",
        "acceptance_gates": "Fixed custody, scope and executed checks precede readiness.",
        "sample_size_budget": "Test nodes confer zero independent human observations.",
    }
    saved = dict(work=work, date=date, start=start, end=end)
    value["replay_input"] = saved
    value["reproducibility_checksum"] = canonical_hash(saved)
    return value


def replay(path: Path) -> Json:
    """Reconstruct claims from bound inputs and refuse even rehashed aggregate edits."""
    value = json.loads(path.read_bytes())
    saved = value["replay_input"]
    work = saved["work"]
    if value["reproducibility_checksum"] != canonical_hash(saved):
        raise ValueError("replay_hash")
    for reference in value["source_references"] + value["code_config_hashes"]:
        s.prior.old.bound(reference)
    if "historical" in work:
        source = next(
            r for r in work["references"] if Path(r["path"]).name == Path(s.PREVIOUS).name
        )
        original = json.loads(s.prior.old.bound(source).read_bytes())
        historical = original["replay_input"]["work"]
        if (
            original != work["historical"]
            or work.get("schedule") != s.schedule(historical)
            or work.get("semantic_controls") != historical["original_semantic_controls"]
            or work.get("count_controls") != historical["adversarial_control_rows"]
        ):
            raise ValueError("schedule_drift")
    if "bundle" in work:
        if work["bundle"] != json.loads(s.prior.old.bound(work["bundle_reference"]).read_bytes()):
            raise ValueError("bundle_drift")
        if probe_rows(work["bundle"]) != work["rows"]:
            raise ValueError("node_reduction_drift")
        if not work.get("measurement_error"):
            s.consume(work["bundle"], work["bundle"]["invocation_id"])
    if value != build(work, saved["date"], saved["start"], saved["end"]):
        raise ValueError("reduction_drift")
    return dict(passed=True, replay_passed=True)


def terminal(path: Path, raw: Path) -> Json:
    """Both unchanged scanners inspect the exact bytes primary_publication will expose."""
    replay(path)
    receipts = [
        s.prior.run_check(
            name, [str(ROOT / ".venv/bin/python"), str(ROOT / script), flag, str(path)], raw
        )
        for name, script, flag in [
            ("adversarial", "scripts/adversarial_verify.py", "--json"),
            ("strict_rows", "scripts/verdict_row_consistency_lint.py", "--strict"),
        ]
    ]
    return dict(passed=all(r["passed"] for r in receipts), checks=receipts)


def main(argv: list[str] | None = None) -> int:
    """Authenticate first, bound measurement, reserve verification once, then publish."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261116")
    parser.add_argument("--input-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (s.NAME + ".json"))
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.replay:
            print(json.dumps(replay(args.replay)), flush=True)
            return 0
        if args.date != "20261116":
            raise ValueError("run_date")
        start = time.monotonic_ns()
        raw = args.output.parent / "raw" / s.NAME / "invocations" / str(time.time_ns())
        raw.mkdir(parents=True, exist_ok=True)
        s.progress("authentication_before")
        work = s.authenticate(args.input_root, raw)
        work["owner_pid"] = os.getpid()
        s.progress("authentication_after")
        work["code_config_hashes"] = [
            s.ref(ROOT / p) for p in [*OWNED, SEAM, s.TEST, "pyproject.toml"]
        ]
        resources = dict(
            disk_free_bytes=shutil.disk_usage(raw).free,
            open_files_limit=resource.getrlimit(resource.RLIMIT_NOFILE)[0],
            imports_available=all(
                importlib.util.find_spec(n) is not None for n in ["pytest", "coverage"]
            ),
        )
        work["resources"] = resources
        for field, minimum in [
            ("disk_free_bytes", 100_000_000),
            ("open_files_limit", 64),
            ("imports_available", True),
        ]:
            observed = resources[field]
            work["checks"].append(
                dict(
                    check=field,
                    upstream="current_host",
                    path=str(raw),
                    sha256=None,
                    artifact_field=field,
                    op=">=",
                    expected=minimum,
                    observed=observed,
                    passed=observed >= minimum,
                )
            )
        if all(c["passed"] for c in work["checks"]):
            config = raw / "coverage.ini"
            config.write_text(
                "[run]\nparallel=true\npatch=subprocess,_exit\ndata_file="
                + str(raw / ".coverage")
                + "\ninclude=\n"
                + "".join("    " + str(ROOT / p) + "\n" for p in [*OWNED, SEAM])
                + "[report]\nexclude_lines=\nomit=\n"
            )
            work["references"].append(s.ref(config))
            atomic_json(raw / "checks.json", dict(commands=commands(raw)))
            work["references"].append(s.ref(raw / "checks.json"))
            os.environ["COVERAGE_RCFILE"] = str(config)
            tracer = Coverage(config_file=str(config), data_suffix=True)
            tracer.start()
            s.progress("measurement_before")
            measure(work, raw / "private")
            work["repository_health"] = s.prior.run_check(
                "repository_health",
                [str(ROOT / ".venv/bin/pytest"), "tests/python", "-q"],
                raw / "health",
                min(120, max(1, start / 1e9 + 3300 - time.monotonic())),
            )
            s.progress("measurement_after")
            tracer.stop()
            tracer.save()
            verification(work, raw)
        value = build(work, args.date, start, time.monotonic_ns())
        s.progress("publication_before")
        publish_primary(args.output, value, lambda p: terminal(p, raw / "terminal"))
        s.progress("publication_after")
        print(
            json.dumps(
                dict(
                    path=str(args.output),
                    schedule_ready_score=value["schedule_ready_score"],
                    honest_verdict=value["honest_verdict"],
                )
            ),
            flush=True,
        )
        return 0
    except (OSError, ValueError, KeyError, StopIteration) as error:
        print("[exp9139] refused=" + str(error), flush=True)
        return 1
