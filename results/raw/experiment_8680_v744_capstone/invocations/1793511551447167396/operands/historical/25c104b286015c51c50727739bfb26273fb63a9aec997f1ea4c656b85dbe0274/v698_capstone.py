"""REQ-REPORT-8069: exact accounting cannot convert readiness into science."""

import argparse
import json
import hashlib
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any

import yaml

from carnot.reporting import v697_capstone as previous
from carnot.reporting import v698_capstone_reduction as reduction
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS, build_current_work_receipt
from carnot.reporting.evidence_features_custody_7980 import checked
from carnot.reporting.primary_publication import publish_primary, read_bound_sidecar, reader_receipt
from carnot.reporting.roadmap_contract import parse_design
from carnot.reporting.v685_authority_lifecycle import tasks_digest
from carnot.reporting.v686_contract_validation import run_check

Json = dict[str, Any]
ROOT = previous.ROOT
CLI = "scripts/experiments/experiment_8069_v698_capstone.py"
TEST = "tests/python/test_v698_capstone_8069.py"
INPUT = "results/experiment_8057_v698_fixture_consumer_contract.json"
OWNED = [
    "python/carnot/reporting/v698_capstone.py",
    "python/carnot/reporting/v698_capstone_reduction.py",
    CLI,
]
holm = reduction.holm
read = previous.read
START = time.monotonic()


def progress(phase: str, units: int = 0, pending: str = "") -> None:
    """Actual units and elapsed time expose stalls without inventing work."""
    print(
        f"[exp8069] phase={phase} elapsed_s={time.monotonic() - START:.3f} completed_units={units} pending={pending}",
        flush=True,
    )


def reference(path: Path, role: str = "input") -> Json:
    """An absent path has a null digest so later appearance cannot pass replay."""
    return dict(path=str(path), sha256=sha256_file(path) if path.is_file() else None, role=role)


def failure(path: Path, upstream: str, field: str, expected: Any, observed: Any) -> Json:
    """Exact operands distinguish absence from a failed measurement criterion."""
    return dict(
        check=field,
        upstream=upstream,
        path=str(path),
        hash=reference(path)["sha256"],
        field=field,
        artifact_field=field,
        op="==",
        expected=expected,
        observed=observed,
        passed=False,
    )


def authorities(root: Path) -> tuple[list[Json], Json]:
    """Read immutable activation bytes, complete prompts and visible table separately."""
    invocation = read(root / INPUT)
    snapshots = invocation["authority_snapshots"]
    active, design = [
        checked(dict(path=snapshots[k]["snapshot_path"], sha256=snapshots[k]["sha256"]))
        for k in ("active", "design")
    ]
    text = design.read_text()
    table, tasks = parse_design(text, milestone="2026.10.698")
    shown = [
        dict(order=i + 1, **{k: t[k] for k in ("id", "title", "phase", "deliverable")})
        for i, t in enumerate(tasks)
    ]
    digest = re.search(r"Canonical (?:full-)?task SHA-256: `([0-9a-f]{64})`", text)
    observed = yaml.safe_load(active.read_bytes())
    if (
        observed["milestone"] != "2026.10.698"
        or observed["tasks"] != tasks
        or table != shown
        or [t["id"].split("-")[0] for t in tasks] != [f"exp{n}" for n in range(8057, 8070)]
        or digest is None
        or digest[1] != invocation["canonical_tasks_sha256"]
        or tasks_digest(tasks) != invocation["canonical_tasks_sha256"]
    ):
        raise ValueError("immutable_authority_drift")
    return tasks, invocation


def skip_path(root: Path, task: Json) -> Path:
    """The conductor names its skip using the task identifier, not the primary title."""
    return root / (
        "results/experiment_"
        + task["id"][3:7]
        + "_"
        + task["id"].split("-", 1)[1].replace("-", "_")
        + ".json"
    )


def collect(root: Path, tasks: list[Json]) -> tuple[list[Json], list[Json], list[Json]]:
    """Authenticate each final primary or actual skip without filling absent science."""
    rows, refs, failures = [], [], []
    log = root / "ops/conductor-log.md"
    text = log.read_text() if log.is_file() else ""
    refs.append(reference(log, "conductor_log"))
    for task in tasks[:-1]:
        path, skip = root / task["deliverable"], skip_path(root, task)
        primary = path.is_file()
        selected = path if primary or not skip.is_file() else skip
        data = read(selected)
        issues = []
        state = data.get("verdict_class", "blocked")
        verdict = data.get("honest_verdict")
        kind = "present" if primary else "conductor_skip_receipt" if data else "missing"
        if primary:
            if data.get("task_id") != task["id"] or data.get("experiment_id") != int(
                task["id"][3:7]
            ):
                raise ValueError("primary_identity")
            side = Path(data.get("terminal_validation_sidecar_path") or "/missing")
            report = read(side)
            refs.append(reference(side, "terminal_sidecar"))
            bound = Path(report.get("publication", report).get("sidecar_path", "/missing"))
            try:
                terminal = read_bound_sidecar(path, bound)
                if terminal["report"].get("passed") is not True:
                    raise ValueError("terminal_report_failed")
            except (ValueError, KeyError, OSError) as error:
                issues.append(
                    failure(bound, task["id"], "terminal_primary_binding", True, str(error))
                )
            refs.append(reference(bound, "terminal_validator"))
            if (
                data.get("flagged_adversarial") is not False
                or data.get("required_checks_passed") is not True
            ):
                issues.append(
                    failure(path, task["id"], "owned_checks_and_adversarial", True, False)
                )
            for g in data.get("gate_check_summary", []):
                issues.append(
                    dict(g, artifact_field=g.get("field", g.get("artifact_field")), passed=False)
                )
        else:
            if data:
                if (
                    data.get("experiment") != int(task["id"][3:7])
                    or data.get("schema") != "blocked_gate_check_v1"
                ):
                    raise ValueError("skip_identity")
                evidence = Path(data["failed_evidence_path"])
                if reference(evidence)["sha256"] != data["failed_evidence_sha256"]:
                    raise ValueError("skip_upstream_hash")
                issues.append(
                    failure(
                        evidence,
                        data["failed_upstream"],
                        data["failed_field"],
                        data["failed_expected"],
                        data["failed_observed"],
                    )
                )
            issues.append(failure(path, task["id"], "declared_primary_present", True, False))
            state = "blocked"
        refs.append(reference(selected, kind))
        refs.append(reference(path, "declared_primary"))
        shards = data.get("raw_shard_hashes", [])
        shards = (
            [dict(path=p, sha256=h) for p, h in shards.items()]
            if isinstance(shards, dict)
            else shards
        )
        for ref in shards:
            p = Path(ref["path"])
            if reference(p)["sha256"] != ref["sha256"]:
                issues.append(
                    failure(
                        p, task["id"], "primitive_sha256", ref["sha256"], reference(p)["sha256"]
                    )
                )
            refs.append(dict(ref, role="primitive_shard"))
        eligible = (
            primary
            and state not in ("blocked", "disqualified")
            and not any(g.get("classification") != "scientific" for g in issues)
        )
        rows.append(
            dict(
                task_id=task["id"],
                id=task["id"],
                unit_id=task["id"],
                unit=task["id"],
                source=str(selected),
                seed=None,
                arm="task_disposition",
                path=str(selected),
                sha256=reference(selected)["sha256"],
                primary_present=primary,
                producer_status=kind,
                producer_honest_verdict=verdict,
                honest_verdict=verdict if primary else "complete_blocked_" + kind,
                verdict_class=state,
                eligible=eligible,
                excluded=not eligible,
                started=primary,
                completed=True,
                failed=state == "disqualified",
                censored=False,
                status="completed",
                numerator=int(eligible),
                denominator=1,
                raw_numerator=int(eligible),
                raw_denominator=1,
                exclusion_reason=None
                if eligible
                else "external_prerequisite_or_invalid_measurement",
                gate_check_summary=issues,
                conductor_log_rows=[
                    line
                    for line in text.splitlines()
                    if task["title"][:48] in line or task["id"] in line
                ],
            )
        )
        failures.extend(issues)
        progress("task_disposition", len(rows), str(12 - len(rows)))
    return rows, refs, failures


def build(root: Path, date: str, durable: Path) -> Json:
    """Freeze the inputs and keep every PRD gap tied to its own evidence."""
    progress("preconditions_before", pending="named resources and immutable authority")
    failures = []
    try:
        tasks, authority = authorities(root)
    except (ValueError, KeyError, TypeError, OSError) as error:
        tasks, authority = authorities(ROOT)
        failures.append(
            failure(
                root / INPUT,
                "exp8057",
                "immutable_authority",
                "authenticated thirteen-task activation",
                str(error),
            )
        )
    names = list(
        dict.fromkeys(
            previous.NAMED
            + [INPUT, "ops/conductor-log.md", "results/experiment_8056_v697_capstone.json"]
        )
    )
    preconditions = [
        dict(reference(root / p), check="resource_exists", passed=(root / p).is_file())
        for p in names
    ]
    preconditions += [
        dict(
            reference(ROOT / ".venv/bin" / tool),
            check="tool_exists",
            passed=(ROOT / ".venv/bin" / tool).is_file(),
        )
        for tool in ("python", "pytest", "coverage", "ruff", "mypy")
    ]
    failures.extend(
        failure(Path(r["path"]), "preconditions", r["check"], True, False)
        for r in preconditions
        if not r["passed"]
    )
    progress("preconditions_after", len(preconditions), "frozen equations")
    frozen = dict(
        task_contract=tasks,
        canonical_tasks_sha256=tasks_digest(tasks),
        margins=[0.01, 0.02, 0.02],
        draws=10000,
        alpha=0.05,
        blocks=[32, 16, 64],
        absent_or_invalid_or_unsafe_p=1,
        source_evaluation_support=[72, 8],
        learning_support=[80, 10],
        retention_support=[48, 8],
        seeds="average within source; no independent environment credit",
    )
    atomic_json(durable / "method_freeze.json", frozen)
    rows, refs, issues = collect(root, tasks)
    failures.extend(issues)
    audits = []
    for row in rows:
        try:
            audit = reduction.independent(read(Path(row["path"])), int(row["task_id"][3:7]))
        except (ValueError, KeyError, TypeError, OSError) as error:
            audit = dict(measurement_available=False, reduction_error=str(error))
            failures.append(
                failure(
                    Path(row["path"]),
                    row["task_id"],
                    "independent_reduction",
                    "valid primitive equations",
                    str(error),
                )
            )
        path = durable / "reductions" / (row["task_id"] + ".json")
        atomic_json(path, audit)
        audits.append(
            dict(
                task_id=row["task_id"],
                **reference(path, "independent_reduction"),
                measurement_available=audit["measurement_available"],
            )
        )
        progress("independent_reduction_after", len(audits), str(12 - len(audits)))
    h3 = read(Path(audits[8]["path"])).get("primary_hypothesis_results", [None])[0]
    family = holm(
        h3, valid=rows[8]["eligible"] and not read(Path(rows[8]["path"])).get("verifier_is_oracle")
    )
    prior = read(root / "results/experiment_8056_v697_capstone.json")
    old_rows = {r["task_id"]: r for r in prior.get("task_dispositions", [])}
    retirements = []
    for task, row in zip(tasks[:-1], rows, strict=True):
        for old in task.get("prior_failures", []):
            history = old_rows.get(old["experiment_id"], {})
            repeat = history.get("honest_verdict") == row["honest_verdict"] and row[
                "verdict_class"
            ] in ("blocked", "disqualified")
            retirements.append(
                dict(
                    task_id=task["id"],
                    prior_failure=old,
                    historical_disposition=history,
                    retire=bool(repeat and old["retire_if_same_verdict"]),
                    retirement_scope="only the unchanged tested mechanism and failed operand",
                    reopen_condition=old["addressed_by"],
                    continuity_obligation=task["id"].startswith(("exp8067-", "exp8068-")),
                )
            )
    gaps = dict(
        useful_source_verification=dict(
            requirements=["FR-06", "FR-12"],
            closed=False,
            decision="blocked_absent_H1_H2_after_failed_repeatability",
            reopen_condition="Pass unchanged 1e-6 duplicate tolerance and token alignment; acquire new fit/tune/evaluation shards, freeze matched heads and complete human targets; pass support, safety and Holm margins.",
        ),
        retained_causal_learning=dict(
            requirements=["FR-11"],
            closed=False,
            decision="null_fresh_admission_no_beneficial_changes_and_reused_guard_retention_failure",
            reopen_condition="Change admission information or candidate mechanism; obtain five beneficial changed later sources and .02 cost margin with per-seed false-accept safety and both guarded retention limits on unexposed evidence.",
        ),
        reproducible_deployment=dict(
            requirements=["FR-05", "FR-08", "FR-09", "FR-10", "NFR-01"],
            closed=False,
            decision="blocked_censored_cache_budget_and_unpriced_complete_service",
            reopen_condition="Complete cold/warm/miss/change/eviction/restart pairs, qualify useful decisions and charge source acquisition, external feedback, transfer and storage; reproduce complete service.",
        ),
    )
    own = dict(
        task_id="exp8069-capstone",
        id="exp8069-capstone",
        unit_id="exp8069-capstone",
        unit="exp8069-capstone",
        source="current_owned_reader",
        arm="task_disposition",
        seed=None,
        path=None,
        sha256=None,
        primary_present=False,
        started=True,
        completed=False,
        eligible=False,
        excluded=False,
        failed=False,
        censored=False,
        status="pending",
        numerator=0,
        denominator=1,
        raw_numerator=0,
        raw_denominator=1,
        exclusion_reason=None,
        honest_verdict="complete_blocked_v698_capstone",
        verdict_class="blocked",
        producer_status="pending_normal_exit",
        gate_check_summary=[],
    )
    rows.append(own)
    retirements.extend(
        dict(
            task_id=tasks[-1]["id"],
            prior_failure=old,
            historical_disposition=old_rows.get(old["experiment_id"], {}),
            retire=False,
            retirement_scope="Required terminal accounting remains open; current source and cache failure operands differ from the previous fixture consumer block.",
            reopen_condition=old["addressed_by"],
            continuity_obligation=True,
        )
        for old in tasks[-1].get("prior_failures", [])
    )
    refs += [reference(root / p, "named_input") for p in names if (root / p).is_file()]
    refs += [
        reference(Path(s["snapshot_path"]), "authority_" + k)
        for k, s in authority["authority_snapshots"].items()
        if s["exists"]
    ]
    # References retain original bytes; separate durable copies support later custody inspection.
    unique = {r["path"]: r for r in refs}
    saved = []
    for index, ref in enumerate(unique.values()):
        saved.append(previous.save(ref, durable))
        if index % 100 == 0:
            progress("durable_input_after", index + 1, str(len(unique) - index - 1))
    value: Json = dict(
        experiment_id=8069,
        task_id="exp8069-capstone",
        milestone="2026.10.698",
        schema="carnot.v698.capstone.v1",
        run_date=date,
        input_root=str(root),
        honest_verdict="complete_blocked_v698_capstone",
        verdict_class="blocked",
        rows=rows,
        task_dispositions=rows,
        task_contract=tasks,
        canonical_tasks_sha256=tasks_digest(tasks),
        gate_check_summary=failures,
        primary_hypothesis_results=family,
        independent_reduction_rows=audits,
        source_artifact_hashes=list(unique.values()),
        raw_shard_hashes=saved,
        code_config_hashes=[reference(ROOT / p, "current_code") for p in OWNED],
        checkpoint_references=[reference(durable / "method_freeze.json", "method_freeze")] + audits,
        random_seed=80698065,
        MODEL_SPECS=[],
        model_specs=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        substrate_declaration=dict(
            custody="aggregation_from_upstream_artifacts",
            pretrained="no_model_load",
            MODEL_SPECS=[],
        ),
        verifier_is_oracle=False,
        claim_scope="Exact task custody and exposed development reductions only; no generalized learning, live solve or board speed credit.",
        generalized_learning_benefit_score=0,
        science_ready=False,
        gap_decisions=gaps,
        retirement_rows=retirements,
        reopen_conditions=[g["reopen_condition"] for g in gaps.values()],
        historical_dispositions=prior.get("task_dispositions", []),
        useful_null_findings=[
            "Fresh guard retained initial state; no beneficial changed later sources.",
            "An empty authenticated ARC frontier satisfies mandatory monitoring.",
            "Valid unchanged board custody remains useful independently of blocked workload bounds.",
            "Warm feature reuse excludes acquisition and feedback costs and cannot close source or learning gaps.",
        ],
        authority_snapshots=authority["authority_snapshots"],
        contract_ready_score=read(root / INPUT).get("contract_ready_score", 0),
        capstone_execution_ready_score=0,
        required_checks_passed=False,
        flagged_adversarial=False,
        validation_receipts=[],
        coverage_statement_counts={},
        preconditions_checked=preconditions,
        terminal_validation_sidecar_path=str(durable / "terminal_validation.json"),
        acceptance_gate_results=dict(validity=False, readiness=0),
        g1=False,
        g2=False,
        g3=False,
        g4=False,
        paper_ready=False,
        unmet_gates=["publication_gate_pending"],
        repository_health=prior.get("repository_health", []),
        sample_size_budget=dict(
            intended=13,
            completed=12,
            eligible=sum(r["eligible"] for r in rows),
            excluded=sum(r["excluded"] for r in rows),
            failed=sum(r["failed"] for r in rows),
            censored=0,
            independent=0,
            unit="administrative_task_disposition",
        ),
    )
    value["reproducibility_checksum"] = canonical_hash(dict(freeze=frozen, inputs=unique))
    atomic_json(durable / "reduction_identity.json", dict(payload=claim(value)))
    value["checkpoint_references"].append(
        reference(durable / "reduction_identity.json", "reduction_identity")
    )
    return value


def claim(value: Json) -> Json:
    """The reduction seal protects conclusions that custody checks alone cannot derive."""
    return {
        k: value[k]
        for k in (
            "task_contract",
            "canonical_tasks_sha256",
            "primary_hypothesis_results",
            "gap_decisions",
            "science_ready",
            "retirement_rows",
            "historical_dispositions",
        )
    }


def cold_replay(value: Json) -> list[str]:
    """Reopen owned and upstream bytes, then repeat every available primitive equation."""
    progress("cold_replay_before", pending="hashes and independent equations")
    seals = [r for r in value["checkpoint_references"] if r["role"] == "claim_seal"]
    if seals and read(Path(seals[-1]["path"])) != previous.previous.previous.claim_payload(value):
        return ["claim_seal_drift"]
    for ref in (
        value["source_artifact_hashes"]
        + value["raw_shard_hashes"]
        + value["code_config_hashes"]
        + value["checkpoint_references"]
    ):
        path = Path(ref["path"])
        if reference(path)["sha256"] != ref["sha256"]:
            return ["source_bytes_changed"]
        if path.name.endswith(".shards.json"):
            shards = read(path)
            digest = hashlib.sha256()
            for shard in shards["shards"]:
                part = Path(shard["path"])
                if reference(part)["sha256"] != shard["sha256"]:
                    return ["source_bytes_changed"]
                digest.update(part.read_bytes())
            if "sha256:" + digest.hexdigest() != shards["original_sha256"]:
                return ["source_bytes_changed"]
    identity = next(r for r in value["checkpoint_references"] if r["role"] == "reduction_identity")
    if read(Path(identity["path"]))["payload"] != claim(value):
        return ["reduction_claim_drift"]
    copies = {r.get("original_path", r["path"]): r["path"] for r in value["raw_shard_hashes"]}
    for row, ref in zip(value["rows"][:-1], value["independent_reduction_rows"], strict=True):
        try:
            observed = reduction.independent(
                read(Path(copies.get(row["path"], row["path"]))), int(row["task_id"][3:7]), copies
            )
        except (ValueError, KeyError, TypeError, OSError) as error:
            observed = dict(measurement_available=False, reduction_error=str(error))
        if observed != read(Path(ref["path"])):
            return ["independent_reduction_drift"]
    progress("cold_replay_after", 12, "none")
    return []


def manifest(private: Path) -> Json:
    """Reuse bounded validation, scoped to the added reader and actual CLI."""
    value = previous.manifest(private)
    mapping = dict(zip(previous.OWNED + [previous.TEST], OWNED + [TEST], strict=True))
    value["commands"] = [s for s in value["commands"] if s["name"] != "full_suite"]
    for spec in value["commands"]:
        spec["argv"] = [mapping.get(a, a) for a in spec["argv"]]
        spec["argv"] = [
            "--include=" + ",".join(OWNED) if a.startswith("--include=") else a
            for a in spec["argv"]
        ]
    value.update(
        coverage_includes=OWNED,
        dependency_hashes={p: sha256_file(ROOT / p) for p in OWNED + [TEST]},
        diagnostic_policy="Reuse the existing bounded full-suite failure; it is not a scientific gate.",
    )
    return value


def complete(value: Json, receipts: list[Json], counts: Json) -> None:
    """An executed reader may qualify while the source and deployment branches block."""
    valid = (
        bool(receipts)
        and all(r["passed"] for r in receipts)
        and all(
            p in counts
            and counts[p]["num_statements"] > 0
            and counts[p]["num_statements"] == counts[p]["covered_lines"]
            for p in OWNED
        )
    )
    value.update(
        validation_receipts=receipts,
        coverage_statement_counts=counts,
        required_checks_passed=valid,
        capstone_execution_ready_score=int(valid),
    )
    if not valid:
        value.update(
            honest_verdict="complete_disqualified_v698_owned_checks", verdict_class="disqualified"
        )
    value["rows"][-1].update(
        completed=True,
        eligible=valid,
        failed=not valid,
        numerator=int(valid),
        raw_numerator=int(valid),
        status="completed",
        producer_status="normal_reduction_child_exit",
        honest_verdict=value["honest_verdict"],
        verdict_class=value["verdict_class"],
    )
    value["acceptance_gate_results"].update(validity=valid, readiness=int(valid))
    budget = value["sample_size_budget"]
    budget.update(
        completed=13,
        eligible=sum(r["eligible"] for r in value["rows"]),
        failed=sum(r["failed"] for r in value["rows"]),
    )
    value.update({k + "_count": v for k, v in budget.items() if k != "unit"})


def publish(value: Json, output: Path, private: Path, durable: Path, *, fixture: bool) -> None:
    """Publish bytes only after child exit and checks; bind both independent readers."""
    candidate = private / output.name
    previous.seal(value, durable / "seals")
    atomic_json(candidate, value)
    receipts = []
    specs = [
        (
            "cold_replay",
            [
                str(ROOT / ".venv/bin/python"),
                "-u",
                str(ROOT / CLI),
                "--cold-replay",
                str(candidate),
            ],
        )
    ]
    if not fixture:
        specs += [
            (
                "adversarial",
                [
                    str(ROOT / ".venv/bin/python"),
                    "scripts/adversarial_verify.py",
                    "--json",
                    str(candidate),
                ],
            ),
            (
                "strict_rows",
                [
                    str(ROOT / ".venv/bin/python"),
                    "scripts/verdict_row_consistency_lint.py",
                    "--strict",
                    str(candidate),
                ],
            ),
        ]
    for name, argv in specs:
        progress("subprocess_before_" + name, len(receipts), "terminal checks")
        receipts.append(
            run_check(
                ROOT,
                dict(name=name, argv=argv, expected_exit=0, deadline_s=180),
                private,
                durable / "terminal_logs",
            )
        )
        progress("subprocess_after_" + name, len(receipts), "terminal checks")
    valid = all(r["passed"] for r in receipts)
    if not fixture:
        valid = valid and read(Path(receipts[-2]["log_path"])).get("flagged_count", 1) == 0
    if not valid:
        raise ValueError("terminal_validation_failed")
    digest = sha256_file(candidate)
    publication = publish_primary(
        output, value, lambda p: dict(passed=sha256_file(p) == digest, receipts=receipts)
    )
    readers = reader_receipt(
        value["task_id"],
        output.parent,
        field="capstone_execution_ready_score",
        expected=value["capstone_execution_ready_score"],
    )
    if not readers["passed"]:
        raise ValueError("published_reader_drift")
    atomic_json(
        Path(value["terminal_validation_sidecar_path"]),
        dict(publication=publication, readers=readers, receipts=receipts, normal_process_exit=True),
    )


def main(argv: list[str] | None = None) -> int:
    """Freeze checks before the reduction child; private routes supply no science."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    progress("exp8069_start", pending="preconditions")
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20261003")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--durable", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--fixture-e2e", action="store_true")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args(argv)
    if args.cold_replay:
        try:
            errors = cold_replay(read(args.cold_replay))
        except (ValueError, KeyError, TypeError, OSError, StopIteration) as error:
            errors = [str(error)]
        print(json.dumps(dict(errors=errors)), flush=True)
        return int(bool(errors))
    if args.date != "20261003":
        raise ValueError("date_must_be_20261003")
    root = args.root.resolve()
    output = args.output or root / "results/experiment_8069_v698_capstone.json"
    durable = args.durable or output.parent / "raw" / output.stem / "invocations" / str(
        time.time_ns()
    )
    if args.worker:
        atomic_json(output, build(root, args.date, durable))
        return 0
    started = time.monotonic_ns()
    with tempfile.TemporaryDirectory(prefix="capstone8069-") as directory:
        private = Path(directory)
        frozen = manifest(private)
        worker = dict(
            name="reduction_normal_exit",
            expected_exit=0,
            deadline_s=300,
            argv=[
                str(ROOT / ".venv/bin/python"),
                "-u",
                str(ROOT / CLI),
                "--worker",
                "--root",
                str(root),
                "--date",
                args.date,
                "--durable",
                str(durable),
                "--output",
                str(private / "candidate.json"),
            ],
        )
        frozen["reduction_command"] = worker
        atomic_json(durable / "validation_manifest.json", frozen)
        progress("subprocess_before_reduction", pending="0/1 children")
        exit_receipt = run_check(ROOT, worker, private, durable / "validation_logs")
        progress("subprocess_after_reduction", 1, "owned validation")
        if not exit_receipt["passed"]:
            raise ValueError("reduction_child_failed")
        value = read(private / "candidate.json")
        receipts = [exit_receipt]
        if not args.fixture_e2e:
            for spec in frozen["commands"]:
                progress("subprocess_before_" + spec["name"], len(receipts), "owned checks")
                receipts.append(run_check(ROOT, spec, private, durable / "validation_logs"))
                progress("subprocess_after_" + spec["name"], len(receipts), "owned checks")
        report = read(private / "coverage.json")
        counts = {p: f["summary"] for p, f in report.get("files", {}).items()}
        complete(value, receipts, counts)
        if args.fixture_e2e:
            value.update(
                verifier_is_oracle=True,
                required_checks_passed=False,
                honest_verdict="complete_blocked_fixture_science",
                verdict_class="blocked",
            )
        else:
            publication = read(Path(receipts[1]["log_path"]))
            value.update(
                paper_ready=publication["paper_ready"],
                unmet_gates=publication["unmet_gates"],
                publication_gate_results=publication,
                **{f"g{i}": publication["gates"][f"G{i}"]["pass"] for i in range(1, 5)},
            )
        ended = time.monotonic_ns()
        value["duration_s"] = (ended - started) / 1e9
        value["phase_spans"] = [
            dict(
                phase="frozen_reduction_and_validation",
                start_s=0,
                end_s=value["duration_s"],
                completed_units=13,
            )
        ]
        value["current_work_receipt"] = build_current_work_receipt(
            run_id=str(started),
            owner_pid=os.getpid(),
            events=[],
            inference_substrate=value["inference_substrate"],
            inference_substrate_details=dict(model_work=False),
            inference_substrate_class="no_model_load",
            execution_venue="host",
            started_monotonic_ns=started,
            ended_monotonic_ns=ended,
        )
        value["checkpoint_references"].append(
            reference(durable / "validation_manifest.json", "command_freeze")
        )
        value["checkpoint_references"].extend(
            reference(Path(r["log_path"]), "owned_validation_log") for r in receipts
        )
        publish(value, output, private / "terminal", durable, fixture=args.fixture_e2e)
    progress("published_exp8069", 13, "none")
    return 0
