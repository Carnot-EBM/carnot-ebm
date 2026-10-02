"""REQ-REPORT-8030: terminal accounting keeps scientific gaps separate.

Immutable invocation records identify this milestone even after the live plan
changes. A correct blocked audit cannot grant readiness to failed producers.
"""

import argparse
import gzip
import json
from pathlib import Path
import tempfile
import time
from typing import Any

import yaml

from carnot.reporting import v694_capstone as previous
from carnot.reporting import v695_capstone_reduction as reduction
from carnot.reporting import v695_contract_methods as authority
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.evidence_features_custody_7980 import checked
from carnot.reporting.primary_publication import publish_primary, reader_receipt, read_bound_sidecar
from carnot.reporting.v686_contract_validation import run_check, dependency_hashes

ROOT = previous.ROOT
CLI = "scripts/experiments/experiment_8030_v695_capstone.py"
OWNED = [
    "python/carnot/reporting/v695_capstone.py",
    "python/carnot/reporting/v695_capstone_reduction.py",
    CLI,
]
TEST = "tests/python/test_experiment_8030_v695_capstone.py"
Json = dict[str, Any]
START = time.monotonic()
read = previous.old.read
reference = previous.old.reference


def progress(phase: str, units: int = 0, pending: str = "") -> None:
    """Real phase boundaries expose progress without inventing runtime evidence."""
    print(
        f"[exp8030] phase={phase} elapsed_s={time.monotonic() - START:.3f} "
        f"completed_units={units} pending={pending}",
        flush=True,
    )


def collect(root: Path, tasks: list[Json]) -> tuple[list[Json], list[Json], list[Json], list[Json]]:
    """Use unchanged gate, sidecar and primitive readers before independent arithmetic."""
    rows, refs, failures, _ = previous.collect(root, tasks)
    audits = []
    for row in rows:
        try:
            data = read(Path(row["path"]))
        except (ValueError, OSError):
            data = {}
        issues = row["gate_check_summary"]
        side = Path(data.get("terminal_validation_sidecar_path") or "/missing")
        saved = read(side)
        if side.is_dir() or "publication" in saved:
            bound = (
                Path(saved["publication"]["sidecar_path"])
                if "publication" in saved
                else side / (row["sha256"][7:] + ".json")
            )
            refs.append(reference(bound, "terminal_publication_sidecar"))
            try:
                report = read_bound_sidecar(Path(row["path"]), bound)
                if report["report"]["passed"] is True:
                    for issue in issues[:]:
                        if issue["artifact_field"] in {
                            "terminal_primary_sha256",
                            "terminal_validation.passed",
                        }:
                            issues.remove(issue)
                            failures.remove(issue)
            except (ValueError, KeyError, OSError):
                pass
        for receipt in data.get("validation_receipts", []):
            if (
                receipt.get("classification") not in {"diagnostic", "repository_health"}
                and receipt.get("scope") != "repository_health"
                and receipt.get("required") is not False
                and receipt.get("passed") is not True
            ):
                item = dict(
                    previous.old.operand(
                        Path(row["path"]),
                        row["task_id"],
                        "validation_receipts." + receipt["name"] + ".passed",
                        True,
                        receipt.get("passed"),
                    ),
                    passed=False,
                    check_result="failed",
                    original_receipt=receipt,
                )
                issues.append(item)
                failures.append(item)
        try:
            result = reduction.independent(data, int(row["task_id"][3:7]))
        except (ValueError, KeyError, TypeError, OSError) as error:
            result = dict(
                measurement_available=False,
                reduction_error=str(error),
                primary=reduction.bootstrap([]),
            )
            item = dict(
                previous.old.operand(
                    Path(row["path"]),
                    row["task_id"],
                    "independent_reduction",
                    "valid original primitives",
                    str(error),
                ),
                passed=False,
                check_result="failed",
            )
            issues.append(item)
            failures.append(item)
        if row["verdict_class"] in {"blocked", "disqualified"}:
            item = dict(
                previous.old.operand(
                    Path(row["path"]),
                    row["task_id"],
                    "verdict_class",
                    ["positive", "null", "circular_positive"],
                    row["verdict_class"],
                ),
                passed=False,
                check_result="failed",
            )
            issues.append(item)
            failures.append(item)
        row.update(
            eligible=not issues and row["verdict_class"] not in {"blocked", "disqualified"},
            excluded=bool(issues),
            numerator=int(not issues),
            raw_numerator=int(not issues),
            exclusion="failed_prerequisite_or_custody" if issues else None,
        )
        audits.append(dict(task_id=row["task_id"], **result))
        progress("independent_task_reduced", len(audits), str(12 - len(audits)))
    return rows, refs, failures, audits


def scientific(audits: list[Json], rows: list[Json]) -> list[Json]:
    """Failed producer gates retain descriptive effects but receive no claim credit."""
    by = {int(r["task_id"][3:7]): r for r in audits}
    states = {int(r["task_id"][3:7]): r for r in rows}
    primary = [by[n].get("primary", reduction.bootstrap([])) for n in (8021, 8024, 8026)]
    qualified = [
        states[n]["eligible"] and by[n]["measurement_available"] for n in (8021, 8024, 8026)
    ]
    qualified[1] = qualified[1] and by[8024].get("diagnostic", {}).get("support_passed", False)
    static_gates = by[8021].get("diagnostic", {}).get("acceptance_gate_results", {})
    qualified[0] = qualified[0] and all(
        static_gates.get(k) is True
        for k in (
            "support",
            "headroom",
            "changed_actions",
            "brier_noninferiority",
            "no_additional_false_accepts",
        )
    )
    return reduction.family(primary, qualified)


def build(root: Path, date: str, durable: Path) -> Json:
    """Freeze authority, roles, methods and exclusions before reducing any targets."""
    progress("freeze_before", pending="immutable invocation authority")
    invocation = read(root / "results/experiment_8018_v695_contract_methods.json")
    snapshots = invocation["authority_snapshots"]
    active = Path(snapshots["active"]["snapshot_path"])
    design = Path(snapshots["design"]["snapshot_path"])
    for s in snapshots.values():
        if s["exists"]:
            checked(dict(path=s["snapshot_path"], sha256=s["sha256"]))
    tasks = yaml.safe_load(active.read_bytes())["tasks"]
    if date != "20261002" or [r["id"].split("-")[0] for r in tasks] != [
        f"exp{n}" for n in range(8018, 8031)
    ]:
        raise ValueError("date_or_thirteen_task_roster_changed")
    assessment = authority.assess(
        design, root / "consumed-staging-absent", active, durable / "authority"
    )
    freeze = dict(
        task_contract=tasks,
        canonical_tasks_sha256=previous.old.authority.lifecycle.tasks_digest(tasks),
        method_freeze=invocation.get("method_freeze"),
        decision_costs=reduction.static.CONFIG["costs"],
        source_support_floors=dict(independent=72, per_class=12),
        roles="original source-disjoint role slots; historical development exposure",
        budgets=dict(tasks=13, pretrained_model_calls=0, bootstrap_draws=10000),
        exclusions=reference(root / "ops/exclusion_manifest.yaml", "exclusions"),
        gates="original bytes, seals, complete targets and class floors; producer owned checks; Holm three primaries",
    )
    freeze_path = durable / "method_freeze.json"
    atomic_json(freeze_path, freeze)
    progress("freeze_after", 13, "terminal task readers")
    rows, refs, failures, audits = collect(root, tasks)
    failures += [
        dict(r, passed=False, check_result="failed") for r in assessment["gate_check_summary"]
    ]
    primary = scientific(audits, rows)
    blocked = bool(failures)
    state = "blocked" if blocked else "null"
    verdict = "complete_" + state + "_v695_capstone"
    gaps = {
        "useful_oracle_distinct_decisions": dict(
            requirements=["FR-12"],
            closed=False,
            decision="static_comparison_disqualified_and_source_test_unmeasured",
            upstream_ids=[8021, 8023, 8024],
            reopen_condition="Qualified complete source evaluation and static owned checks; beat strongest same-information controls with cost and Brier safety.",
        ),
        "durable_self_learning": dict(
            requirements=["FR-11"],
            closed=False,
            decision="trajectory_measured_retention_audit_disqualified",
            upstream_ids=[8025, 8026],
            reopen_condition="Freeze retention before any label preview; equal-budget updates improve later decisions and retained targets after restart.",
        ),
        "validated_affordable_deployment": dict(
            requirements=["FR-05", "FR-08", "NFR-01"],
            closed=False,
            decision="no_qualified_complete_service_speedup",
            upstream_ids=[8027, 8029],
            reopen_condition="Qualified useful workload plus complete acquisition, scoring, update, durable storage and restart timing; board custody cannot replace service evidence.",
        ),
    }
    own = dict(
        task_id=tasks[-1]["id"],
        id=tasks[-1]["id"],
        unit_id=tasks[-1]["id"],
        arm="task_disposition",
        metric="owned_validation",
        path=None,
        sha256=None,
        honest_verdict=verdict,
        verdict_class=state,
        producer_status="pending_owned_validation",
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
        exclusion=None,
        censor_reason=None,
    )
    rows.append(own)
    receipt = previous.old.build_current_work_receipt(
        run_id="exp8030-" + date,
        owner_pid=__import__("os").getpid(),
        events=[],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        inference_substrate_details=dict(
            work="CPU cached primitive and checkpoint reduction; no fitting"
        ),
        execution_venue="host",
        started_monotonic_ns=time.monotonic_ns(),
        ended_monotonic_ns=time.monotonic_ns(),
    )
    receipt["started_monotonic_timestamp_ns"] = receipt.pop("started_monotonic_ns")
    receipt["ended_monotonic_timestamp_ns"] = receipt.pop("ended_monotonic_ns")
    value = dict(
        receipt,
        experiment_id=8030,
        task_id=tasks[-1]["id"],
        milestone="2026.10.695",
        run_date=date,
        execution_date=date,
        schema="carnot.v695.capstone.v1",
        honest_verdict=verdict,
        verdict_class=state,
        gate_check_summary=failures,
        rows=rows,
        task_dispositions=rows,
        task_contract=tasks,
        input_root=str(root),
        independent_reduction_rows=audits,
        primary_hypothesis_results=primary,
        source_artifact_hashes=refs,
        cited_upstream_artifacts=[
            r for r in refs if r["role"] in {"present", "conductor_skip_receipt", "missing"}
        ],
        raw_shard_hashes=[r for r in refs if r["role"] == "primitive_shard"],
        checkpoint_references=[reference(freeze_path, "method_freeze")],
        code_config_hashes=[reference(ROOT / p, "current_code") for p in OWNED],
        authority_snapshots=assessment["authority_snapshots"],
        authority_rows=assessment["contract_rows"],
        canonical_tasks_sha256=freeze["canonical_tasks_sha256"],
        model_specs=[],
        trained_head_specs=[],
        model_invocation_counts=receipt["invocation_counts"],
        random_seed=6958030,
        verifier_is_oracle=False,
        claim_scope=dict(
            current="thirteen terminal dispositions; exposed development arithmetic",
            independent_benefit=False,
            source="likelihood information differs from energy architecture; absent evaluation stays unmeasured",
            historical="board, inference and publication claims retain original limits",
        ),
        positive_control_results=reduction.learning.controls(),
        genuine_headroom=dict(
            static=audits[3].get("diagnostic", {}).get("genuine_headroom"),
            scope="exposed development baseline errors; no generalized credit",
        ),
        acceptance_gate_results=dict(
            validity=False,
            decision_benefit=False,
            durable_learning=False,
            affordable_deployment=False,
            readiness=0,
        ),
        capstone_execution_ready_score=0,
        gap_decisions=gaps,
        science_ready=False,
        paper_ready=False,
        g1=False,
        g2=False,
        g3=False,
        g4=False,
        unmet_gates=["publication_gate_pending"],
        generalized_learning_benefit_score=0,
        retirement_rows=previous.old.retirements(tasks, rows),
        reopen_conditions=[g["reopen_condition"] for g in gaps.values()],
        flagged_adversarial=False,
        validation_receipts=[],
        coverage_statement_counts={},
        terminal_validation_sidecar_path=str(durable / "terminal_validation.json"),
        sample_size_budget=dict(
            intended=13,
            eligible=sum(r["eligible"] for r in rows),
            started=sum(r["started"] for r in rows),
            completed=12,
            excluded=sum(r["excluded"] for r in rows),
            failed=sum(r["failed"] for r in rows),
            censored=0,
            independent=0,
            unit="administrative_task_disposition",
        ),
    )
    previous.authenticate_retirements(root, value["retirement_rows"])
    for retirement in value["retirement_rows"]:
        if retirement["prior_path"]:
            refs.append(reference(Path(retirement["prior_path"]), "prior_verdict"))
    historical = read(root / "results/experiment_8017_v694_capstone.json")
    value["historical_required_failures"] = []
    for old_receipt in historical.get("validation_receipts", []):
        if old_receipt.get("passed") is False:
            source = root / old_receipt["log_path"]
            saved = (
                durable
                / "historical"
                / (old_receipt["name"] + "-" + sha256_file(source)[7:] + ".log")
            )
            saved.parent.mkdir(parents=True, exist_ok=True)
            saved.write_bytes(source.read_bytes())
            value["checkpoint_references"].append(reference(saved, "historical_failed_check"))
            value["historical_required_failures"].append(
                dict(
                    old_receipt,
                    original_primary_sha256=sha256_file(
                        root / "results/experiment_8017_v694_capstone.json"
                    ),
                    frozen_log_path=str(saved),
                    log_authenticated=sha256_file(saved) == old_receipt["log_sha256"],
                    resolved=False,
                )
            )
    value["reproducibility_checksum"] = canonical_hash(
        dict(sources=refs, freeze=freeze, code=value["code_config_hashes"])
    )
    value["independent_reduction_sha256"] = canonical_hash(audits)
    summaries = []
    for audit in audits:
        path = durable / "reductions" / (audit["task_id"] + ".json")
        atomic_json(path, audit)
        value["checkpoint_references"].append(reference(path, "independent_reduction"))
        diagnostic = audit.get("diagnostic", {})
        summaries.append(
            dict(
                task_id=audit["task_id"],
                measurement_available=audit["measurement_available"],
                reduction_error=audit.get("reduction_error"),
                path=str(path),
                sha256=sha256_file(path),
                primary={
                    k: v for k, v in audit.get("primary", {}).items() if k != "centered_errors"
                },
                support=diagnostic.get("retention_support"),
                acceptance_gate_results=diagnostic.get("acceptance_gate_results"),
                retention_drift_rows=diagnostic.get("retention_drift_rows"),
            )
        )
    value["independent_reduction_rows"] = summaries
    return value


def cold_replay(value: Json) -> list[str]:
    """A fresh process reopens every durable primitive and redoes scientific arithmetic."""
    progress("cold_replay_before", pending="durable primitive reduction")
    for ref in (
        value["source_artifact_hashes"]
        + value["code_config_hashes"]
        + value["checkpoint_references"]
    ):
        path = Path(ref["path"])
        if (sha256_file(path) if path.is_file() else None) != ref["sha256"]:
            return ["source_bytes_changed"]
        if ref.get("role") == "command_freeze":
            for name, digest in read(path)["dependency_hashes"].items():
                if sha256_file(ROOT / name) != digest:
                    return ["code_configuration_changed"]
    active = value["authority_snapshots"]["active"]
    checked(dict(path=active["snapshot_path"], sha256=active["sha256"]))
    tasks = yaml.safe_load(Path(active["snapshot_path"]).read_bytes())["tasks"]
    rows, _, _, audits = collect(Path(value["input_root"]), tasks)
    errors = []
    if (
        rows != value["task_dispositions"][:-1]
        or canonical_hash(audits) != value["independent_reduction_sha256"]
        or scientific(audits, rows) != value["primary_hypothesis_results"]
    ):
        errors.append("independent_reduction_drift")
    if value["science_ready"] is not False or value["generalized_learning_benefit_score"] != 0:
        errors.append("unsupported_generalization")
    if (
        value["verdict_class"] in {"blocked", "disqualified"}
        and value["capstone_execution_ready_score"]
    ):
        errors.append("unsafe_readiness")
    if previous.old.authority.lifecycle.tasks_digest(tasks) != value["canonical_tasks_sha256"]:
        errors.append("authority_drift")
    seal = next(r for r in value["checkpoint_references"] if r["role"] == "claim_seal")
    if read(Path(seal["path"])) != claim_payload(value):
        errors.append("claim_seal_drift")
    progress("cold_replay_after", 13, "none")
    return errors


def manifest(private: Path) -> Json:
    """Reuse the existing frozen validation plan with coverage only for current code."""
    value = previous.manifest(private)
    for spec in value["commands"]:
        spec["argv"] = [
            a.replace(previous.CLI, CLI)
            .replace(previous.TEST, TEST)
            .replace(previous.OWNED[0], OWNED[0])
            for a in spec["argv"]
        ]
        spec["argv"] = [
            "--include=" + ",".join(OWNED) if a.startswith("--include=") else a
            for a in spec["argv"]
        ]
        if spec["name"] in {"ruff_check", "ruff_format", "strict_mypy", "spec_coverage"}:
            spec["argv"].append(OWNED[1])
    value.update(
        dependency_hashes=dependency_hashes(ROOT, paths=OWNED + [TEST]),
        coverage_includes=OWNED,
    )
    return value


def claim_payload(value: Json) -> Json:
    """Bind all claim fields while keeping the seal's own reference out of its bytes."""
    return {
        k: v for k, v in value.items() if k not in {"checkpoint_references", "field_principles"}
    }


def apply_checks(value: Json, receipts: list[Json], counts: Json) -> None:
    """Owned correctness completes accounting; blocked science still has readiness zero."""
    required = [r for r in receipts if r.get("classification") != "diagnostic"]
    covered = all(
        p in counts
        and counts[p]["num_statements"] > 0
        and counts[p]["num_statements"] == counts[p]["covered_lines"]
        for p in OWNED
    )
    valid = bool(required) and all(r["passed"] for r in required) and covered
    value.update(
        validation_receipts=required,
        coverage_statement_counts=counts,
        required_checks_passed=valid,
        repository_health=[r for r in receipts if r.get("classification") == "diagnostic"],
    )
    if not valid:
        value.update(
            verdict_class="disqualified", honest_verdict="complete_disqualified_v695_owned_checks"
        )
    value["acceptance_gate_results"].update(
        validity=valid, readiness=int(valid and value["verdict_class"] == "null")
    )
    value["capstone_execution_ready_score"] = value["acceptance_gate_results"]["readiness"]
    previous.old.terminal_disposition(value)
    previous.authenticate_retirements(Path(value["input_root"]), value["retirement_rows"])
    value["rows"][-1]["numerator"] = int(valid)


def publish(value: Json, output: Path, private: Path, durable: Path) -> None:
    """Existing terminal consumers inspect candidate and published final bytes."""
    candidate = private / output.name
    atomic_json(candidate, value)
    specs = [
        ("cold_replay", [str(ROOT / CLI), "--cold-replay", str(candidate)]),
        ("adversarial", ["scripts/adversarial_verify.py", "--json", str(candidate)]),
        ("strict_rows", ["scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)]),
    ]
    receipts = []
    for prefix, target in (("candidate", candidate), ("published", output)):
        for name, args in specs:
            progress("subprocess_before_" + prefix + "_" + name, len(receipts), "terminal checks")
            argv = [
                str(ROOT / ".venv/bin/python"),
                *[str(target) if a == str(candidate) else a for a in args],
            ]
            receipt = run_check(
                ROOT,
                dict(name=prefix + "_" + name, argv=argv, expected_exit=0, deadline_s=180),
                private,
                durable / "logs",
            )
            receipts.append(receipt)
            progress("subprocess_after_" + prefix + "_" + name, len(receipts), "terminal checks")
        if not all(r["passed"] for r in receipts) or read(Path(receipts[-2]["log_path"])).get(
            "flagged_count", 0
        ):
            atomic_json(durable / "failed_terminal.json", dict(receipts=receipts))
            raise ValueError("terminal_validation_failed")
        if prefix == "candidate":
            digest = sha256_file(candidate)
            published = publish_primary(
                output, value, lambda p: dict(passed=sha256_file(p) == digest, receipts=receipts[:])
            )
            selected = reader_receipt(
                value["task_id"],
                output.parent,
                field="capstone_execution_ready_score",
                expected=value["capstone_execution_ready_score"],
            )
            if not selected["passed"] or selected["gate_sha256"] != digest:
                raise ValueError("published_reader_drift")
            atomic_json(
                Path(value["terminal_validation_sidecar_path"]),
                dict(passed=True, **published, readers=selected, receipts=receipts[:]),
            )
    atomic_json(
        durable / "published_recheck.json",
        dict(primary_sha256=sha256_file(output), receipts=receipts),
    )


def prepare_fixture(root: Path) -> None:
    """Private authority and terminal bytes test readers without natural measurements."""
    root.mkdir(parents=True, exist_ok=True)
    for name in ("design.md", "active.yaml"):
        (root / name).write_bytes(
            gzip.decompress((ROOT / "tests/fixtures/v695" / (name + ".gz")).read_bytes())
        )
    tasks = yaml.safe_load((root / "active.yaml").read_bytes())["tasks"]
    fields: Json = {t["id"]: {} for t in tasks}
    for task in tasks:
        for gate in task.get("gated_on", []):
            if gate["artifact_field"] not in {"verdict_class", "flagged_adversarial"}:
                fields[gate["upstream"]][gate["artifact_field"]] = gate["value"]
    snapshots = {
        k: dict(exists=True, snapshot_path=str(root / n), sha256=sha256_file(root / n))
        for k, n in (("active", "active.yaml"), ("design", "design.md"))
    }
    for task in tasks[:-1]:
        path = root / task["deliverable"]
        side = root / "sidecars" / (task["id"] + ".json")
        atomic_json(
            path,
            dict(
                experiment_id=int(task["id"][3:7]),
                task_id=task["id"],
                honest_verdict="complete_null_private",
                verdict_class="null",
                rows=[],
                flagged_adversarial=False,
                terminal_validation_sidecar_path=str(side),
                **fields[task["id"]],
                **(dict(authority_snapshots=snapshots) if task == tasks[0] else {}),
            ),
        )
        atomic_json(side, dict(passed=True, primary_sha256=sha256_file(path)))


def main(argv: list[str] | None = None) -> int:
    """Run bounded owned checks before publication; replay uses the same raw readers."""
    progress("cli_start", pending="arguments")
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20261002")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fixture-e2e", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        try:
            errors = cold_replay(read(args.cold_replay))
        except (ValueError, KeyError, TypeError, OSError) as error:
            errors = [str(error)]
        print(json.dumps(dict(errors=errors)), flush=True)
        return int(bool(errors))
    if args.date != "20261002":
        raise ValueError("date_must_be_20261002")
    root = args.root.resolve()
    output = args.output or root / "results/experiment_8030_v695_capstone.json"
    durable = output.parent / "raw" / output.stem
    started = time.monotonic()
    started_ns = time.monotonic_ns()
    with tempfile.TemporaryDirectory(prefix="exp8030-owned-") as directory:
        private = Path(directory)
        frozen = manifest(private)
        atomic_json(durable / "validation_manifest.json", frozen)
        value = build(root, args.date, durable)
        receipts = []
        counts = {}
        if args.fixture_e2e:
            receipts = [
                dict(name="private_fixture", passed=True, argv=["private_fixture"], actual_exit=0)
            ]
            counts = {p: dict(num_statements=1, covered_lines=1) for p in OWNED}
            value["claim_scope"]["current"] = "private circular reader control only"
        else:
            for spec in frozen["commands"]:
                progress("subprocess_before_" + spec["name"], len(receipts), "owned validation")
                receipts.append(run_check(ROOT, spec, private, durable / "validation_logs"))
                progress(
                    "subprocess_after_" + spec["name"], len(receipts), "remaining owned checks"
                )
            report = read(private / "coverage.json")
            counts = {p: f["summary"] for p, f in report.get("files", {}).items()}
            publication = read(Path(receipts[0]["log_path"]))
            value.update(
                publication_gate_results=publication,
                paper_ready=publication["paper_ready"],
                unmet_gates=publication["unmet_gates"],
                **{f"g{i}": publication["gates"][f"G{i}"]["pass"] for i in range(1, 5)},
            )
        apply_checks(value, receipts, counts)
        value["duration_s"] = time.monotonic() - started
        value["started_monotonic_timestamp_ns"] = started_ns
        value["ended_monotonic_timestamp_ns"] = time.monotonic_ns()
        value["phase_spans"] = [
            dict(
                phase="frozen_authority_reduction_and_owned_validation",
                start_s=0,
                end_s=value["duration_s"],
                completed_units=13,
            )
        ]
        value["checkpoint_references"].append(
            reference(durable / "validation_manifest.json", "command_freeze")
        )
        value["field_principles"] = {
            k: "Exact bytes bind current work; owned correctness, exposed science and historical publication remain separate."
            for k in value
        }
        seal_path = durable / "claim_seal.json"
        atomic_json(seal_path, claim_payload(value))
        value["checkpoint_references"].append(reference(seal_path, "claim_seal"))
        if args.fixture_e2e:
            atomic_json(output, value)
        else:
            publish(value, output, private / "terminal", durable)
    progress("published_final_bytes", 13, "none")
    return 0
