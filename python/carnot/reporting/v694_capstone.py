"""REQ-REPORT-8017: retain exact V694 evidence without transferring science credit.

This invocation reads cached bytes. Historical model work and fitted heads
remain upstream evidence, so they cannot inflate current work or sample counts.
"""

from collections import Counter
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import yaml

from carnot.reporting import v693_capstone as old
from carnot.reporting import v693_capstone_reduction as reduction
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v686_contract_validation import run_check
from carnot.verify.qwen_completion_7932 import parse_response
from scripts.conductor_gates import _eval_op
from scripts.experiments.experiment_8005_v694_contract_methods import assess

ROOT = old.ROOT
CLI = "scripts/experiments/experiment_8017_v694_capstone.py"
OWNED = ["python/carnot/reporting/v694_capstone.py", CLI]
TEST = "tests/python/test_experiment_8017_v694_capstone.py"
Json = dict[str, Any]
START = time.monotonic()


def progress(phase: str, units: int = 0, pending: str = "") -> None:
    """Visible boundaries expose real work without inventing heartbeat evidence."""
    print(
        f"[exp8017] phase={phase} elapsed_s={time.monotonic() - START:.3f} "
        f"completed_units={units} pending={pending}",
        flush=True,
    )


def reduce_primitives(data: Json, number: int) -> Json:
    """Recompute operands from rows and original responses before reading summaries."""
    result: Json = dict(measurement_available=bool(data.get("rows")))
    if number == 8009:
        result.update(
            static=reduction.score(data.get("rows", []), {}),
            action_counts=dict(
                Counter(
                    r.get("action", r.get("decision", "escalate")) for r in data.get("rows", [])
                )
            ),
        )
    if number in (8012, 8013):
        result["learning"] = reduction.audit(data, {})
    if number == 8011 and data.get("rows"):
        checkpoints = data.get("checkpoint_references", [])
        raw = [old.read(Path(r["path"])) for r in checkpoints]
        if len(raw) != len(data["rows"]):
            raise ValueError("source_checkpoint_roster_drift")
        groups: Json = {}
        for index, (row, saved) in enumerate(zip(data["rows"], raw, strict=True)):
            if index % 32 == 0:
                progress("source_response_reduction", index, str(len(raw) - index))
            if any(row.get(k) != v for k, v in saved.items()):
                raise ValueError("source_probability_drift")
            if (
                "parsed" in saved
                and parse_response(saved["raw_response"], saved["visible_ids"]) != saved["parsed"]
            ):
                raise ValueError("source_parse_drift")
            if (
                row.get("status", "generated") != "generated"
                or row.get("parsed", {}).get("completed") is False
            ):
                continue
            response = json.loads(saved["raw_response"]["choices"][0]["message"]["content"])
            probability = response["unsupported_probability"]
            if probability != row["probability"] or not 0 <= probability <= 1:
                raise ValueError("source_probability_drift")
            cells = groups.setdefault(row["group_id"], {})
            if row["arm"] in cells:
                raise ValueError("source_duplicate_arm")
            cells[row["arm"]] = probability
        contrasts = []
        for group, cells in groups.items():
            if set(cells) != {"original", "swap", "duplicate"}:
                continue
            contrasts.append(
                dict(
                    group_id=group,
                    complete=True,
                    swap_minus_original=cells["swap"] - cells["original"],
                    duplicate_minus_original=cells["duplicate"] - cells["original"],
                    swap_minus_duplicate=cells["swap"] - cells["duplicate"],
                )
            )
        producer = {r["group_id"]: dict(r) for r in data.get("sensitivity_rows", [])}
        for row in data.get("duplicate_control_rows", []):
            producer.setdefault(row["group_id"], {}).update(row)
        for row in contrasts:
            if row["group_id"] not in producer or any(
                not math.isclose(row[k], producer[row["group_id"]][k], abs_tol=1e-12)
                for k in ("swap_minus_original", "duplicate_minus_original", "swap_minus_duplicate")
            ):
                raise ValueError("source_summary_drift")
        result["source"] = dict(
            independent_sources=len(contrasts),
            intended_rows=len(raw),
            completed_triplets=len(contrasts),
            contrasts=contrasts,
            mean_swap_minus_duplicate=sum(r["swap_minus_duplicate"] for r in contrasts)
            / len(contrasts)
            if contrasts
            else None,
            source_dependence_only=True,
            closes_deployment_gap=False,
        )
    return result


def collect(root: Path, tasks: list[Json]) -> tuple[list[Json], list[Json], list[Json], list[Json]]:
    """Exact skip receipts explain absence; they never provide scientific primitives."""
    rows, refs, failures, audits = [], [], [], []
    loaded = []
    for task in tasks[:-1]:
        path = root / task["deliverable"]
        skip = root / (
            "results/experiment_"
            + task["id"][3:7]
            + "_"
            + task["id"].split("-", 1)[1].replace("-", "_")
            + ".json"
        )
        status = (
            "present"
            if path.is_file()
            else "conductor_skip_receipt"
            if skip.is_file()
            else "missing"
        )
        selected = skip if status == "conductor_skip_receipt" else path
        try:
            data = old.read(selected)
        except (ValueError, OSError):
            data = dict(
                verdict_class="disqualified",
                honest_verdict="complete_disqualified_malformed_producer",
            )
        loaded.append((data, path, selected, status))
    by_id = {t["id"]: item for t, item in zip(tasks[:-1], loaded, strict=True)}
    for index, (task, (data, path, selected, status)) in enumerate(
        zip(tasks[:-1], loaded, strict=True)
    ):
        issues = []

        def fail(
            field: str,
            expected: Any,
            observed: Any,
            source: Path = selected,
            upstream: str = task["id"],
            op: str = "==",
        ) -> None:
            issues.append(
                dict(
                    old.operand(source, upstream, field, expected, observed),
                    op=op,
                    passed=False,
                    check_result="failed",
                )
            )

        for gate in task.get("gated_on", []):
            upstream_data, upstream_path, _, _ = by_id[gate["upstream"]]
            actual = upstream_data.get(gate["artifact_field"], "contract_error_missing_field")
            if not _eval_op(actual, gate["op"], gate["value"])[0]:
                fail(
                    gate["artifact_field"],
                    gate["value"],
                    actual,
                    upstream_path,
                    gate["upstream"],
                    gate["op"],
                )
        state = data.get("verdict_class", "blocked")
        verdict = data.get("honest_verdict", "complete_blocked_missing_declared_producer")
        if status != "present":
            state, verdict = "blocked", "complete_blocked_" + status
            fail("declared_primary_present", True, False, path)
        elif data.get("flagged_adversarial") is not False:
            fail(
                "flagged_adversarial",
                False,
                data.get("flagged_adversarial", "contract_error_missing_field"),
            )
        refs.append(old.reference(selected, status, list(data)))
        if status == "present":
            side = Path(data.get("terminal_validation_sidecar_path") or "/missing")
            saved = old.read(side)
            refs.append(old.reference(side, "terminal_sidecar"))
            if saved.get("primary_sha256", saved.get("candidate_sha256")) != sha256_file(path):
                fail(
                    "terminal_primary_sha256", sha256_file(path), saved.get("primary_sha256"), side
                )
            terminal = saved
            if saved.get("sidecar_path"):
                side = Path(saved["sidecar_path"])
                terminal = old.read(side)
                refs.append(old.reference(side, "terminal_validator"))
            terminal_passed = terminal.get("passed", terminal.get("report", {}).get("passed"))
            if terminal_passed is not True:
                fail("terminal_validation.passed", True, terminal_passed, side)
            for item in data.get("gate_check_summary", []):
                if isinstance(item, dict) and item.get("passed") is False:
                    issues.append(
                        dict(
                            item,
                            path=item.get("path", item.get("artifact_path")),
                            hash=item.get("hash", item.get("artifact_hash")),
                            op=item.get("op", "=="),
                            upstream_id=item.get("upstream_id", task["id"]),
                            check_result="failed",
                        )
                    )
        shards = data.get("raw_shard_hashes", [])
        if isinstance(shards, dict):
            shards = [dict(path=p, sha256=h) for p, h in shards.items()]
        primitives = shards + data.get("checkpoint_references", [])
        for ref in primitives:
            p = Path(ref["path"])
            refs.append(old.reference(p, "primitive_shard"))
            if refs[-1]["sha256"] != ref["sha256"]:
                fail("primitive_sha256", ref["sha256"], refs[-1]["sha256"], p)
        try:
            audit = reduce_primitives(data, int(task["id"][3:7]))
        except (ValueError, KeyError, TypeError, OSError) as error:
            audit = dict(measurement_available=False, reduction_error=str(error))
            fail("independent_reduction", "valid", str(error))
        audits.append(dict(task_id=task["id"], **audit))
        rows.append(
            dict(
                task_id=task["id"],
                id=task["id"],
                unit_id=task["id"],
                arm="task_disposition",
                metric="qualified_disposition",
                path=str(selected),
                sha256=sha256_file(selected) if selected.is_file() else None,
                verdict_class=state,
                honest_verdict=verdict,
                producer_honest_verdict=data.get("honest_verdict"),
                producer_status=status,
                started=status == "present",
                completed=True,
                eligible=not issues and state not in {"blocked", "disqualified"},
                excluded=bool(issues),
                failed=state == "disqualified",
                censored=False,
                status="completed",
                raw_numerator=int(not issues),
                raw_denominator=1,
                numerator=int(not issues),
                denominator=1,
                exclusion="failed_prerequisite_or_custody" if issues else None,
                censor_reason=None,
                gate_check_summary=issues,
                producer_gate_check_summary=data.get("gate_check_summary", []),
            )
        )
        failures.extend(issues)
        progress("task_reduced", index + 1, str(12 - index - 1))
    return rows, refs, failures, audits


def build(root: Path, design: Path, date: str, durable: Path) -> Json:
    """Freeze roles and gates before labels; administrative completion supplies no benefit."""
    started = time.monotonic_ns()
    progress("freeze_authority", pending="methods and roles")
    active = root / "research-roadmap.yaml"
    tasks = yaml.safe_load(active.read_bytes())["tasks"]
    if date != "20261002" or [t["id"].split("-")[0] for t in tasks] != [
        f"exp{n}" for n in range(8005, 8018)
    ]:
        raise ValueError("date_or_thirteen_task_roster_changed")
    authority = assess(design, root / "research-roadmap-next.yaml", active, durable / "authority")
    frozen = dict(
        task_contract=tasks,
        canonical_tasks_sha256=old.authority.lifecycle.tasks_digest(tasks),
        role_hashes={t["id"]: canonical_hash(t) for t in tasks},
        methods="source means; exact fixed costs; checkpoint response contrasts; causal update equations",
        cost_matrix=dict(false_accept=5, false_reject=1, escalation=0.25),
        budgets=dict(task_dispositions=13, independent_current_scientific_units=0),
        acceptance_gates="exact declared fields, final byte custody, all owned checks; no generalization transfer",
    )
    atomic_json(durable / "method_freeze.json", frozen)
    rows, refs, failures, audits = collect(root, tasks)
    for item in authority["gate_check_summary"]:
        failures.append(
            dict(item, upstream_id="V694_authority", passed=False, check_result="failed")
        )
    state = (
        "blocked"
        if failures or any(r["verdict_class"] in {"blocked", "disqualified"} for r in rows)
        else "null"
    )
    verdict = (
        "complete_blocked_v694_prerequisites"
        if state == "blocked"
        else "complete_null_v694_development_only"
    )
    gaps = {
        "useful_oracle_distinct_decisions": dict(
            closed=False,
            decision="blocked_static_measurement",
            reopen_condition="New source-disjoint targets with both calibration classes and measurable decision headroom; beat the strongest same-information control.",
        ),
        "durable_self_learning": dict(
            closed=False,
            decision="blocked_updates_and_retention",
            reopen_condition="Qualified initial head and equal-budget causal updates improve future decisions and unexposed retention after restart.",
        ),
        "validated_affordable_deployment": dict(
            closed=False,
            decision="no_qualified_update_or_device_custody",
            reopen_condition="Useful independent workload plus authenticated board execution with complete acquisition, update and durable service costs.",
        ),
    }
    for gap in gaps.values():
        gap.update(
            next_action="Stop unchanged no-headroom extensions; collect new information only against the stated falsifiable gate.",
            scientific_benefit=False,
        )
    own = dict(
        task_id=tasks[-1]["id"],
        id=tasks[-1]["id"],
        unit_id=tasks[-1]["id"],
        path=None,
        sha256=None,
        arm="task_disposition",
        metric="owned_validation",
        numerator=0,
        denominator=1,
        raw_numerator=0,
        raw_denominator=1,
        honest_verdict=verdict,
        verdict_class=state,
        producer_status="pending_owned_validation",
        started=True,
        completed=False,
        eligible=False,
        failed=False,
        excluded=False,
        censored=False,
        exclusion=None,
        censor_reason=None,
        status="pending",
    )
    rows.append(own)
    value = old.build_current_work_receipt(
        run_id="exp8017-" + date,
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        inference_substrate_details=dict(
            work="cached primitive arithmetic; no fitted head scoring"
        ),
        execution_venue="host",
        started_monotonic_ns=started,
        ended_monotonic_ns=time.monotonic_ns(),
    )
    atomic_json(durable / "current_work_receipt.json", value)
    value["started_monotonic_timestamp_ns"] = value.pop("started_monotonic_ns")
    value["ended_monotonic_timestamp_ns"] = value.pop("ended_monotonic_ns")
    value.update(
        experiment_id=8017,
        task_id=tasks[-1]["id"],
        milestone="2026.10.694",
        run_date=date,
        execution_date=date,
        schema="v694_capstone_v1",
        honest_verdict=verdict,
        verdict_class=state,
        gate_check_summary=failures,
        rows=rows,
        task_dispositions=rows,
        independent_reduction_rows=audits,
        source_artifact_hashes=refs,
        cited_upstream_artifacts=[
            r for r in refs if r["role"] in {"present", "conductor_skip_receipt", "missing"}
        ],
        raw_shard_hashes=[r for r in refs if r["role"] == "primitive_shard"],
        checkpoint_references=[
            old.reference(durable / n, "current_checkpoint")
            for n in ("method_freeze.json", "current_work_receipt.json")
        ],
        code_config_hashes=[old.reference(ROOT / p, "current_code") for p in OWNED],
        authority_snapshots=authority["authority_snapshots"],
        authority_activated=authority["activated"],
        task_contract=tasks,
        canonical_tasks_sha256=frozen["canonical_tasks_sha256"],
        input_root=str(root),
        model_specs=[],
        model_invocation_counts=value["invocation_counts"],
        trained_head_specs=[],
        random_seed=6948017,
        verifier_is_oracle=False,
        claim_scope=dict(
            current="thirteen terminal dispositions and finite development arithmetic",
            historical_models="citations only",
            source_sensitivity="information dependence, not mitigation or deployment",
            retention="exposed development evidence",
            independent_benefit=False,
        ),
        positive_control_results=reduction.controls(),
        genuine_headroom=False,
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
        retirement_rows=old.retirements(tasks, rows),
        reopen_conditions=[g["reopen_condition"] for g in gaps.values()],
        flagged_adversarial=False,
        validation_receipts=[],
        coverage_statement_counts={},
        terminal_validation_sidecar_path=None,
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
    authenticate_retirements(root, value["retirement_rows"])
    value["phase_spans"] = [
        dict(
            phase="authority_and_primitives",
            start_s=0,
            end_s=value["duration_s"],
            completed_units=12,
        )
    ]
    value["reproducibility_checksum"] = canonical_hash(
        dict(sources=refs, methods=frozen, code=value["code_config_hashes"])
    )
    value["field_principles"] = {
        k: "Bind exact evidence; preserve blockers and finite nulls; readiness supplies no scientific credit."
        for k in value
    }
    return value


def authenticate_retirements(root: Path, rows: list[Json]) -> None:
    """Compare actual byte-bound terminal verdicts and retain the declared scope change."""
    old.authenticate_retirements(root, rows)
    for row in rows:
        prior = old.read(Path(row["prior_path"])) if row["prior_path"] else {}
        side = old.read(Path(prior.get("terminal_validation_sidecar_path") or "/missing"))
        current = old.read(Path(row["producer_path"])) if row["producer_path"] else {}
        current_side = old.read(Path(current.get("terminal_validation_sidecar_path") or "/missing"))
        row.update(
            prior_observed_verdict=prior.get("honest_verdict"),
            changed_scope=row["reopen_condition"],
            prior_authenticated=bool(
                row["prior_authenticated"]
                and side.get("primary_sha256", side.get("candidate_sha256")) == row["prior_sha256"]
            ),
            current_authenticated=bool(
                current.get("honest_verdict") == row["terminal_verdict"]
                and current.get("flagged_adversarial") is False
                and current_side.get("primary_sha256", current_side.get("candidate_sha256"))
                == row["producer_sha256"]
            ),
        )
        row["retire"] = bool(
            row["retire"] and row["prior_authenticated"] and row["current_authenticated"]
        )
        row["scope"] = row["task_id"] + ": terminal " + row["terminal_verdict"]
        row["retirement_decision"] = (
            "retire_authenticated_exact_repeat"
            if row["retire"]
            else "retain_changed_or_unauthenticated_scope"
        )


def cold_replay(value: Json) -> list[str]:
    """Re-read source bytes and checkpoints so edited reductions cannot pass a seal."""
    progress("cold_replay_start", pending="source custody")
    refs = (
        value["source_artifact_hashes"]
        + value["code_config_hashes"]
        + value["checkpoint_references"]
    )
    for ref in refs:
        path = Path(ref["path"])
        if (sha256_file(path) if path.is_file() else None) != ref["sha256"]:
            return ["source_bytes_changed"]
        if ref.get("role") == "owned_reduction":
            saved = old.read(path)
            if saved != dict(
                rows=value["rows"], independent_reduction_rows=value["independent_reduction_rows"]
            ):
                return ["owned_primitive_drift"]
    for snapshot in value["authority_snapshots"].values():
        if (
            snapshot["exists"]
            and sha256_file(Path(snapshot["snapshot_path"])) != snapshot["sha256"]
        ):
            return ["authority_snapshot_changed"]
    active = value["authority_snapshots"]["active"]
    tasks = yaml.safe_load(Path(active["snapshot_path"]).read_bytes())["tasks"]
    rows, _, _, audits = collect(Path(value["input_root"]), tasks)
    errors = []
    if rows != value["task_dispositions"][:-1] or audits != value["independent_reduction_rows"]:
        errors.append("independent_reduction_drift")
    if value["generalized_learning_benefit_score"] != 0 or value["science_ready"] is not False:
        errors.append("unsupported_generalization")
    if (
        value["verdict_class"] in {"blocked", "disqualified"}
        and value["capstone_execution_ready_score"]
    ):
        errors.append("unsafe_readiness")
    if old.authority.lifecycle.tasks_digest(tasks) != value["canonical_tasks_sha256"]:
        errors.append("authority_drift")
    return errors


def manifest(private: Path) -> Json:
    """Freeze exact owned checks and private coverage paths before measurement."""
    py, cov, pytest, ruff, mypy = [
        str(ROOT / ".venv/bin" / n) for n in ("python", "coverage", "pytest", "ruff", "mypy")
    ]
    prefix = [cov, "run", f"--data-file={private / '.coverage'}", "--include=" + ",".join(OWNED)]
    common = ["-n", "0", "-o", "addopts=", "--no-cov", "-q"]
    specs = [
        ("publication_gate", [py, "scripts/publication_gate.py", "--json"], 0, 60),
        (
            "owned_tests",
            prefix + ["-m", "pytest", *common, TEST, f"--basetemp={private / 'tests'}"],
            0,
            300,
        ),
        (
            "real_cli_missing_replay",
            prefix[:2]
            + ["--append"]
            + prefix[2:]
            + [CLI, "--date", "20261002", "--cold-replay", str(private / "absent.json")],
            1,
            60,
        ),
        (
            "E2E-018_and_consumers",
            [
                pytest,
                *common,
                "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
                "tests/python/test_primary_publication_7928.py",
                "tests/python/test_conductor_gates.py",
                "tests/python/test_in_process_doc_reconcile.py",
                f"--basetemp={private / 'consumers'}",
            ],
            0,
            300,
        ),
        (
            "coverage_json",
            [
                cov,
                "json",
                f"--data-file={private / '.coverage'}",
                "-o",
                str(private / "coverage.json"),
            ],
            0,
            60,
        ),
        (
            "coverage_report",
            [
                cov,
                "report",
                f"--data-file={private / '.coverage'}",
                "--show-missing",
                "--fail-under=100",
            ],
            0,
            60,
        ),
        ("ruff_check", [ruff, "check", *OWNED, TEST], 0, 60),
        ("ruff_format", [ruff, "format", "--check", *OWNED, TEST], 0, 60),
        ("strict_mypy", [mypy, "--strict", "--follow-imports=silent", *OWNED], 0, 120),
        ("spec_coverage", [py, "scripts/check_spec_coverage.py", TEST, *OWNED], 0, 60),
        ("full_suite", [pytest, "tests/python", "-q"], 0, 180),
    ]
    return dict(
        commands=[
            dict(
                name=n,
                argv=a,
                expected_exit=e,
                deadline_s=d,
                classification="diagnostic" if n == "full_suite" else "required",
            )
            for n, a, e, d in specs
        ],
        dependency_hashes={p: sha256_file(ROOT / p) for p in OWNED + [TEST]},
        coverage_includes=OWNED,
        scratch_root=str(private),
        policy="owned checks required; repository health separate",
    )


def apply_checks(value: Json, receipts: list[Json], counts: Json) -> None:
    """Observed validation can complete owned work while external blockers remain blocked."""
    required = [r for r in receipts if r.get("classification") != "diagnostic"]
    complete = all(
        p in counts
        and counts[p]["num_statements"] > 0
        and counts[p]["num_statements"] == counts[p]["covered_lines"]
        for p in OWNED
    )
    valid = bool(required) and all(r["passed"] for r in required) and complete
    value.update(
        validation_receipts=required,
        repository_health=[r for r in receipts if r.get("classification") == "diagnostic"],
        coverage_statement_counts=counts,
        required_checks_passed=valid,
    )
    value["acceptance_gate_results"].update(
        validity=valid, readiness=int(valid and value["verdict_class"] == "null")
    )
    value["capstone_execution_ready_score"] = value["acceptance_gate_results"]["readiness"]
    if not valid:
        value.update(
            verdict_class="disqualified",
            honest_verdict="complete_disqualified_v694_owned_checks",
            capstone_execution_ready_score=0,
        )
    old.terminal_disposition(value)
    authenticate_retirements(Path(value["input_root"]), value["retirement_rows"])
    value["rows"][-1]["numerator"] = int(valid)


def publish(value: Json, output: Path, private: Path, durable: Path) -> None:
    """Publish only final bytes accepted by cold replay, validators and both readers."""
    value["terminal_validation_sidecar_path"] = str(durable / "terminal_validation.json")
    candidate = private / output.name
    atomic_json(candidate, value)
    specs = [
        (
            "cold_replay",
            [
                str(ROOT / ".venv/bin/python"),
                str(ROOT / CLI),
                "--date",
                "20261002",
                "--cold-replay",
                str(candidate),
            ],
        ),
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
    receipts = [
        run_check(
            ROOT,
            dict(name=n, argv=a, expected_exit=0, deadline_s=90, classification="required"),
            private / n,
            durable / "logs",
        )
        for n, a in specs
    ]
    report = old.read(Path(receipts[1]["log_path"]))
    if not all(r["passed"] for r in receipts) or report.get("flagged_count", 0):
        atomic_json(durable / "failed_terminal.json", dict(receipts=receipts))
        raise ValueError("terminal_validation_failed")
    digest = sha256_file(candidate)
    publication = publish_primary(
        output, value, lambda p: dict(passed=sha256_file(p) == digest, receipts=receipts)
    )
    selected = reader_receipt(
        value["task_id"],
        output.parent,
        field="capstone_execution_ready_score",
        expected=value["capstone_execution_ready_score"],
    )
    if (
        not selected["passed"]
        or selected["gate_sha256"] != digest
        or selected["document_sha256"] != digest
    ):
        raise ValueError("published_reader_drift")
    atomic_json(
        Path(value["terminal_validation_sidecar_path"]),
        dict(passed=True, **publication, receipts=receipts, readers=selected),
    )
    rechecks = [
        run_check(
            ROOT,
            dict(
                name="published_" + n,
                argv=[str(output) if a == str(candidate) else a for a in argv],
                expected_exit=0,
                deadline_s=90,
            ),
            private / ("published_" + n),
            durable / "logs",
        )
        for n, argv in specs
    ]
    atomic_json(
        durable / "published_recheck.json",
        dict(primary_sha256=sha256_file(output), receipts=rechecks),
    )
    if sha256_file(output) != digest or not all(r["passed"] for r in rechecks):
        raise ValueError("published_validation_failed")


def append_retirements(rows: list[Json]) -> None:
    """Keep historical bytes and append only authenticated exact repeated verdicts."""
    path = ROOT / "ops/exclusion_manifest.yaml"
    original = path.read_text()
    additions = []
    for row in rows:
        identity = (
            "v694_exact_repeat_"
            + canonical_hash([row["task_id"], row["prior_experiment_id"]])[7:23]
        )
        if row["retire"] and row.get("prior_authenticated") and identity not in original:
            additions.append(
                dict(
                    id=identity,
                    experiment_scope=row["scope"],
                    reason="Authenticated repeated verdict: " + row["terminal_verdict"],
                    retired_milestone="2026.10.694",
                    retired_by_artifact="results/experiment_8017_v694_capstone.json",
                    retire_if_same_verdict=True,
                    prior_path=row["prior_path"],
                    prior_sha256=row["prior_sha256"],
                    producer_path=row["producer_path"],
                    producer_sha256=row["producer_sha256"],
                    reopening_condition=row["reopen_condition"],
                )
            )
    if additions:
        path.write_text(original + "\n" + yaml.safe_dump(additions, sort_keys=False))


def qualify(root: Path, design: Path, date: str, output: Path) -> int:
    """Run one frozen validation plan with private scratch and durable published evidence."""
    started = time.monotonic_ns()
    durable = output.parent / "raw" / output.stem
    with tempfile.TemporaryDirectory(prefix="carnot-8017-") as directory:
        private = Path(directory)
        retained_health = old.read(output).get("repository_health", [])
        frozen = manifest(private)
        if retained_health:
            frozen["commands"] = [
                s for s in frozen["commands"] if s["classification"] != "diagnostic"
            ]
        frozen["retained_repository_health"] = retained_health
        atomic_json(durable / "validation_manifest.json", frozen)
        value = build(root, design, date, durable)
        receipts = [
            run_check(ROOT, spec, private / spec["name"], durable / "validation_logs")
            for spec in frozen["commands"]
        ] + retained_health
        publication = old.read(Path(receipts[0]["log_path"]))
        value.update(
            publication_gate_results=publication,
            paper_ready=publication["paper_ready"],
            unmet_gates=publication["unmet_gates"],
            **{f"g{i}": publication["gates"][f"G{i}"]["pass"] for i in range(1, 5)},
        )
        report = old.read(private / "coverage.json")
        counts = {p: f["summary"] for p, f in report.get("files", {}).items()}
        apply_checks(value, receipts, counts)
        value["repository_health_reused"] = bool(retained_health)
        value["checkpoint_references"].append(
            old.reference(durable / "validation_manifest.json", "command_freeze")
        )
        ended = time.monotonic_ns()
        value.update(
            duration_s=(ended - started) / 1e9,
            started_monotonic_timestamp_ns=started,
            ended_monotonic_timestamp_ns=ended,
        )
        value["phase_spans"].append(
            dict(
                phase="owned_validation",
                start_s=value["phase_spans"][0]["end_s"],
                end_s=value["duration_s"],
                completed_units=len(receipts),
            )
        )
        atomic_json(
            durable / "primitive_rows.json",
            dict(
                rows=value["rows"], independent_reduction_rows=value["independent_reduction_rows"]
            ),
        )
        value["checkpoint_references"].append(
            old.reference(durable / "primitive_rows.json", "owned_reduction")
        )
        value["field_principles"].update(
            {
                k: "Current checks bind exact bytes; historical paper readiness does not supply milestone science."
                for k in value
            }
        )
        publish(value, output, private / "terminal", durable)
        if root == ROOT:
            append_retirements(value["retirement_rows"])
    progress("published_final_bytes", 13)
    return 0
