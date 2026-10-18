"""REQ-REPORT-8043: complete custody does not grant scientific success.

This reader binds one invocation to exact producer bytes. Owned execution,
scientific benefit and historical publication keep separate acceptance gates.
"""

import argparse
import json
import re
from pathlib import Path
import tempfile
import time
from typing import Any

import yaml

from carnot.reporting import v695_capstone as previous
from carnot.reporting import v696_capstone_reduction as reduction
from carnot.reporting import v685_authority_lifecycle as lifecycle
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.evidence_features_custody_7980 import checked
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.roadmap_contract import parse_design
from carnot.reporting.v686_contract_validation import dependency_hashes, run_check

Json = dict[str, Any]
ROOT = previous.ROOT
CLI = "scripts/experiments/experiment_8043_v696_capstone.py"
TEST = "tests/python/test_experiment_8043_v696_capstone.py"
OWNED = [
    "python/carnot/reporting/v696_capstone.py",
    "python/carnot/reporting/v696_capstone_reduction.py",
    CLI,
]
START = time.monotonic()


def progress(phase: str, units: int = 0, pending: str = "") -> None:
    """Flush real counts so bounded work stays visible without invented duration."""
    print(
        f"[exp8043] phase={phase} elapsed_s={time.monotonic() - START:.3f} completed_units={units} pending={pending}",
        flush=True,
    )


def authorities(root: Path) -> tuple[list[Json], Json]:
    """Use invocation snapshots when staging was consumed instead of recreating it."""
    invocation = previous.read(root / "results/experiment_8031_v696_contract_methods.json")
    snapshots = invocation["authority_snapshots"]
    active = checked(
        dict(path=snapshots["active"]["snapshot_path"], sha256=snapshots["active"]["sha256"])
    )
    design = checked(
        dict(path=snapshots["design"]["snapshot_path"], sha256=snapshots["design"]["sha256"])
    )
    observed = yaml.safe_load(active.read_bytes())
    table, machine = parse_design(design.read_text(), milestone="2026.10.696")
    tasks = observed["tasks"]
    shown = [
        dict(order=i + 1, **{k: t[k] for k in ("id", "title", "phase", "deliverable")})
        for i, t in enumerate(tasks)
    ]
    digest = re.search(r"Canonical full-task SHA-256: `([0-9a-f]{64})`", design.read_text())
    if (
        observed["milestone"] != "2026.10.696"
        or len(table) != 13
        or machine != tasks
        or table != shown
        or digest is None
        or digest[1] != invocation["canonical_tasks_sha256"]
        or [t["id"].split("-")[0] for t in tasks] != [f"exp{n}" for n in range(8031, 8044)]
        or lifecycle.tasks_digest(tasks) != invocation["canonical_tasks_sha256"]
    ):
        raise ValueError("immutable_authority_drift")
    return tasks, invocation


def save_reference(ref: Json, durable: Path) -> Json:
    """Keep each imported byte copy under this raw directory with its original path."""
    path = Path(ref["path"])
    frozen = lifecycle._snapshot(
        path, path.read_bytes() if path.is_file() else None, durable / "inputs", path.name
    )
    return dict(
        path=frozen.get("snapshot_path", str(durable / "absent" / path.name)),
        sha256=frozen["sha256"],
        original_path=str(path),
        role=ref.get("role", "input"),
    )


def operand(path: Path, upstream: str, field: str, expected: Any, observed: Any) -> Json:
    """Missing fields are contract errors, so failed gates retain exact operands."""
    return dict(
        previous.previous.old.operand(path, upstream, field, expected, observed),
        sha256=sha256_file(path) if path.is_file() else None,
        passed=False,
        check_name=field,
    )


def collect(root: Path, tasks: list[Json]) -> tuple[list[Json], list[Json], list[Json], list[Json]]:
    """Reuse actual primary/skip readers, then authenticate producer code and equations."""
    rows, refs, failures, _ = previous.collect(root, tasks)
    audits = []
    for row in rows:
        data = previous.read(Path(row["path"]))
        hashes = data.get("code_config_hashes", [])
        hashes = (
            [dict(path=str(root / p), sha256=h) for p, h in hashes.items()]
            if isinstance(hashes, dict)
            else hashes
        )
        issues = []
        for ref in hashes:
            if Path(ref.get("original_path", ref["path"])).suffix not in {
                ".py",
                ".yaml",
                ".json",
                ".toml",
            }:
                continue
            p = Path(ref.get("original_path", ref["path"]))
            observed = sha256_file(p) if p.is_file() else None
            refs.append(dict(path=ref["path"], role="producer_code", sha256=ref["sha256"]))
            if observed != ref["sha256"]:
                issues.append(
                    operand(
                        p, row["task_id"], "producer_code_config_sha256", ref["sha256"], observed
                    )
                )
        try:
            audit = reduction.independent(data, int(row["task_id"][3:7]))
        except (ValueError, KeyError, TypeError, OSError) as error:
            audit = dict(measurement_available=False, reduction_error=str(error))
            issues.append(
                operand(
                    Path(row["path"]),
                    row["task_id"],
                    "independent_reduction",
                    "authenticated primitives",
                    str(error),
                )
            )
        row["gate_check_summary"].extend(issues)
        failures.extend(issues)
        if issues:
            row.update(
                eligible=False,
                excluded=True,
                numerator=0,
                raw_numerator=0,
                exclusion="failed_prerequisite_or_custody",
            )
        audits.append(dict(task_id=row["task_id"], **audit))
        progress("independent_task_reduced", len(audits), str(12 - len(audits)))
    return rows, refs, failures, audits


def retirements(root: Path, tasks: list[Json], rows: list[Json]) -> list[Json]:
    """Bind historical verdicts and retire failed operands even after a spelling change."""
    result = previous.previous.old.retirements(tasks, rows)
    previous.previous.authenticate_retirements(root, result)
    paths = {}
    for n in (8030, 8017):
        p = root / f"results/experiment_{n}_v{695 if n == 8030 else 694}_capstone.json"
        for t in previous.read(p).get("task_contract", []):
            paths[t["id"]] = root / t["deliverable"]
    for r in result:
        p = paths.get(
            r["prior_experiment_id"],
            Path(r["prior_path"]) if r.get("prior_path") else root / "absent",
        )
        if not p.is_file():
            identity, slug = r["prior_experiment_id"].split("-", 1)
            skip = (
                root
                / "results"
                / ("experiment_" + identity[3:] + "_" + slug.replace("-", "_") + ".json")
            )
            p = skip if skip.is_file() else p
        prior = previous.read(p)
        r.update(
            prior_path=str(p),
            prior_sha256=sha256_file(p) if p.is_file() else None,
            prior_observed_verdict=prior.get("honest_verdict"),
            prior_authenticated=prior.get("honest_verdict") == r["prior_verdict"],
            changed_scope=r["reopen_condition"],
            resolved=False,
        )
        current = previous.read(Path(r["producer_path"])) if r["producer_path"] else {}
        failed_fields = lambda d: {
            x["artifact_field"]
            for x in d.get("gate_check_summary", [])
            if isinstance(x, dict) and x.get("passed") is False
        }
        same = bool(
            current.get("verdict_class") in {"blocked", "disqualified"}
            and failed_fields(prior) & failed_fields(current)
        )
        repeated_scoring = (
            "likelihood_calibration" in r["prior_verdict"]
            and "scoring_isolation" in r["terminal_verdict"]
            and current.get("verdict_class") == "disqualified"
            and current.get("acceptance_gate_results", {}).get("changed_condition") is False
        )
        r["retire"] = bool(
            r["prior_authenticated"]
            and r["retire_if_same_verdict"]
            and (r["repeated_exact_verdict"] or same or repeated_scoring)
        )
        r["retirement_decision"] = (
            "retire_scoped_failed_mechanism"
            if r["retire"]
            else "changed_mechanism_or_unavailable_authority"
        )
        r["mechanism_repeat"] = same or repeated_scoring
    return result


def build(root: Path, date: str, durable: Path) -> Json:
    """Freeze every decision rule before reading measured rows and evaluator releases."""
    progress("freeze_before", pending="invocation methods and code")
    tasks, invocation = authorities(root)
    owned_refs = [
        save_reference(previous.reference(ROOT / p, "current_code"), durable) for p in OWNED
    ]
    freeze = dict(
        task_contract=tasks,
        canonical_tasks_sha256=lifecycle.tasks_digest(tasks),
        methods=invocation["method_freeze"],
        margins=[0.01, 0.02, 0.02],
        draws=10000,
        alpha=0.05,
        multiplicity="one-sided Holm H1/H2/H3",
        absent_p=1,
        seed=6968043,
        blocks=[32, 16, 64],
        code_config_hashes=dependency_hashes(ROOT, paths=OWNED + [TEST]),
        exclusions_sha256=sha256_file(ROOT / "ops/exclusion_manifest.yaml"),
        pretrained_model_calls=0,
        data_access="authenticate original prediction seals before evaluator targets; exposed development only",
    )
    atomic_json(durable / "method_freeze.json", freeze)
    progress("freeze_after", 13, "producer custody and primitive equations")
    rows, refs, failures, audits = collect(root, tasks)
    h3 = audits[8]
    primaries = [
        reduction.bootstrap([], 0.01),
        reduction.bootstrap([], 0.02),
        h3.get("primary", reduction.bootstrap([], 0.02, 32)),
    ]
    primary = reduction.family(
        primaries, [False, False, bool(rows[8]["eligible"] and h3.get("scientific_qualified"))]
    )
    blocked = bool(failures) or any(r["verdict_class"] in {"blocked", "disqualified"} for r in rows)
    state = "blocked" if blocked else "null"
    verdict = "complete_" + state + "_v696_capstone"
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
    gaps = {
        "useful_source_verification": dict(
            requirements=["FR-06", "FR-12"],
            closed=False,
            decision="blocked_unqualified_repeatability_and_absent_H1_H2",
            upstream_ids=[8033, 8034, 8035, 8036, 8037],
            reopen_condition="A changed scorer must pass original token alignment and duplicate tolerance 1e-6 before new fit and evaluation captures, frozen heads and complete evaluator targets.",
        ),
        "retained_causal_self_learning": dict(
            requirements=["FR-11"],
            closed=False,
            decision="valid_new_window_mechanism_null_with_failed_safety_and_retention",
            upstream_ids=[8038, 8039],
            reopen_condition="Change the diagnosed update mechanism or information, then pass later .02 cost margin, per-seed false-accept safety and retained .01 Brier/.02 cost gates on unexposed sources.",
        ),
        "reproducible_deployment": dict(
            requirements=["FR-05", "FR-08", "FR-09", "FR-10", "NFR-01"],
            closed=False,
            decision="local_transaction_null_and_cpu_fallback_custody_without_complete_service",
            upstream_ids=[8040, 8042],
            reopen_condition="Establish useful qualified workload, then reproduce complete acquisition, scoring, updates, storage, restart and device transfer costs; preserve host fallback.",
        ),
    }
    durable_refs = []
    for i, ref in enumerate(refs):
        durable_refs.append(save_reference(ref, durable))
        if i % 100 == 0:
            progress("raw_input_custody", i, str(len(refs) - i))
    audit_refs = []
    for audit in audits:
        p = durable / "reductions" / (audit["task_id"] + ".json")
        atomic_json(p, audit)
        audit_refs.append(dict(path=str(p), sha256=sha256_file(p), role="independent_reduction"))
    summary = [
        {
            k: v
            for k, v in a.items()
            if k not in {"rows", "retention_rows", "primary", "block_sensitivity"}
        }
        for a in audits
    ]
    value: Json = dict(
        experiment_id=8043,
        task_id=tasks[-1]["id"],
        milestone="2026.10.696",
        schema="carnot.v696.capstone.v1",
        run_date=date,
        execution_date=date,
        input_root=str(root),
        honest_verdict=verdict,
        verdict_class=state,
        gate_check_summary=[
            dict(
                r,
                sha256=r.get("sha256", r.get("hash", r.get("artifact_hash"))),
                check_name=r.get("check_name", r["artifact_field"]),
            )
            for r in failures
        ],
        rows=rows,
        task_dispositions=rows,
        task_contract=tasks,
        independent_reduction_rows=summary,
        primary_hypothesis_results=primary,
        independent_reduction_sha256=canonical_hash(audits),
        checkpoint_references=audit_refs
        + [
            dict(
                path=str(durable / "method_freeze.json"),
                sha256=sha256_file(durable / "method_freeze.json"),
                role="method_freeze",
            )
        ],
        source_artifact_hashes=durable_refs,
        cited_upstream_artifacts=[
            r for r in durable_refs if r["role"] in {"present", "missing", "conductor_skip_receipt"}
        ],
        raw_shard_hashes=durable_refs,
        code_config_hashes=owned_refs,
        canonical_tasks_sha256=lifecycle.tasks_digest(tasks),
        authority_snapshots=invocation["authority_snapshots"],
        random_seed=6968043,
        model_specs=[],
        MODEL_SPECS=[],
        trained_head_specs=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        substrate_declaration=dict(
            custody="aggregation_from_upstream_artifacts",
            numerical="verifier_scoring",
            pretrained="no_model_load",
            MODEL_SPECS=[],
            pretrained_model_calls=0,
        ),
        claim_scope="This invocation binds thirteen administrative dispositions and exposed finite-trajectory reductions. Missing source evidence and complete deployment remain unavailable; historical publication is separate.",
        verifier_is_oracle=False,
        genuine_headroom=dict(source=None, learning=h3.get("producer_gates")),
        positive_control_results=reduction.learning.controls(),
        generalized_learning_benefit_score=0,
        acceptance_gate_results=dict(
            validity=False,
            readiness=0,
            source_benefit=False,
            retained_learning=False,
            deployment=False,
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
        retirement_rows=retirements(root, tasks, rows),
        reopen_conditions=[g["reopen_condition"] for g in gaps.values()],
        historical_required_failures=previous.read(
            root / "results/experiment_8030_v695_capstone.json"
        ).get("gate_check_summary", []),
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
    value["reproducibility_checksum"] = canonical_hash(
        dict(sources=durable_refs, freeze=freeze, code=value["code_config_hashes"])
    )
    return value


def complete(value: Json, receipts: list[Json], counts: Json) -> None:
    """A checked reader can finish while outside scientific prerequisites remain blocked."""
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
        capstone_execution_ready_score=int(valid),
    )
    if not valid:
        value.update(
            verdict_class="disqualified", honest_verdict="complete_disqualified_v696_owned_checks"
        )
    value["acceptance_gate_results"].update(validity=valid, readiness=int(valid))
    own = value["rows"][-1]
    own.update(
        completed=True,
        status="completed",
        producer_status="current_capstone_validated",
        eligible=valid,
        failed=not valid,
        numerator=int(valid),
        raw_numerator=int(valid),
        honest_verdict=value["honest_verdict"],
        verdict_class=value["verdict_class"],
    )
    value["sample_size_budget"].update(
        completed=13,
        eligible=sum(r["eligible"] for r in value["rows"]),
        failed=sum(r["failed"] for r in value["rows"]),
    )
    value.update(
        {
            k + "_count": v
            for k, v in value["sample_size_budget"].items()
            if k not in {"started", "unit"}
        }
    )


def seal(value: Json, durable: Path) -> None:
    """A claim seal supplements raw replay so changed claims cannot pass custody."""
    value["field_principles"] = {
        k: "Bind this invocation's exact evidence; owned execution, scientific utility and historical publication remain separate."
        for k in value
    }
    value["field_principles"].update(
        capstone_execution_ready_score="Owned reader checks and full added-statement coverage; external science cannot change this administrative score.",
        primary_hypothesis_results="Frozen .01/.02/.02 margins, 10000 draws and Holm .05; absent or unsafe hypotheses receive p=1.",
        gap_decisions="Source utility, retained causal learning and complete deployment require separate qualified evidence.",
        paper_ready="Historical G1-G4 conjunction; it never supplies current scientific qualification.",
        science_ready="Qualified independent current benefit and safety are required; completion gives no scientific credit.",
        sample_size_budget="Thirteen administrative dispositions are zero new independent scientific observations.",
        generalized_learning_benefit_score="An exposed finite trajectory cannot establish generalized learning benefit.",
        model_invocation_counts="Zero current pretrained calls; imported model measurements and replayed small-head equations remain distinct.",
        gate_check_summary="Each failure names the exact operand and byte identity; a missing field is a contract error.",
        retirement_rows="Authenticate prior verdict bytes and failed mechanisms; changed valid nulls do not retire unrelated learning.",
        validation_receipts="Actual argv, process exits and durable log hashes establish owned correctness.",
        coverage_statement_counts="Only added module and real CLI statements must reach 100 percent.",
    )
    path = durable / "claim_seal.json"
    atomic_json(path, previous.claim_payload(value))
    value["checkpoint_references"].append(
        dict(path=str(path), sha256=sha256_file(path), role="claim_seal")
    )


def cold_replay(value: Json) -> list[str]:
    """Reopen durable input copies and recompute producer equations in a fresh process."""
    progress("cold_replay_before", pending="durable inputs and checkpoints")
    for ref in (
        value["source_artifact_hashes"]
        + value["code_config_hashes"]
        + value["checkpoint_references"]
    ):
        p = Path(ref["path"])
        if (sha256_file(p) if p.is_file() else None) != ref["sha256"]:
            return ["source_bytes_changed"]
        if (
            ref.get("role") == "current_code"
            and sha256_file(Path(ref["original_path"])) != ref["sha256"]
        ):
            return ["code_configuration_changed"]
        if ref.get("role") in {"method_freeze", "command_freeze"}:
            for name, digest in (
                previous.read(p)
                .get("code_config_hashes", previous.read(p).get("dependency_hashes", {}))
                .items()
            ):
                if sha256_file(ROOT / name) != digest:
                    return ["code_configuration_changed"]
    tasks, _ = authorities(Path(value["input_root"]))
    rows, _, _, audits = collect(Path(value["input_root"]), tasks)
    h3 = audits[8]
    family = reduction.family(
        [
            reduction.bootstrap([], 0.01),
            reduction.bootstrap([], 0.02),
            h3.get("primary", reduction.bootstrap([], 0.02, 32)),
        ],
        [False, False, bool(rows[8]["eligible"] and h3.get("scientific_qualified"))],
    )
    errors = []
    if (
        rows != value["task_dispositions"][:-1]
        or canonical_hash(audits) != value["independent_reduction_sha256"]
        or family != value["primary_hypothesis_results"]
    ):
        errors.append("independent_reduction_drift")
    if lifecycle.tasks_digest(tasks) != value["canonical_tasks_sha256"]:
        errors.append("authority_drift")
    claim = next(r for r in value["checkpoint_references"] if r["role"] == "claim_seal")
    if previous.read(Path(claim["path"])) != previous.claim_payload(value):
        errors.append("claim_seal_drift")
    progress("cold_replay_after", 13, "none")
    return errors


def manifest(private: Path) -> Json:
    """Reuse bounded supervision with coverage limited to this reader and its CLI."""
    value = previous.manifest(private)
    for spec in value["commands"]:
        spec["argv"] = [
            a.replace(previous.CLI, CLI)
            .replace(previous.TEST, TEST)
            .replace(previous.OWNED[0], OWNED[0])
            .replace(previous.OWNED[1], OWNED[1])
            for a in spec["argv"]
        ]
        spec["argv"] = [
            "--include=" + ",".join(OWNED) if a.startswith("--include=") else a
            for a in spec["argv"]
        ]
        if spec["name"] == "full_suite":
            spec["deadline_s"] = 90
    value.update(
        dependency_hashes=dependency_hashes(ROOT, paths=OWNED + [TEST]), coverage_includes=OWNED
    )
    return value


def publish(value: Json, output: Path, private: Path, durable: Path) -> None:
    """Validate candidate and published bytes through unchanged terminal consumers."""
    candidate = private / output.name
    atomic_json(candidate, value)
    py = str(ROOT / ".venv/bin/python")
    checks = [
        ("cold_replay", [py, "-u", str(ROOT / CLI), "--cold-replay", str(candidate)]),
        ("adversarial", [py, "scripts/adversarial_verify.py", "--json", str(candidate)]),
        (
            "strict_rows",
            [py, "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)],
        ),
    ]
    receipts = []
    digest = sha256_file(candidate)
    for prefix, target in (("candidate", candidate), ("published", output)):
        for name, argv in checks:
            progress("subprocess_before_" + prefix + "_" + name, len(receipts), "terminal checks")
            spec = dict(
                name=prefix + "_" + name,
                argv=[str(target) if a == str(candidate) else a for a in argv],
                expected_exit=0,
                deadline_s=180,
            )
            receipts.append(run_check(ROOT, spec, private, durable / "terminal_logs"))
            progress("subprocess_after_" + prefix + "_" + name, len(receipts), "terminal checks")
        flagged = previous.read(Path(receipts[-2]["log_path"])).get("flagged_count", 0)
        if not all(r["passed"] for r in receipts) or flagged:
            atomic_json(durable / "failed_terminal.json", dict(receipts=receipts))
            raise ValueError("terminal_validation_failed")
        if prefix == "candidate":
            published = publish_primary(
                output, value, lambda p: dict(passed=sha256_file(p) == digest, receipts=receipts[:])
            )
            readers = reader_receipt(
                value["task_id"],
                output.parent,
                field="capstone_execution_ready_score",
                expected=value["capstone_execution_ready_score"],
            )
            if (
                not readers["passed"]
                or readers["gate_sha256"] != digest
                or readers["document_sha256"] != digest
            ):
                raise ValueError("published_reader_drift")
            atomic_json(
                Path(value["terminal_validation_sidecar_path"]),
                dict(passed=True, publication=published, readers=readers, receipts=receipts[:]),
            )
    atomic_json(
        durable / "published_recheck.json",
        dict(primary_sha256=sha256_file(output), receipts=receipts),
    )


def main(argv: list[str] | None = None) -> int:
    """Run frozen owned checks once, preserving unrelated repository health separately."""
    progress("cli_start", pending="arguments")
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20261002")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        try:
            errors = cold_replay(previous.read(args.cold_replay))
        except (ValueError, KeyError, TypeError, OSError, StopIteration) as error:
            errors = [str(error)]
        print(json.dumps(dict(errors=errors)), flush=True)
        return int(bool(errors))
    if args.date != "20261002":
        raise ValueError("date_must_be_20261002")
    root = args.root.resolve()
    output = args.output or root / "results/experiment_8043_v696_capstone.json"
    durable = output.parent / "raw" / output.stem
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="exp8043-owned-") as directory:
        private = Path(directory)
        frozen = manifest(private)
        atomic_json(durable / "validation_manifest.json", frozen)
        value = build(root, args.date, durable)
        receipts = []
        for spec in frozen["commands"]:
            progress("subprocess_before_" + spec["name"], len(receipts), "owned validation")
            receipts.append(run_check(ROOT, spec, private, durable / "validation_logs"))
            progress("subprocess_after_" + spec["name"], len(receipts), "owned validation")
        report = previous.read(private / "coverage.json")
        counts = {p: f["summary"] for p, f in report.get("files", {}).items()}
        complete(value, receipts, counts)
        if not value["required_checks_passed"]:
            atomic_json(durable / "failed_owned_checks.json", value)
            raise ValueError("owned_validation_failed")
        publication = previous.read(Path(receipts[0]["log_path"]))
        value.update(
            publication_gate_results=publication,
            paper_ready=publication["paper_ready"],
            unmet_gates=publication["unmet_gates"],
            **{f"g{i}": publication["gates"][f"G{i}"]["pass"] for i in range(1, 5)},
        )
        value["retirement_rows"] = retirements(root, value["task_contract"], value["rows"])
        value["duration_s"] = time.monotonic() - started
        value["phase_spans"] = [
            dict(
                phase="frozen_custody_reduction_and_owned_checks",
                start_s=0,
                end_s=value["duration_s"],
                completed_units=13,
            )
        ]
        value["checkpoint_references"].append(
            previous.reference(durable / "validation_manifest.json", "command_freeze")
        )
        seal(value, durable)
        publish(value, output, private / "terminal", durable)
    progress("published_final_bytes", 13, "none")
    return 0
