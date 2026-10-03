"""REQ-REPORT-8056: reader execution, finite benefit and publication stay separate."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import time
from typing import Any

import yaml

from carnot.reporting import v696_capstone as previous
from carnot.reporting import v697_capstone_reduction as reduction
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
from carnot.reporting.v697_evidence import operand

Json = dict[str, Any]
ROOT = previous.ROOT
CLI = "scripts/experiments/experiment_8056_v697_capstone.py"
TEST = "tests/python/test_experiment_8056_v697_capstone.py"
INPUT = "results/experiment_8044_v697_contract_methods.json"
OWNED = [
    "python/carnot/reporting/v697_capstone.py",
    "python/carnot/reporting/v697_capstone_reduction.py",
    CLI,
]
NAMED = [
    "AGENTS.md",
    "CLAUDE.md",
    "CODEX.md",
    "ops/e2e-test-plan.md",
    "openspec/capabilities/research-reporting/spec.md",
    "scripts/experiment_template.py",
    "python/carnot/reporting/current_work_receipt.py",
    "python/carnot/reporting/primary_publication.py",
    "python/carnot/reporting/v696_capstone.py",
    "python/carnot/reporting/v696_capstone_reduction.py",
    "python/carnot/reporting/roadmap_contract.py",
    "python/carnot/reporting/v685_authority_lifecycle.py",
    "scripts/publication_gate.py",
    "scripts/failure_ledger.py",
    "ops/exclusion_manifest.yaml",
    "ops/verifier_gaps.md",
    INPUT,
    "openspec/change-proposals/research-roadmap-vNEXT.md",
]
START = time.monotonic()
read = previous.previous.read


def progress(phase: str, units: int = 0, pending: str = "") -> None:
    """Expose real phase boundaries so the conductor can detect stalled work."""
    print(
        f"[exp8056] phase={phase} elapsed_s={time.monotonic() - START:.3f} completed_units={units} pending={pending}",
        flush=True,
    )


def preconditions(root: Path) -> list[Json]:
    """Check resources before reduction; absence is an operand, never a measurement."""
    progress("preconditions_before", pending="named inputs and Python tools")
    rows = [
        operand(root / p, "local_preconditions", "resource_exists", True, (root / p).is_file())
        for p in NAMED
        + [".venv/bin/" + n for n in ("python", "pytest", "coverage", "ruff", "mypy")]
    ]
    progress("preconditions_after", len(rows), "immutable authority")
    return rows


def authorities(root: Path) -> tuple[list[Json], Json]:
    """Bind full prompts and task order to the original invocation, after activation."""
    invocation = read(root / INPUT)
    snapshots = invocation["authority_snapshots"]
    active = checked(
        dict(path=snapshots["active"]["snapshot_path"], sha256=snapshots["active"]["sha256"])
    )
    design = checked(
        dict(path=snapshots["design"]["snapshot_path"], sha256=snapshots["design"]["sha256"])
    )
    observed = yaml.safe_load(active.read_bytes())
    table, tasks = parse_design(design.read_text(), milestone="2026.10.697")
    shown = [
        dict(order=i + 1, **{k: t[k] for k in ("id", "title", "phase", "deliverable")})
        for i, t in enumerate(tasks)
    ]
    digest = re.search(r"Canonical full-task SHA-256: `([0-9a-f]{64})`", design.read_text())
    if (
        observed["milestone"] != "2026.10.697"
        or observed["tasks"] != tasks
        or table != shown
        or digest is None
        or digest[1] != invocation["canonical_tasks_sha256"]
        or lifecycle.tasks_digest(tasks) != digest[1]
        or [t["id"].split("-")[0] for t in tasks] != [f"exp{n}" for n in range(8044, 8057)]
    ):
        raise ValueError("immutable_authority_drift")
    return tasks, invocation


def save(ref: Json, durable: Path) -> Json:
    """Preserve original directory structure so cold equations read durable copies."""
    source = Path(ref["path"]).resolve()
    target = durable / "inputs" / source.relative_to("/")
    if source.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.stat().st_size >= 90_000_000:
            parts = []
            progress("large_input_sharding_before", pending=str(source))
            with source.open("rb") as stream:
                while chunk := stream.read(80_000_000):
                    part = target.with_name(target.name + f".part{len(parts):04d}")
                    part.write_bytes(chunk)
                    parts.append(
                        dict(path=str(part), sha256=sha256_file(part), size_bytes=len(chunk))
                    )
                    progress("large_input_shard_after", len(parts), str(source))
            target = target.with_name(target.name + ".shards.json")
            atomic_json(
                target,
                dict(original_path=str(source), original_sha256=sha256_file(source), shards=parts),
            )
        else:
            shutil.copyfile(source, target)
    return dict(
        path=str(target),
        original_path=str(source),
        sha256=sha256_file(target) if target.is_file() else None,
        role=ref.get("role", "input"),
    )


def collect(root: Path, tasks: list[Json]) -> tuple[list[Json], list[Json], list[Json], list[Json]]:
    """Reuse shipped primary, terminal, skip, gate and producer code readers."""
    rows, refs, failures, _ = previous.collect(root, tasks)
    for row in rows:
        for g in row["gate_check_summary"]:
            g.update(
                sha256=g.get("sha256", g.get("hash")),
                check_name=g.get("check_name", g["artifact_field"]),
            )
    audits = []
    for row in rows:
        try:
            audit = reduction.independent(read(Path(row["path"])), int(row["task_id"][3:7]))
        except (ValueError, KeyError, TypeError, OSError) as error:
            audit = dict(measurement_available=False, reduction_error=str(error))
            gate = operand(
                Path(row["path"]),
                row["task_id"],
                "independent_reduction",
                "authenticated primitives",
                str(error),
            )
            failures.append(gate)
            row["gate_check_summary"].append(gate)
            row.update(
                eligible=False,
                excluded=True,
                numerator=0,
                raw_numerator=0,
                exclusion="failed_primitive_reduction",
            )
        audits.append(dict(task_id=row["task_id"], **audit))
        progress("independent_task_after", len(audits), str(12 - len(audits)))
    return rows, refs, failures, audits


def hypotheses(audits: list[Json], rows: list[Json]) -> list[Json]:
    """Keep all registered tests in Holm, including absent and unsafe hypotheses."""
    h3 = audits[8]
    return reduction.family(
        [
            reduction.bootstrap([], 0.01),
            reduction.bootstrap([], 0.02),
            h3.get("primary", reduction.bootstrap([], 0.02, 32)),
        ],
        [False, False, bool(rows[8]["eligible"] and h3.get("scientific_qualified"))],
    )


def manifest(private: Path) -> Json:
    """Freeze bounded owned checks separately from the one repository diagnostic."""
    value = previous.manifest(private)
    for spec in value["commands"]:
        spec["argv"] = [
            a.replace(previous.CLI, CLI)
            .replace(previous.TEST, TEST)
            .replace(previous.OWNED[0], OWNED[0])
            .replace(previous.OWNED[1], OWNED[1])
            .replace("20261002", "20261003")
            for a in spec["argv"]
        ]
        spec["argv"] = [
            "--include=" + ",".join(OWNED) if a.startswith("--include=") else a
            for a in spec["argv"]
        ]
        if spec["name"] == "full_suite":
            spec["argv"] = ["timeout", "--signal=TERM", "--kill-after=10s", "90s", *spec["argv"]]
    value.update(
        dependency_hashes=dependency_hashes(ROOT, paths=OWNED + [TEST]), coverage_includes=OWNED
    )
    return value


def build(root: Path, date: str, durable: Path) -> Json:
    """Freeze the invocation before measuring and keep each scientific gap separate."""
    gates = preconditions(root)
    failures = [g for g in gates if not g["passed"]]
    try:
        tasks, invocation = authorities(root)
    except (ValueError, KeyError, TypeError, OSError) as error:
        tasks, invocation = [], {}
        failures.append(
            operand(
                root / INPUT,
                "exp8044",
                "immutable_authority",
                "exact thirteen-task invocation",
                str(error),
            )
        )
    progress("freeze_before", pending="methods, identities and code")
    freeze = dict(
        task_contract=tasks,
        canonical_tasks_sha256=lifecycle.tasks_digest(tasks),
        methods=invocation.get("method_freeze"),
        margins=[0.01, 0.02, 0.02],
        draws=10000,
        alpha=0.05,
        multiplicity="one-sided Holm H1/H2/H3",
        absent_p=1,
        seed=6968043,
        blocks=[32, 16, 64],
        code_config_hashes=dependency_hashes(ROOT, paths=OWNED + [TEST]),
        input_identities=[
            dict(
                path=str(root / t["deliverable"]),
                sha256=sha256_file(root / t["deliverable"])
                if (root / t["deliverable"]).is_file()
                else None,
            )
            for t in tasks[:-1]
        ],
        exclusions_sha256=sha256_file(root / "ops/exclusion_manifest.yaml")
        if (root / "ops/exclusion_manifest.yaml").is_file()
        else None,
        pretrained_model_calls=0,
        exposure="development; seeds and guards add no independent samples",
    )
    atomic_json(durable / "method_freeze.json", freeze)
    progress("freeze_after", len(tasks), "primitive reduction")
    rows, refs, producer_failures, audits = collect(root, tasks) if tasks else ([], [], [], [])
    failures.extend(producer_failures)
    primary = (
        hypotheses(audits, rows)
        if audits
        else reduction.family([reduction.bootstrap([], m) for m in (0.01, 0.02, 0.02)], [False] * 3)
    )
    state = "blocked" if failures or any(not h["qualified"] for h in primary) else "null"
    verdict = "complete_" + state + "_v697_capstone"
    if not tasks:
        verdict = "complete_blocked_immutable_authority"
    gaps = {
        "useful_source_verification": dict(
            requirements=["FR-06", "FR-12"],
            closed=False,
            decision="blocked_absent_H1_H2_source_primitives",
            upstream_ids=[8045, 8047, 8048, 8049, 8050],
            reopen_condition="Qualify scorer evidence as non-oracle positive/null under the original gate; collect new token shards, frozen heads, complete human targets and source support before margin tests.",
        ),
        "retained_causal_self_learning": dict(
            requirements=["FR-11"],
            closed=False,
            decision="tested_guard_mechanism_null_with_future_safety_and_retention_failures",
            upstream_ids=[8051, 8052],
            reopen_condition="Change feedback information or the diagnosed candidate-acceptance mechanism; pass later .02 cost, every-seed false-accept safety and retained .01 Brier/.02 cost on new unexposed sources.",
        ),
        "reproducible_deployment": dict(
            requirements=["FR-05", "FR-08", "FR-09", "FR-10", "NFR-01"],
            closed=False,
            decision="blocked_useful_complete_service_absent",
            upstream_ids=[8053, 8055],
            reopen_condition="First qualify useful decisions; then include likelihood acquisition, external feedback, transfer, scoring, rejected updates, guard scans, storage and restart in reproducible service costs.",
        ),
    }
    own = dict(
        task_id="exp8056-capstone",
        id="exp8056-capstone",
        unit_id="exp8056-capstone",
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
    retirements = previous.retirements(root, tasks, rows) if tasks else []
    for r in retirements:
        if r["prior_path"]:
            refs.append(previous.previous.reference(Path(r["prior_path"]), "prior_verdict"))
    for role, snap in invocation.get("authority_snapshots", {}).items():
        if snap["exists"]:
            refs.append(dict(path=snap["snapshot_path"], role="authority_" + role))
    refs.extend(dict(path=str(root / p), role="named_input") for p in NAMED if (root / p).is_file())
    frozen_refs = []
    for i, ref in enumerate(refs):
        frozen_refs.append(save(ref, durable))
        if i % 100 == 0:
            progress("durable_inputs", i, str(len(refs) - i))
    codes = [save(dict(path=str(ROOT / p), role="current_code"), durable) for p in OWNED]
    checkpoints = [previous.previous.reference(durable / "method_freeze.json", "method_freeze")]
    summaries = []
    for audit in audits:
        p = durable / "reductions" / (audit["task_id"] + ".json")
        atomic_json(p, audit)
        checkpoints.append(previous.previous.reference(p, "independent_reduction"))
        summaries.append(
            dict(
                task_id=audit["task_id"],
                measurement_available=audit["measurement_available"],
                path=str(p),
                sha256=sha256_file(p),
                operands={
                    k: v
                    for k, v in audit.items()
                    if k
                    in {
                        "producer_gates",
                        "per_seed_false_accept_rows",
                        "retention_drift_rows",
                        "later_support",
                        "retention_support",
                        "condition_metric_rows",
                        "costs",
                        "board_rows",
                        "guard_fallback_counts",
                        "guard_fallback_fraction",
                        "missing_cost_components",
                    }
                },
            )
        )
    value: Json = dict(
        experiment_id=8056,
        task_id="exp8056-capstone",
        milestone="2026.10.697",
        schema="carnot.v697.capstone.v1",
        run_date=date,
        input_root=str(root),
        honest_verdict=verdict,
        verdict_class=state,
        gate_check_summary=failures,
        rows=rows,
        task_dispositions=rows,
        task_contract=tasks,
        primary_hypothesis_results=primary,
        independent_reduction_rows=summaries,
        independent_reduction_sha256=canonical_hash(audits),
        source_artifact_hashes=frozen_refs,
        cited_upstream_artifacts=[
            r for r in frozen_refs if r["role"] in {"present", "missing", "conductor_skip_receipt"}
        ],
        raw_shard_hashes=frozen_refs,
        code_config_hashes=codes,
        checkpoint_references=checkpoints,
        authority_snapshots=invocation.get("authority_snapshots", {}),
        canonical_tasks_sha256=lifecycle.tasks_digest(tasks),
        random_seed=6968043,
        MODEL_SPECS=[],
        model_specs=[],
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
        claim_scope="This invocation binds actual V697 dispositions and exposed finite-source equations. Reader completion, empirical guards, ARC nulls, hardware custody and historical publication confer no generalized scientific credit.",
        verifier_is_oracle=False,
        genuine_headroom=dict(scope="imported exposed development"),
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
        retirement_rows=retirements,
        reopen_conditions=[g["reopen_condition"] for g in gaps.values()],
        science_ready=False,
        paper_ready=False,
        g1=False,
        g2=False,
        g3=False,
        g4=False,
        unmet_gates=["publication_gate_pending"],
        preconditions_checked=gates,
        flagged_adversarial=False,
        validation_receipts=[],
        coverage_statement_counts={},
        terminal_validation_sidecar_path=str(durable / "terminal_validation.json"),
        sample_size_budget=dict(
            intended=13,
            eligible=sum(r["eligible"] for r in rows),
            started=sum(r["started"] for r in rows),
            completed=len(rows) - 1,
            excluded=sum(r["excluded"] for r in rows),
            failed=sum(r["failed"] for r in rows),
            censored=0,
            independent=0,
            unit="administrative_task_disposition",
        ),
    )
    value["reproducibility_checksum"] = canonical_hash(
        dict(sources=frozen_refs, freeze=freeze, code=codes)
    )
    return value


def complete(value: Json, receipts: list[Json], counts: Json) -> None:
    """Complete owned work without changing external scientific failure operands."""
    required = [r for r in receipts if r.get("classification") != "diagnostic"]
    valid = (
        bool(required)
        and all(r["passed"] for r in required)
        and all(
            p in counts
            and counts[p]["num_statements"] > 0
            and counts[p]["num_statements"] == counts[p]["covered_lines"]
            for p in OWNED
        )
    )
    value.update(
        validation_receipts=required,
        coverage_statement_counts=counts,
        required_checks_passed=valid,
        capstone_execution_ready_score=int(valid),
        repository_health=[r for r in receipts if r.get("classification") == "diagnostic"],
    )
    if not valid:
        value.update(
            verdict_class="disqualified", honest_verdict="complete_disqualified_v697_owned_checks"
        )
    value["acceptance_gate_results"].update(validity=valid, readiness=int(valid))
    value["rows"][-1].update(
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
        completed=len(value["rows"]),
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
    """Use the shipped claim seal, with its complete per-field explanations."""
    previous.seal(value, durable)


def cold_replay(value: Json) -> list[str]:
    """Fresh equations reopen durable source copies rather than reported means."""
    progress("cold_replay_before", pending="durable inputs and primitive equations")
    copies = {}
    for ref in (
        value["source_artifact_hashes"]
        + value["code_config_hashes"]
        + value["checkpoint_references"]
    ):
        path = Path(ref["path"])
        if (sha256_file(path) if path.is_file() else None) != ref["sha256"]:
            return ["source_bytes_changed"]
        if path.name.endswith(".shards.json"):
            manifest = read(path)
            digest = hashlib.sha256()
            for shard in manifest["shards"]:
                part = Path(shard["path"])
                if not part.is_file() or sha256_file(part) != shard["sha256"]:
                    return ["source_bytes_changed"]
                digest.update(part.read_bytes())
            if "sha256:" + digest.hexdigest() != manifest["original_sha256"]:
                return ["source_bytes_changed"]
        if (
            ref.get("role") == "current_code"
            and sha256_file(Path(ref["original_path"])) != ref["sha256"]
        ):
            return ["code_configuration_changed"]
        copies[ref.get("original_path", ref["path"])] = ref["path"]
    audits = [
        dict(
            task_id=r["task_id"],
            **reduction.independent(
                read(Path(copies.get(r["path"], r["path"]))), int(r["task_id"][3:7]), copies
            ),
        )
        for r in value["rows"][:-1]
    ]
    errors = []
    if canonical_hash(audits) != value["independent_reduction_sha256"] or (
        audits and hypotheses(audits, value["rows"]) != value["primary_hypothesis_results"]
    ):
        errors.append("independent_reduction_drift")
    claim = next(r for r in reversed(value["checkpoint_references"]) if r["role"] == "claim_seal")
    if read(Path(claim["path"])) != previous.previous.claim_payload(value):
        errors.append("claim_seal_drift")
    progress("cold_replay_after", len(audits), "none")
    return errors


def publish(value: Json, output: Path, private: Path, durable: Path) -> None:
    """Expose only bytes accepted by cold reduction, both linters and both readers."""
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
        flagged = read(Path(receipts[-2]["log_path"])).get("flagged_count", 0)
        if not all(r["passed"] for r in receipts) or flagged:
            atomic_json(durable / "failed_terminal.json", dict(receipts=receipts))
            raise ValueError("terminal_validation_failed")
        if prefix == "candidate":
            publication = publish_primary(
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
                dict(passed=True, publication=publication, readers=readers, receipts=receipts[:]),
            )
    atomic_json(
        durable / "published_recheck.json",
        dict(primary_sha256=sha256_file(output), receipts=receipts),
    )


def main(argv: list[str] | None = None) -> int:
    """Run frozen owned checks once; preserve unrelated repository health separately."""
    progress("start", pending="preconditions")
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20261003")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cold-replay", type=Path)
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
    output = args.output or root / "results/experiment_8056_v697_capstone.json"
    durable = output.parent / "raw" / output.stem
    began = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="exp8056-owned-") as directory:
        private = Path(directory)
        frozen = manifest(private)
        atomic_json(durable / "validation_manifest.json", frozen)
        value = build(root, args.date, durable)
        receipts = []
        for spec in frozen["commands"]:
            progress("subprocess_before_" + spec["name"], len(receipts), "owned validation")
            receipts.append(run_check(ROOT, spec, private, durable / "validation_logs"))
            progress("subprocess_after_" + spec["name"], len(receipts), "owned validation")
        report = read(private / "coverage.json")
        counts = {p: f["summary"] for p, f in report.get("files", {}).items()}
        complete(value, receipts, counts)
        if not value["required_checks_passed"]:
            atomic_json(durable / "failed_owned_checks.json", value)
            raise ValueError("owned_validation_failed")
        publication = read(Path(receipts[0]["log_path"]))
        value.update(
            publication_gate_results=publication,
            paper_ready=publication["paper_ready"],
            unmet_gates=publication["unmet_gates"],
            **{f"g{i}": publication["gates"][f"G{i}"]["pass"] for i in range(1, 5)},
        )
        value["duration_s"] = time.monotonic() - began
        value["phase_spans"] = [
            dict(
                phase="frozen_custody_reduction_and_owned_checks",
                start_s=0,
                end_s=value["duration_s"],
                completed_units=len(value["rows"]),
            )
        ]
        value["checkpoint_references"].append(
            previous.previous.reference(durable / "validation_manifest.json", "command_freeze")
        )
        seal(value, durable)
        publish(value, output, private / "terminal", durable)
    progress("published_final_bytes", 13, "none")
    return 0
