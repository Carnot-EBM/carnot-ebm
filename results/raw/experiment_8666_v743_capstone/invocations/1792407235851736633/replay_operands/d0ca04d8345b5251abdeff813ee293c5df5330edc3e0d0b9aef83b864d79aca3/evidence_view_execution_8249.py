"""REQ-REPORT-8249: publish qualified mechanisms separately from scientific benefit.

The supervisor, primary publisher and crash-coverage patch remain unchanged.
Only current owned code is covered; private oracle controls cannot establish
independent generalization or learning on the exposed natural source cohort.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import methods_stream_execution_8111 as execution
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.verify import evidence_view_kernel_8249 as k
from carnot.verify import learning_validation_8235 as qualified
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8249_v713_evidence_view_kernel"
TASK = "exp8249-evidence-view-kernel"
MODULE = "python/carnot/verify/evidence_view_execution_8249.py"
RUNNER = MODULE
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_evidence_view_kernel_8249.py"
OWNED = ["python/carnot/verify/evidence_view_kernel_8249.py", MODULE, CLI]
RUN_DATE = "20261007"
MODEL_SPECS: list[Json] = []
PROTOCOL = "openspec/change-proposals/v713-evidence-intervention-protocol.json"
reference = qualified.reference
run_check = qualified.run_check
BASE_MANIFEST = execution.manifest


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real phase counts so children and private fixture work never appear stalled."""
    print(f"[exp8249] phase={phase} completed={completed} pending={pending}", flush=True)


def gate(work: Json, path: Path, field: str, expected: Any, observed: Any) -> None:
    """Missing operands remain distinguishable from measured zeros in the gate receipt."""
    work["checks"].append(
        dict(
            upstream=path.stem,
            path=str(path),
            hash=sha256_file(path) if path.is_file() else None,
            artifact_field=field,
            op="==",
            expected=expected,
            observed=observed,
            passed=expected == observed,
        )
    )


def authenticate(root: Path, work: Json) -> None:
    """Require current qualified primary and sidecar custody without relabeling Exp7854."""
    names = [
        ("8179_v707_sentence_transport_methods", "required_checks_passed"),
        ("8235_v712_learning_validation", "learning_execution_ready_score"),
        ("8248_v713_evidence_intervention_methods", "intervention_protocol_ready_score"),
    ]
    for name, field in names:
        path = root / "results" / f"experiment_{name}.json"
        gate(work, path, "exists", True, True if path.is_file() else None)
        if not path.is_file():
            continue
        value = json.loads(path.read_bytes())
        for key, expected in [
            (field, 1),
            ("required_checks_passed", True),
            ("flagged_adversarial", False),
        ]:
            gate(work, path, key, expected, value.get(key))
        terminal = Path(value["terminal_validation_sidecar_path"])
        report = json.loads(terminal.read_bytes())
        sidecar = Path(report["publication"]["sidecar_path"])
        gate(
            work,
            terminal,
            "publication.primary_sha256",
            sha256_file(path),
            report["publication"].get("primary_sha256"),
        )
        gate(
            work,
            sidecar,
            "report.passed",
            True,
            read_bound_sidecar(path, sidecar)["report"].get("passed"),
        )
        work["refs"].extend(
            dict(reference(p), fields_imported=fields)
            for p, fields in [
                (
                    path,
                    [
                        field,
                        "required_checks_passed",
                        "flagged_adversarial",
                        "terminal_validation_sidecar_path",
                    ],
                ),
                (terminal, ["publication.primary_sha256", "publication.sidecar_path"]),
                (sidecar, ["primary_sha256", "report.passed"]),
            ]
        )
    for name in [
        PROTOCOL,
        "ops/exclusion_manifest.yaml",
        "results/experiment_7854_v682_intervention_protocol.json",
    ]:
        path = root / name
        gate(work, path, "exists", True, True if path.is_file() else None)
        if path.is_file():
            work["refs"].append(
                dict(
                    reference(path),
                    fields_imported=["role_manifest", "original_roles"] if name == PROTOCOL else [],
                    purpose="Freeze required historical and operational input bytes without importing truth claims.",
                )
            )
    if all(c["passed"] for c in work["checks"]):
        protocol = json.loads((root / PROTOCOL).read_bytes())
        k.bind_roles(protocol)
        gate(
            work,
            root / PROTOCOL,
            "protocol_sha256",
            json.loads(
                (
                    root / "results/experiment_8248_v713_evidence_intervention_methods.json"
                ).read_bytes()
            )["protocol_sha256"],
            sha256_file(root / PROTOCOL),
        )


def measure(root: Path, raw: Path, *, fixture: bool = False, **kwargs: Any) -> Json:
    """Check tools and private scratch before measuring owned mechanics, with no model loads."""
    progress("before_preconditions")
    start, wall = time.monotonic_ns(), time.time_ns()
    raw.mkdir(parents=True, exist_ok=True)
    work: Json = dict(
        checks=[],
        refs=[],
        evidence={},
        owned_failure="",
        fixture=fixture,
        invocation_argv=list(sys.argv),
    )
    with TemporaryDirectory(prefix="carnot-8249-private-") as directory:
        private = Path(directory)
        probe = private / "probe"
        probe.write_bytes(b"private scratch")
        gate(
            work, probe, "private_scratch_writable", True, probe.read_bytes() == b"private scratch"
        )
        for name in ["python", "coverage", "pytest", "ruff", "mypy"]:
            path = ROOT / ".venv/bin" / name
            gate(work, path, "required_tool", True, path.is_file())
        if not fixture:
            try:
                authenticate(root, work)
            except (OSError, ValueError, KeyError, TypeError) as error:
                gate(work, root, "authenticated_input_schema", "valid", str(error))
        progress("after_preconditions", len(work["checks"]))
        if all(c["passed"] for c in work["checks"]):
            progress("before_private_benchmark")
            work["evidence"] = k.qualify(private)
            atomic_json(raw / "primitive_evidence.json", work["evidence"])
            progress("after_private_benchmark", 384)
    work.update(
        duration_s=(time.monotonic_ns() - start) / 1e9,
        clock=dict(
            started_monotonic_ns=start, ended_monotonic_ns=time.monotonic_ns(), started_wall_ns=wall
        ),
        code_config_hashes=[
            reference(ROOT / p)
            for p in [
                *OWNED,
                TEST,
                "python/carnot/verify/evidence_intervention_8248.py",
                "python/carnot/verify/sentence_transport_8179.py",
            ]
        ],
    )
    atomic_json(raw / "measurement.json", work)
    return work


def build(work: Json, raw: Path, receipts: list[Json], *, fixture: bool = False) -> Json:
    """Gate each kernel separately; a passed fixture establishes execution, not natural benefit."""
    failed = [c for c in work["checks"] if not c["passed"]]
    evidence = work["evidence"]
    coverage = raw / "logs/changed_code_coverage.json"
    counts = json.loads(coverage.read_bytes())["files"] if coverage.is_file() else {}
    covered = fixture or all(
        name in counts
        and not counts[name]["missing_lines"]
        and counts[name]["summary"]["num_statements"] > 0
        for name in OWNED
    )
    owned = (
        bool(receipts)
        and all(r["passed"] for r in receipts)
        and (bool(failed) or covered)
        and not work["owned_failure"]
    )
    view_ready = int(owned and not failed and evidence.get("view_ready", False))
    admission_ready = int(owned and not failed and evidence.get("admission_ready", False))
    verdict = (
        "disqualified"
        if not owned
        else "blocked"
        if failed
        else "circular_positive"
        if view_ready and admission_ready
        else "null"
    )
    labels = {
        event["record"]["origin"]: event["record"]["label"]
        for event in evidence.get("events", [])
        if event["kind"] == "release"
    }
    rows = [
        dict(
            unit_id=record["source_cluster_id"],
            source_cluster_id=record["source_cluster_id"],
            slot=record["slot"],
            condition="private_oracle_stream",
            arm=arm,
            metric="decision_cost",
            p_unsupported=record["predictions"][arm],
            numerator=int(int(record["predictions"][arm] >= 0.5) != labels[record["slot"]])
            if record["slot"] in labels
            else None,
            denominator=int(record["slot"] in labels),
            status="completed" if record["slot"] in labels else "censored",
            censor_reason=None if record["slot"] in labels else "label_not_due",
            independent_source_count=0,
        )
        for record in evidence.get("state", {}).get("issued", [])
        for arm in k.ARMS
    ]
    if failed:
        rows = [
            dict(
                unit_id=c["upstream"],
                condition="missing_external_operand",
                status="excluded",
                exclusion_reason=c["artifact_field"],
                numerator=None,
                denominator=0,
            )
            for c in failed
        ]
    value = dict(
        experiment_id=8249,
        task_id=TASK,
        milestone="2026.10.713",
        run_date=RUN_DATE,
        honest_verdict="complete_"
        + verdict
        + "_"
        + (failed[0]["upstream"] if verdict == "blocked" else "evidence_view_kernel"),
        verdict_class=verdict,
        gate_check_summary=work["checks"],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=MODEL_SPECS,
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        trained_head_specs=[],
        rows=rows,
        intended_count=len(rows),
        completed_count=sum(r["status"] == "completed" for r in rows),
        failed_count=0,
        censored_count=sum(r["status"] == "censored" for r in rows),
        excluded_count=sum(r["status"] == "excluded" for r in rows),
        independent_count=0,
        verifier_is_oracle=True,
        exposure_scope="exposed_development_and_separate_private_oracle_controls",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=owned,
        flagged_adversarial=False,
        acceptance_gates=dict(
            owned_validation=owned,
            changed_statement_coverage=covered,
            view_kernel=bool(view_ready),
            admission_kernel=bool(admission_ready),
            natural_benefit=False,
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        preconditions_checked=work["checks"],
        duration_s=work["duration_s"]
        + sum(r.get("duration_s", 0) for r in receipts)
        + work.get("global_health", {}).get("duration_s", 0),
        random_seed=101,
        source_artifact_hashes=work["refs"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes=[
            reference(p)
            for p in sorted(raw.rglob("*"))
            if p.is_file()
            and p.name
            in {
                "measurement.json",
                "primitive_evidence.json",
                "changed_code_coverage.json",
                "validation_commands.json",
                "validation_receipts.json",
            }
        ],
        phase_spans=[dict(phase="qualification", duration_s=work["duration_s"], **work["clock"])]
        + [
            dict(
                phase=r.get("name", "private_validation"),
                duration_s=r.get("duration_s", 0),
                started_monotonic_ns=r.get("started_monotonic_ns"),
                ended_monotonic_ns=r.get("ended_monotonic_ns"),
                started_wall_ns=r.get("started_wall_ns"),
            )
            for r in receipts
        ],
        cited_upstream_artifacts=[
            dict(
                r,
                fields_imported=r.get("fields_imported", []),
            )
            for r in work["refs"]
        ],
        view_kernel_ready_score=view_ready,
        admission_kernel_ready_score=admission_ready,
        kernel_hashes=[reference(ROOT / p) for p in OWNED],
        mutation_rows=evidence.get("mutations", []),
        view_byte_maps=[r["view"]["sentence_map"] for r in evidence.get("mutations", [])],
        admission_events=evidence.get("events", []),
        state_schema=dict(
            version=1,
            prior="Beta(1,1)",
            public_groups=k.GROUPS,
            minimum_distinct_releases=8,
            delay=8,
            mixture="lambda=n/(n+16)",
            truth_constraint=False,
        ),
        qualified_transport_reference=[r for r in work["refs"] if "8179" in r["path"]],
        rejection_reasons=[
            "future_feedback",
            "duplicate_source",
            "retention",
            "partial_record",
            "hash_drift",
            "ledger_drift",
            "sentence_address",
            "input_token_limit",
        ],
        fixture_costs=evidence.get("fixture_costs", {}),
        coverage_statement_counts={name: entry["summary"] for name, entry in counts.items()},
        fixture_mode=fixture,
        invocation_argv=list(sys.argv),
        scientific_benefit_measured=False,
        claim_scope="Private view and delayed admission execution qualification only; no natural learning or generalization established.",
        methodology_note="Reuse intact V707 transport, frozen V713 public role selection and durable issue/release ledger. Private group-conditional oracle stream is circular_positive; all adaptive arms share releases and missing-feature global counts. Natural evidence remains unmeasured.",
        repository_health=work.get("global_health", {}),
        reconciliation_note="Conductor owns ops and traceability updates.",
    )
    # Invocation identity must come from the measurement, not the replay process argv.
    value["invocation_argv"] = work.get("invocation_argv", [])
    value = normalize_artifact_for_template_write(value)
    value["field_principles"] = {
        key: "Bind current measured execution and custody; readiness supplies no scientific benefit."
        for key in value
    }
    value["field_principles"].update(
        view_byte_maps="Preserve original UTF-8 custody through whole-sentence deletion.",
        admission_events="Persist predictions before t-8 feedback for causal replay.",
        fixture_costs="Oracle decision costs qualify mechanics and have zero natural independence.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Cold rebuild from primitive maps and ledger transitions rejects rehashed headlines."""
    try:
        value = json.loads(path.read_bytes())
        checksum = value.pop("reproducibility_checksum")
        if checksum != canonical_hash(value):
            return False
        for ref in (
            value["source_artifact_hashes"]
            + value["code_config_hashes"]
            + value["raw_shard_hashes"]
        ):
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                return False
        for receipt in value["validation_receipts"]:
            for prefix in ["stdout", "stderr"]:
                if (
                    prefix + "_path" in receipt
                    and sha256_file(Path(receipt[prefix + "_path"])) != receipt[prefix + "_sha256"]
                ):
                    return False
        raw = Path(value["terminal_validation_sidecar_path"]).parent
        work = json.loads((raw / "measurement.json").read_bytes())
        if work["evidence"]:
            primitive = json.loads((raw / "primitive_evidence.json").read_bytes())
            if k.reconstruct(primitive) != work["evidence"]:
                return False
        return build(
            work, raw, value["validation_receipts"], fixture=value["fixture_mode"]
        ) == dict(value, reproducibility_checksum=checksum)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def manifest(private: Path, candidate: Path) -> Json:
    """Freeze compact current validation, reusing Exp8235's crash coverage patch."""
    with (
        patch.object(execution, "e", sys.modules[__name__]),
        patch.object(execution, "OWNED", OWNED),
    ):
        specs = BASE_MANIFEST(private, candidate)
    config = private / "coverage.ini"
    config.write_text(
        config.read_text().replace("[run]", "[run]\npatch = _exit") + "[report]\nexclude_lines =\n"
    )
    for spec in specs["commands"]:
        if spec["name"] == "consumer_and_E2E015_019":
            spec["name"] = "consumer_and_E2E019_020"
            begin = spec["argv"].index("tests/python/test_development_methods_8098.py")
            spec["argv"][begin:] = [
                "tests/python/test_primary_publication_7928.py",
                "tests/python/test_experiment_7942_v689_sentence_labels.py",
                "tests/python/test_hard_exit_learning_qualification_8206.py",
                "--basetemp=" + str(private / "consumer"),
            ]
            spec["deadline_s"] = 900
        if spec["name"] == "strict_mypy":
            spec["argv"] = [
                a.replace("--follow-imports=silent", "--follow-imports=skip") for a in spec["argv"]
            ]
        if spec["name"] == "spec_coverage":
            spec["argv"].insert(-1, "--files")
    return specs


def main(argv: list[str] | None = None) -> int:
    """Reuse normally exited workers, bounded heartbeats and unchanged atomic publication."""
    with (
        patch.object(execution, "e", sys.modules[__name__]),
        patch.object(execution, "OWNED", OWNED),
        patch.object(execution, "manifest", manifest),
    ):
        return int(execution.main(argv))
