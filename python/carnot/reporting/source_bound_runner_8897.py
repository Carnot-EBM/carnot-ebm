"""REQ-REPORT-8897: retain private source operands and publish only checked terminal bytes."""

import argparse
import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from zipfile import ZipFile

from carnot.reporting import source_bound_decisions_8897 as m
from carnot.reporting import size_aware_verifier_8871 as authority
from carnot.reporting import evidence_loss_runner_8883 as inherited
from carnot.reporting import ordered_evidence_runner_8831 as ordered
from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    sha256_file,
    build_current_work_receipt,
    validate_current_work_receipt,
)
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.selective_runner_8816 import stream, unqualified
from carnot.reporting.selective_methods_runner_8829 import qualified
from carnot.reporting.v746_retained_evidence import reference, load
from carnot.reporting.v748_collection_contract import namespace
from scripts.experiment_template import ExperimentTemplate, normalize_artifact_for_template_write

Json = dict[str, Any]
SOURCE = "experiment_8883_v760_evidence_loss_boundary"
PIN = "sha256:bfe0873cc66771673ad71b0df1fe633112c0b229c75d926ccbd4e8c0f035e011"
DESIGN = "openspec/change-proposals/research-methods-v761-20261107.json"
DESIGN_PIN = "sha256:e02928f8cce705aa5e62d57ad3a591f9b3cb9cda1c011fddd1867245fb0efe88"
HEALTH = Path("/tmp/exp8897-health/receipt.json")


def preconditions(raw: Path, sources: Path) -> Json:
    """Authenticate the original boundary producer, all imported bytes and private capacity."""
    checker = namespace(authority, "source8897_authority")
    checker.HISTORY = [(sources / (SOURCE + ".json"), PIN, "boundary_rows_ready_score")]
    resolutions = []

    def original_bytes(ref: Json) -> Json:
        value = load(ref)
        for operand in value.get("source_artifact_hashes", []):
            path = Path(operand["path"])
            relative = str(path.relative_to(m.ROOT))
            if (
                relative
                in {
                    "openspec/change-proposals/research-roadmap-vNEXT.md",
                    "research-roadmap.yaml",
                    "research-references.md",
                }
                and sha256_file(path) != operand["sha256"]
            ):
                m.progress("before_historical_bytes_" + path.name)
                argv = ["git", "show", "c94186ade2:" + relative]
                child = subprocess.run(
                    argv, cwd=m.ROOT, capture_output=True, timeout=15, check=False
                )
                target = raw / "historical" / path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(child.stdout)
                if sha256_file(target) != operand["sha256"]:
                    raise ValueError("historical_input_drift")
                resolutions.append(
                    dict(
                        original_path=str(path),
                        resolved_reference=reference(target),
                        argv=argv,
                        exit_code=child.returncode,
                        producer_sha256=ref["sha256"],
                    )
                )
                operand["path"] = str(target)
                m.progress("after_historical_bytes_" + path.name)
        return value

    checker.load = original_bytes
    pre = dict(checker.preconditions(raw))
    pre["historical_reference_resolutions"] = resolutions
    design = m.ROOT / DESIGN
    observed = sha256_file(design) if design.is_file() else None
    if observed != DESIGN_PIN:
        pre["external_ready"] = False
        pre["diagnostics"].append(
            dict(
                check="missing_file" if observed is None else "rejected_producer",
                upstream="v761_execution_design",
                artifact_field="sha256",
                expected=DESIGN_PIN,
                observed=observed,
                path=str(design),
                sha256=observed,
            )
        )
    else:
        pre["references"].append(reference(design))
    return pre


def installed(raw: Path, requests: list[Json]) -> Json:
    """Install a private wheel with the opt-in module; no release or source wheel is changed."""
    raw.mkdir(parents=True, exist_ok=True)
    original = load(reference(m.ROOT / "results/experiment_8871_v759_size_aware_verifier.json"))[
        "package_hash"
    ]
    if sha256_file(Path(original["path"])) != original["sha256"]:
        raise ValueError("owned_installed_original_bytes")
    wheel = raw / Path(original["path"]).name
    module = m.ROOT / m.OWNED[0]
    name = "carnot/verify/source_bound_witness.py"
    m.progress("before_private_wheel")
    with ZipFile(original["path"]) as source, ZipFile(wheel, "w") as target:
        record = next(n for n in source.namelist() if n.endswith(".dist-info/RECORD"))
        for entry in source.infolist():
            if entry.filename != record:
                target.writestr(entry, source.read(entry.filename))
        data = module.read_bytes()
        target.writestr(name, data)
        records = list(csv.reader(io.StringIO(source.read(record).decode())))
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        records.append([name, "sha256=" + digest, str(len(data))])
        buffer = io.StringIO()
        csv.writer(buffer).writerows(records)
        target.writestr(record, buffer.getvalue())
    m.progress("after_private_wheel")
    atomic_json(raw / "requests.json", dict(requests=requests))
    receipts = []
    with TemporaryDirectory(prefix="exp8897-installed-") as directory:
        private = Path(directory)
        python = private / "venv/bin/python"
        for phase, argv in [
            ("venv", ["uv", "venv", str(private / "venv"), "--python", sys.executable]),
            ("install", ["uv", "pip", "install", "--python", str(python), "--no-deps", str(wheel)]),
            (
                "installed_callable",
                [
                    "env",
                    "-u",
                    "PYTHONPATH",
                    str(python),
                    "-I",
                    "-u",
                    "-c",
                    "import os,runpy,sys; os.chdir(sys.argv.pop(1)); sys.argv[0]=sys.argv.pop(1); runpy.run_path(sys.argv[0],run_name='__main__')",
                    str(private),
                    str(m.ROOT / m.OWNED[-2]),
                    str(raw / "requests.json"),
                    str(raw / "responses.json"),
                ],
            ),
        ]:
            m.progress("before_" + phase, 0, len(requests))
            receipts.append(stream(phase, argv, raw / "logs", deadline=180))
            m.progress("after_" + phase, len(requests), 0)
            if not receipts[-1]["passed"]:
                raise ValueError("owned_installed_child_failure")
        response = load(reference(raw / "responses.json"))
        for ref, expected in zip(
            response["imported"],
            [module, m.ROOT / "python/carnot/verify/size_aware_witness.py"],
            strict=True,
        ):
            if not Path(ref["path"]).is_relative_to(private) or ref["sha256"] != sha256_file(
                expected
            ):
                raise ValueError("owned_installed_import_drift")
    return dict(
        response,
        passed=True,
        receipts=receipts,
        package=reference(wheel),
        original_package=original,
    )


def prepare(raw: Path, sources: Path, count: int | None) -> Json:
    """Seal new sources before serving and preserve the full scheduled roster on failure."""
    started = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw.chmod(0o700)
    m.progress("before_preconditions")
    pre = preconditions(raw / "authorities", sources)
    m.progress("after_preconditions")
    exposed: set[str] = set()
    if pre["external_ready"]:
        original = load(reference(raw / "authorities" / (SOURCE + ".json")))
        previous = load(original["work_reference"])
        exposed = {m.old.base.relevant(p["source"]) for w in previous["worlds"] for p in w["pairs"]}
    template = ExperimentTemplate(
        8897, "Separate typed-source claims", "results/" + m.NAME + ".json"
    )
    with template.phase("freeze_before_serving"):
        worlds = m.freeze(exposed, count=count or 30)
        atomic_json(raw / "worlds.json", dict(worlds=worlds, exposed=sorted(exposed)))
    result: Json = dict(
        rows=m.scheduled(worlds, "external_or_resource_gate", "censored"),
        source_view_pairs=[],
        ledger_validation_rows=[],
        per_world_errors=[],
        unsupported_claim_count=0,
        controls=dict.fromkeys(
            [
                "semantic_parity",
                "provenance_rejection",
                "order_rename_invariance",
                "no_unsupported_claims",
            ],
            False,
        ),
    )
    installation: Json = dict(passed=False, receipts=[], rows=[])
    errors = []
    if pre["external_ready"] and pre["resources"]["passed"]:
        try:
            with template.phase("installed_source_boundary"):
                m.progress("before_installed_benchmark", 0, len(worlds))
                installation = installed(raw / "installed", m.requests(worlds))
                m.progress("after_installed_benchmark", len(worlds), 0)
                result = m.evaluate(worlds, installation["rows"])
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(str(error))
            result["rows"] = m.scheduled(worlds, str(error), "failed")
    ended = time.monotonic_ns()
    substrate = "no_model_load" if pre["external_ready"] else "blocked_no_run"
    receipt = build_current_work_receipt(
        run_id=m.TASK + "/" + str(started),
        owner_pid=os.getpid(),
        events=[],
        inference_substrate=substrate,
        inference_substrate_class=substrate,
        inference_substrate_details=dict(
            compute="CPU exact signed-record decisions and typed-ledger equality",
            model_invoked=False,
        ),
        execution_venue="host",
        started_monotonic_ns=started,
        ended_monotonic_ns=ended,
        phase_spans=[
            dict(phase="authenticated_source_boundary", start_s=0, end_s=(ended - started) / 1e9)
        ],
    )
    return dict(
        result,
        worlds=worlds,
        private_world_limit=count,
        preconditions_checked=pre,
        internal_errors=errors,
        installed_callable_receipt=installation,
        current_work_receipt=receipt,
        producer_invocation=dict(argv=sys.argv, cwd=str(Path.cwd()), python=sys.executable),
        code_config_hashes=[
            reference(m.ROOT / name)
            for name in m.OWNED
            + m.TESTS
            + [
                "python/carnot/verify/size_aware_witness.py",
                "python/carnot/verify/_indexed_witness_core.py",
                "python/carnot/verify/_indexed_witness_meter.py",
                "python/carnot/reporting/current_work_receipt.py",
                "python/carnot/reporting/primary_publication.py",
                "scripts/experiment_template.py",
                "scripts/adversarial_verify.py",
                "scripts/verdict_row_consistency_lint.py",
            ]
        ],
        raw_shard_hashes=[reference(path) for path in sorted(raw.rglob("*")) if path.is_file()],
    )


def validate(raw: Path, private: Path) -> Json:
    """Reuse exact-node coverage and original private E2E-018/021 without changing validators."""
    checker = namespace(inherited, "source8897_validation")
    checker.m = m
    return dict(checker.validate(raw, private))


def artifact(work: Json, validation: Json, date: str, work_ref: Json, validation_ref: Json) -> Json:
    """Administrative readiness qualifies constructed typed records, never natural-text fidelity."""
    pre, receipt, rows = work["preconditions_checked"], work["current_work_receipt"], work["rows"]
    gates = dict(
        work["controls"],
        authorities=pre["external_ready"],
        resources=pre["resources"]["passed"],
        full_roster=work["private_world_limit"] is None
        and len(work["worlds"]) == 120
        and len({world["cluster_id"] for world in work["worlds"]}) == 120
        and len(rows) == 120 * 5 * len(m.ARMS),
        installation=work["installed_callable_receipt"]["passed"],
        owned_execution=not work["internal_errors"],
        owned_validation=qualified(validation),
    )
    ready = int(all(gates.values()))
    verdict = (
        "blocked" if not pre["external_ready"] else "circular_positive" if ready else "disqualified"
    )
    value: Json = dict(
        experiment_id=8897,
        task_id=m.TASK,
        milestone="2026.11.761",
        run_date=date,
        honest_verdict="complete_" + verdict + "_source_bound_decisions",
        verdict_class=verdict,
        inference_substrate=receipt["inference_substrate"],
        inference_substrate_class=receipt["inference_substrate_class"],
        execution_venue="host",
        MODEL_SPECS=[],
        model_invocation_counts=receipt["invocation_counts"],
        numerical_training_calls=0,
        verifier_is_oracle=True,
        exposure_scope="Constructed signed records and caller-supplied typed sources only; no natural text fidelity claim.",
        independent_generalization_score=0,
        duration_s=receipt["duration_s"],
        phase_spans=receipt["phase_spans"],
        preconditions_checked=pre,
        random_seed=m.SEED,
        source_artifact_hashes=pre["references"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes=work["raw_shard_hashes"]
        + validation["raw_references"]
        + [work_ref, validation_ref],
        cited_upstream_artifacts=pre["references"],
        rows=rows,
        intended_count=len(rows),
        completed_count=sum(row["status"] == "completed" for row in rows),
        failed_count=sum(row["status"] == "failed" for row in rows),
        censored_count=sum(row["status"] == "censored" for row in rows),
        excluded_count=0,
        independent_count=len(work["worlds"]),
        gate_check_summary=pre["diagnostics"]
        + [
            dict(
                check="owned_gate_failure",
                upstream=m.TASK,
                artifact_field=key,
                expected=True,
                observed=passed,
                path=work_ref["path"],
                sha256=work_ref["sha256"],
            )
            for key, passed in sorted(gates.items())
            if not passed
        ],
        acceptance_gates=gates,
        required_checks_passed=qualified(validation),
        validation_receipts=validation["receipts"] + work["installed_callable_receipt"]["receipts"],
        owned_statement_coverage=validation["coverage"],
        source_boundary_ready_score=ready,
        record_decision={
            decision: sum(row["record_decision"] == decision for row in rows)
            for decision in ["support", "refute", "insufficient"]
        },
        source_coverage_status={
            status: sum(row["source_coverage_status"] == status for row in rows)
            for status in ["complete", "incomplete", "invalid", "unknown"]
        },
        source_semantics_status="unknown",
        source_view_pairs=work["source_view_pairs"],
        ledger_validation_rows=work["ledger_validation_rows"],
        unsupported_claim_count=work["unsupported_claim_count"],
        per_world_errors=work["per_world_errors"],
        installed_callable_receipt=work["installed_callable_receipt"],
        current_work_receipt=receipt,
        producer_invocation=work["producer_invocation"],
        work_reference=work_ref,
        validation_reference=validation_ref,
        repository_health=work.get(
            "repository_health", dict(status="not_run", scope="unrelated_global_health")
        ),
        methodology="Freeze120 new worlds in four query groups before scoring. Reuse the paired source/view recipe with a new seed, excluding previous source structures. Send visible facts and optional caller-owned typed-source ledgers to a private wheel installation. Independently score full-source truth only after serving. Compare view-only and opt-in decisions across five views, eight corrupted ledger controls, order and entity renaming. Equal typed-source ledgers establish coverage of that supplied source only. Exact constructed checks are circular evidence; natural-text semantics remain unknown.",
        protocol=dict(
            worlds=120,
            views=m.old.VIEWS,
            arms=m.ARMS,
            seed=m.SEED,
            cap=200000,
            model_calls=0,
            pilot_limit_s=4200,
            hard_cap_s=4800,
            inference="exact signed records",
            natural_language_fidelity=False,
        ),
    )
    value["field_principles"] = {
        key: "Bind actual producer bytes and scheduled units; constructed record truth cannot certify natural text."
        for key in value
    }
    value["field_principles"].update(
        source_boundary_ready_score="Administrative parity and provenance gates only; no scientific benefit or release claim.",
        record_decision="Counts reduce exact visible-record decisions from per-unit rows.",
        source_coverage_status="Counts qualify only equality to the caller-supplied complete typed source.",
        source_semantics_status="Always unknown; hashes, ledgers and quoted spans cannot certify natural-language interpretation.",
        independent_count="120 independent constructed source worlds; views, arms and renamings do not increase n.",
        source_view_pairs="Full-source labels are evaluator-only and never sent to the installed callable.",
    )
    value = normalize_artifact_for_template_write(value)
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Rebuild deterministic claims so rehashing a false aggregate cannot authorize it."""
    try:
        value = load(reference(path))
        work, validation = load(value["work_reference"]), load(value["validation_reference"])
        for ref in (
            value["source_artifact_hashes"]
            + value["code_config_hashes"]
            + value["raw_shard_hashes"]
        ):
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                return False
        frozen = load(reference(Path(value["work_reference"]["path"]).parent / "worlds.json"))
        if work["worlds"] != m.freeze(
            set(frozen["exposed"]), count=work["private_world_limit"] or 30
        ):
            return False
        if validate_current_work_receipt(work["current_work_receipt"]):
            return False
        if work["installed_callable_receipt"]["passed"]:
            observations = [m.observe(request) for request in m.requests(work["worlds"])]
            rebuilt = m.evaluate(work["worlds"], observations)
            if observations != work["installed_callable_receipt"]["rows"] or any(
                rebuilt[key] != work[key] for key in rebuilt
            ):
                return False
        if value["required_checks_passed"]:
            checker = namespace(ordered.checker, "source8897_coverage")
            checker.m = m
            if not checker.validation_evidence(validation):
                return False
        return bool(
            value
            == artifact(
                work,
                validation,
                value["run_date"],
                value["work_reference"],
                value["validation_reference"],
            )
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def terminal(path: Path, raw: Path) -> Json:
    """Reuse unchanged scanners and an actual outside-checkout CLI replay."""
    checker = namespace(ordered, "source8897_terminal")
    checker.m = m
    return dict(checker.terminal(path, raw))


def main(argv: list[str] | None = None) -> int:
    """Keep checkpoints private until terminal checks accept the prospective primary bytes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261107"], default="20261107")
    parser.add_argument("--output", type=Path, default=m.ROOT / "results" / (m.NAME + ".json"))
    parser.add_argument("--private-output", action="store_true")
    parser.add_argument("--private-world-limit", type=int)
    parser.add_argument("--sources", type=Path, default=m.ROOT / "results")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = replay(args.cold_replay)
        m.progress("replay_passed" if passed else "replay_rejected")
        return int(not passed)
    output = args.output.absolute()
    if args.private_output and output.resolve().is_relative_to(m.ROOT):
        parser.error("private output must be outside checkout")
    if args.private_world_limit is not None and (
        not args.private_output or args.private_world_limit < 1
    ):
        parser.error("subset requires private output and a positive count")
    raw = output.parent / "raw" / m.NAME / "invocations" / str(time.time_ns())
    work = prepare(raw, args.sources, args.private_world_limit)
    if HEALTH.is_file():
        work["repository_health"] = load(reference(HEALTH))
        atomic_json(raw / "repository_health.json", work["repository_health"])
        work["raw_shard_hashes"].append(reference(raw / "repository_health.json"))
        health_receipt = work["repository_health"].get("receipt", {})
        work["raw_shard_hashes"] += [
            dict(path=health_receipt[side + "_path"], sha256=health_receipt[side + "_sha256"])
            for side in ["stdout", "stderr"]
            if side + "_path" in health_receipt
        ]
    atomic_json(raw / "work.json", work)
    m.progress("before_owned_validation")
    with TemporaryDirectory(prefix="exp8897-validation-") as directory:
        validation = (
            unqualified() if args.private_output else validate(raw / "validation", Path(directory))
        )
    m.progress("after_owned_validation")
    atomic_json(raw / "validation.json", validation)
    value = artifact(
        work,
        validation,
        args.date,
        reference(raw / "work.json"),
        reference(raw / "validation.json"),
    )
    m.progress("before_terminal_validation")
    publication = publish_primary(output, value, lambda candidate: terminal(candidate, raw))
    atomic_json(raw / "publication_receipt.json", publication)
    for operand in raw.rglob("*"):
        if operand.is_file():
            operand.chmod(0o444)
    m.progress("complete_" + value["verdict_class"], value["completed_count"], 0)
    return 0
