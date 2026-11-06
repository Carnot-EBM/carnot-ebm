"""REQ-REPORT-8883: retain private paired operands and publish only validated boundary evidence."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from zipfile import ZipFile
import hashlib

from carnot.reporting import evidence_loss_boundary_8883 as m
from carnot.reporting import size_aware_verifier_8871 as authority
from carnot.reporting import selective_methods_runner_8829 as inherited
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
SOURCE = "experiment_8868_v759_conditional_risk_methods"
UPSTREAM = {
    SOURCE: (
        "sha256:af45370397fc99a00514d511a3c81d29790a879e3bb6bafcf55701e0088d5207",
        "conditional_methods_ready_score",
    ),
    "experiment_8870_v759_conditional_risk_evaluation": (
        "sha256:5dc1020bfc6fa45d40a1465c2353555291479b250d71f0f7cf2242458a9a8960",
        "conditional_rows_ready_score",
    ),
    "experiment_8871_v759_size_aware_verifier": (
        "sha256:079a01a6b8a926a65ffa2e43db3b76c32de507adf8ab7113e1a3931fc3c5b176",
        "size_dispatch_ready_score",
    ),
    "experiment_8878_v760_branch_contract": (
        "sha256:697c14905bb02ed4f2fce51028575e7a71bb8ad58b0e5eb8a9bb532f53532a21",
        "contract_ready_score",
    ),
}
DESIGN = "openspec/change-proposals/research-methods-v760-20261106.json"
DESIGN_PIN = "sha256:c99296bd441c10200bad04e77643649bd65f62fb2199029302b24473af6da3d7"
HEALTH = Path("/tmp/exp8883-health/receipt.json")


def preconditions(raw: Path, sources: Path = m.ROOT / "results") -> Json:
    """Authenticate original producers, retained bytes, execution authority and private capacity."""
    checker = namespace(authority, "boundary8883_authority")
    checker.HISTORY = [
        (sources / (n + ".json"), pin, field) for n, (pin, field) in UPSTREAM.items()
    ]
    pre = dict(checker.preconditions(raw))
    path = m.ROOT / DESIGN
    observed = sha256_file(path) if path.is_file() else None
    if observed != DESIGN_PIN:
        pre["diagnostics"].append(
            dict(
                check="missing_file" if observed is None else "rejected_producer",
                upstream="v760_design",
                artifact_field="sha256",
                expected=DESIGN_PIN,
                observed=observed,
                path=str(path),
                sha256=observed,
            )
        )
        pre["external_ready"] = False
    else:
        pre["references"].append(reference(path))
    return pre


def installed(raw: Path, pairs: list[Json], package: Json) -> Json:
    """Install retained wheel bytes privately; send only visible facts and queries to the child."""
    raw.mkdir(parents=True, exist_ok=True)
    wheel = raw / Path(package["path"]).name
    shutil.copyfile(package["path"], wheel)
    atomic_json(raw / "requests.json", dict(payloads=[p["input"] for p in pairs]))
    receipts = []
    with TemporaryDirectory(prefix="exp8883-installed-") as directory:
        private = Path(directory)
        python = private / "venv/bin/python"
        for name, command in [
            ("venv", ["uv", "venv", str(private / "venv"), "--python", sys.executable]),
            ("install", ["uv", "pip", "install", "--python", str(python), "--no-deps", str(wheel)]),
            (
                "exact_views",
                [
                    "env",
                    "-u",
                    "PYTHONPATH",
                    str(python),
                    "-u",
                    str(m.ROOT / m.OWNED[2]),
                    str(raw / "requests.json"),
                    str(raw / "responses.json"),
                ],
            ),
        ]:
            m.progress("before_" + name, 0, len(pairs))
            receipts.append(stream(name, command, raw / "logs", deadline=120))
            m.progress("after_" + name, len(pairs), 0)
            if not receipts[-1]["passed"]:
                raise ValueError("owned_installed_child_failure")
        response = load(reference(raw / "responses.json"))
        with ZipFile(wheel) as archive:
            expected = (
                "sha256:"
                + hashlib.sha256(archive.read("carnot/verify/size_aware_witness.py")).hexdigest()
            )
        if (
            not response["imported_path"].startswith(str(private))
            or response["imported_sha256"] != expected
        ):
            raise ValueError("owned_installed_import_drift")
    return dict(response, receipts=receipts, package=reference(wheel))


def measurement(raw: Path, worlds: list[Json]) -> Json:
    """Retain frozen heads and rules before serving, then open independent source/view labels."""
    source = load(reference(raw / "authorities" / (SOURCE + ".json")))
    fitted = load(source["fit_manifest"])
    certificate = next(
        ref for ref in source["raw_shard_hashes"] if Path(ref["path"]).name == "certificates.json"
    )
    choices = m.rules(load(certificate))
    atomic_json(raw / "frozen_policies.json", dict(fitted=fitted, choices=choices))
    installed_source = load(
        reference(raw / "authorities/experiment_8871_v759_size_aware_verifier.json")
    )
    pairs = [p for w in worlds for p in w["pairs"]]
    started = time.monotonic()
    m.progress("before_installed_benchmark", 0, len(pairs))
    responses = installed(raw / "installed", pairs, installed_source["package_hash"])
    m.progress("after_installed_benchmark", len(pairs), 0)
    decisions = {
        canonical_hash(p["input"]): d for p, d in zip(pairs, responses["rows"], strict=True)
    }
    m.progress("before_frozen_prediction_and_independent_labels", 0, len(pairs))
    result = m.evaluate(
        worlds, fitted, choices, lambda payload, **kw: decisions[canonical_hash(payload)]
    )
    m.progress("after_frozen_prediction_and_independent_labels", len(pairs), 0)
    result["installation_receipts"] = responses["receipts"]
    result["package_hash"] = responses["package"]
    result["pilot_runtime"] = dict(
        duration_s=time.monotonic() - started,
        projected_total_s=(time.monotonic() - started) + 1800,
        limit_s=4200,
    )
    if result["pilot_runtime"]["projected_total_s"] >= 4200:
        raise ValueError("owned_runtime_budget")
    return result


def prepare(raw: Path, sources: Path, limit: int | None) -> Json:
    """Keep all scheduled slots when an external input blocks or an owned control fails."""
    started = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw.chmod(0o700)
    m.progress("before_preconditions")
    pre = preconditions(raw / "authorities", sources)
    m.progress("after_preconditions")
    template = ExperimentTemplate(
        8883, "Source and visible evidence boundary", "results/" + m.NAME + ".json"
    )
    with template.phase("freeze_before_scoring"):
        exposed: set[str] = set()
        if pre["external_ready"]:
            original = load(
                next(ref for ref in pre["references"] if Path(ref["path"]).name == SOURCE + ".json")
            )
            cohort = load(original["population_manifest"])
            exposed = set(cohort["exposed_invariants"]) | {
                s for w in cohort["worlds"] for s in w["relevant_structures"]
            }
        worlds = m.freeze(exposed, count=limit or 30)
        atomic_json(raw / "frozen_worlds.json", dict(worlds=worlds, exposed=sorted(exposed)))
    result: Json = dict(
        rows=m.scheduled(worlds, "external_or_resource_gate", "censored"),
        truth_rows=[],
        controls=dict(passed=False),
        installation_receipts=[],
    )
    errors = []
    if pre["external_ready"] and pre["resources"]["passed"]:
        try:
            with template.phase("paired_evidence_boundary"):
                result = measurement(raw, worlds)
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(str(error))
            result["rows"] = m.scheduled(worlds, str(error), "failed")
    ended = time.monotonic_ns()
    substrate = "blocked_no_run" if not pre["external_ready"] else "no_model_load"
    receipt = build_current_work_receipt(
        run_id=m.TASK + "/" + str(started),
        owner_pid=os.getpid(),
        events=[],
        inference_substrate=substrate,
        inference_substrate_class=substrate,
        inference_substrate_details=dict(
            compute="CPU frozen numerical heads and installed exact records", model_invoked=False
        ),
        execution_venue="host",
        started_monotonic_ns=started,
        ended_monotonic_ns=ended,
        phase_spans=[
            dict(phase="authenticated_paired_boundary", start_s=0, end_s=(ended - started) / 1e9)
        ],
    )
    return dict(
        result,
        worlds=worlds,
        private_world_limit=limit,
        preconditions_checked=pre,
        internal_errors=errors,
        current_work_receipt=receipt,
        protocol=m.protocol(),
        producer_invocation=dict(argv=sys.argv, cwd=str(Path.cwd()), python=sys.executable),
        code_config_hashes=[
            reference(m.ROOT / p)
            for p in m.OWNED
            + m.TESTS
            + [
                "python/carnot/reporting/current_work_receipt.py",
                "python/carnot/reporting/primary_publication.py",
                "scripts/experiment_template.py",
                "scripts/adversarial_verify.py",
                "scripts/verdict_row_consistency_lint.py",
            ]
        ],
        raw_shard_hashes=[reference(p) for p in sorted(raw.rglob("*")) if p.is_file()],
    )


def validation_stream(name: str, argv: list[str], raw: Path, **kwargs: Any) -> Json:
    """Allow bounded checkout-guard startup while preserving every collected test and check."""
    if name == "collect_once":
        kwargs["deadline"] = 180
    return dict(stream(name, argv, raw, **kwargs))


def validate(raw: Path, private: Path) -> Json:
    """Retain measured unit/child coverage, strict checks and private original E2E-018/021."""
    checker = namespace(inherited, "boundary8883_validation")
    checker.m = m
    checker.stream = validation_stream
    return dict(checker.validate(raw, private))


def artifact(work: Json, validation: Json, date: str, work_ref: Json, validation_ref: Json) -> Json:
    """Administrative paired readiness cannot become a source-extraction fidelity claim."""
    rows, pre, receipt = work["rows"], work["preconditions_checked"], work["current_work_receipt"]
    summary = m.summarize(rows)
    gates = dict(
        authorities=pre["external_ready"],
        resources=pre["resources"]["passed"],
        full_roster=work["private_world_limit"] is None
        and len(work["worlds"]) == len({w["cluster_id"] for w in work["worlds"]}) == 120
        and len(rows) == 9000
        and all(sum(w["group"] == g for w in work["worlds"]) == 30 for g in m.base.GROUPS),
        paired_controls=summary["omission_controls"]["passed"],
        owned_execution=not work["internal_errors"],
        owned_validation=qualified(validation),
    )
    ready = int(all(gates.values()))
    verdict = (
        "blocked" if not pre["external_ready"] else "circular_positive" if ready else "disqualified"
    )
    pairs = [p for w in work["worlds"] for p in w["pairs"]]
    value: Json = dict(
        summary,
        experiment_id=8883,
        task_id=m.TASK,
        milestone="2026.11.760",
        run_date=date,
        honest_verdict="complete_" + verdict + "_evidence_loss_boundary",
        verdict_class=verdict,
        inference_substrate=receipt["inference_substrate"],
        inference_substrate_class=receipt["inference_substrate_class"],
        execution_venue="host",
        MODEL_SPECS=[],
        model_invocation_counts=receipt["invocation_counts"],
        numerical_training_calls=0,
        verifier_is_oracle=True,
        exposure_scope="Constructed signed-record worlds in four exposed query-topology groups; no natural text or extraction certification.",
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
        cited_upstream_artifacts=[reference(m.ROOT / "results" / (n + ".json")) for n in UPSTREAM],
        rows=rows,
        intended_count=len(rows),
        completed_count=sum(r["status"] == "completed" for r in rows),
        failed_count=sum(r["status"] == "failed" for r in rows),
        censored_count=sum(r["status"] == "censored" for r in rows),
        excluded_count=0,
        independent_count=len(work["worlds"]),
        gate_check_summary=pre["diagnostics"]
        + [
            dict(
                check="owned_gate_failure",
                upstream=m.TASK,
                artifact_field=k,
                expected=True,
                observed=v,
                path=work_ref["path"],
                sha256=work_ref["sha256"],
            )
            for k, v in gates.items()
            if not v
        ],
        acceptance_gates=gates,
        required_checks_passed=qualified(validation),
        validation_receipts=validation["receipts"] + work["installation_receipts"],
        owned_statement_coverage=validation["coverage"],
        boundary_rows_ready_score=ready,
        source_fidelity_established_score=0,
        current_work_receipt=receipt,
        source_view_pairs=[
            {
                k: p[k]
                for k in ("unit_id", "cluster_id", "group", "view", "source_sha256", "view_sha256")
            }
            for p in pairs
        ],
        transformation_rows=[dict(unit_id=p["unit_id"], **p["transformation"]) for p in pairs],
        full_source_truth={p["unit_id"]: p["source"] for p in work["truth_rows"]},
        visible_view_truth={p["unit_id"]: p["visible"] for p in work["truth_rows"]},
        protocol=work["protocol"],
        producer_invocation=work["producer_invocation"],
        work_reference=work_ref,
        validation_reference=validation_ref,
        repository_health=work.get(
            "repository_health", dict(status="not_run", scope="unrelated_global_health")
        ),
        scientific_motivation=dict(
            url="https://arxiv.org/abs/2605.03534v2",
            scope="SURE-RAG motivates missing-hop and conflict controls; none of its performance claims are imported.",
        ),
        methodology="Freeze120 worlds before scoring; retain five paired source/view transformations. Use five unchanged V759 heads and replicate0 grouped cutoffs; no refitting or selection. Evaluate the retained installed exact solver using only visible records. Open independently enumerated relational source/view truth after serving. Report source acceptance errors separately from visible semantic errors. Average heads within each world; shared exploratory bootstrap draws preserve pairing across arms and views. Certificates for the old population do not transfer. Supported records can omit source conflicts. No general text or source-fidelity guarantee follows.",
    )
    value["field_principles"] = {
        k: "Bind actual producer bytes, scheduled units and measured work; constructed record correctness does not certify extraction."
        for k in value
    }
    value["field_principles"].update(
        boundary_rows_ready_score="Complete controlled paired rows and required owned checks only.",
        source_fidelity_established_score="Fixed zero: even zero observed errors cannot certify extraction or natural text.",
        full_source_truth="Independent hidden complete-source labels used only after serving.",
        visible_view_truth="Independent entailment from the exact visible record set.",
        source_view_pairs="Bind each view to its actual source, including the conflicting counterpart.",
        paired_intervals="Pointwise exploratory whole-world bootstrap; seeds and views cannot inflate independent n.",
        omission_controls="Irrelevant removal must preserve decisions; missing necessary hops force exact abstention.",
        undetectable_omissions="Identical visible records cannot reveal an omitted contradictory or irrelevant source fact.",
    )
    value = normalize_artifact_for_template_write(value)
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Reconstruct hidden/visible semantics and reductions so rehashing false assertions cannot pass."""
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
        frozen = load(
            next(
                ref
                for ref in work["raw_shard_hashes"]
                if Path(ref["path"]).name == "frozen_worlds.json"
            )
        )
        if work["protocol"] != m.protocol() or work["worlds"] != m.freeze(
            set(frozen["exposed"]), count=work["private_world_limit"] or 30
        ):
            return False
        if validate_current_work_receipt(work["current_work_receipt"]):
            return False
        if work["truth_rows"]:
            with TemporaryDirectory(prefix="exp8883-replay-") as directory:
                fresh = Path(directory)
                shutil.copytree(
                    Path(value["work_reference"]["path"]).parent / "authorities",
                    fresh / "authorities",
                )
                repeated = measurement(fresh, work["worlds"])
            if any(repeated[k] != work[k] for k in ("rows", "truth_rows", "controls")):
                return False
        if value["required_checks_passed"]:
            checker = namespace(ordered.checker, "boundary8883_coverage_replay")
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
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return False


def terminal(path: Path, raw: Path) -> Json:
    """The unchanged scanners and an actual foreign-directory CLI check prospective bytes."""
    checker = namespace(ordered, "boundary8883_terminal")
    checker.m = m
    return dict(checker.terminal(path, raw))


def main(argv: list[str] | None = None) -> int:
    """Keep checkpoints private until terminal validation; private subsets never grant readiness."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261106"], default="20261106")
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
    health = HEALTH
    if health.is_file():
        shutil.copytree(health.parent, raw / "repository_health")
        work["repository_health"] = dict(
            load(reference(raw / "repository_health/receipt.json")),
            logs=[reference(p) for p in sorted((raw / "repository_health").iterdir())],
        )
    atomic_json(raw / "work.json", work)
    m.progress("before_owned_validation")
    with TemporaryDirectory(prefix="exp8883-validation-") as directory:
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
    m.progress("before_publication")
    publication = publish_primary(output, value, lambda candidate: terminal(candidate, raw))
    atomic_json(raw / "publication_receipt.json", publication)
    for operand in raw.rglob("*"):
        if operand.is_file():
            operand.chmod(0o444)
    m.progress("complete_" + value["verdict_class"], value["completed_count"], 0)
    return 0
