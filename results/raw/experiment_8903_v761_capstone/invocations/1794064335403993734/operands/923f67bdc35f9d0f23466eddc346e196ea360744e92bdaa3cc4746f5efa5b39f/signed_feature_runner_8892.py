"""REQ-REPORT-8892: retain private feature pilots until exact terminal validation."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import signed_feature_methods_8892 as m
from carnot.reporting import topology_memory_runner_8881 as inherited
from carnot.reporting import v761_branch_contract as authority
from carnot.reporting import topology_memory_methods_8881 as old
from carnot.reporting.causal_learning_audit_8810 import require
from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    sha256_file,
    build_current_work_receipt,
    validate_current_work_receipt,
)
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.selective_runner_8816 import unqualified
from carnot.reporting.v746_retained_evidence import reference
from carnot.reporting.v748_collection_contract import namespace
from scripts.experiment_template import ExperimentTemplate, normalize_artifact_for_template_write

Json = dict[str, Any]
UPSTREAM = {
    "experiment_8881_v760_topology_memory_methods": (
        "sha256:4d43ee0ea01ae703efb1087b5a2f2a253c529f46d3e2c9d7f87ff2b028d3c1e3",
        "topology_methods_ready_score",
        1,
    ),
    "experiment_8886_v760_label_benefit_audit": (
        "sha256:77a50605e84f41a93aa6c2c9eac6f98fd2c5a6c8e8c130430ea8f7d6e82fa23e",
        "label_benefit_score",
        0,
    ),
}


def preconditions(raw: Path, sources: Path) -> Json:
    """Authenticate original assertions and operands, without depending on unrelated runtime repairs."""
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    refs, diagnostics = [], []
    for name, (pin, field, expected) in UPSTREAM.items():
        source = sources / (name + ".json")
        diagnostic = dict(
            check="missing_file",
            upstream=name,
            artifact_field=field,
            expected=expected,
            observed=None,
            path=str(source),
            hash=None,
        )
        try:
            data = source.read_bytes()
            diagnostic.update(check="rejected_producer")
            value = json.loads(data)
            diagnostic.update(check="missing_field", hash=sha256_file(source))
            actual = value[field]
            diagnostic.update(
                check="observed_zero" if actual == 0 else "rejected_producer", observed=actual
            )
            require(actual == expected, "upstream_assertion")
            diagnostic.update(
                check="rejected_producer",
                artifact_field="sha256",
                expected=pin,
                observed=sha256_file(source),
            )
            require(
                sha256_file(source) == pin
                and value["experiment_id"] == int(name.split("_")[1])
                and value["task_id"].startswith(f"exp{value['experiment_id']}-")
                and value["required_checks_passed"] is True,
                "producer_identity",
            )
            operands = [reference(source)] + [
                r
                for key in ("source_artifact_hashes", "code_config_hashes", "raw_shard_hashes")
                for r in value[key]
            ]
            for i, ref in enumerate(operands):
                path = Path(ref["path"])
                diagnostic.update(path=str(path), expected=ref["sha256"], artifact_field="sha256")
                diagnostic["observed"] = sha256_file(path)
                require(diagnostic["observed"] == ref["sha256"], "original_operand_drift")
                target = raw / "operands" / ref["sha256"].split(":")[1]
                target.parent.mkdir(exist_ok=True)
                if not target.exists():
                    shutil.copyfile(path, target)
                refs.append(dict(reference(target), original_path=str(path)))
                if i % 100 == 0:
                    m.progress("authenticate_" + name, i, len(operands) - i)
        except (OSError, ValueError, KeyError, TypeError) as error:
            diagnostics.append(
                dict(
                    diagnostic,
                    check="missing_file"
                    if isinstance(error, OSError)
                    else "missing_field"
                    if isinstance(error, KeyError)
                    else diagnostic["check"],
                    error=str(error),
                )
            )
    contract = authority.authority(m.ROOT)
    if not contract["ready"]:
        diagnostics.append(
            dict(
                check="rejected_producer",
                upstream="V761",
                artifact_field="execution_authority",
                expected=True,
                observed=False,
                path=str(m.ROOT / authority.DESIGN),
                hash=None,
            )
        )
    for name in (authority.DESIGN, authority.ACTIVE, authority.STAGED, authority.METHODS):
        path = m.ROOT / name
        if path.is_file():
            target = raw / "authority" / path.name
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(path, target)
            refs.append(dict(reference(target), original_path=str(path)))
    atomic_json(raw / "scratch_probe.json", dict(private=True))
    resources: Json = dict(
        private_scratch=True,
        free_disk_bytes=shutil.disk_usage(raw).free,
        tools_available=all(
            (m.ROOT / ".venv/bin" / t).is_file() for t in ("python", "pytest", "ruff", "mypy")
        ),
        numerical_backend="JAX CPU",
        model_loads=0,
        pilot_budget_s=300,
        total_budget_s=4200,
        hard_cap_s=4800,
    )
    resources["passed"] = (
        resources["tools_available"] and resources["free_disk_bytes"] >= 512 * 1024**2
    )
    return dict(
        external_ready=not diagnostics,
        references=refs,
        diagnostics=diagnostics,
        resources=resources,
        authority_ready=contract["ready"],
    )


def validate(raw: Path, private: Path) -> Json:
    """Reuse original exact-node, child coverage, lint, type and private E2E checks."""
    runner = namespace(inherited, "signed8892_validation")
    runner.m = m
    return dict(runner.validate(raw, private))


def terminal(candidate: Path, raw: Path) -> list[Json]:
    """Run the unchanged scanners and the real foreign-directory cold replay."""
    runner = namespace(inherited, "signed8892_terminal")
    runner.m = m
    return list(runner.terminal(candidate, raw))


def census(reason: str) -> Json:
    """Keep all scheduled pilot slots when external evidence or an owned check fails."""
    p = m.protocol()
    p["cohort_manifest"] = p["pilot_manifest"]
    rows = [
        dict(r, arm=f + ":" + r["arm"], feature_arm=f)
        for f in ("old", "signed")
        for r in old.census(p, reason)
    ]
    return dict(
        rows=rows,
        numerical_training_calls=0,
        pilots=[],
        control_rows=[],
        feature_collision_rows=[],
        feature_work_counts=[],
        development_manifest={},
        schema_compatibility_rows=[],
        matched_work_controls=[],
        fit_rows=[],
    )


def measure(raw: Path) -> Json:
    """Seal development and exactly two pilots; successor evaluation never opens here."""
    p = m.protocol()
    atomic_json(raw / "protocol.json", p)
    dev = m.development(p)
    atomic_json(raw / "development.json", dev)
    fitted = m.fit(dev)
    atomic_json(raw / "fit.json", fitted)
    rows, pilots, controls, work = [], [], m.controls(), []
    for stream in p["pilot_manifest"]:
        result = m.pilot(raw / "checkpoints" / stream["stream_id"], stream, fitted, p)
        atomic_json(raw / (stream["stream_id"] + ".json"), result)
        pilots.append(reference(raw / (stream["stream_id"] + ".json")))
        rows.extend(result["rows"])
        controls.extend(result["controls"])
        for arm in result["arms"]:
            observations = arm["observations"]
            work.append(
                dict(
                    cluster_id=stream["stream_id"],
                    feature_arm=arm["feature_arm"],
                    observations=len(observations),
                    dummy_mask_coordinates=2 * len(observations)
                    if arm["feature_arm"] == "old"
                    else 0,
                    deployed_vector_coordinates=9 * len(observations),
                    **{k: sum(o["work"][k] for o in observations) for k in observations[0]["work"]},
                )
            )
    return dict(
        rows=rows,
        numerical_training_calls=2784,
        pilots=pilots,
        control_rows=controls,
        feature_collision_rows=dev["feature_collision_rows"],
        feature_work_counts=work,
        development_manifest=dev["manifest"],
        schema_compatibility_rows=m.controls()[-1:],
        matched_work_controls=controls,
        fit_rows=fitted["fit_rows"],
    )


def recover(raw: Path, reason: str) -> Json:
    """Retain completed pilot units and actual fitting calls after an owned interruption."""
    result = census(reason)
    completed: dict[tuple[str, str], Json] = {}
    if (raw / "fit.json").is_file():
        fitted = json.loads((raw / "fit.json").read_bytes())
        result.update(
            numerical_training_calls=fitted["numerical_training_calls"], fit_rows=fitted["fit_rows"]
        )
    for stream in m.protocol()["pilot_manifest"]:
        path = raw / (stream["stream_id"] + ".json")
        if path.is_file():
            pilot = json.loads(path.read_bytes())
            completed.update({(r["unit_id"], r["arm"]): r for r in pilot["rows"]})
            result["pilots"].append(reference(path))
            result["control_rows"].extend(pilot["controls"])
            result["numerical_training_calls"] += pilot["numerical_training_calls"]
    result["rows"] = [completed.get((r["unit_id"], r["arm"]), r) for r in result["rows"]]
    return result


def verify_inputs(value: Json) -> None:
    """Require the complete pinned operand closure, even when a caller rehashes a false manifest."""
    code = {r["path"]: r for r in value["code_config_hashes"]}
    require(all(str(m.ROOT / p) in code for p in m.OWNED + m.TESTS), "owned_source_inventory")
    require(
        value["producer_invocation"]["executed_sources"]
        == [code[str(m.ROOT / p)] for p in m.OWNED],
        "executed_source_drift",
    )
    refs = value["preconditions_checked"]["references"]
    require(refs == value["source_artifact_hashes"], "source_inventory_drift")
    retained = {(r["original_path"], r["sha256"]) for r in refs}
    if value["preconditions_checked"]["external_ready"]:
        for name, (pin, field, expected) in UPSTREAM.items():
            source = next(
                (
                    r
                    for r in refs
                    if r["sha256"] == pin and Path(r["original_path"]).name == name + ".json"
                ),
                None,
            )
            if source is None:
                raise ValueError("upstream_pin")
            original = json.loads(Path(source["path"]).read_bytes())
            require(
                original[field] == expected and original["required_checks_passed"] is True,
                "upstream_assertion",
            )
            require(
                all(
                    (r["path"], r["sha256"]) in retained
                    for key in ("source_artifact_hashes", "code_config_hashes", "raw_shard_hashes")
                    for r in original[key]
                ),
                "upstream_operand_inventory",
            )


def replay(path: Path) -> Json:
    """Rebuild features, fitting and causal pilot transitions so rehashed false rows fail."""
    value = json.loads(path.read_bytes())
    checksum = value.pop("reproducibility_checksum")
    require(canonical_hash(value) == checksum, "artifact_checksum")
    require(
        (value["experiment_id"], value["task_id"], value["milestone"], value["run_date"])
        == (8892, m.TASK, "2026.11.761", "20261107"),
        "producer_identity",
    )
    require(
        canonical_hash(value["frozen_protocol"]) == canonical_hash(m.protocol()), "protocol_drift"
    )
    for ref in (
        value["source_artifact_hashes"] + value["code_config_hashes"] + value["raw_shard_hashes"]
    ):
        require(sha256_file(Path(ref["path"])) == ref["sha256"], "operand_hash")
    verify_inputs(value)
    require(not validate_current_work_receipt(value["current_work_receipt"]), "current_receipt")
    require(
        value["duration_s"] == value["current_work_receipt"]["duration_s"]
        and value["model_invocation_counts"] == value["current_work_receipt"]["invocation_counts"]
        and value["verifier_is_oracle"] is True
        and value["independent_generalization_score"] == 0,
        "compute_scope",
    )
    raw = Path(value["raw_directory"])
    result = json.loads((raw / "measurement.json").read_bytes())
    for key, expected in result.items():
        require(canonical_hash(value[key]) == canonical_hash(expected), "reduction_drift:" + key)
    if len(result["pilots"]) == 2:
        with TemporaryDirectory(prefix="carnot8892-replay-") as directory:
            rebuilt = measure(Path(directory))
            for key in result:
                if key != "pilots":
                    require(
                        canonical_hash(result[key]) == canonical_hash(rebuilt[key]),
                        "semantic_drift:" + key,
                    )
            for original, reproduced in zip(result["pilots"], rebuilt["pilots"], strict=True):
                require(
                    canonical_hash(json.loads(Path(original["path"]).read_bytes()))
                    == canonical_hash(json.loads(Path(reproduced["path"]).read_bytes())),
                    "causal_pilot_drift",
                )
    elif result["pilots"]:
        fitted = json.loads((raw / "fit.json").read_bytes())
        require(
            canonical_hash(fitted) == canonical_hash(m.fit(m.development(m.protocol()))),
            "partial_fit_drift",
        )
        reason = next(r["missing_reason"] for r in result["rows"] if r["status"] == "censored")
        require(
            canonical_hash(result) == canonical_hash(recover(raw, reason)),
            "partial_reduction_drift",
        )
        with TemporaryDirectory(prefix="carnot8892-partial-replay-") as directory:
            for stream, ref in zip(m.protocol()["pilot_manifest"], result["pilots"]):
                reproduced = m.pilot(
                    Path(directory) / stream["stream_id"], stream, fitted, m.protocol()
                )
                require(
                    canonical_hash(reproduced)
                    == canonical_hash(json.loads(Path(ref["path"]).read_bytes())),
                    "partial_pilot_drift",
                )
    completed = sum(r["status"] == "complete" for r in value["rows"])
    require(
        len(value["rows"]) == value["intended_count"] == 3072
        and value["completed_count"] == completed
        and value["excluded_count"] == 0
        and value["censored_count"] == 3072 - completed
        and value["failed_count"] == 0
        and value["independent_count"] == 0,
        "count_drift",
    )
    valid = value["required_checks_passed"]
    ready = int(
        value["preconditions_checked"]["external_ready"]
        and value["preconditions_checked"]["resources"]["passed"]
        and valid
        and completed == 3072
        and not value["owned_errors"]
        and all(r["passed"] for r in value["control_rows"])
    )
    require(value["signed_methods_ready_score"] == ready, "readiness_drift")
    expected = (
        "blocked"
        if not value["preconditions_checked"]["external_ready"]
        else "null"
        if ready
        else "disqualified"
    )
    require(value["verdict_class"] == expected, "verdict_drift")
    require(
        value["feature_schema"] == m.protocol()["feature_schema"]
        and value["original_acceptance_thresholds"] == old.GATES,
        "schema_or_threshold_drift",
    )
    return dict(passed=True, status="replay_passed", sha256=sha256_file(path))


def main(argv: list[str] | None = None) -> int:
    """Run the finite development-only protocol and publish one checked terminal artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261107"], default="20261107")
    parser.add_argument("--output", type=Path, default=m.ROOT / "results" / (m.NAME + ".json"))
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--sources", type=Path, default=m.ROOT / "results")
    parser.add_argument("--replay", "--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.replay:
        try:
            print(json.dumps(replay(args.replay)), flush=True)
            return 0
        except (OSError, ValueError, KeyError, TypeError) as error:
            print("replay_rejected: " + str(error), flush=True)
            return 1
    output = (args.private_output or args.output).absolute()
    if args.private_output and output.resolve().is_relative_to(m.ROOT):
        parser.error("private outputs must remain outside checkout")
    started = time.monotonic_ns()
    raw = output.parent / "raw" / m.NAME / "invocations" / str(started)
    raw.mkdir(parents=True, mode=0o700)
    invocation = dict(
        executable=sys.executable,
        arguments=argv if argv is not None else sys.argv[1:],
        cwd=os.getcwd(),
        pid=os.getpid(),
        executed_sources=[reference(m.ROOT / n) for n in m.OWNED],
    )
    atomic_json(raw / "invocation.json", invocation)
    template = ExperimentTemplate(
        8892, "Signed feature methods", "results/" + m.NAME + ".json", repo_root=m.ROOT, seed=m.SEED
    )
    m.progress("preconditions_before", 0, 1)
    pre = preconditions(raw, args.sources)
    atomic_json(raw / "preconditions.json", pre)
    m.progress("preconditions_after", 1, 0)
    result, errors = census("external_or_owned_precondition"), []
    if pre["external_ready"] and pre["resources"]["passed"]:
        try:
            with template.phase("development_pilots"):
                result = measure(raw)
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            errors.append(str(error))
            result = recover(raw, str(error))
    atomic_json(raw / "measurement.json", result)
    measured = time.monotonic_ns()
    m.progress("validation_before", 0, 1)
    with TemporaryDirectory(prefix="carnot8892-validation-") as directory:
        validation = (
            unqualified() if args.private_output else validate(raw / "validation", Path(directory))
        )
    valid = bool(
        validation["coverage"]["complete"]
        and validation["union"]["passed"]
        and all(r["passed"] for r in validation["receipts"])
    )
    ended = time.monotonic_ns()
    spans = [
        dict(phase="preconditions_and_pilots", start_s=0.0, end_s=(measured - started) / 1e9),
        dict(phase="validation", start_s=(measured - started) / 1e9, end_s=(ended - started) / 1e9),
    ]
    receipt = build_current_work_receipt(
        run_id=str(started),
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="cpu_linear_normalized_energy_head",
        inference_substrate_class="no_model_load"
        if result["numerical_training_calls"]
        else "blocked_no_run",
        execution_venue="host",
        inference_substrate_details=dict(backend="JAX CPU", pretrained_loads=0),
        started_monotonic_ns=started,
        ended_monotonic_ns=ended,
        phase_spans=spans,
        small_ebm_training=dict(
            performed=result["numerical_training_calls"] > 0,
            numerical_training_calls=result["numerical_training_calls"],
        ),
    )
    completed = sum(r["status"] == "complete" for r in result["rows"])
    ready = int(
        pre["external_ready"]
        and pre["resources"]["passed"]
        and valid
        and not errors
        and completed == 3072
        and all(r["passed"] for r in result["control_rows"])
    )
    verdict = "blocked" if not pre["external_ready"] else "null" if ready else "disqualified"
    p = m.protocol()
    helpers = [
        "scripts/experiment_template.py",
        "python/carnot/reporting/current_work_receipt.py",
        "python/carnot/reporting/primary_publication.py",
        "scripts/adversarial_verify.py",
        "scripts/verdict_row_consistency_lint.py",
        "python/carnot/reporting/topology_memory_methods_8881.py",
        "python/carnot/reporting/topology_memory_runner_8881.py",
    ]
    value = dict(
        result,
        experiment_id=8892,
        task_id=m.TASK,
        milestone="2026.11.761",
        run_date=args.date,
        honest_verdict=f"complete_{verdict}_signed_feature_methods",
        verdict_class=verdict,
        inference_substrate=receipt["inference_substrate"],
        inference_substrate_class=receipt["inference_substrate_class"],
        execution_venue="host",
        MODEL_SPECS=[],
        model_invocation_counts=receipt["invocation_counts"],
        numerical_training_calls=result["numerical_training_calls"],
        verifier_is_oracle=True,
        exposure_scope="constructed signed-record development only; no evaluation labels opened",
        independent_generalization_score=0,
        duration_s=receipt["duration_s"],
        phase_spans=spans,
        preconditions_checked=pre,
        random_seed=m.SEED,
        producer_invocation=invocation,
        current_work_receipt=receipt,
        source_artifact_hashes=pre["references"],
        code_config_hashes=[reference(m.ROOT / n) for n in m.OWNED + m.TESTS + helpers],
        raw_shard_hashes=[reference(path) for path in sorted(raw.rglob("*.json"))]
        + validation["raw_references"],
        raw_directory=str(raw),
        intended_count=3072,
        completed_count=completed,
        failed_count=0,
        censored_count=3072 - completed,
        excluded_count=0,
        independent_count=0,
        gate_check_summary=pre["diagnostics"],
        acceptance_gates=dict(
            methods_controls_required=True,
            owned_statement_coverage=100,
            benefit_required_for_method_readiness=False,
        ),
        required_checks_passed=valid,
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        owned_errors=errors,
        signed_methods_ready_score=ready,
        feature_schema=p["feature_schema"],
        frozen_protocol=p,
        original_acceptance_thresholds=old.GATES,
        cited_upstream_artifacts=[
            dict(experiment_id=int(n.split("_")[1]), artifact_field=f, expected=e, sha256=h)
            for n, (h, f, e) in UPSTREAM.items()
        ],
        methodology="Two development pilots; fixed linear categorical energy objective and SGD .1. Old-feature controls mask two coordinates of a common observation.24 paired successor streams and27 simultaneous intervals remain frozen and unopened.",
        methodology_note="Exact constructed labels are oracle supervision. Fitting Brier and collision changes are diagnostics; methods readiness is not future benefit or natural-language generalization.",
        flagged_adversarial=False,
        field_principles={},
    )
    health = Path("/tmp/carnot-8892-repository-health.log")
    if health.is_file():
        shutil.copyfile(health, raw / "repository_health.log")
        value["repository_health"] = dict(
            scope="repository_health",
            command=".venv/bin/pytest tests/python -q",
            **reference(raw / "repository_health.log"),
            interpretation="Exact bounded whole-suite output; separate from owned qualification.",
        )
        value["raw_shard_hashes"].append(reference(raw / "repository_health.log"))
    value = normalize_artifact_for_template_write(value)
    value["field_principles"] = {
        k: "Actual producer metadata or immutable operand; constructed development scope only."
        for k in value
    }
    value["field_principles"].update(
        signed_methods_ready_score="Schema, causal controls and owned checks qualify methods; no future benefit claim.",
        independent_count="Zero evaluation streams opened. Two development pilots do not establish independent generalization.",
        feature_collision_rows="Development-only equal-feature groups with exact constructed labels; no population lower bound.",
        feature_work_counts="All fact scans, binding and candidate comparisons; dummy masking separate from deployed vector work.",
        numerical_training_calls="2400 fixed development updates and384 pilot updates; child validation owns separate receipts.",
        frozen_protocol="24 fresh streams, existing topology limits, original nine gates per arm and nine contrasts; no evaluation outcomes.",
        reproducibility_checksum="Canonical artifact hash plus independent feature, fit and transition reconstruction.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    candidate = raw / "prospective.json"
    atomic_json(candidate, value)
    if not args.private_output:
        m.progress("terminal_validation_before", 0, 3)
        reports = terminal(candidate, raw)
        value["validation_receipts"] += reports
        value["raw_shard_hashes"] += [
            reference(Path(r[k + "_path"])) for r in reports for k in ("stdout", "stderr")
        ]
        if not all(r["passed"] for r in reports):
            value.update(
                honest_verdict="complete_disqualified_signed_feature_methods",
                verdict_class="disqualified",
                signed_methods_ready_score=0,
                required_checks_passed=False,
            )
        final = time.monotonic_ns()
        spans.append(
            dict(
                phase="terminal_validation",
                start_s=(ended - started) / 1e9,
                end_s=(final - started) / 1e9,
            )
        )
        receipt = build_current_work_receipt(
            run_id=str(started),
            owner_pid=os.getpid(),
            events=[],
            inference_substrate=value["inference_substrate"],
            inference_substrate_class=value["inference_substrate_class"],
            execution_venue="host",
            inference_substrate_details=dict(backend="JAX CPU", pretrained_loads=0),
            started_monotonic_ns=started,
            ended_monotonic_ns=final,
            phase_spans=spans,
            small_ebm_training=dict(
                performed=result["numerical_training_calls"] > 0,
                numerical_training_calls=result["numerical_training_calls"],
            ),
        )
        value.update(
            duration_s=receipt["duration_s"], phase_spans=spans, current_work_receipt=receipt
        )
        value.pop("reproducibility_checksum")
        value["reproducibility_checksum"] = canonical_hash(value)
        m.progress("terminal_validation_after", 3, 0)
    publish_primary(output, value, replay)
    for path in raw.rglob("*"):
        if path.is_file():
            path.chmod(0o400)
    m.progress("published_" + value["verdict_class"], completed, 3072 - completed)
    return 0
