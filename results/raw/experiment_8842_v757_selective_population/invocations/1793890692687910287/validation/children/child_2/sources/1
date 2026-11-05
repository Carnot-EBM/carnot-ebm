"""REQ-REPORT-8842: execute, authenticate and cold-replay fixed-population evidence."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from urllib.request import urlopen

import jax
import numpy as np
import scipy
import yaml

from carnot.reporting import selective_population_8842 as m
from carnot.reporting import selective_methods_runner_8829 as inherited
from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    sha256_file,
    build_current_work_receipt,
    validate_current_work_receipt,
)
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.selective_runner_8816 import stream, unqualified
from carnot.reporting.v746_retained_evidence import reference, load
from carnot.reporting.v748_collection_contract import namespace
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]
UPSTREAM = m.ROOT / "results/experiment_8829_v756_selective_methods.json"
PIN = "sha256:80b78a54f399eed31035062bcde37659e46ee259ce339755067c9819939cad9c"
SOURCE_URL = "https://arxiv.org/abs/2610.09541v1"


def preconditions(raw: Path, upstream: Path) -> Json:
    """Reject absent external authorities before work; local resource failures disqualify."""
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw.chmod(0o700)
    atomic_json(raw / "writable.json", dict(probe=True))
    diagnostic: Json = dict(
        upstream="exp8829-selective-methods",
        artifact_field="selective_protocol_ready_score",
        expected=1,
        observed=None,
        path=str(upstream),
        sha256=None,
    )
    refs, errors = [], []
    try:
        value = json.loads(upstream.read_bytes())
        diagnostic["sha256"] = sha256_file(upstream)
        diagnostic["observed"] = value["selective_protocol_ready_score"]
        if value["selective_protocol_ready_score"] != 1:
            raise ValueError("observed_zero")
        if value["task_id"] != "exp8829-selective-methods" or value["verdict_class"] != "null":
            raise ValueError("rejected_producer")
        if diagnostic["sha256"] != PIN:
            raise ValueError("rejected_producer_hash")
        refs = (
            [reference(upstream)]
            + value["source_artifact_hashes"]
            + value["code_config_hashes"]
            + value["raw_shard_hashes"]
            + [
                dict(
                    path=str(Path(m.legacy.old.__file__)),
                    sha256="sha256:1dca08d7ab959e324795e26c31875415921b6e63567fdbb1447a81713c2b2b7f",
                )
            ]
        )
        for ref in refs:
            diagnostic.update(path=ref["path"], artifact_field="sha256", expected=ref["sha256"])
            diagnostic["observed"] = sha256_file(Path(ref["path"]))
            if diagnostic["observed"] != ref["sha256"]:
                raise ValueError("rejected_producer")
        historical = m.ROOT / "results/experiment_8832_v756_selective_availability.json"
        quarantine = json.loads(historical.read_bytes())
        if (
            quarantine["verdict_class"] != "disqualified"
            or quarantine["inference_substrate_class"] != "small_ebm_training"
        ):
            raise ValueError("quarantine_decision_drift")
        refs.append(reference(historical))
        for ref in quarantine["raw_shard_hashes"]:
            diagnostic.update(path=ref["path"], artifact_field="sha256", expected=ref["sha256"])
            diagnostic["observed"] = sha256_file(Path(ref["path"]))
            if diagnostic["observed"] != ref["sha256"]:
                raise ValueError("rejected_quarantined_operand")
            refs.append(ref)
        authority_path = m.ROOT / "research-roadmap.yaml"
        diagnostic.update(
            upstream="frozen_v757_contract",
            path=str(authority_path),
            artifact_field="milestone_and_task",
            expected="2026.11.757/" + m.TASK,
        )
        authority_bytes = authority_path.read_bytes()
        authority = yaml.safe_load(authority_bytes)
        if authority["milestone"] != "2026.11.757" or not any(
            t["id"] == m.TASK for t in authority["tasks"]
        ):
            raise ValueError("rejected_v757_authority")
        for name, content in [
            ("frozen_v757_roadmap.yaml", authority_bytes),
            ("frozen_v757_design.md", (m.ROOT / authority["milestone_doc"]).read_bytes()),
        ]:
            retained = raw / name
            retained.write_bytes(content)
            refs.append(reference(retained))
        diagnostic.update(
            upstream="Certified by Abstention v1",
            path=SOURCE_URL,
            artifact_field="retrieved_source",
            expected="primary abstract with matching title",
            observed=None,
        )
        m.progress("before_method_authority_retrieval")
        with urlopen(SOURCE_URL, timeout=25) as response:
            source = response.read()
        if b"Certified by Abstention" not in source:
            raise ValueError("rejected_method_authority")
        (raw / "method_authority.html").write_bytes(source)
        refs.append(reference(raw / "method_authority.html"))
        m.progress("after_method_authority_retrieval")
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors.append(
            dict(
                diagnostic,
                check="missing_file"
                if isinstance(error, OSError)
                else "missing_field"
                if isinstance(error, KeyError)
                else "observed_zero"
                if str(error) == "observed_zero"
                else "rejected_producer",
                error=str(error),
            )
        )
    resources: Json = dict(
        repository_virtualenv=(m.ROOT / ".venv/bin/python").is_file(),
        dependencies=dict(jax=jax.__version__, numpy=np.__version__, scipy=scipy.__version__),
        backend=jax.default_backend(),
        private_output_writable=(raw / "writable.json").is_file(),
        private_output_mode=oct(raw.stat().st_mode & 0o777),
        free_disk_bytes=shutil.disk_usage(raw).free,
        population_pilot_deadline_s=300,
        measurement_deadline_s=900,
        validation_deadline_per_child_s=600,
    )
    resources["passed"] = (
        resources["repository_virtualenv"]
        and resources["private_output_writable"]
        and resources["backend"] == "cpu"
        and resources["free_disk_bytes"] >= 512 * 1024 * 1024
    )
    return dict(resources=resources, references=refs, diagnostics=errors, external_ready=not errors)


def measure(raw: Path, pre: Json) -> Json:
    """Seal construction, fitting, draws and predictions before releasing population truth."""
    started = time.monotonic()
    atomic_json(
        raw / "protocol.json",
        dict(
            seed=m.SEED,
            seeds=m.SEEDS,
            steps=240,
            population=4096,
            fit=120,
            canaries=12,
            replicates=30,
            budgets=m.BUDGETS,
            evaluation_draws=240,
            thresholds=m.THRESHOLDS,
            alpha=m.ALPHA,
            selection="maximum accepted coverage subject to exact upper risk<=.05; tie lowest cutoff; no acceptance means no certificate",
            code=[reference(m.ROOT / p) for p in m.OWNED],
            census_custody="cutoff choice reads calibration draw labels only; census follows sealed choices",
        ),
    )
    exposed = inherited.exposed_structures(pre["references"])
    cohort = m.freeze(exposed)
    atomic_json(raw / "population.json", cohort)
    fitted = m.fit(cohort)
    atomic_json(raw / "fit.json", fitted)
    population_size = sum(w["split"] == "population" for w in cohort["worlds"])
    trace = m.draws(population_size)
    atomic_json(raw / "draws.json", trace)
    predictions = m.predict(cohort, fitted)
    atomic_json(raw / "predictions.json", dict(rows=predictions))
    truth = m.labels(cohort["inputs"])
    atomic_json(raw / "private_labels.json", dict(labels=truth))
    result = m.reduce(predictions, truth, trace)
    atomic_json(raw / "reduction.json", result)
    atomic_json(
        raw / "population_scored.json",
        dict(
            rows=[
                dict(
                    row,
                    truth=truth[row["unit_id"]],
                    metric=int(
                        m.legacy.old.decide(row["probabilities"], 0.5)
                        == m.legacy.ACTIONS[
                            m.legacy.old.target(row["candidate"], truth[row["unit_id"]])
                        ]
                    ),
                    denominator=1,
                    status="completed",
                    missing_reason=None,
                    scope="descriptive unselected .5 action; selected rules audited separately",
                )
                for row in predictions
            ]
        ),
    )
    m.audit_predictions(predictions, cohort, truth, fitted)
    canaries = []
    for unit in cohort["canaries"]:
        text = "\n".join(json.dumps(f) for f in unit["input"]["facts"])
        source = raw / "canary_sources" / (unit["world_id"].split("/")[-1] + ".txt")
        source.parent.mkdir(exist_ok=True)
        source.write_text(text)
        canaries.append(
            dict(
                unit,
                source=reference(source),
                span_start=0,
                span_end=len(text.encode()),
                truth=m.graph.truth(unit["input"]),
            )
        )
    atomic_json(raw / "canaries.json", dict(rows=canaries))
    if time.monotonic() - started > pre["resources"]["measurement_deadline_s"]:
        raise ValueError("measurement_deadline_exceeded")
    return dict(
        result=result,
        training_calls=fitted["numerical_training_calls"],
        population_size=population_size,
        fit_worlds=sum(w["split"] == "fit" for w in cohort["worlds"]),
        canary_worlds=len(canaries),
        distinct_world_count=len(
            {
                i
                for r in trace["replicates"]
                for s in ["calibration", "evaluation"]
                for i in r[s]["indices"]
            }
        ),
        independent_draw_count=len(trace["replicates"]) * 600,
    )


def validate(raw: Path, private: Path) -> Json:
    """Keep existing private consumer assertions and actual unit/child statement accounting."""
    checker = namespace(inherited, "selective8842_validation")
    checker.m = m
    return dict(checker.validate(raw, private))


def replay(path: Path) -> Json:
    """Recompute predictions, labels, draws and reductions instead of trusting a self-hash."""
    value = json.loads(path.read_bytes())
    if value["experiment_id"] != 8842 or value["task_id"] != m.TASK:
        raise ValueError("producer_identity")
    expected = value.pop("reproducibility_checksum")
    if canonical_hash(value) != expected:
        raise ValueError("artifact_checksum")
    for ref in (
        value["source_artifact_hashes"] + value["code_config_hashes"] + value["raw_shard_hashes"]
    ):
        if sha256_file(Path(ref["path"])) != ref["sha256"]:
            raise ValueError("operand_hash")
    if validate_current_work_receipt(value["current_work_receipt"]):
        raise ValueError("current_receipt")
    if value["completed_count"] != len(value["rows"]):
        raise ValueError("count_drift")
    if value["population_manifest"]:
        cohort, fitted, trace = [
            load(value[k]) for k in ["population_manifest", "fit_manifest", "draw_manifest"]
        ]
        truth = load(value["labels_manifest"])["labels"]
        predictions = load(value["predictions_manifest"])["rows"]
        m.audit_predictions(predictions, cohort, truth, fitted)
        calculated = m.reduce(predictions, truth, trace)
        for key, observed in calculated.items():
            if canonical_hash(observed) != canonical_hash(value[key]):
                raise ValueError("reduction_drift:" + key)
        for unit in load(value["canary_manifest"])["rows"]:
            if sha256_file(Path(unit["source"]["path"])) != unit["source"]["sha256"]:
                raise ValueError("canary_source_hash")
    return dict(passed=True, status="replay_passed", sha256=sha256_file(path))


def main(argv: list[str] | None = None) -> int:
    """Publish one terminal experiment without activating any roadmap or changing history."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261105"], default="20261105")
    parser.add_argument("--output", type=Path, default=m.ROOT / "results" / (m.NAME + ".json"))
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--upstream", type=Path, default=UPSTREAM)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    m.progress("start_no_model_load")
    if args.cold_replay:
        try:
            print(json.dumps(replay(args.cold_replay)), flush=True)
            return 0
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(f"replay_rejected: {error}", flush=True)
            return 1
    output = (args.private_output or args.output).absolute()
    if args.private_output and output.resolve().is_relative_to(m.ROOT):
        parser.error("private outputs must remain outside checkout")
    raw = output.parent / "raw" / m.NAME / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    started = time.monotonic_ns()
    pre = preconditions(raw, args.upstream)
    health = Path("/tmp/carnot-8842-repository-health/receipt.json")
    health_ref = None
    if health.is_file():
        health_payload = json.loads(health.read_bytes())
        for suffix in ["stdout", "stderr"]:
            target = raw / ("repository_health." + suffix)
            shutil.copyfile(health_payload[suffix + "_path"], target)
            health_payload[suffix + "_path"] = str(target)
        atomic_json(raw / "repository_health.json", health_payload)
        health_ref = reference(raw / "repository_health.json")
    m.progress("after_preconditions")
    work: Json = dict(
        result=dict(
            rows=[],
            certificate_menu=[],
            selection_sha256=None,
            census_released_after_selection=False,
            certificate_firing_count=0,
            certificate_failure_count=0,
        ),
        training_calls=0,
        population_size=0,
        fit_worlds=0,
        canary_worlds=0,
        distinct_world_count=0,
        independent_draw_count=0,
    )
    owned_errors = []
    if pre["external_ready"] and pre["resources"]["passed"]:
        try:
            work = measure(raw, pre)
        except (OSError, ValueError, KeyError, TypeError) as error:
            owned_errors.append(str(error))
    measured = time.monotonic_ns()
    m.progress("before_owned_validation")
    with TemporaryDirectory(prefix="exp8842-validation-") as directory:
        validation = (
            unqualified() if args.private_output else validate(raw / "validation", Path(directory))
        )
    m.progress("after_owned_validation")
    valid = bool(
        validation["coverage"]["complete"]
        and validation["union"]["passed"]
        and all(r["passed"] for r in validation["receipts"])
    )
    verdict = (
        "disqualified"
        if owned_errors or not pre["resources"]["passed"]
        else "blocked"
        if not pre["external_ready"]
        else "null"
        if valid
        else "disqualified"
    )
    ready = int(verdict == "null" and work["population_size"] == 4096)
    ended = time.monotonic_ns()
    phases = [
        dict(phase="preconditions_measurement", start_s=0, end_s=(measured - started) / 1e9),
        dict(
            phase="owned_validation",
            start_s=(measured - started) / 1e9,
            end_s=(ended - started) / 1e9,
        ),
    ]
    substrate_class = "blocked_no_run" if not work["population_size"] else "no_model_load"
    receipt = build_current_work_receipt(
        run_id=str(started),
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="deterministic_local",
        inference_substrate_class=substrate_class,
        inference_substrate_details=dict(
            scope="in-process CPU Gibbs heads; exact encoded labels and binomial arithmetic"
        ),
        execution_venue="host",
        started_monotonic_ns=started,
        ended_monotonic_ns=ended,
        phase_spans=phases,
        small_ebm_training=dict(
            performed=work["training_calls"] > 0, numerical_training_calls=work["training_calls"]
        ),
    )
    atomic_json(raw / "current_work_receipt.json", receipt)
    manifests = {
        field: reference(raw / name) if (raw / name).is_file() else None
        for field, name in [
            ("population_manifest", "population.json"),
            ("fit_manifest", "fit.json"),
            ("draw_manifest", "draws.json"),
            ("labels_manifest", "private_labels.json"),
            ("predictions_manifest", "predictions.json"),
            ("canary_manifest", "canaries.json"),
        ]
    }
    sources = [
        "consensus_energy_8806.py",
        "consensus_selective_energy_8819.py",
        "selective_methods_8829.py",
        "evidence_decisive_8818.py",
        "current_work_receipt.py",
        "primary_publication.py",
    ]
    from scripts.adversarial_verify import SUBSTRATE_CLASSES

    value = dict(
        work["result"],
        **manifests,
        experiment_id=8842,
        task_id=m.TASK,
        milestone="2026.11.757",
        run_date=args.date,
        honest_verdict=f"complete_{verdict}_selective_population",
        verdict_class=verdict,
        inference_substrate=receipt["inference_substrate"],
        inference_substrate_class=substrate_class,
        execution_venue="host",
        MODEL_SPECS=[],
        model_invocation_counts=receipt["invocation_counts"],
        numerical_training_calls=work["training_calls"],
        verifier_is_oracle=True,
        exposure_scope="conditional on a sealed finite constructed population; no novel-structure, natural-text or hidden-game generalization",
        independent_generalization_score=0,
        duration_s=receipt["duration_s"],
        random_seed=m.SEED,
        phase_spans=phases,
        current_work_receipt=receipt,
        preconditions_checked=pre,
        source_artifact_hashes=pre["references"],
        code_config_hashes=[reference(m.ROOT / p) for p in m.OWNED + m.TESTS]
        + [reference(m.ROOT / "python/carnot/reporting" / p) for p in sources]
        + [reference(m.ROOT / "scripts/experiment_template.py")],
        raw_shard_hashes=[reference(p) for p in raw.glob("*.json")]
        + [reference(p) for p in (raw / "canary_sources").glob("*.txt")]
        + [reference(p) for p in raw.glob("repository_health.*") if p.suffix != ".json"]
        + validation["raw_references"],
        intended_count=630,
        completed_count=len(work["result"]["rows"]),
        failed_count=len(owned_errors),
        censored_count=0,
        excluded_count=0,
        independent_count=work["independent_draw_count"],
        independent_draw_count=work["independent_draw_count"],
        distinct_world_count=work["distinct_world_count"],
        gate_check_summary=pre["diagnostics"],
        owned_errors=owned_errors,
        acceptance_gates=dict(
            population=dict(
                expected=4096,
                observed=work["population_size"],
                principle="Never shrink the fixed population",
            ),
            risk=dict(
                upper=0.05,
                delta=0.01,
                principle="Familywise risk bounds are conditional on the frozen population",
            ),
            validation=dict(passed=valid, principle="Failed owned checks disqualify readiness"),
        ),
        required_checks_passed=valid,
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        selective_protocol_ready_score=ready,
        canary_protocol_ready_score=int(
            valid
            and pre["external_ready"]
            and pre["resources"]["passed"]
            and not owned_errors
            and work["canary_worlds"] == 12
        ),
        familywise_allocation=dict(
            delta=0.01,
            policies=7,
            budgets=3,
            thresholds=6,
            alpha_per_bound=m.ALPHA,
            scope="within each independent replicate; no across-replicate simultaneous guarantee",
        ),
        binomial_boundary_rows=m.boundaries(),
        threshold_menu=m.THRESHOLDS,
        policy_menu=[dict(arm="learned_energy", seed=s, training_steps=240) for s in m.SEEDS]
        + [dict(arm=a, seed=0, training_steps=0) for a in ["agreement_only", "fixed_evidence"]],
        source_method_map=[
            dict(
                url=SOURCE_URL,
                source=next(
                    (
                        r
                        for r in pre["references"]
                        if Path(r["path"]).name == "method_authority.html"
                    ),
                    None,
                ),
                method="Bonferroni exact one-sided Clopper-Pearson bound; separately report certificate firing and census conditional error",
                limitations="No imported paper performance or Carnot generalization theorem",
            )
        ]
        + [
            dict(
                source=reference(Path(m.legacy.old.__file__)),
                method="unchanged five count features, Gibbs initialization and class-balanced NCE training; seeds11/23/37/47/59;240 steps; unchanged consensus and fixed-evidence action probabilities",
                output_contract="accept/reject/escalate for one hash-fixed candidate and variant per world",
            )
        ],
        compute_contract=dict(
            closed_substrate_vocabulary=sorted(SUBSTRATE_CLASSES),
            invocation_counts="zero model loads and generations; numerical training recorded separately",
            outputs=["accept", "reject", "escalate"],
            labels=["support", "refute", "insufficient"],
        ),
        methodology="120 disjoint fitting worlds; independent with-replacement finite-population index draws; unchanged five-feature Gibbs NCE heads; exact binomial risk selection; post-selection full census; separate synthetic source-span canaries.",
        repository_health=health_ref,
        flagged_adversarial=False,
        field_principles={},
    )
    value["field_principles"] = {
        key: "Measured producer operand or independently recomputable reduction; byte hashes bind evidence and do not prove scientific validity."
        for key in value
    }
    value["field_principles"].update(
        population_manifest="All4096 units and their hash-fixed queries seal before independent index draws; construction rejection does not carry into sampling.",
        fit_manifest="Only120 fitting worlds supply labels to unchanged240-step heads at five fixed seeds.",
        draw_manifest="Independent spawned streams retain initial/final RNG states and every replacement index, including repeats.",
        independent_draw_count="Count independent scheduled RNG indices once; budgets are nested and every policy shares those indices.",
        distinct_world_count="Count distinct drawn population indices separately; seeds and repeated variants add no worlds.",
        familywise_allocation="The frozen .01 budget covers7 policies times3 budgets times6 thresholds within each replicate.",
        binomial_boundary_rows="SciPy inversion and direct logarithms check the integer zero-error floor; zero acceptance never certifies.",
        certificate_firing_count="Count issued rules separately from failures among issued rules; abstention may hide low availability.",
        certificate_failure_count="A selected certificate fails exactly when post-selection census accepted risk exceeds .05.",
        rows="Each replicate, policy and calibration budget retains evaluation counts; complete per-world policy operands remain bound under raw/.",
        canary_protocol_ready_score="Separate synthetic source-span custody qualifies canary consumers; calibration certificates do not establish canary success.",
        selective_protocol_ready_score="Complete population, draw protocol and owned checks qualify finite-population consumers independently of certificate availability.",
        source_method_map="Retrieved original source motivates availability versus conditional risk; no published performance is imported.",
        compute_contract="Use the validator's actual closed enum; CPU training does not load a pretrained checkpoint.",
        repository_health="One bounded full-suite diagnostic is retained separately without weakening owned validation.",
        owned_statement_coverage="Actual unit and real child coverage must execute every new statement with zero exclusions.",
    )
    value = normalize_artifact_for_template_write(value)
    value["reproducibility_checksum"] = canonical_hash(value)
    prospective = raw / "prospective.json"
    atomic_json(prospective, value)
    if not args.private_output:
        m.progress("before_terminal_checks")
        reports = [
            stream(
                name,
                [
                    sys.executable,
                    "-u",
                    str(m.ROOT / "scripts" / script),
                    *options,
                    str(prospective),
                ],
                raw / "checks",
                deadline=180,
            )
            for name, script, options in [
                ("adversarial", "adversarial_verify.py", ["--json"]),
                ("rows", "verdict_row_consistency_lint.py", ["--strict"]),
            ]
        ]
        reports.append(
            stream(
                "cold_replay",
                [
                    "env",
                    "-u",
                    "PYTHONPATH",
                    "--chdir=/tmp",
                    sys.executable,
                    "-u",
                    str(m.ROOT / m.CLI),
                    "--cold-replay",
                    str(prospective),
                ],
                raw / "checks",
                deadline=600,
            )
        )
        value["validation_receipts"] += reports
        if not all(r["passed"] for r in reports):
            value.update(
                honest_verdict="complete_disqualified_selective_population",
                verdict_class="disqualified",
                selective_protocol_ready_score=0,
                canary_protocol_ready_score=0,
                required_checks_passed=False,
            )
        del value["reproducibility_checksum"]
        value["reproducibility_checksum"] = canonical_hash(value)
        m.progress("after_terminal_checks")
    publish_primary(output, value, replay)
    m.progress("complete_" + value["verdict_class"], value["completed_count"], 0)
    return 0
