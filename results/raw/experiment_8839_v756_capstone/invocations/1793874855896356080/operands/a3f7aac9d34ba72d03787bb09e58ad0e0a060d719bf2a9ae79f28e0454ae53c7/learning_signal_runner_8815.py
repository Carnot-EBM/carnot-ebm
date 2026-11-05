"""REQ-REPORT-8815: finite diagnosis with private evidence and terminal publication.

The existing validation runner streams children and keeps real deadlines.
Optional output locations change custody only, never the scientific protocol.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

from coverage import Coverage
import jax
import jax.numpy as jnp
import numpy as np

from carnot.reporting import causal_learning_audit_8810 as audit
from carnot.reporting import learning_signal_diagnosis_8815 as d
from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v753_qualification_runner import coverage_summary

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8815_v755_learning_signal_diagnosis"
CLI = "scripts/experiments/" + NAME + ".py"
OWNED = [
    "python/carnot/reporting/learning_signal_diagnosis_8815.py",
    "python/carnot/reporting/learning_signal_runner_8815.py",
    CLI,
]
TESTS = ["tests/python/test_learning_signal_diagnosis_8815.py"]
AUDIT = "results/experiment_8810_v754_causal_learning_audit.json"
AUDIT_PIN = "sha256:1e4f9a49a7a715d439b05503da0b324b07b25ef5b8f785bde5dabcef3585a87a"
CODE = OWNED + [
    "python/carnot/memory/causal_memory_8802.py",
    "python/carnot/reporting/evidence_intervention_corpus_8791.py",
    "python/carnot/reporting/evidence_set_corpus_8768.py",
    "python/carnot/reporting/evidence_set_methods_8768.py",
    "python/carnot/learning/causal_template_streams_8804.py",
    "python/carnot/models/gibbs/__init__.py",
    "python/carnot/training/nce.py",
    "python/carnot/reporting/causal_learning_audit_8810.py",
    "python/carnot/reporting/current_work_receipt.py",
    "python/carnot/reporting/primary_publication.py",
    "python/carnot/reporting/experiment_7303_validation_scope.py",
]


def authenticate() -> tuple[list[Json], list[Json], set[str]]:
    """Recheck sealed predictions through the existing independent audit before diagnosis."""
    d.progress("authenticate_before", 0, 1)
    ref = dict(path=str(ROOT / AUDIT), sha256=AUDIT_PIN)
    old = d.load(ref)
    d.require(old["required_checks_passed"] is True, "rejected_producer_8810")
    source_ref, frozen_ref = old["raw_shard_hashes"][:2]
    d.require(source_ref["sha256"] == audit.PIN, "historical_authority")
    source, frozen = d.load(source_ref), d.load(frozen_ref)
    checked = audit.reduce(source, frozen)
    d.require(all(checked["leakage_checks"]["checks"].values()), "sealed_audit_failed")
    rows, excluded = [], set()
    for index, (stream, public_ref) in enumerate(zip(frozen["streams"], source["input_refs"])):
        public = d.load(public_ref)
        for split, key in [("online", "worlds"), ("retention", "retention")]:
            for recipe, payload in zip(stream[key], public[split], strict=True):
                structure = d.structure(payload)
                excluded.add(structure)
                for mode in ("off", "on"):
                    counts = d.m.observe(
                        payload, [] if mode == "off" else [d.m.normalize(payload)]
                    )["counts"]
                    rows.append(
                        dict(
                            unit_id=recipe["unit_id"],
                            cluster_id=stream["stream_id"],
                            structure=structure,
                            template_mode=mode,
                            counts=counts,
                            truth=d.m.decide(payload)["decision"],
                            full_structure=recipe["structural_hash"],
                        )
                    )
        d.progress("diagnostic_worlds", index + 1, len(frozen["streams"]) - index - 1)
    d.progress("authenticate_after", len(rows), 0)
    return rows, [ref, source_ref, frozen_ref], excluded


def execute(raw: Path) -> Json:
    """Run the finite registered comparison once; never launch a new stream."""
    rows, refs, excluded = authenticate()
    d.progress("development_before", 0, 60)
    development, split = d.development(excluded)
    d.progress("development_after", 60, 0)
    d.progress("controls_before", 0, 1)
    controls = d.controls()
    d.progress("controls_after", len(controls), 0)
    fits = []
    train = [r for r in development if r["split"] == "fit"]
    valid = [r for r in development if r["split"] == "validation"]
    for seed in d.SEEDS:
        for objective in d.OBJECTIVES:
            for lr in d.LRS:
                d.progress(f"fit_before_seed{seed}_{objective}_lr{lr}", len(fits), 45 - len(fits))
                fits.append(d.fit(train, valid, seed, objective, lr))
                d.progress("fit_after", len(fits), 45 - len(fits))
    evidence = dict(
        diagnostic_rows=rows,
        development_rows=development,
        split_manifest=split,
        fits=fits,
        controls=controls,
        source_artifact_hashes=refs,
    )
    atomic_json(raw / "evidence.json", evidence)
    return evidence


def reduce(evidence: Json) -> Json:
    """Recompute all terminal reductions from owned rows and selected numeric heads."""
    tables, floors = [], {}
    for mode in ("off", "on"):
        table, floor = d.collisions(
            [r for r in evidence["diagnostic_rows"] if r["template_mode"] == mode], mode
        )
        tables.extend(table)
        floors[mode] = floor
    split = evidence["split_manifest"]
    disjoint = (
        len(set(split["fit"])) == 40
        and len(set(split["validation"])) == 20
        and set(split["fit"]).isdisjoint(split["validation"])
        and not set(split["fit"] + split["validation"]).intersection(split.get("excluded", []))
    )
    control_ok = all(r["passed"] for r in evidence["controls"])
    selected = d.selection(evidence["fits"], disjoint, control_ok)
    rows = []
    valid = [r for r in evidence["development_rows"] if r["split"] == "validation"]
    for fitted in evidence["fits"]:
        objective = fitted["objective"]
        if fitted["lr"] != selected["learning_rates"][objective] or not valid:
            continue
        x, y = d.dataset(valid)
        params = [jnp.asarray(p) for p in fitted["parameters"]]
        probabilities = np.asarray(
            jax.nn.softmax(jax.vmap(lambda a: d.logits(params, a, objective))(x), axis=-1)
        )
        losses = np.sum((probabilities - np.asarray(jax.nn.one_hot(y, 3))) ** 2, axis=1)
        for row, value, probability in zip(valid, losses, probabilities, strict=True):
            rows.append(
                dict(
                    unit_id=row["unit_id"],
                    cluster_id=row["structure"],
                    arm=objective,
                    seed=fitted["seed"],
                    metric=float(value),
                    denominator=1,
                    status="completed",
                    missing_reason=None,
                    prediction=probability.tolist(),
                )
            )
    return dict(
        collision_rows=tables,
        empirical_feature_floor=floors,
        per_seed_fit_rows=evidence["fits"],
        gradient_control_rows=evidence["controls"],
        objective_selection=selected,
        selected_configuration_hash=canonical_hash(selected["configuration"]),
        objective_candidate_ready_score=selected["ready"],
        diagnosis_rows_ready_score=1,
        mechanism_supported=dict(
            observed_feature_conflict=any(r["conflicting"] for r in tables),
            categorical_development_improvement=selected["winning_seeds"],
            controls_passed=control_ok,
            disjoint=disjoint,
            effective_query_structures=len({r["structure"] for r in evidence["diagnostic_rows"]}),
            distinct_full_structures=len(
                {r.get("full_structure", r["structure"]) for r in evidence["diagnostic_rows"]}
            ),
            unresolved="Objective comparison cannot uniquely attribute V754 harm; no future benefit measured.",
        ),
        rows=rows,
        intended_count=len(rows),
        completed_count=len(rows),
        failed_count=0,
        censored_count=0,
        excluded_count=0,
        independent_count=len(set(split["fit"] + split["validation"])),
    )


def replay(path: Path) -> Json:
    """Reproduce features, fits and reductions; repaired outer hashes cannot grant trust."""
    value: Json = json.loads(path.read_bytes())
    for ref in value["code_config_hashes"]:
        d.require(sha256_file(Path(ref["path"])) == ref["sha256"], "code_hash")
    evidence = d.load(value["raw_shard_hashes"][0])
    diagnostic, refs, excluded = authenticate()
    development, split = d.development(excluded)
    d.require(
        evidence["diagnostic_rows"] == diagnostic
        and evidence["source_artifact_hashes"] == refs
        and evidence["development_rows"] == development
        and evidence["split_manifest"] == split,
        "evidence_drift",
    )
    train = [r for r in development if r["split"] == "fit"]
    valid = [r for r in development if r["split"] == "validation"]
    for index, fitted in enumerate(evidence["fits"]):
        reproduced = d.fit(train, valid, fitted["seed"], fitted["objective"], fitted["lr"])
        d.require(fitted == reproduced, "fit_drift")
        d.progress("replay_fits", index + 1, len(evidence["fits"]) - index - 1)
    d.require(d.controls() == evidence["controls"], "control_drift")
    expected = reduce(evidence)
    for key, observed in expected.items():
        d.require(value[key] == observed, "reduction_drift:" + key)
    for receipt in value.get("validation_receipts", []):
        d.require(sha256_file(Path(receipt["log_path"])) == receipt["log_sha256"], "validation_log")
    return dict(passed=True, replay_passed=True, primary_sha256=sha256_file(path))


def commands(plan: list[tuple[str, list[str], str, float]], logs: Path) -> list[Json]:
    """Reuse bounded, output-forwarding validation with a heartbeat independent of deadlines."""
    specs = [
        CommandSpec(name, tuple(argv), scope, deadline) for name, argv, scope, deadline in plan
    ]
    return run_commands(ROOT, specs, log_dir=logs, heartbeat_s=30)


def validate(raw: Path) -> Json:
    """Measure owned statements in real unit and CLI processes with zero exclusions."""
    import tempfile

    private = Path(tempfile.mkdtemp(prefix="carnot8815-coverage-"))
    config = private / "coverage.ini"
    datafile = private / ".coverage"
    config.write_text(
        "[run]\nparallel=true\npatch=subprocess\ndata_file="
        + str(datafile)
        + "\ninclude=\n"
        + "".join("    " + str(ROOT / p) + "\n" for p in OWNED)
        + "[report]\nexclude_lines=\n"
    )
    py = str(ROOT / ".venv/bin/python")
    plan = [
        (
            "owned_pytest",
            [
                py,
                "-m",
                "coverage",
                "run",
                "--rcfile=" + str(config),
                "-m",
                "pytest",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                *TESTS,
            ],
            "owned",
            300,
        ),
        (
            "private_E2E_018_021",
            [
                py,
                "-m",
                "pytest",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
                "tests/python/test_restricted_decision_audit_8210.py",
                "tests/python/test_primary_publication_7928.py",
            ],
            "owned",
            300,
        ),
        ("ruff_check", [str(ROOT / ".venv/bin/ruff"), "check", *OWNED, *TESTS], "owned", 60),
        (
            "ruff_format",
            [str(ROOT / ".venv/bin/ruff"), "format", "--check", *OWNED, *TESTS],
            "owned",
            60,
        ),
        (
            "strict_mypy",
            [str(ROOT / ".venv/bin/mypy"), "--strict", "--follow-imports=silent", *OWNED],
            "owned",
            120,
        ),
        ("spec_coverage", [py, "scripts/check_spec_coverage.py", *TESTS], "owned", 60),
        (
            "global_python_health",
            [str(ROOT / ".venv/bin/pytest"), "tests/python", "-q"],
            "repository_health",
            180,
        ),
    ]
    atomic_json(raw / "validation_manifest.json", dict(plan=plan, owned=OWNED, tests=TESTS))
    receipts = commands(plan, raw / "validation")
    cov = Coverage(data_file=str(datafile), config_file=False)
    cov.combine(data_paths=[str(private)], keep=True)
    cov.save()
    retained = raw / "coverage"
    shutil.copytree(private, retained, dirs_exist_ok=True)
    coverage = coverage_summary(retained / ".coverage", OWNED)
    result = dict(
        passed=coverage["complete"] and all(r["passed"] for r in receipts if r["scope"] == "owned"),
        receipts=receipts,
        coverage=coverage,
    )
    atomic_json(raw / "validation.json", result)
    return result


def terminal(candidate: Path, raw: Path) -> Json:
    """Run unchanged scanners and replay from /tmp without PYTHONPATH before publication."""
    py = str(ROOT / ".venv/bin/python")
    receipts = commands(
        [
            (
                "adversarial",
                [py, "scripts/adversarial_verify.py", str(candidate), "--json"],
                "owned",
                60,
            ),
            (
                "row_consistency",
                [py, "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)],
                "owned",
                60,
            ),
            (
                "cold_replay",
                [
                    "env",
                    "-u",
                    "PYTHONPATH",
                    "-C",
                    "/tmp",
                    py,
                    "-u",
                    str(ROOT / CLI),
                    "--replay",
                    str(candidate),
                ],
                "owned",
                600,
            ),
        ],
        raw / "terminal",
    )
    return dict(passed=all(r["passed"] for r in receipts), receipts=receipts)


def main(argv: list[str] | None = None) -> int:
    """Keep output location and replay as custody choices independent of the experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261104")
    parser.add_argument(
        "--output", "--private-output", type=Path, default=ROOT / "results" / (NAME + ".json")
    )
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args(argv)
    if args.replay:
        try:
            print(json.dumps(replay(args.replay)), flush=True)
            return 0
        except (ValueError, OSError, KeyError) as error:
            print(f"replay_rejected:{error}", flush=True)
            return 1
    if args.date != "20261104":
        print("date_rejected: expected 20261104", flush=True)
        return 1
    start = time.monotonic_ns()
    output = args.output.absolute()
    raw = (
        args.raw_dir or output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    ).absolute()
    raw.mkdir(parents=True, exist_ok=True)
    raw.chmod(0o700)
    gate, evidence = [], None
    try:
        evidence = execute(raw)
        value = reduce(evidence)
        verdict = "null"
    except (FileNotFoundError, KeyError, ValueError) as error:
        verdict = "blocked" if isinstance(error, (FileNotFoundError, KeyError)) else "disqualified"
        operand = Path(getattr(error, "filename", None) or ROOT / AUDIT)
        gate = [
            dict(
                check="historical_operands",
                upstream="exp8810-causal-learning-audit",
                artifact_field="required_checks_passed",
                expected=True,
                observed=None,
                path=str(operand),
                hash=sha256_file(operand) if operand.is_file() else None,
                status=(
                    "missing_file"
                    if isinstance(error, FileNotFoundError)
                    else "missing_field"
                    if isinstance(error, KeyError)
                    else "rejected_producer"
                ),
                reason=str(error),
            )
        ]
        value = dict(
            diagnosis_rows_ready_score=0,
            objective_candidate_ready_score=0,
            collision_rows=[],
            empirical_feature_floor={},
            per_seed_fit_rows=[],
            gradient_control_rows=[],
            objective_selection={},
            selected_configuration_hash=None,
            mechanism_supported={},
            rows=[],
            intended_count=0,
            completed_count=0,
            failed_count=0,
            censored_count=0,
            excluded_count=0,
            independent_count=0,
        )
    checked = validate(raw)
    owned_ok = checked["passed"]
    if not owned_ok:
        verdict = "disqualified"
        value.update(diagnosis_rows_ready_score=0, objective_candidate_ready_score=0)
    end = time.monotonic_ns()
    phases = [
        dict(
            phase="finite_diagnosis_and_validation",
            started_monotonic_ns=start,
            ended_monotonic_ns=end,
            duration_s=(end - start) / 1e9,
        )
    ]
    receipt = build_current_work_receipt(
        run_id=raw.name,
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="deterministic_verifier",
        inference_substrate_details=dict(
            backend="JAX CPU", architecture="Gibbs9x4; multinomial linear diagnostic"
        ),
        inference_substrate_class="no_model_load",
        execution_venue="local_cpu",
        started_monotonic_ns=start,
        ended_monotonic_ns=end,
        phase_spans=phases,
        small_ebm_training=dict(performed=evidence is not None),
    )
    atomic_json(raw / "current_work_receipt.json", receipt)
    value.update(
        experiment_id=8815,
        task_id="exp8815-learning-signal-diagnosis",
        milestone="2026.11.755",
        run_date=args.date,
        honest_verdict="complete_" + verdict + "_finite_learning_signal_diagnosis",
        verdict_class=verdict,
        inference_substrate="deterministic_verifier",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=receipt["invocation_counts"],
        numerical_training_calls=(45 * 400 + 5 * 400 + 1) if evidence is not None else 0,
        verifier_is_oracle=True,
        duration_s=(end - start) / 1e9,
        random_seed=8815,
        reproducibility_checksum=canonical_hash(evidence),
        source_artifact_hashes=evidence["source_artifact_hashes"] if evidence else [],
        code_config_hashes=[d.reference(ROOT / p) for p in CODE],
        raw_shard_hashes=([d.reference(raw / "evidence.json")] if evidence else [])
        + [d.reference(raw / "current_work_receipt.json")],
        phase_spans=phases,
        preconditions_checked=dict(historical_seals=evidence is not None),
        gate_check_summary=gate,
        acceptance_gates=dict(
            owned_checks=owned_ok,
            feature_diagnosis=value["diagnosis_rows_ready_score"] == 1,
            objective_continuation=value["objective_candidate_ready_score"] == 1,
        ),
        required_checks_passed=owned_ok,
        validation_receipts=checked["receipts"],
        owned_statement_coverage=checked["coverage"],
        exposure_scope="Exposed synthetic development and retrospective V754 labels only.",
        independent_generalization_score=0,
        cited_upstream_artifacts=evidence["source_artifact_hashes"] if evidence else [],
        methodology_note="Finite feature floor; three objectives;60 canonical worlds;40/20 split;45 fits;400 SGD updates;development-only LR choice. No population bound,FR-11 benefit or new stream.",
        repository_health_debt=[
            r for r in checked["receipts"] if r["scope"] == "repository_health"
        ],
        current_work_receipt=d.reference(raw / "current_work_receipt.json"),
    )
    value["field_principles"] = {
        key: "Hash-bound finite development evidence qualifies a mechanism, not future generalization."
        for key in value
    }
    candidate = raw / "candidate.json"
    atomic_json(candidate, value)
    final = terminal(candidate, raw)
    atomic_json(raw / "terminal_validation.json", final)
    if not final["passed"]:
        value.update(
            verdict_class="disqualified",
            honest_verdict="complete_disqualified_terminal_check",
            diagnosis_rows_ready_score=0,
            objective_candidate_ready_score=0,
            required_checks_passed=False,
        )
        atomic_json(raw / "rejected_candidate.json", json.loads(candidate.read_bytes()))
        value["acceptance_gates"].update(
            owned_checks=False, feature_diagnosis=False, objective_continuation=False
        )
    publish_primary(
        output,
        value,
        lambda path: dict(
            passed=canonical_hash(json.loads(path.read_bytes())) == canonical_hash(value)
            and (
                final["passed"]
                or (
                    value["verdict_class"] == "disqualified"
                    and value["diagnosis_rows_ready_score"] == 0
                    and value["objective_candidate_ready_score"] == 0
                )
            ),
            qualification=d.reference(raw / "terminal_validation.json"),
            candidate_sha256=sha256_file(path),
        ),
    )
    d.progress("published", 1, 0)
    return 0
