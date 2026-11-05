"""REQ-VERIFY-8819: authenticate, seal, validate and publish selective energy evidence."""

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
from typing import Any

from carnot.reporting import consensus_selective_energy_8819 as m
from carnot.reporting import evidence_evaluation_runner_8779 as shared
from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    build_current_work_receipt,
)
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.selective_runner_8816 import stream, unqualified
from carnot.reporting.v746_retained_evidence import reference, load
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8819_v755_consensus_selective_energy"
TASK = "exp8819-consensus-selective-energy"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/consensus_selective_energy_8819.py",
    "python/carnot/reporting/consensus_selective_runner_8819.py",
    CLI,
]
TESTS = ["tests/python/test_consensus_selective_energy_8819.py"]
SOURCE = ROOT / "results/experiment_8816_v755_selective_evidence_methods.json"
SOURCE_PIN = "sha256:3a474cb1eab956509047d686bf11c0998df99e536f2634dc1e2f874859ae7898"


def admit(path: Path) -> Json:
    """A replaced or unready producer cannot quietly become a different experiment."""
    diagnostic = dict(
        check="missing_file",
        upstream="exp8816-selective-evidence-methods",
        artifact_field="consensus_protocol_ready_score",
        expected=1,
        observed=None,
        path=str(path.absolute()),
        sha256=None,
    )
    try:
        ref = reference(path)
        diagnostic.update(check="rejected_producer", sha256=ref["sha256"])
        source = json.loads(path.read_bytes())
        if not isinstance(source, dict):
            raise ValueError("producer_object")
        diagnostic.update(
            check="missing_field", observed=source.get("consensus_protocol_ready_score")
        )
        if "consensus_protocol_ready_score" not in source:
            raise ValueError("missing_field")
        diagnostic["check"] = "observed_zero"
        if source["consensus_protocol_ready_score"] != 1:
            raise ValueError("readiness")
        diagnostic.update(
            check="rejected_producer",
            artifact_field="producer_sha256",
            expected=SOURCE_PIN,
            observed=ref["sha256"],
        )
        if ref["sha256"] != SOURCE_PIN or source["required_checks_passed"] is not True:
            raise ValueError("producer_identity_or_validation")
        refs = [ref, *source["raw_shard_hashes"], *source["code_config_hashes"]]
        for operand in refs:
            diagnostic.update(path=operand["path"], sha256=operand["sha256"])
            if reference(Path(operand["path"])) != operand:
                raise ValueError("bound_operand_drift")
        return dict(passed=True, producer=source, references=refs, diagnostics=[])
    except (OSError, ValueError, KeyError, TypeError) as error:
        return dict(passed=False, references=[], diagnostics=[dict(diagnostic, error=str(error))])


def validate(raw: Path, private: Path) -> Json:
    """Use the existing exact-node validator, including real unit/child coverage."""
    validator = namespace(shared, "consensus8819_validation")
    validator.e = SimpleNamespace(ROOT=ROOT, OWNED=OWNED, TESTS=TESTS)
    validator.original = SimpleNamespace(
        CONSUMERS=[
            "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
            "tests/python/test_restricted_decision_audit_8210.py",
        ]
    )
    validator.child = stream
    return dict(validator.validate(raw, private))


def prepare(raw: Path, private: bool = False) -> Json:
    """Freeze protocol and weights before calibration; seal decisions before labels open."""
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw.chmod(0o700)
    admitted = admit(SOURCE)
    atomic_json(raw / "protocol.json", m.protocol())
    work = dict(
        preconditions_checked=dict(
            passed=admitted["passed"],
            diagnostics=admitted["diagnostics"],
            optional_branches_required=False,
            private_output=private,
        ),
        source_artifact_hashes=admitted["references"],
        code_config_hashes=[reference(ROOT / p) for p in OWNED + TESTS],
        fit=dict(
            parameters=[],
            selected_threshold_rows=[],
            shortcut_controls=[],
            numerical_training_calls=0,
        ),
        per_candidate_rows=[],
        cost_rows=[],
        raw_shard_hashes=[reference(raw / "protocol.json")],
    )
    if admitted["passed"]:
        worlds = [
            w for w in admitted["producer"]["split_manifest"]["worlds"] if w["split"] != "canary"
        ]
        if private:
            worlds = [w for w in worlds if int(w["world_id"].split("/")[-1]) < 3]
        m.old.progress("8819_before_construct", 0, len(worlds))
        cohort = m.construct(worlds)
        atomic_json(raw / "cohort.json", cohort)
        m.old.progress("8819_after_construct", len(worlds), 0)
        work["fit"] = m.fit(cohort, seeds=[1] if private else m.SEEDS, steps=2 if private else 240)
        atomic_json(raw / "fit.json", work["fit"])
        atomic_json(raw / "selectors.json", dict(rows=work["fit"]["selected_threshold_rows"]))
        held = [x for x in cohort["inputs"] if x["split"] == "evaluation"]
        m.old.progress("8819_before_generation", 0, len(held))
        predictions, checks = m.predict(
            held, work["fit"]["parameters"], work["fit"]["selected_threshold_rows"]
        )
        committed = dict(rows=predictions, checks=checks)
        atomic_json(raw / "predictions.json", committed)
        seal = canonical_hash(committed)
        atomic_json(
            raw / "prediction_seal.json",
            dict(sha256=seal, predictions=reference(raw / "predictions.json")),
        )
        m.old.progress("8819_after_generation_before_label_release", len(held), 0)
        # Read the committed bytes, not the mutable in-memory predictions, for scoring.
        sealed = load(reference(raw / "predictions.json"))
        rows, oracle = m.release(cohort, sealed["rows"], sealed["checks"], seal)
        work.update(
            per_candidate_rows=rows,
            cost_rows=[oracle],
            cohort_reference=reference(raw / "cohort.json"),
            prediction_seal=seal,
        )
        work["raw_shard_hashes"] += [
            reference(raw / p)
            for p in [
                "cohort.json",
                "fit.json",
                "selectors.json",
                "predictions.json",
                "prediction_seal.json",
            ]
        ]
    end = time.monotonic_ns()
    work["current_work_receipt"] = build_current_work_receipt(
        run_id=str(start),
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="cpu_energy_training",
        inference_substrate_details=dict(scope="constructed Gibbs selector; no LLM calls"),
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=start,
        ended_monotonic_ns=end,
        phase_spans=[
            dict(phase="construct_fit_calibrate_seal_score", start_s=0.0, end_s=(end - start) / 1e9)
        ],
        small_ebm_training=dict(
            performed=bool(work["fit"]["parameters"]),
            numerical_training_calls=work["fit"]["numerical_training_calls"],
        ),
    )
    return work


def build(work: Json, validation: Json) -> Json:
    """Valid evidence can be a null; neither abstention nor the oracle grants benefit."""
    qualified = bool(
        validation["union"]["passed"]
        and validation["coverage"]["complete"]
        and all(x["passed"] for x in validation["receipts"])
    )
    external = work["preconditions_checked"]["passed"]
    fit = work["fit"]
    controls = bool(fit["shortcut_controls"]) and all(x["passed"] for x in fit["shortcut_controls"])
    summary = m.summarize(work["per_candidate_rows"], fit["selected_threshold_rows"])
    ready = int(qualified and external and controls)
    verdict = (
        "blocked"
        if not external
        else "disqualified"
        if not ready
        else "positive"
        if summary["benefit_passed"]
        else "null"
    )
    candidates = work["per_candidate_rows"]
    grouped: dict[str, list[Json]] = {}
    for row in candidates:
        if row["arm"] == "learned_energy":
            grouped.setdefault(row["unit_id"], []).append(row)
    rows = [
        dict(
            unit_id=u,
            cluster_id=g[0]["cluster_id"],
            arm="learned_energy",
            seed=None,
            metric=sum(x["metric"] for x in g) / len(g),
            denominator=len(g),
            status="completed",
            missing_reason=None,
        )
        for u, g in grouped.items()
    ]
    receipt = work["current_work_receipt"]
    value = dict(
        experiment_id=8819,
        task_id=TASK,
        milestone="2026.11.755",
        run_date="20261105",
        honest_verdict="complete_" + verdict + "_consensus_selective_energy",
        verdict_class=verdict,
        inference_substrate=receipt["inference_substrate"],
        inference_substrate_class=receipt["inference_substrate_class"],
        MODEL_SPECS=[],
        target_model="GibbsModel(input_dim=5, hidden_dims=[4]); three NCE heads per seed",
        model_invocation_counts=receipt["invocation_counts"],
        numerical_training_calls=fit["numerical_training_calls"],
        verifier_is_oracle=False,
        duration_s=receipt["duration_s"],
        random_seed=m.SEEDS[0],
        source_artifact_hashes=work["source_artifact_hashes"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes=work["raw_shard_hashes"],
        phase_spans=receipt["phase_spans"],
        preconditions_checked=work["preconditions_checked"],
        rows=rows,
        intended_count=2160,
        completed_count=len(rows),
        failed_count=0,
        censored_count=2160 - len(rows),
        excluded_count=0,
        gate_check_summary=work["preconditions_checked"]["diagnostics"],
        required_checks_passed=qualified,
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        exposure_scope="constructed six-vertex relational evidence and oracle-constructed agreement; no natural-text or hidden-game generalization",
        independent_generalization_score=0,
        cited_upstream_artifacts=work["source_artifact_hashes"],
        consensus_rows_ready_score=ready,
        selective_benefit_score=int(ready and summary["benefit_passed"]),
        selected_threshold_rows=fit["selected_threshold_rows"],
        calibration_eligibility_rows=[
            dict(arm=x["arm"], seed=x["seed"], **row)
            for x in fit["selected_threshold_rows"]
            for row in x["calibration_rows"]
        ],
        per_candidate_rows=candidates,
        shortcut_controls=fit["shortcut_controls"],
        cost_rows=fit.get("cost_rows", []) + work["cost_rows"],
        current_work_receipt=receipt,
        protocol=m.protocol(),
        oracle_verdict_class="circular_positive",
        methodology_note="Five 99-percent whole-world bootstrap intervals; nested seeds averaged within worlds. Zero decisive risk is undefined. No fixed safe cutoff makes contrasts not estimable. Matched random comparisons are descriptive. Constructed oracle reference is separately costed and grants no learned benefit.",
        global_test_spec_debt=work.get(
            "global_test_spec_debt",
            {"status": "not_supplied; private fixture qualification is separate"},
        ),
        **summary,
    )
    return value


def candidate_value(work: Json, validation: Json, raw: Path) -> Json:
    """Bind the primitive work and actual check receipt before hashing terminal claims."""
    value = build(work, validation)
    value.update(
        work_reference=reference(raw / "work.json"),
        validation_reference=reference(raw / "validation.json"),
    )
    value["field_principles"] = {
        k: "Freeze selectors before evaluation; preserve denominators, actual costs and immutable custody; seeds are nested within worlds."
        for k in value
    }
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Replay input-only serving and reductions, rejecting repaired aggregate hashes."""
    try:
        value = json.loads(path.read_bytes())
        work, validation = load(value["work_reference"]), load(value["validation_reference"])
        refs = (
            work["raw_shard_hashes"]
            + work["code_config_hashes"]
            + work["source_artifact_hashes"]
            + validation["raw_references"]
        )
        if any(reference(Path(x["path"])) != x for x in refs):
            return False
        if work["preconditions_checked"]["passed"]:
            cohort = load(work["cohort_reference"])
            if cohort != m.construct(cohort["worlds"]):
                return False
            fit = work["fit"]
            calibration = [x for x in cohort["inputs"] if x["split"] == "calibration"]
            predicted, _ = m.predict(calibration, fit["parameters"], [])
            scored = [dict(x, truth=cohort["labels"][x["unit_id"]]) for x in predicted]
            selected = [
                m.select(scored, a, s) for a, s in sorted({(x["arm"], x["seed"]) for x in scored})
            ]
            if (
                selected != fit["selected_threshold_rows"]
                or scored != fit["calibration_predictions"]
            ):
                return False
            held = [x for x in cohort["inputs"] if x["split"] == "evaluation"]
            predictions, checks = m.predict(held, fit["parameters"], selected)
            if canonical_hash(dict(rows=predictions, checks=checks)) != work["prediction_seal"]:
                return False
            rows, _ = m.release(cohort, predictions, checks, work["prediction_seal"])
            if rows != work["per_candidate_rows"]:
                return False
        return bool(
            canonical_hash(value)
            == canonical_hash(
                candidate_value(work, validation, Path(value["work_reference"]["path"]).parent)
            )
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def terminal(candidate: Path, raw: Path) -> Json:
    """Existing scanners and an outside-checkout interpreter inspect exact candidate bytes."""
    receipts = []
    for name, argv in [
        (
            "adversarial",
            [str(ROOT / ".venv/bin/python"), "-u", "scripts/adversarial_verify.py", str(candidate)],
        ),
        (
            "rows_strict",
            [
                str(ROOT / ".venv/bin/python"),
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                str(candidate),
                "--strict",
            ],
        ),
        (
            "cold_replay",
            [
                "env",
                "-u",
                "PYTHONPATH",
                str(ROOT / ".venv/bin/python"),
                "-u",
                str(ROOT / CLI),
                "--cold-replay",
                str(candidate),
            ],
        ),
    ]:
        command = (
            argv if name != "cold_replay" else ["bash", "-c", 'cd /tmp && exec "$@"', "cold", *argv]
        )
        receipts.append(stream(name, command, raw / "terminal_checks", deadline=180))
    return dict(passed=replay(candidate) and all(x["passed"] for x in receipts), receipts=receipts)


def main(argv: list[str] | None = None) -> int:
    """Private fixtures stay unqualified; production runs the owned checks before publication."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261105"], default="20261105")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--private-output", action="store_true")
    parser.add_argument("--repository-health", type=Path)
    parser.add_argument("--cold-replay", "--replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = replay(args.cold_replay)
        m.old.progress("8819_replay_passed" if passed else "8819_replay_rejected")
        return int(not passed)
    output = args.output.absolute()
    if args.private_output and output.resolve().is_relative_to(ROOT):
        parser.error("private outputs must remain outside checkout")
    raw = output.parent / "raw" / NAME / "invocations" / str(time.time_ns())
    m.old.progress("8819_start")
    work = prepare(raw, args.private_output)
    if args.repository_health:
        health = reference(args.repository_health)
        work["global_test_spec_debt"] = load(health)
        work["raw_shard_hashes"] += [health, *work["global_test_spec_debt"]["raw_references"]]
    atomic_json(raw / "work.json", work)
    with TemporaryDirectory(prefix="exp8819-") as directory:
        m.old.progress("8819_before_owned_validation", 0, 1)
        validation = (
            unqualified() if args.private_output else validate(raw / "validation", Path(directory))
        )
        m.old.progress("8819_after_owned_validation", 1, 0)
    atomic_json(raw / "validation.json", validation)
    value = candidate_value(work, validation, raw)
    m.old.progress("8819_before_publication", 0, 1)
    publish_primary(output, value, lambda candidate: terminal(candidate, raw))
    m.old.progress("8819_complete", 1, 0)
    return 0
