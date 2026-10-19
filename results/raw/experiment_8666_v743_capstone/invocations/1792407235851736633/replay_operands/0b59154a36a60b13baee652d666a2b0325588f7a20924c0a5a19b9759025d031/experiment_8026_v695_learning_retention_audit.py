"""REQ-REPORT-8026: independent learning retention audit with terminal custody.

Finite exposed development replay and private durability checks are different
claims. Prior audit target access is retained and disqualifies benefit credit.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

from carnot import experiment_8025_v695_causal_online_updates as prior
from carnot.experiment_8007_v694_conditioning_diagnosis import copy_evidence
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.evidence_features_custody_7980 import checked, reference
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.learning_store_8026 import recover as store_recover, worker
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.verify import learning_retention_audit_8026 as m

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[2]
NAME = "experiment_8026_v695_learning_retention_audit"
TASK = "exp8026-learning-retention-audit"
MODEL_SPECS: list[str] = []
OWNED = [
    f"python/carnot/{NAME}.py",
    "python/carnot/verify/learning_retention_audit_8026.py",
    "python/carnot/reporting/learning_store_8026.py",
    f"scripts/experiments/{NAME}.py",
]
TEST = "tests/python/test_learning_retention_audit_8026.py"
INPUTS = [
    ("experiment_8006_v694_independent_replay", "replay_reader_ready_score"),
    ("experiment_8019_v695_eligible_targets", "retention_targets_ready_score"),
    ("experiment_8020_v695_qualified_energy_fit", "energy_fit_ready_score"),
    (prior.NAME, "learning_measurement_ready_score"),
]


def load_inputs(root: Path, raw: Path) -> tuple[Json, list[Json]]:
    """Authenticate producers before copying primitive evidence and public roles."""
    refs, failures, producers = [], [], {}
    for name, field in INPUTS:
        path = root / "results" / (name + ".json")
        value = json.loads(path.read_text()) if path.is_file() else {}
        for key, expected in ((field, 1), ("flagged_adversarial", False)):
            observed = value.get(key, "MISSING_CONTRACT_FIELD")
            gate = dict(
                upstream_id=value.get("task_id", name),
                path=str(path),
                hash=sha256_file(path) if path.is_file() else None,
                artifact_field=key,
                expected=expected,
                observed=observed,
                passed=observed == expected,
            )
            if not gate["passed"]:
                failures.append(gate)
                break
        if path.is_file():
            refs.append(reference(path))
        producers[name] = value
    data = dict(references=refs)
    if failures:
        return data, failures
    try:
        trajectory = producers[prior.NAME]
        targets = producers[INPUTS[1][0]]
        for key, expected in (("retained_labels_opened", False), ("verdict_class", "null")):
            m.equal("producer/" + key, expected, trajectory.get(key, "MISSING_CONTRACT_FIELD"))
        m.equal("producer/config", prior.m.CONFIG, trajectory["config"])
        directory = Path(trajectory["trajectory_directory"])
        target = raw / "trajectory"
        target.mkdir(parents=True, exist_ok=True)
        for filename in ("inputs.json", "methods.json", "ledger.sqlite"):
            ref = copy_evidence(reference(directory / filename), raw)
            shutil.copyfile(ref["path"], target / filename)
            refs.append(ref)
        checkpoint_dir = raw / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        for ref in trajectory["checkpoint_references"]:
            source = checked(ref)
            shutil.copyfile(source, checkpoint_dir / source.name)
        shutil.copytree(directory / "checkpoints", checkpoint_dir, dirs_exist_ok=True)
        inputs = json.loads((target / "inputs.json").read_text())
        publics = {}
        for role in ("fit", "stream", "retention"):
            ref = copy_evidence(targets["public_manifests"][role], raw)
            refs.append(ref)
            publics[role] = json.loads(checked(ref).read_text())["rows"]
        m.equal("stream/original_feature_bytes", publics["stream"], inputs["sources"])
        m.equal("stream/original_slots", list(range(256)), [r["slot"] for r in publics["stream"]])
        stream = copy_evidence(targets["evaluator_manifests"]["stream"], raw)
        retained = copy_evidence(targets["evaluator_manifests"]["retention"], raw)
        refs += [stream, retained, copy_evidence(targets["exclusion_manifest"], raw)]
        for row in targets["historical_failure_logs"]:
            refs.append(copy_evidence(dict(path=row["path"], sha256=row["hash"]), raw))
        data.update(
            trajectory=str(target),
            checkpoint_directory=str(checkpoint_dir),
            stream_target=stream,
            retention_target=retained,
            retention_public=publics["retention"],
            fit_public=publics["fit"],
            prefreeze_exposure=True,
        )
    except (ValueError, KeyError, OSError) as error:
        failures.append(
            dict(
                upstream_id=prior.TASK,
                path=str(root / "results" / (prior.NAME + ".json")),
                hash=refs[-1].get("sha256") if refs else None,
                artifact_field="primitive_custody_contract",
                expected="valid original durable bytes",
                observed=str(error),
                passed=False,
            )
        )
    return data, failures


def base(failures: list[Json]) -> Json:
    """Terminal metadata keeps measurement readiness separate from scientific benefit."""
    return dict(
        experiment_id=8026,
        task_id=TASK,
        milestone="2026.10.695",
        run_date="20261002",
        schema="carnot.v695.learning_retention_audit.v1",
        honest_verdict="complete_blocked_learning_retention_audit"
        if failures
        else "complete_null_learning_retention_audit",
        verdict_class="blocked" if failures else "null",
        gate_check_summary=failures,
        claim_scope="One historically exposed development trajectory; 20 schedules are not independent environments. Retention overlap is descriptive. No generalized or deployment benefit.",
        random_seed=69526,
        verifier_is_oracle=False,
        flagged_adversarial=False,
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_specs=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        trained_head_specs=[],
        generalized_learning_benefit_score=0,
        learning_audit_ready_score=0,
        learning_benefit_score=0,
        genuine_headroom={},
        positive_control_results={},
        acceptance_gate_results=dict(
            measurement=False,
            retention_support=False,
            recovery=False,
            owned_checks=False,
            benefit=False,
        ),
        rows=[],
        independent_issue_rows=[],
        budget_comparison_rows=[],
        retention_rows=[],
        block_bootstrap_intervals=[],
        overlap_strata=[],
        crash_restart_rows=[],
        checkpoint_references=[],
        cited_upstream_artifacts=[],
        raw_shard_hashes=[],
        code_config_hashes=[],
        validation_receipts=[],
        coverage_statement_counts={},
        repository_health=[],
        phase_spans=[],
        prefreeze_retention_exposure=False,
        sample_size_budget=dict(
            intended=320,
            eligible=0,
            started=0,
            completed=0,
            excluded=0,
            failed=0,
            censored=320,
            independent=0,
        ),
    )


def recover(data: Json, held: Path, scratch: Path) -> list[Json]:
    """Use the same runnable entry point for private kills and real restart."""
    return store_recover(data, held, scratch, ROOT, ROOT / OWNED[-1])


def validation_plan(scratch: Path) -> list[CommandSpec]:
    """Reuse the working bounded runner and consumer checks with owned-only coverage."""
    commands = prior.eligible.validation_plan(scratch)
    (scratch / "coverage.ini").write_text(
        "[run]\nparallel = True\ndata_file = "
        + str(scratch / ".coverage")
        + "\ninclude =\n    "
        + "\n    ".join(str(ROOT / p) for p in OWNED)
        + "\n"
    )
    result = []
    for command in commands:
        argv = tuple(
            a.replace(prior.eligible.NAME, NAME).replace(prior.eligible.TEST, TEST)
            for a in command.argv
        )
        if command.name in ("ruff_check", "ruff_format", "strict_mypy"):
            argv = (*argv, *OWNED[1:3])
        result.append(
            replace(
                command,
                argv=argv,
                timeout_s=60 if command.scope == "repository_health" else command.timeout_s,
            )
        )
    return result


def apply_validation(value: Json, receipts: list[Json], counts: Json) -> None:
    """A complete audited null can be ready; owned or exposure failures cannot."""
    owned = [r for r in receipts if r["scope"] == "owned"]
    good = bool(owned) and all(r["passed"] for r in owned) and set(counts) == set(OWNED)
    good = good and all(
        r["num_statements"] > 0 and r["missing_lines"] == 0 for r in counts.values()
    )
    value["validation_receipts"] = owned
    value["repository_health"] = [r for r in receipts if r["scope"] == "repository_health"]
    value["coverage_statement_counts"] = counts
    value["acceptance_gate_results"]["owned_checks"] = good
    measured = all(
        value["acceptance_gate_results"][k]
        for k in ("measurement", "retention_support", "recovery")
    )
    if value["verdict_class"] != "blocked" and (not good or value["prefreeze_retention_exposure"]):
        value.update(
            verdict_class="disqualified",
            honest_verdict="complete_disqualified_learning_retention_audit",
        )
    value["learning_audit_ready_score"] = int(
        good and measured and value["verdict_class"] not in ("blocked", "disqualified")
    )
    value["learning_benefit_score"] = int(
        value["learning_audit_ready_score"] == 1 and value["acceptance_gate_results"]["benefit"]
    )
    if value["learning_benefit_score"]:
        value.update(
            verdict_class="circular_positive" if value["verifier_is_oracle"] else "positive",
            honest_verdict="complete_circular_positive_learning_retention_audit"
            if value["verifier_is_oracle"]
            else "complete_positive_learning_retention_audit",
        )


def replay(path: Path) -> Json:
    """Fresh process reduction binds every primitive and claim to exact bytes."""
    value = json.loads(path.read_text())
    for ref in (
        value["raw_shard_hashes"] + value["code_config_hashes"] + value["cited_upstream_artifacts"]
    ):
        checked(ref)
    if value["acceptance_gate_results"]["measurement"]:
        bundle = json.loads(checked(value["audit_bundle"]).read_text())
        reduced = m.reduce(bundle)
        for key, observed in reduced.items():
            if value[key] != observed:
                raise ValueError("reduction_drift:" + key)
    if value["learning_audit_ready_score"] and (
        value["verdict_class"] in ("blocked", "disqualified")
        or not value["acceptance_gate_results"]["owned_checks"]
    ):
        raise ValueError("unsafe_readiness")
    return dict(passed=True)


def terminal(path: Path) -> Json:
    """Existing cold, adversarial and strict row readers check exact candidate bytes."""
    py = str(ROOT / ".venv/bin/python")
    commands = [
        CommandSpec(
            "cold_reduction",
            (py, "-u", str(ROOT / OWNED[-1]), "--cold-replay", str(path)),
            "terminal",
            120,
        ),
        CommandSpec(
            "adversarial",
            (py, "scripts/adversarial_verify.py", str(path), "--json"),
            "terminal",
            120,
        ),
        CommandSpec(
            "strict_rows",
            (py, "scripts/verdict_row_consistency_lint.py", "--strict", str(path)),
            "terminal",
            120,
        ),
    ]
    receipts = run_commands(
        ROOT,
        commands,
        log_dir=path.parent / "terminal_logs" / sha256_file(path).split(":")[-1],
        heartbeat_s=30,
        extra_env=dict(JAX_PLATFORMS="cpu", OPENBLAS_NUM_THREADS="1"),
    )
    return dict(passed=all(r["passed"] for r in receipts), receipts=receipts)


def main(argv: list[str] | None = None) -> int:
    """Run frozen audit, cold replay, or the actual task-owned learning worker CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20261002", choices=["20261002"])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fixture-input", type=Path)
    parser.add_argument("--validation-worker", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--store-worker", type=Path)
    parser.add_argument("--store-dir", type=Path)
    parser.add_argument("--boundary", choices=["before", "after", "none"], default="none")
    args = parser.parse_args(argv)
    if args.store_worker:
        print(json.dumps(worker(args.store_worker, args.store_dir, args.boundary)), flush=True)
        return 0
    if args.cold_replay:
        print(json.dumps(replay(args.cold_replay)), flush=True)
        return 0
    started = time.monotonic()
    output = (args.output or args.root / "results" / (NAME + ".json")).absolute()
    raw = output.parent / "raw" / output.stem
    raw.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="carnot-8026-") as temporary:
        scratch = Path(temporary)
        commands = validation_plan(scratch) if not args.validation_worker else []
        atomic_json(
            raw / "methods.json",
            dict(
                config=m.CONFIG,
                commands=[asdict(c) for c in commands],
                source_roles="original stream, original fit features, original retention; no reselection",
                prior_exposure="historical development; audit annotation preview before freeze",
                artifact_guard_enabled=True,
                pretrained_load_budget=0,
                code_config_hashes=[reference(ROOT / p) for p in OWNED],
            ),
        )
        m.progress("preconditions_before")
        data, failures = (
            (json.loads(args.fixture_input.read_text()), [])
            if args.fixture_input
            else load_inputs(args.root, raw)
        )
        value = base(failures)
        value["cited_upstream_artifacts"] = data["references"]
        value["prefreeze_retention_exposure"] = data.get("prefreeze_exposure", False)
        value["verifier_is_oracle"] = bool(args.fixture_input)
        m.progress("preconditions_after")
        if not failures:
            atomic_json(raw / "audit_bundle.json", data)
            value["audit_bundle"] = reference(raw / "audit_bundle.json")
            try:
                value.update(m.reduce(data, raw / "retention_prediction_seal.json"))
                value["acceptance_gate_results"].update(
                    measurement=True,
                    retention_support=value["retention_support"]["passed"],
                    benefit=any(r["benefit_passed"] for r in value["benefit_rows"]),
                )
                inputs = json.loads((Path(data["trajectory"]) / "inputs.json").read_text())
                value["trained_head_specs"] = [
                    dict(
                        arm=a,
                        parameters=110,
                        pretrained=False,
                        scope="private task-owned recovery worker",
                        optimizer="unchanged calibrated BCE sparse SGD",
                        calibration_fixed=True,
                    )
                    for a in prior.m.ARMS[1:]
                ]
                labels = {
                    r["family_id"]: r["eligible_y"]
                    for r in json.loads(checked(data["stream_target"]).read_text())["rows"]
                }
                held = scratch / "held.json"
                shutil.copyfile(checked(data["retention_target"]), held)
                value["crash_restart_rows"] = recover(
                    dict(inputs, labels=labels), held, scratch / "crashes"
                )
                value["acceptance_gate_results"]["recovery"] = all(
                    r["passed"] for r in value["crash_restart_rows"]
                )
                atomic_json(raw / "crash_restart_rows.json", dict(rows=value["crash_restart_rows"]))
                value["rows"] = value["independent_issue_rows"]
                rs = [
                    r
                    for r in value["rows"]
                    if r["arm"] == "uniform" and r["seed"] == inputs["seeds"][0]
                ]
                held_rows = [
                    r
                    for r in value["retention_rows"]
                    if r["arm"] == "uniform" and r["seed"] == inputs["seeds"][0]
                ]
                observed = rs + held_rows
                value["sample_size_budget"] = dict(
                    intended=len(observed),
                    started=len(observed),
                    completed=len(observed),
                    eligible=sum(r["eligibility"] for r in observed),
                    excluded=sum(r["exclusion_reason"] is not None for r in observed),
                    censored=sum(r["censor_reason"] is not None for r in observed),
                    failed=0,
                    independent=len({r["source_cluster_id"] for r in observed if r["eligibility"]}),
                    independent_environments=1,
                    seeds_are_independent=False,
                )
                value["genuine_headroom"] = dict(
                    measured=True,
                    frozen_later_cost=sum(
                        r["cost"]
                        for r in value["rows"]
                        if r["arm"] == "frozen_no_write"
                        and r["seed"] == inputs["seeds"][0]
                        and r["slot"] >= 36
                        and r["eligibility"]
                    ),
                )
                if value["prefreeze_retention_exposure"]:
                    value["gate_check_summary"].append(
                        dict(
                            upstream_id=TASK,
                            path=str(raw / "methods.json"),
                            hash=sha256_file(raw / "methods.json"),
                            artifact_field="audit_protocol.retention_annotations_previewed_before_overlap_freeze",
                            expected=False,
                            observed=True,
                            passed=False,
                        )
                    )
            except m.AuditFailure as error:
                value.update(
                    verdict_class="disqualified",
                    honest_verdict="complete_disqualified_learning_retention_audit",
                )
                value["gate_check_summary"].append(
                    dict(error.operand, upstream_id=prior.TASK, path=data["trajectory"], hash=None)
                )
        m.progress("owned_validation_before")
        if not args.validation_worker:
            receipts = run_commands(
                ROOT,
                commands,
                log_dir=raw / "validation_logs",
                heartbeat_s=30,
                extra_env=dict(
                    JAX_PLATFORMS="cpu",
                    OPENBLAS_NUM_THREADS="1",
                    COVERAGE_FILE=str(scratch / ".coverage-health"),
                ),
            )
            coverage = (
                json.loads((scratch / "coverage.json").read_text())
                if (scratch / "coverage.json").is_file()
                else {}
            )
            counts = {p: r["summary"] for p, r in coverage.get("files", {}).items() if p in OWNED}
            apply_validation(value, receipts, counts)
        m.progress("owned_validation_after")
        value["code_config_hashes"] = [reference(ROOT / p) for p in OWNED] + [
            reference(Path(prior.m.__file__)),
            reference(Path(prior.m.conditioned.__file__)),
        ]
        value["raw_shard_hashes"] = [
            reference(p)
            for p in raw.rglob("*")
            if p.is_file() and p.suffix in (".json", ".sqlite", ".log")
        ]
        value["reproducibility_checksum"] = canonical_hash(
            dict(config=m.CONFIG, raw=value["raw_shard_hashes"], code=value["code_config_hashes"])
        )
        value["duration_s"] = time.monotonic() - started
        value["phase_spans"] = [
            dict(phase="independent_audit_and_owned_checks", duration_s=value["duration_s"])
        ]
        value["terminal_validation_sidecar_path"] = str(raw / "terminal_validation.json")
        value["methodology_note"] = (
            "No model loads or generation. Original slots and costs; fixed calibrated sparse learner. 10000 moving-block draws average seeds within slot. Historical development and early audit annotation access prevent inferential benefit credit. Private crashes test actual FULL synchronous SQLite commits."
        )
        value["field_principles"] = {
            k: f"Record {k} for this invocation with its source, denominator and evidential scope."
            for k in value
        }
        value["field_principles"].update(
            learning_audit_ready_score="One requires complete valid primitives, retention support, real recovery and all owned checks; disqualification forces zero.",
            learning_benefit_score="Every registered causal, retention and uncertainty gate must pass; protocol exposure bars inferential credit.",
            generalized_learning_benefit_score="Exposed development and private fixtures cannot close generalization.",
            overlap_strata="Public-fit cutpoints diagnose locality; shared coefficients and decay prevent a proof of nonforgetting.",
            model_invocation_counts="Current pretrained calls are zero; imported training history is not current work.",
            sample_size_budget="Count original source groups once; schedules do not add environments.",
        )
        m.progress("publication_before")
        receipt = publish_primary(
            output, value, lambda p: replay(p) if args.validation_worker else terminal(p)
        )
        post = replay(output) if args.validation_worker else terminal(output)
        readers = reader_receipt(
            TASK,
            output.parent,
            field="learning_audit_ready_score",
            expected=value["learning_audit_ready_score"],
        )
        atomic_json(
            raw / "terminal_validation.json",
            dict(publication=receipt, published=post, readers=readers),
        )
        if not post["passed"] or not readers["passed"]:
            raise ValueError("published_validation_failed")
        m.progress("publication_after")
    return 0
