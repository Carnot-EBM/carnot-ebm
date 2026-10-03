"""REQ-REPORT-8039: independently audit a sealed exposed-development trajectory."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import selectors
import shutil
import sqlite3
import subprocess
import tempfile
import time
from typing import Any

from carnot import experiment_8032_v696_sealed_methods as sealed
from carnot import experiment_8038_v696_windowed_online_learning as producer
from carnot.experiment_artifacts import artifact_output_root
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.evidence_features_custody_7980 import checked, reference
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.learning_store_8026 import worker
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.verify import learning_benefit_8039 as a
from carnot.verify.evidence_features_7980 import normalized

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[2]
NAME = "experiment_8039_v696_learning_benefit_audit"
TASK = "exp8039-learning-benefit-audit"
SCRIPT = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_learning_benefit_8039.py"
OWNED = [f"python/carnot/{NAME}.py", "python/carnot/verify/learning_benefit_8039.py", SCRIPT]
MODEL_SPECS: list[str] = []


def load_inputs(root: Path, raw: Path) -> tuple[Json, list[Json]]:
    """Check protocol, publication and final seals without opening evaluator targets."""
    refs, gates, values = [], [], {}
    path = root / "results" / (sealed.NAME + ".json")
    try:
        for n, name, ready in (
            (8032, sealed.NAME, "learning_inputs_ready_score"),
            (8038, producer.NAME, "learning_trajectory_ready_score"),
        ):
            path = root / "results" / (name + ".json")
            sealed.require(path, "primary_exists", True, path.is_file())
            value = json.loads(path.read_text())
            binding = json.loads(Path(value["terminal_validation_sidecar_path"]).read_text())[
                "publication"
            ]
            sidecar = json.loads(Path(binding["sidecar_path"]).read_text())
            for field, expected, observed in (
                ("experiment_id", n, value.get("experiment_id", "MISSING_CONTRACT_FIELD")),
                (ready, 1, value.get(ready, "MISSING_CONTRACT_FIELD")),
                (
                    "flagged_adversarial",
                    False,
                    value.get("flagged_adversarial", "MISSING_CONTRACT_FIELD"),
                ),
                ("primary_sha256", sha256_file(path), binding["primary_sha256"]),
                ("sidecar.primary_sha256", sha256_file(path), sidecar["primary_sha256"]),
                ("primary_path", str(path), binding["primary_path"]),
                ("sidecar.primary_path", str(path), sidecar["primary_path"]),
                ("report.passed", True, sidecar["report"]["passed"]),
            ):
                sealed.require(path, field, expected, observed)
                gates.append(
                    dict(
                        upstream_id=value["task_id"],
                        path=str(path),
                        sha256=sha256_file(path),
                        artifact_field=field,
                        expected=expected,
                        observed=observed,
                        passed=True,
                        check_name=field,
                    )
                )
            for p in (
                path,
                Path(value["terminal_validation_sidecar_path"]),
                Path(binding["sidecar_path"]),
            ):
                refs.append(sealed.copy_bound(reference(p), raw))
            values[n] = value
        protocol_ref = sealed.copy_bound(values[8032]["methods_reference"], raw)
        refs.append(protocol_ref)
        protocol = json.loads(checked(protocol_ref).read_text())
        sealed.require(path, "protocol.methods", sealed.METHODS, protocol["methods"])
        sealed.require(path, "protocol.frozen", True, protocol["frozen"])
        for access in values[8032]["evaluator_access_log"]:
            sealed.require(
                path,
                "protocol_before_evaluator_access",
                True,
                access["protocol_frozen_before_access"]
                and access["opened_at_ns"] > protocol["frozen_at_ns"],
            )
        sealed.require(
            path, "retained_labels_opened", False, values[8038]["retained_labels_opened"]
        )
        sealed.require(
            path,
            "producer_protocol_binding",
            True,
            any(
                r["sha256"] == protocol_ref["sha256"]
                for r in values[8038]["cited_upstream_artifacts"]
            ),
        )
        trajectory = Path(values[8038]["trajectory_directory"])
        checked(values[8038]["trajectory_seal"])
        manifest = json.loads((trajectory / "seal.json").read_text())
        sealed.require(path, "trajectory.sealed", True, manifest["sealed"])
        for ref in manifest["references"]:
            checked(ref)
        for ref in values[8038]["checkpoint_references"]:
            checked(ref)
        public = {}
        for role in ("fit", "retention"):
            ref = sealed.copy_bound(values[8032]["role_manifests"]["public"][role], raw)
            refs.append(ref)
            public[role] = json.loads(checked(ref).read_text())["rows"]
            for r in public[role]:
                r["source_cluster_id"] = normalized(bytes.fromhex(r["source_bytes"]))
        inputs = json.loads((trajectory / "inputs.json").read_text())
        for field in ("parameters", "calibration", "geometry"):
            sealed.require(
                path, "starting_head." + field, protocol["head"][field], inputs["head"][field]
            )
        return dict(
            trajectory=str(trajectory),
            fit_public=public["fit"],
            retention_public=public["retention"],
            retention_target=values[8032]["role_manifests"]["evaluator"]["retention"],
            references=refs,
            gate_checks=gates,
            historical_exposure=values[8032]["historical_exposure"],
            repository_health=values[8038]["repository_health"],
        ), []
    except sealed.Contract as error:
        return dict(references=refs), [error.gate]
    except (KeyError, ValueError, OSError) as error:
        return dict(references=refs), [
            sealed.Contract(
                path,
                str(error.args[0]) if isinstance(error, KeyError) else "input_contract",
                "complete byte-bound evidence",
                "MISSING_CONTRACT_FIELD" if isinstance(error, KeyError) else str(error),
            ).gate
        ]


def validation_plan(scratch: Path) -> list[CommandSpec]:
    """Reuse bounded checks and limit coverage to added modules and actual CLI."""
    commands = producer.validation_plan(scratch)
    (scratch / "coverage.ini").write_text(
        "[run]\nparallel = True\ndata_file = "
        + str(scratch / ".coverage")
        + "\ninclude =\n    "
        + "\n    ".join(str(ROOT / p) for p in OWNED)
        + "\n"
    )
    return [
        replace(
            c,
            argv=tuple(
                x.replace(producer.NAME, NAME)
                .replace(producer.TEST, TEST)
                .replace("python/carnot/verify/windowed_online_8038.py", OWNED[1])
                for x in c.argv
            ),
        )
        for c in commands
    ]


def terminal(path: Path) -> Json:
    """Cold replay and existing adversarial readers check candidate and final bytes."""
    py = str(ROOT / ".venv/bin/python")
    commands = [
        CommandSpec(
            "cold_reduction",
            (py, "-u", str(ROOT / SCRIPT), "--cold-replay", str(path)),
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
    raw = path.parent / "raw" / NAME if path.parent.name == "results" else path.parent
    receipts = run_commands(
        ROOT,
        commands,
        log_dir=raw / "terminal_logs" / sha256_file(path).split(":")[1],
        heartbeat_s=30,
    )
    return dict(passed=all(r["passed"] for r in receipts), receipts=receipts)


def recover(
    initial: Json, scratch: Path, durable: Path, *, boundary_timeout_s: float = 60
) -> list[Json]:
    """Real task-owned child crashes establish fixture recovery, never scientific benefit."""
    source = scratch / "fixture.json"
    row = dict(family_id="private", q=0.5, features=[0.3] * 8, public_eligible=True)
    rows = []
    env = dict(
        os.environ,
        PYTHONUNBUFFERED="1",
        JAX_PLATFORMS="cpu",
        CARNOT_EXPERIMENT_ARTIFACT_ROOT=str(scratch),
    )
    for arm in a.ARMS[:3]:
        atomic_json(
            source,
            dict(
                head=initial,
                arm=arm,
                seed=101,
                next_source=row,
                releases=[dict(source=row, family_id=f"private-{i}", y=i % 2) for i in range(4)],
            ),
        )
        base = (
            str(ROOT / ".venv/bin/python"),
            "-u",
            str(ROOT / SCRIPT),
            "--store-worker",
            str(source),
        )
        control = scratch / arm / "control"
        receipts = run_commands(
            ROOT,
            [CommandSpec("uninterrupted", (*base, "--store-dir", str(control)), "fixture", 60)],
            log_dir=scratch / arm / "control_logs",
            extra_env=env,
            heartbeat_s=30,
        )
        a.equal("uninterrupted_exit", True, receipts[0]["passed"])
        expected = json.loads((control / "outcome.json").read_text())
        for boundary in ("before", "after"):
            directory = scratch / arm / boundary
            argv = (*base, "--store-dir", str(directory), "--boundary", boundary)
            a.progress(
                "crash_" + arm + "_" + boundary + "_before_subprocess", len(rows), 6 - len(rows)
            )
            child = subprocess.Popen(
                argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, bufsize=0
            )
            assert child.stdout is not None
            selector = selectors.DefaultSelector()
            selector.register(child.stdout, selectors.EVENT_READ)
            output = b""
            began = time.monotonic()
            try:
                while b"COMMIT_BOUNDARY " + boundary.encode() not in output:
                    if time.monotonic() - began > boundary_timeout_s:
                        raise TimeoutError("commit_boundary_timeout")
                    if selector.select(timeout=20):
                        chunk = os.read(child.stdout.fileno(), 4096)
                        if not chunk:
                            raise ValueError("commit_boundary_missing")
                        output += chunk
                    a.progress("crash_child_pending", len(rows), 6 - len(rows))
                child.kill()
                exit_code = child.wait(timeout=10)
            finally:
                selector.close()
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=10)
                child.stdout.close()
            a.progress(
                "crash_" + arm + "_" + boundary + "_after_subprocess", len(rows), 6 - len(rows)
            )
            log = directory / "killed.log"
            log.write_bytes(output)
            db = sqlite3.connect(directory / "ledger.sqlite")
            count = db.execute(
                "SELECT count(*) FROM events WHERE kind='learning_commit'"
            ).fetchone()[0]
            db.close()
            restart = (*base, "--store-dir", str(directory))
            receipts = run_commands(
                ROOT,
                [CommandSpec("restart", restart, "fixture", 60)],
                log_dir=directory / "restart_logs",
                extra_env=env,
                heartbeat_s=30,
            )
            actual = json.loads((directory / "outcome.json").read_text())
            rows.append(
                dict(
                    arm=arm,
                    boundary=boundary,
                    command_argv=list(argv),
                    restart_argv=list(restart),
                    killed_exit_code=exit_code,
                    restart_exit_code=receipts[0]["exit_code"],
                    commits_before_restart=count,
                    exactly_once=actual["exactly_once"],
                    next_prediction_matches=actual["next_probability"]
                    == expected["next_probability"],
                    actual_state=actual,
                    expected_state=expected,
                    release_ids=actual["release_ids"],
                    log_sha256=sha256_file(log),
                    killed_log_path=str(durable / log.relative_to(scratch)),
                    validation_receipts=receipts,
                    passed=exit_code == -9
                    and receipts[0]["passed"]
                    and actual == expected
                    and count == int(boundary == "after"),
                    scope="private fixture; real calibrated optimizer and FULL synchronous commit; recovery only",
                )
            )
    shutil.copytree(scratch, durable)
    # Replace private receipt paths with their copied, durable equivalents.
    for r in rows:
        for receipt in r["validation_receipts"]:
            receipt["log_path"] = str(durable / Path(receipt["log_path"]).relative_to(scratch))
    atomic_json(durable / "recovery_rows.json", dict(rows=rows))
    return rows


def base(failures: list[Json]) -> Json:
    """Terminal absence and complete measurements share one explicit artifact contract."""
    value: Json = dict(
        experiment_id=8039,
        task_id=TASK,
        milestone="2026.10.696",
        schema="carnot.v696.learning_benefit_audit.v1",
        run_date="20261002",
        honest_verdict="complete_blocked_learning_trajectory"
        if failures
        else "complete_null_learning_benefit",
        verdict_class="blocked" if failures else "null",
        claim_scope="This invocation independently audits one finite exposed-development CPU trajectory. Recovery fixtures establish storage behavior only. No generalization or deployment claim.",
        gate_check_summary=failures,
        learning_audit_ready_score=0,
        learning_benefit_score=0,
        generalized_learning_benefit_score=0,
        effective_independent_streams=1,
        random_seed=a.CONFIG["seed"],
        config=a.CONFIG,
        verifier_is_oracle=False,
        genuine_headroom=dict(measured=False),
        positive_control_results={},
        acceptance_gate_results=dict(
            validity=False, retention=False, recovery=False, owned_checks=False
        ),
        inference_substrate="verifier_scoring",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_specs=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        trained_head_specs=[],
        substrate_declaration=dict(
            custody="aggregation_from_upstream_artifacts",
            numerical="verifier_scoring",
            pretrained="no_model_load",
        ),
        historical_exposure={},
        validation_receipts=[],
        repository_health=[],
        coverage_statement_counts={},
        flagged_adversarial=False,
        sample_size_budget=dict(
            intended=256,
            eligible=0,
            started=0,
            completed=0,
            excluded=0,
            censored=256,
            failed=0,
            independent=0,
        ),
    )
    for key in (
        "rows",
        "independent_replay_rows",
        "later_loss_rows",
        "retention_rows",
        "overlap_strata_rows",
        "recovery_rows",
        "primary_hypothesis_results",
        "prefreeze_access_checks",
        "checkpoint_references",
        "raw_shard_hashes",
        "code_config_hashes",
        "cited_upstream_artifacts",
    ):
        value[key] = []
    return value


def replay(path: Path) -> Json:
    """Fresh reduction checks each final aggregate and guards readiness from mutations."""
    value = json.loads(path.read_text())
    for ref in value["raw_shard_hashes"] + value["code_config_hashes"]:
        checked(ref)
    if value["acceptance_gate_results"]["validity"]:
        bundle = json.loads(checked(value["audit_bundle"]).read_text())
        reduced = a.replay_trajectory(Path(bundle["trajectory"]))
        retained = a.retention(
            bundle, reduced, Path(value["retention_prediction_seal"]["path"]), cold=True
        )
        compared = a.compare(reduced["rows"], retained["retention_rows"])
        for fields in (reduced, retained, compared):
            for key, observed in fields.items():
                a.equal("reduction_drift:" + key, observed, value[key])
        recovery = json.loads(checked(value["recovery_reference"]).read_text())["rows"]
        a.equal("recovery_drift", recovery, value["recovery_rows"])
        a.equal(
            "recovery_gate",
            all(r["passed"] for r in recovery),
            value["acceptance_gate_results"]["recovery"],
        )
    a.equal(
        "unsafe_readiness",
        True,
        not value["learning_audit_ready_score"]
        or (
            value["verdict_class"] in ("null", "positive")
            and all(value["acceptance_gate_results"].values())
        ),
    )
    a.equal("generalization_claim", 0, value["generalized_learning_benefit_score"])
    return dict(passed=True, sha256=sha256_file(path))


def main(argv: list[str] | None = None) -> int:
    """Freeze code and access rules, measure, validate and atomically publish once."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261002"], default="20261002")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--fixture-bundle", type=Path)
    parser.add_argument("--validation-worker", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--store-worker", type=Path)
    parser.add_argument("--store-dir", type=Path)
    parser.add_argument("--boundary", choices=["none", "before", "after"], default="none")
    args = parser.parse_args(argv)
    started = time.monotonic()
    a.progress("begin_no_pretrained_loads_or_generations")
    try:
        if args.store_worker:
            if args.store_dir is None:
                raise ValueError("store_directory_required")
            worker(args.store_worker, args.store_dir, args.boundary)
            return 0
        if args.cold_replay:
            print(json.dumps(replay(args.cold_replay)), flush=True)
            return 0
        output = artifact_output_root(root=args.root) / (NAME + ".json")
        raw = output.parent / "raw" / NAME / "invocations" / str(time.time_ns())
        raw.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="carnot-8039-") as temporary:
            scratch = Path(temporary)
            commands = validation_plan(scratch)
            a.progress("input_seals_before")
            bundle, failures = (
                (json.loads(args.fixture_bundle.read_text()), [])
                if args.fixture_bundle
                else load_inputs(args.root, raw)
            )
            value = base(failures)
            value["gate_check_summary"] = failures or bundle.get("gate_checks", [])
            value["cited_upstream_artifacts"] = bundle.get("references", [])
            value["historical_exposure"] = bundle.get(
                "historical_exposure", dict(artificial=bool(args.fixture_bundle))
            )
            value["repository_health"] = bundle.get("repository_health", [])
            dependencies = [
                Path(a.math.__file__),
                Path(producer.w.__file__),
                Path(producer.w.old.__file__),
                ROOT / "python/carnot/reporting/learning_store_8026.py",
                ROOT / "python/carnot/reporting/current_work_receipt.py",
                ROOT / "python/carnot/reporting/primary_publication.py",
            ]
            value["code_config_hashes"] = [
                sealed.copy_bound(reference(p), raw, "code")
                for p in [*(ROOT / p for p in OWNED + [TEST]), *dependencies]
            ]
            atomic_json(
                raw / "configuration.json",
                dict(
                    config=a.CONFIG,
                    code=value["code_config_hashes"],
                    task_id=TASK,
                    identity=8039,
                    frozen=True,
                    frozen_at_ns=time.time_ns(),
                    target_identity=bundle.get("retention_target"),
                    artifact_guard_enabled=True,
                    acceptance="coefficient/probability<=1e-10; gain>=.02; no added false accepts; cost drift<=.02; Brier drift<=.01",
                    task_overrides="User Exp8039 limits replace Exp8032 stricter retention diagnostics for H3 only.",
                    data_access="replay released stream labels; retention only after prediction and final checkpoint seals",
                    commands=[
                        dict(name=c.name, argv=c.argv, timeout_s=c.timeout_s) for c in commands
                    ],
                ),
            )
            frozen = time.monotonic()
            bundle["audit_protocol"] = reference(raw / "configuration.json")
            a.progress("methods_code_budgets_identity_frozen")
            if not failures:
                shutil.copytree(Path(bundle["trajectory"]), raw / "trajectory")
                bundle["trajectory"] = str(raw / "trajectory")
                reduced = a.replay_trajectory(raw / "trajectory")
                value.update(reduced)
                prediction_seal = raw / "retention_predictions.json"
                value.update(a.retention(bundle, reduced, prediction_seal))
                value.update(a.compare(reduced["rows"], value["retention_rows"]))
                bundle["retention_target"] = sealed.copy_bound(
                    bundle["retention_target"], raw, "evaluator"
                )
                atomic_json(raw / "bundle.json", bundle)
                value["audit_bundle"] = reference(raw / "bundle.json")
                value["retention_prediction_seal"] = reference(prediction_seal)
                recovery_scratch = scratch / "recovery"
                recovery_scratch.mkdir()
                value["recovery_rows"] = recover(
                    reduced["initial_head"], recovery_scratch, raw / "recovery"
                )
                value["recovery_reference"] = reference(raw / "recovery/recovery_rows.json")
                value["positive_control_results"] = a.controls()
                atomic_json(
                    raw / "independent_measurement.json",
                    {
                        key: value[key]
                        for key in (
                            "rows",
                            "independent_replay_rows",
                            "later_loss_rows",
                            "retention_rows",
                            "retention_drift_rows",
                            "overlap_strata_rows",
                            "primary_hypothesis_results",
                            "recovery_rows",
                        )
                    },
                )
                value["acceptance_gate_results"].update(
                    validity=True,
                    retention=value["retention_support"]["passed"],
                    recovery=all(r["passed"] for r in value["recovery_rows"]),
                )
                value["genuine_headroom"] = dict(
                    measured=True,
                    scope="natural issued later decisions",
                    cost_numerator=sum(
                        r["cost"]
                        for r in reduced["rows"]
                        if r["arm"] == "cumulative" and r["eligibility"] and r["post_first_update"]
                    ),
                    cost_denominator=sum(
                        r["arm"] == "cumulative" and r["eligibility"] and r["post_first_update"]
                        for r in reduced["rows"]
                    ),
                )
                value["trained_head_specs"] = [
                    dict(
                        arm=r["arm"],
                        seed=r["seed"],
                        parameters=110,
                        device="cpu",
                        pretrained=False,
                        current_training=False,
                        imported_updates=r["actual_gradient_count"],
                    )
                    for r in reduced["final_checkpoint_rows"]
                ]
            measured = time.monotonic()
            if not args.validation_worker:
                a.progress("owned_validation_before")
                receipts = run_commands(
                    ROOT,
                    commands,
                    log_dir=raw / "validation_logs",
                    heartbeat_s=30,
                    extra_env=dict(
                        CARNOT_8039_COVERAGE_CONFIG=str(scratch / "coverage.ini"),
                        COVERAGE_FILE=str(scratch / ".coverage-health"),
                        OPENBLAS_NUM_THREADS="1",
                        JAX_PLATFORMS="cpu",
                    ),
                )
                value["validation_receipts"] = [r for r in receipts if r["scope"] == "owned"]
                value["repository_health"] += [
                    r for r in receipts if r["scope"] == "repository_health"
                ]
                coverage = (
                    json.loads((scratch / "coverage.json").read_text())
                    if (scratch / "coverage.json").exists()
                    else dict(files={})
                )
                counts = {k: r["summary"] for k, r in coverage["files"].items()}
                value["coverage_statement_counts"] = counts
                value["acceptance_gate_results"]["owned_checks"] = (
                    len(counts) == len(OWNED)
                    and all(r["missing_lines"] == 0 for r in counts.values())
                    and all(r["passed"] for r in value["validation_receipts"])
                )
                if not failures and not value["acceptance_gate_results"]["owned_checks"]:
                    value.update(
                        verdict_class="disqualified",
                        honest_verdict="complete_disqualified_owned_validation",
                    )
                a.progress("owned_validation_after")
            if args.fixture_bundle:
                value.update(
                    verifier_is_oracle=True,
                    verdict_class="circular_positive",
                    honest_verdict="complete_circular_positive_learning_fixture",
                    learning_benefit_score=0,
                )
            elif value["verdict_class"] == "null":
                value["learning_audit_ready_score"] = int(
                    all(value["acceptance_gate_results"].values())
                )
                if value["learning_benefit_score"] and value["learning_audit_ready_score"]:
                    value.update(
                        verdict_class="positive",
                        honest_verdict="complete_positive_local_learning_benefit",
                    )
            for key in (
                "intended",
                "eligible",
                "completed",
                "excluded",
                "failed",
                "censored",
                "independent",
            ):
                value[key + "_count"] = value["sample_size_budget"][key]
            value["raw_shard_hashes"] = [
                reference(p) for p in sorted(raw.rglob("*")) if p.is_file()
            ]
            value["reproducibility_checksum"] = canonical_hash(
                dict(
                    config=a.CONFIG, code=value["code_config_hashes"], raw=value["raw_shard_hashes"]
                )
            )
            value["duration_s"] = time.monotonic() - started
            value["phase_spans"] = [
                dict(phase="admission_and_freeze", duration_s=frozen - started),
                dict(phase="independent_audit_and_fixture_recovery", duration_s=measured - frozen),
                dict(phase="owned_validation", duration_s=time.monotonic() - measured),
            ]
            value["methodology_note"] = (
                "No pretrained loads or generation. Separate cubic/BCE equations replay all states. Fit-only overlap and retention predictions precede targets. Seeds average within slots; 10000 moving blocks test .02 margin. Capstone must combine H1/H2/H3 with Holm .05. Exposed development and recovery fixtures cannot establish generalized learning."
            )
            value["terminal_validation_sidecar_path"] = str(raw / "terminal_validation.json")
            value["field_principles"] = {
                k: f"Bind {k} to this invocation, original input bytes, denominator and evidence scope."
                for k in value
            }
            value["field_principles"].update(
                learning_audit_ready_score="Complete valid evidence and all owned checks qualify readiness; positive benefit is a separate gate.",
                learning_benefit_score="One requires every local H3 support, margin, false-accept and retention gate. Capstone family credit remains separate.",
                generalized_learning_benefit_score="Exposed development, one trajectory and private fixtures cannot establish generalization.",
                primary_hypothesis_results="H3 tests the .02 margin on seed-averaged slots. H1/H2/H3 require Holm .05 in capstone.",
                recovery_rows="Real fixture crashes establish exactly-once recovery only; they supply no natural benefit.",
                historical_exposure="Prior annotations and original V695 failures stay exposed; a fresh invocation cannot make them unseen.",
                numerical_agreement="Measured probability and coefficient agreement comes from separate equations checked against original producer bytes.",
                trained_head_specs="Imported CPU-head updates differ from current fixture optimizer work and pretrained model invocations.",
                sample_size_budget="Count original groups once, preserve exclusions and delay censoring, and give seeds no independent-stream credit.",
            )
            a.progress("publication_before")
            published = publish_primary(
                output, value, replay if args.validation_worker else terminal
            )
            post = replay(output) if args.validation_worker else terminal(output)
            readers = reader_receipt(
                TASK,
                output.parent,
                field="learning_audit_ready_score",
                expected=value["learning_audit_ready_score"],
            )
            atomic_json(
                Path(value["terminal_validation_sidecar_path"]),
                dict(publication=published, published=post, readers=readers),
            )
            a.equal("terminal_validation", True, post["passed"] and readers["passed"])
            a.progress("publication_after", 1, 0)
        return 0
    except (ValueError, KeyError, OSError, TimeoutError) as error:
        print(f"[exp8039] failed={error}", flush=True)
        return 1
