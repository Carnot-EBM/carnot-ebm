"""REQ-REPORT-8064: publish verified development-only fresh-feedback trajectories."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot import experiment_8051_v697_feedback_constrained_learning as historical
from carnot import experiment_8058_v698_sealed_evidence_methods as prior
from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.evidence_features_custody_7980 import checked, reference
from carnot.reporting.primary_publication import publish_primary, read_bound_sidecar
from carnot.reporting.v686_contract_validation import run_check
from carnot.verify import fresh_feedback_8064 as m

Json = dict[str, Any]
ROOT = prior.ROOT
NAME = "experiment_8064_v698_fresh_feedback_learning"
TASK = "exp8064-fresh-feedback-learning"
CLI = f"scripts/experiments/{NAME}.py"
MODULE = f"python/carnot/{NAME}.py"
TEST = "tests/python/test_fresh_feedback_8064.py"
OWNED = [MODULE, CLI, "python/carnot/verify/fresh_feedback_8064.py"]
MODEL_SPECS: list[str] = []


def load_inputs(root: Path, raw: Path) -> tuple[Json, list[Json]]:
    """Bind the CPU branch without requiring current Qwen scoring or model work."""
    data, failures = historical.load_inputs(root, raw)
    failures = [
        dict(
            check=r["check_name"],
            upstream=r.get("upstream_id"),
            path=r["path"],
            hash=r.get("sha256"),
            field=r.get("artifact_field"),
            op="==",
            expected=r["expected"],
            observed=r["observed"],
        )
        for r in failures
    ]
    path = root / "results" / (prior.NAME + ".json")
    try:
        value = json.loads(path.read_text())
        side = Path(value["terminal_validation_sidecar_path"])
        binding = json.loads(side.read_text())["publication"]
        report_path = Path(binding["sidecar_path"])
        report = read_bound_sidecar(path, report_path)
        for field, expected, observed in [
            ("experiment_id", 8058, value.get("experiment_id")),
            ("learning_protocol_ready_score", 1, value.get("learning_protocol_ready_score")),
            ("flagged_adversarial", False, value.get("flagged_adversarial")),
            ("primary_sha256", sha256_file(path), binding.get("primary_sha256")),
            ("primary_path", str(path.absolute()), report.get("primary_path")),
            ("report.passed", True, report["report"]["passed"]),
            ("frozen_learning_methods", prior.methods(), value["method_freeze"]["methods"]),
        ]:
            if observed != expected:
                failures.append(
                    dict(
                        check="input_authentication",
                        upstream=prior.TASK,
                        path=str(path),
                        hash=sha256_file(path),
                        field=field,
                        op="==",
                        expected=expected,
                        observed=observed,
                    )
                )
        for p in (
            path,
            side,
            report_path,
            ROOT / "ops/exclusion_manifest.yaml",
            ROOT / "python/carnot/verify/feedback_constrained_8051.py",
            ROOT / "python/carnot/experiment_8051_v697_feedback_constrained_learning.py",
        ):
            data["references"].append(historical.prior.upstream.copy_bound(reference(p), raw))
        if not failures:
            eligibility = {r["family_id"]: r for r in value["rows"] if r["role"] == "stream"}
            public = value["role_manifests"]["stream"]
            retention = value["role_manifests"]["retention"]
            for ref in (public, retention):
                data["references"].append(historical.prior.upstream.copy_bound(ref, raw))
            sources = json.loads(checked(public).read_text())["rows"]
            if [r["family_id"] for r in sources] != [r["family_id"] for r in data["sources"]]:
                raise ValueError("original_stream_identity")
            data["sources"] = [
                dict(r, eligible=eligibility[r["family_id"]]["eligible"]) for r in sources
            ]
            data["retention"] = json.loads(checked(retention).read_text())["rows"]
            parent = json.loads(
                (root / "results" / "experiment_8032_v696_sealed_methods.json").read_text()
            )
            data["retention_target"] = historical.prior.upstream.copy_bound(
                parent["role_manifests"]["evaluator"]["retention"], raw
            )
            data["references"].append(data["retention_target"])
            data["repository_health"] = value["repository_health"]
            data["methods"] = value["method_freeze"]
            health = root / "results/raw" / NAME / "preflight/repository_health.json"
            if health.is_file():
                data["current_repository_health"] = json.loads(health.read_text())["receipts"]
                for p in [
                    health,
                    *(Path(r["log_path"]) for r in data["current_repository_health"]),
                ]:
                    data["references"].append(
                        historical.prior.upstream.copy_bound(reference(p), raw)
                    )
    except (OSError, KeyError, ValueError) as error:
        failures.append(
            dict(
                check="frozen_learning_inputs",
                upstream=prior.TASK,
                path=str(path),
                hash=sha256_file(path) if path.is_file() else None,
                field="complete_authenticated_operands",
                op="==",
                expected=True,
                observed=str(error),
            )
        )
    return data, failures


def manifest(private: Path) -> list[Json]:
    """Reuse bounded repository checks and cover only this experiment's statements."""
    specs = prior.manifest(private)
    config = private / "coverage.ini"
    config.write_text(
        "[run]\nparallel=true\ndata_file="
        + str(private / ".coverage")
        + "\ninclude=\n"
        + "".join("    " + str(ROOT / p) + "\n" for p in OWNED)
    )
    for spec in specs:
        spec["argv"] = [
            a.replace(prior.MODULE, MODULE).replace(prior.CLI, CLI).replace(prior.TEST, TEST)
            for a in spec["argv"]
        ]
        if spec["name"].startswith("coverage_"):
            spec["argv"] = [
                "--include=" + ",".join(str(ROOT / p) for p in OWNED)
                if a.startswith("--include=")
                else a
                for a in spec["argv"]
            ]
        if spec["name"] in ("ruff_check", "ruff_format", "strict_mypy"):
            spec["argv"].append(OWNED[2])
    return specs


def retention(data: Json, result: Json, raw: Path) -> list[Json]:
    """All terminal heads and retention predictions seal before targets are opened."""
    predictions = []
    for seal in result["final_head_seals"]:
        head = dict(data["head"], parameters=seal["parameters"], decay_scale=1.0)
        for r in data["retention"]:
            p = m.old.probability(head, m.old.design(head, r)) if r["public_eligible"] else None
            predictions.append(
                dict(
                    seed=seal["seed"],
                    arm=seal["arm"],
                    slot=r["slot"],
                    source=r["source_cluster_id"],
                    family_id=r["family_id"],
                    probability=p,
                    action=m.action(p),
                    head_hash=seal["head_hash"],
                )
            )
    atomic_json(
        raw / "retention_predictions.json",
        dict(
            rows=predictions,
            final_head_seals=result["final_head_seals"],
            retention_labels_opened=False,
        ),
    )
    targets = data.get("retention_labels")
    if targets is None:
        targets = {
            r["family_id"]: r["eligible_y"]
            for r in json.loads(checked(data["retention_target"]).read_text())["rows"]
        }
    rows = []
    for r in predictions:
        y = targets[r["family_id"]]
        valid = y is not None and r["probability"] is not None
        rows.append(
            dict(
                r,
                y=y,
                unit=f"retention/{r['slot']}",
                numerator=m.loss(r["action"], y) if valid else None,
                denominator=int(valid),
                brier=(r["probability"] - y) ** 2 if valid else None,
                status="completed" if valid else "excluded",
                exclusion_reason=None if valid else "ineligible_complete_target",
            )
        )
    atomic_json(raw / "retention_rows.json", dict(rows=rows))
    return rows


def build(
    data: Json,
    failures: list[Json],
    raw: Path,
    receipts: list[Json],
    coverage: Json,
    fixture: bool,
    started: int,
) -> Json:
    """Completeness can qualify a null method; fixture outcomes never earn science."""
    complete = (raw / "trajectory" / "final_head_seals.json").is_file() and (
        raw / "retention_rows.json"
    ).is_file()
    passed = (
        bool(receipts)
        and all(r["passed"] for r in receipts)
        and all(p in coverage and coverage[p]["summary"]["missing_lines"] == 0 for p in OWNED)
    )
    owned_failure = any(r.get("classification") == "owned" for r in failures)
    kind = (
        "disqualified"
        if owned_failure
        else "blocked"
        if failures
        else "circular_positive"
        if fixture
        else "null"
        if passed and complete
        else "disqualified"
    )
    value = historical.prior.base(failures)
    value.update(
        experiment_id=8064,
        task_id=TASK,
        milestone="2026.10.698",
        schema="carnot.v698.fresh_feedback_learning.v1",
        run_date="20261003",
        honest_verdict="complete_" + kind + "_fresh_feedback_learning",
        verdict_class=kind,
        claim_scope="Finite historically exposed development stream. Trajectory readiness certifies causal completeness only. No scientific gain, independent environment, future safety or deployment credit.",
        verifier_is_oracle=fixture,
        required_checks_passed=passed,
        learning_trajectory_ready_score=int(kind == "null" and complete),
        acceptance_certificate="empirical_only",
        config=m.CONFIG,
        frozen_methods=data.get("methods", prior.methods()),
        methodology_note="Original slot+20 labels follow durable predictions. Three commitments use four full-batch calibrated BCE gradients on newest released update-only rows. Admission rows are one-use, guarded comparisons include initial and incumbent, and decision times are shared. Rejected gradients and all alpha scans count as work.",
        source_artifact_hashes=data.get("references", []),
        validation_receipts=receipts,
        coverage_statement_counts={k: v["summary"] for k, v in coverage.items()},
        gate_check_summary=failures,
        repository_health=data.get("repository_health", []),
        current_repository_health=data.get("current_repository_health", []),
        inference_substrate="verifier_ensemble_against_cached_candidates"
        if complete
        else "aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        trained_head_specs=[],
        substrate_declaration=dict(
            inference_substrate="verifier_ensemble_against_cached_candidates",
            no_model_load=True,
            MODEL_SPECS=[],
            trained_head_specs=[],
        ),
        future_access_tests=[
            dict(
                test_path=TEST,
                scenarios=["SCENARIO-REPORT-8064-CAUSAL", "SCENARIO-REPORT-8064-DURABLE"],
                passed=passed,
                scope="Private mutations and process deaths; no future safety certificate",
            )
        ],
        terminal_validation_sidecar_path=str(raw.parent / "terminal_validation.json"),
        trajectory_directory=str(raw / "trajectory"),
        retention_rows=[],
        retained_labels_opened=complete,
        acceptance_gate_results=dict(trajectory=complete, owned_checks=passed),
        preconditions_checked=data.get("gate_checks", []) + failures,
    )
    for field in m.FIELDS.values():
        value[field] = []
    value.update(per_seed_false_accept_rows=[], pending_update_rows=[], later_prediction_rows=[])
    if complete:
        value.update(m.reduce(raw / "trajectory"))
        value["retention_rows"] = json.loads((raw / "retention_rows.json").read_text())["rows"]
        value["trained_head_specs"] = [
            dict(
                arm=r["arm"],
                seed=r["seed"],
                head_hash=r["head_hash"],
                parameters=len(r["parameters"]),
                device="cpu",
                pretrained=False,
            )
            for r in value["final_head_seals"]
        ]
        value["substrate_declaration"]["trained_head_specs"] = value["trained_head_specs"]
    if failures and not owned_failure:
        value["honest_verdict"] = "complete_blocked_" + str(failures[0]["check"]).replace(".", "_")
    if not passed and not fixture and not failures:
        value["gate_check_summary"] = [
            dict(
                check=r["name"],
                upstream=TASK,
                path=r["log_path"],
                hash=r["log_sha256"],
                field="exit_code",
                op="==",
                expected=r["expected_exit"],
                observed=r["exit_code"],
            )
            for r in receipts
            if not r["passed"]
        ]
    ended = time.monotonic_ns()
    value["duration_s"] = (ended - started) / 1e9
    value["phase_spans"] = data.get("phase_spans", [])
    value["current_work_receipt"] = build_current_work_receipt(
        run_id=str(started),
        owner_pid=os.getpid(),
        events=[],
        inference_substrate=value["inference_substrate"],
        inference_substrate_details=dict(operation="CPU energy head only", generators_frozen=True),
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=started,
        ended_monotonic_ns=ended,
        phase_spans=value["phase_spans"],
        small_ebm_training=dict(
            performed=complete,
            attempted_gradients=sum(r["gradients"] for r in value["update_budget_rows"]),
        ),
    )
    value["code_config_hashes"] = [reference(ROOT / p) for p in [*OWNED, TEST]]
    value["code_config_hashes"] += [
        reference(Path(p))
        for p in (
            m.old.__file__,
            prior.__file__,
            historical.__file__,
            ROOT / "python/carnot/reporting/current_work_receipt.py",
            ROOT / "python/carnot/reporting/primary_publication.py",
        )
    ]
    value["raw_shard_hashes"] = [reference(p) for p in sorted(raw.rglob("*")) if p.is_file()]
    value["reproducibility_checksum"] = canonical_hash(
        dict(config=m.CONFIG, code=value["code_config_hashes"], raw=value["raw_shard_hashes"])
    )
    value["field_principles"] = {}
    value["field_principles"] = {
        k: "Trace this field to actual owned evidence; finite exposed labels and seeds cannot establish independent lifelong benefit."
        for k in value
    }
    return value


def replay(path: Path) -> Json:
    """Cold arithmetic and byte checks reject mutation and unsafe readiness."""
    value = json.loads(path.read_text())
    for ref in (
        value["raw_shard_hashes"] + value["code_config_hashes"] + value["source_artifact_hashes"]
    ):
        checked(ref)
    raw = Path(value["trajectory_directory"])
    plan = json.loads((raw.parent / "plan.json").read_text())
    validation = json.loads((raw.parent / "validation.json").read_text())
    receipts, coverage = validation["receipts"], validation["coverage"]
    passed = (
        bool(receipts)
        and [r["name"] for r in receipts] == plan["validation_manifest"]
        and all(r["passed"] for r in receipts)
        and all(p in coverage and coverage[p]["summary"]["missing_lines"] == 0 for p in OWNED)
    )
    if (
        value["validation_receipts"] != receipts
        or value["required_checks_passed"] != passed
        or value["verifier_is_oracle"] != plan["fixture"]
    ):
        raise ValueError("validation_readiness_drift")
    expected_ready = int(
        passed and not plan["fixture"] and not plan["failures"] and bool(value["final_head_seals"])
    )
    if value["learning_trajectory_ready_score"] != expected_ready:
        raise ValueError("validation_readiness_drift")
    if value["final_head_seals"]:
        data = plan["data"]
        targets = data.get("labels")
        if targets is None:
            targets = {
                r["family_id"]: r["eligible_y"]
                for r in json.loads(checked(data["target_reference"]).read_text())["rows"]
            }
        for field, observed in m.reduce(raw).items():
            if value[field] != observed:
                raise ValueError("reduction_drift:" + field)
        if any(r["y"] != targets[r["family_id"]] for r in value["feedback_release_rows"]):
            raise ValueError("released_target_drift")
        seal = json.loads((raw.parent / "retention_predictions.json").read_text())
        if seal["final_head_seals"] != value["final_head_seals"] or seal["retention_labels_opened"]:
            raise ValueError("retention_seal_drift")
        targets = data.get("retention_labels")
        if targets is None:
            targets = {
                r["family_id"]: r["eligible_y"]
                for r in json.loads(checked(data["retention_target"]).read_text())["rows"]
            }
        retained = json.loads((raw.parent / "retention_rows.json").read_text())["rows"]
        if value["retention_rows"] != retained:
            raise ValueError("retention_target_drift")
        heads = {(r["seed"], r["arm"]): r for r in value["final_head_seals"]}
        sources = {r["family_id"]: r for r in data["retention"]}
        for r, p in zip(value["retention_rows"], seal["rows"], strict=True):
            source = sources[r["family_id"]]
            head = dict(
                data["head"], parameters=heads[r["seed"], r["arm"]]["parameters"], decay_scale=1.0
            )
            probability = (
                m.old.probability(head, m.old.design(head, source))
                if source["public_eligible"]
                else None
            )
            if (
                r["y"] != targets[r["family_id"]]
                or r["probability"] != probability
                or r["denominator"] != int(r["y"] is not None and probability is not None)
            ):
                raise ValueError("retention_target_drift")
            if any(r[k] != v for k, v in p.items()) or (
                r["denominator"]
                and (
                    r["numerator"] != m.loss(r["action"], r["y"])
                    or r["brier"] != (r["probability"] - r["y"]) ** 2
                )
            ):
                raise ValueError("retention_reduction_drift")
    if value["learning_trajectory_ready_score"] and (
        value["verdict_class"] != "null"
        or not value["required_checks_passed"]
        or not value["final_head_seals"]
    ):
        raise ValueError("unsafe_readiness")
    return dict(passed=True, sha256=sha256_file(path))


def terminal(path: Path) -> Json:
    """Independent subprocesses finish normally before primary bytes are visible."""
    py = str(ROOT / ".venv/bin/python")
    raw = Path(json.loads(path.read_text())["terminal_validation_sidecar_path"]).parent
    commands = [
        ("cold_replay", [py, "-u", str(ROOT / CLI), "--cold-replay", str(path)]),
        ("adversarial", [py, str(ROOT / "scripts/adversarial_verify.py"), "--json", str(path)]),
        (
            "strict_rows",
            [py, str(ROOT / "scripts/verdict_row_consistency_lint.py"), "--strict", str(path)],
        ),
    ]
    with tempfile.TemporaryDirectory(prefix="carnot-8064-terminal-") as temp:
        receipts = [
            run_check(
                ROOT,
                dict(name=n, argv=a, deadline_s=120, expected_exit=0),
                Path(temp),
                raw / "terminal_logs" / path.name,
            )
            for n, a in commands
        ]
    return dict(passed=all(r["passed"] for r in receipts), receipts=receipts)


def main(argv: list[str] | None = None) -> int:
    """Freeze checks, finish measurements, and publish one terminal disposition."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    m.progress("start_preconditions")
    started = time.monotonic_ns()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261003"], default="20261003")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--fixture-output", type=Path)
    parser.add_argument("--fixture-input", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.cold_replay:
            print(json.dumps(replay(args.cold_replay)), flush=True)
            return 0
        output = (args.fixture_output or args.output).absolute()
        if output.exists():
            raise ValueError("existing_primary_preserved")
        raw = args.resume or output.parent / "raw" / output.stem / str(time.time_ns())
        with tempfile.TemporaryDirectory(prefix="carnot-8064-") as temp:
            private = Path(temp)
            specs = manifest(private)
            atomic_json(
                raw
                / (
                    "validation_commands_resume_" + str(time.time_ns()) + ".json"
                    if args.resume
                    else "validation_commands.json"
                ),
                dict(
                    commands=specs,
                    code={p: sha256_file(ROOT / p) for p in [*OWNED, TEST]},
                    terminal_deadline_s=120,
                    terminal_commands=[
                        dict(
                            name="cold_replay",
                            argv=[
                                str(ROOT / ".venv/bin/python"),
                                "-u",
                                str(ROOT / CLI),
                                "--cold-replay",
                                str(raw.parent / "terminal_candidate.json"),
                            ],
                        ),
                        dict(
                            name="adversarial",
                            argv=[
                                str(ROOT / ".venv/bin/python"),
                                str(ROOT / "scripts/adversarial_verify.py"),
                                "--json",
                                str(raw.parent / "terminal_candidate.json"),
                            ],
                        ),
                        dict(
                            name="strict_rows",
                            argv=[
                                str(ROOT / ".venv/bin/python"),
                                str(ROOT / "scripts/verdict_row_consistency_lint.py"),
                                "--strict",
                                str(raw.parent / "terminal_candidate.json"),
                            ],
                        ),
                    ],
                ),
            )
            m.progress("inputs_before")
            data, failures = (
                (json.loads(args.fixture_input.read_text()), [])
                if args.fixture_input
                else load_inputs(args.root, raw)
            )
            if args.resume:
                original = json.loads((raw / "plan.json").read_text())
                if original["fixture"] != bool(args.fixture_input):
                    raise ValueError("resume_scope_drift")
                data, failures = original["data"], original["failures"]
            frozen = time.monotonic_ns()
            preconditions = (
                []
                if args.fixture_output
                else [
                    run_check(ROOT, spec, private, raw / "validation_logs")
                    for spec in specs
                    if spec["name"] == "python_environment"
                ]
            )
            for receipt in preconditions:
                if not receipt["passed"]:
                    failures.append(
                        dict(
                            check="python_environment",
                            upstream=TASK,
                            path=receipt["log_path"],
                            hash=receipt["log_sha256"],
                            field="exit_code",
                            op="==",
                            expected=0,
                            observed=receipt["exit_code"],
                        )
                    )
            if not args.resume:
                atomic_json(
                    raw / "plan.json",
                    dict(
                        data=data,
                        failures=failures,
                        fixture=bool(args.fixture_input),
                        validation_manifest=[r["name"] for r in specs],
                    ),
                )
            m.progress("inputs_after", len(data.get("sources", [])), 0)
            if not failures:
                try:
                    result = m.measure(data, raw / "trajectory")
                    m.progress(
                        "retention_after_head_seals_before",
                        len(result["final_head_seals"]),
                        len(data["retention"]),
                    )
                    retention(data, result, raw)
                    m.progress("retention_after_head_seals_after", len(data["retention"]), 0)
                except (ValueError, TimeoutError, OSError) as error:
                    failures.append(
                        dict(
                            check="numerical_work",
                            upstream=TASK,
                            path=str(raw / "trajectory"),
                            hash=None,
                            field="complete_valid_trajectories",
                            op="==",
                            expected=True,
                            observed=str(error),
                            classification="owned",
                        )
                    )
                    atomic_json(raw / "numerical_failure.json", dict(failures=failures))
            measured = time.monotonic_ns()
            os.environ["CARNOT_8064_COVERAGE_CONFIG"] = str(private / "coverage.ini")
            m.progress("validation_before", 0, len(specs))
            receipts = (
                []
                if args.fixture_output
                else preconditions
                + [
                    run_check(ROOT, spec, private, raw / "validation_logs")
                    for spec in specs
                    if spec["name"] != "python_environment"
                ]
            )
            coverage = (
                json.loads((private / "coverage.json").read_text())["files"]
                if (private / "coverage.json").is_file()
                else {}
            )
            atomic_json(raw / "validation.json", dict(receipts=receipts, coverage=coverage))
            data["phase_spans"] = [
                dict(phase="freeze", duration_s=(frozen - started) / 1e9),
                dict(phase="numerical", duration_s=(measured - frozen) / 1e9),
                dict(phase="validation", duration_s=(time.monotonic_ns() - measured) / 1e9),
            ]
            m.progress("validation_after", len(receipts), 0)
            value = build(
                data, failures, raw, receipts, coverage, bool(args.fixture_input), started
            )
            m.progress("publication_before")
            publication = publish_primary(output, value, terminal)
            atomic_json(
                Path(value["terminal_validation_sidecar_path"]),
                dict(publication=publication, owned_invocation_exit=0),
            )
            m.progress("complete", len(value["rows"]), 0)
        return 0
    except (OSError, KeyError, ValueError, TimeoutError) as error:
        m.progress("rejected_" + str(error))
        return 1
