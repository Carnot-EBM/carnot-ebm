"""REQ-REPORT-8023: measure fixed-answer evidence effects and freeze small heads.

Evaluation and retention targets remain sealed. Fitting readiness is a numerical
contract, not evidence of correctness, deployment or generalized learning benefit.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot import experiment_8022_v695_likelihood_protocol as protocol
from carnot.inference import likelihood_capture_8023 as runtime
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.evidence_features_custody_7980 import checked, reference
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.verify import likelihood_calibration_8023 as c

Json = dict[str, Any]
ROOT = protocol.ROOT
NAME = "experiment_8023_v695_likelihood_calibration"
TASK = "exp8023-likelihood-calibration"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    f"python/carnot/{NAME}.py",
    "python/carnot/verify/likelihood_calibration_8023.py",
    "python/carnot/inference/likelihood_capture_8023.py",
    CLI,
]
TEST = "tests/python/test_likelihood_calibration_8023.py"
METHODS = dict(
    c.CONFIG,
    source_roles=dict(fit=64, tune=32),
    views=list(protocol.s.ARMS),
    forward_budget=384,
    call_seconds=120,
    model_work_seconds=2400,
    authenticity_floor_seconds=2,
    support_floors=dict(fit=[48, 8], tune=[24, 4]),
    duplicate_tolerance=1e-6,
)


def byte_hash(value: bytes) -> str:
    """Original annotation hashes bind actual answer bytes, not JSON encodings."""
    return "sha256:" + hashlib.sha256(value).hexdigest()


def authenticate(root: Path) -> Json:
    """Public planning reads references but never opens private outcome rows."""
    plan: Json = dict(
        checks=[],
        references=[],
        panel=dict(rows=[], slots=[], exclusions=[]),
        target_references={},
        gguf_sha256=None,
        protocol_fingerprint=None,
        methods=METHODS,
    )
    for identity, name, gates in [
        (8022, protocol.NAME, dict(likelihood_protocol_ready_score=1, token_scoring_ready_score=1)),
        (8019, "experiment_8019_v695_eligible_targets", dict(fit_targets_ready_score=1)),
    ]:
        path = root / "results" / (name + ".json")
        value = json.loads(path.read_text()) if path.is_file() else {}
        for field, expected in dict(
            experiment_id=identity, flagged_adversarial=False, **gates
        ).items():
            plan["checks"].append(
                dict(
                    upstream_id=f"exp{identity}",
                    path=str(path),
                    hash=sha256_file(path) if path.is_file() else None,
                    artifact_field=field,
                    expected=expected,
                    observed=value.get(field, "missing_field_contract_error"),
                    passed=value.get(field) == expected,
                )
            )
        if not value:
            continue
        plan["references"].append(
            dict(reference(path), scope="historical", imported_fields=list(gates))
        )
        if identity == 8022:
            try:
                protocol.replay(value)
                if value["method_map"] != protocol.s.METHODS:
                    raise ValueError("scorer_protocol_changed")
                panel = value["public_panel_manifest"]
                rows = [r for r in panel["rows"] if r["role"] in ("fit", "tune")]
                if {
                    role: sum(r["role"] == role for r in rows) for role in ("fit", "tune")
                } != METHODS["source_roles"]:
                    raise ValueError("fit_tune_roster")
                if len({r["source_normalized_hash"] for r in rows}) != 96:
                    raise ValueError("source_role_overlap")
                plan.update(
                    panel=dict(
                        rows=rows,
                        slots=[s for s in panel["slots"] if s["role"] in ("fit", "tune")],
                        exclusions=[s for s in panel["exclusions"] if s["role"] in ("fit", "tune")],
                    ),
                    gguf_sha256=value["gguf_sha256"],
                    protocol_fingerprint=value["protocol_fingerprint"],
                )
            except (OSError, ValueError, KeyError) as error:
                plan["checks"].append(
                    dict(
                        upstream_id="exp8022",
                        path=str(path),
                        hash=sha256_file(path),
                        artifact_field="durable_protocol_replay",
                        expected="unchanged complete scorer",
                        observed=str(error),
                        passed=False,
                    )
                )
        else:
            for role in ("fit", "tune"):
                for scope, key in [
                    ("public", "public_manifests"),
                    ("evaluator", "evaluator_manifests"),
                ]:
                    ref = value.get(key, {}).get(role, {})
                    try:
                        checked(ref)
                        plan["target_references"].setdefault(role, {})[scope] = ref
                        plan["references"].append(
                            dict(ref, scope="historical", imported_fields=[role, scope])
                        )
                    except (OSError, ValueError, KeyError) as error:
                        plan["checks"].append(
                            dict(
                                upstream_id="exp8019",
                                path=str(path),
                                hash=sha256_file(path),
                                artifact_field=f"{key}.{role}",
                                expected="bound original bytes",
                                observed=str(error),
                                passed=False,
                            )
                        )
    return plan


def join(plan: Json, reduced: Json) -> list[Json]:
    """Only sealed fit/tune features can join complete original human annotations."""
    features = {r["family_id"]: r for r in reduced["source_feature_rows"]}
    rows = []
    for role, refs in plan["target_references"].items():
        public = {
            r["family_id"]: r for r in json.loads(checked(refs["public"]).read_text())["rows"]
        }
        labels = {
            r["family_id"]: r for r in json.loads(checked(refs["evaluator"]).read_text())["rows"]
        }
        for item in (r for r in plan["panel"]["rows"] if r["role"] == role):
            fid = item["family_id"]
            p, label = public[fid], labels[fid]
            if any(p[k] != item[k] for k in ("source_bytes", "answer_bytes")) or label[
                "response_sha256"
            ] != byte_hash(bytes.fromhex(item["answer_bytes"])):
                raise ValueError("original_answer_or_source_changed")
            y = (
                label["eligible_y"]
                if label["completely_annotated"] and label["custody_passed"]
                else None
            )
            rows.append(
                dict(
                    features[fid],
                    q=p["q"],
                    y=y,
                    target_reason=None if y is not None else "incomplete_original_annotation",
                )
            )
    return rows


def commands(scratch: Path) -> list[CommandSpec]:
    """Shared supervision keeps required checks and broad health evidence distinct."""
    with patch.object(protocol, "OWNED", OWNED), patch.object(protocol, "TEST", TEST):
        specs = protocol.commands(scratch)
    specs.insert(
        -1,
        CommandSpec(
            "Exp8022_fixtures",
            (
                str(ROOT / ".venv/bin/pytest"),
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                "tests/python/test_likelihood_protocol_8022.py",
            ),
            "owned",
            180,
        ),
    )
    return specs


def build(
    plan: Json,
    measured: Json,
    fitted: Json,
    raw: Path,
    receipts: list[Json],
    coverage: Json,
    elapsed: float,
) -> Json:
    """Readiness binds complete measurement and numerical states, never benefit."""
    rows = measured.get("qualification", {}).get("rows", measured.get("rows", []))
    reduced = c.reduce(plan["panel"], rows)
    failures = [r for r in plan["checks"] + measured.get("checks", []) if not r["passed"]]
    owned = [r for r in receipts if r["scope"] == "owned"]
    frozen = json.loads((raw / "validation_commands.json").read_text())
    expected_checks = [r["name"] for r in frozen["commands"] if r["scope"] == "owned"]
    counters = measured.get("model_invocation_counts", ZERO_INVOCATION_COUNTS)
    gates = dict(
        measurement=reduced["passed"] and len(plan["panel"]["rows"]) == 96,
        support=bool(fitted.get("role_support"))
        and all(r["passed"] for r in fitted["role_support"].values()),
        states=fitted.get("ready", False),
        owned_checks=bool(owned)
        and [r["name"] for r in owned] == expected_checks
        and all(r["passed"] for r in owned),
        changed_statement_coverage=coverage.get("percent_covered") == 100,
        authenticity=measured.get("duration_s", 0) >= 2,
        current_model=counters["model_loads_completed"] >= 1
        and measured.get("gguf_sha256") == plan["gguf_sha256"]
        and measured.get("offload_evidence", {}).get("supported", False)
        and measured.get("cleanup", {}).get("model_closed", False)
        and measured.get("cleanup", {}).get("lease_released", False),
        scientific_benefit=False,
    )
    ready = int(not failures and all(v for k, v in gates.items() if k != "scientific_benefit"))
    if not gates["measurement"]:
        failures.append(
            dict(
                upstream_id="exp8023_current_capture",
                path=str(raw / "capture.json"),
                hash=sha256_file(raw / "capture.json"),
                artifact_field="complete_qualified_fit_tune_views",
                expected=dict(completed=384, duplicate_drift_max=1e-6),
                observed=dict(
                    completed=sum(r["status"] == "completed" for r in rows),
                    duplicate_drift_max=max(
                        (r["drift"] for r in reduced["duplicate_drift_rows"]), default=None
                    ),
                ),
                passed=False,
            )
        )
    for role, support_value in fitted.get("role_support", {}).items():
        if not support_value["passed"]:
            failures.append(
                dict(
                    upstream_id="exp8019_original_targets",
                    path=plan["target_references"][role]["evaluator"]["path"],
                    hash=plan["target_references"][role]["evaluator"]["sha256"],
                    artifact_field=f"role_support.{role}",
                    expected=METHODS["support_floors"][role],
                    observed=support_value,
                    passed=False,
                )
            )
    duplicate_failed = any(not r["passed"] for r in reduced["duplicate_drift_rows"])
    verdict = (
        "null"
        if ready
        else "disqualified"
        if duplicate_failed
        else "blocked"
        if failures
        else "disqualified"
    )
    loads = counters["model_loads_attempted"]
    counts = dict(
        intended=384,
        eligible=4 * len(plan["panel"]["rows"]),
        started=sum(r["status"] in {"completed", "failed"} for r in rows),
        completed=sum(r["status"] == "completed" for r in rows),
        excluded=len(plan["panel"]["exclusions"]),
        failed=sum(r["status"] == "failed" for r in rows),
        censored=sum(r["status"] == "censored" for r in rows) + max(0, 384 - len(rows)),
        independent=len(
            {r["family_id"] for r in reduced["source_feature_rows"] if r["status"] == "completed"}
        ),
        unit="teacher_forced_view; independent unit is normalized source group",
        seeds_are_independent=False,
    )
    value = dict(
        experiment_id=8023,
        task_id=TASK,
        milestone="2026.10.695",
        run_date="20261002",
        schema="carnot.v695.likelihood_calibration.v1",
        honest_verdict=f"complete_{verdict}_likelihood_calibration",
        verdict_class=verdict,
        claim_scope="Current teacher-forced fit/tune intervention capture and numerical head readiness on exposed development; no correctness, deployment or learning benefit claim.",
        gate_check_summary=failures,
        preconditions_checked=plan["checks"] + measured.get("checks", []),
        rows=rows,
        sample_size_budget=counts,
        random_seed=69523,
        reproducibility_checksum=canonical_hash(dict(plan=plan, rows=rows, heads=fitted)),
        verifier_is_oracle=False,
        acceptance_gate_results=gates,
        genuine_headroom=None,
        positive_control_results=dict(
            scope="private_CPU_arithmetic_and_identity_fixtures", natural_benefit_credit=False
        ),
        generalized_learning_benefit_score=0,
        likelihood_calibration_ready_score=ready,
        token_rows=[dict(id=r["id"], tokens=r["token_rows"]) for r in rows],
        source_feature_rows=reduced["source_feature_rows"],
        role_support=fitted.get("role_support", {}),
        duplicate_drift_rows=reduced["duplicate_drift_rows"],
        fitted_heads=fitted.get("heads", []),
        calibration_config=METHODS,
        selected_comparator=fitted.get("selected_comparator"),
        eval_label_access_count=0,
        cited_upstream_artifacts=plan["references"],
        MODEL_SPECS=[protocol.runtime.MODEL],
        model_specs=[protocol.runtime.MODEL],
        trained_head_specs=[
            dict(arm=h["arm"], parameter_count=h["parameter_count"], checkpoint=r)
            for h, r in zip(fitted.get("heads", []), fitted.get("checkpoints", []), strict=True)
        ],
        inference_substrate="live_llm_inference",
        inference_substrate_class="model_load_no_generation" if loads else "blocked_no_run",
        planned_inference_substrate_class="model_load_no_generation",
        model_invocation_counts={
            **{k: v for k, v in counters.items() if not k.startswith("generation_calls_")},
            "generation": {
                k.removeprefix("generation_calls_"): v
                for k, v in counters.items()
                if k.startswith("generation_calls_")
            },
        },
        model_identity_receipt=measured.get("model_identity_receipt", {}),
        gguf_sha256=measured.get("gguf_sha256"),
        gpu_lease_receipt=measured.get("gpu_lease_receipt", {}),
        offload_evidence=measured.get("offload_evidence", {}),
        forward_pass_counts=reduced["forward_pass_counts"],
        scored_tokens=reduced["scored_tokens"],
        generated_tokens=0,
        current_invocation_ledger=measured.get("current_invocation_ledger", [])
        + [
            dict(r, operation="teacher_forced_scoring", scope="current")
            for r in rows
            if r["status"] in {"completed", "failed"}
        ],
        cleanup=measured.get("cleanup", {}),
        duration_s=elapsed,
        phase_spans=measured.get("phase_spans", []),
        measured_duration_s=measured.get("duration_s", 0),
        raw_shard_hashes=[
            reference(raw / p)
            for p in [
                "plan.json",
                "capture.json",
                "fitted.json",
                "validation_receipts.json",
                "validation_commands.json",
            ]
        ],
        checkpoint_references=[reference(p) for p in sorted((raw / "forwards").glob("*.json"))]
        + fitted.get("checkpoints", [])
        + fitted.get("trials", []),
        code_config_hashes=json.loads((raw / "validation_commands.json").read_text())[
            "code_config_hashes"
        ],
        validation_receipts=owned,
        repository_health=[r for r in receipts if r["scope"] == "repository_health"],
        coverage_statement_counts=coverage,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        flagged_adversarial=False,
        methodology_note="Readiness and sigmoid equality are contract checks. No scientific benefit is tested; duplicate views and seeds add no independent groups.",
    )
    value["raw_shard_hashes"] += (
        [reference(raw / "joined.json")] if (raw / "joined.json").is_file() else []
    )
    value["field_principles"] = {
        k: "Bind current work to primitive evidence and distinguish exposed readiness from independent science."
        for k in value
    }
    value["field_principles"].update(
        likelihood_calibration_ready_score="Integer readiness requires complete current scoring, original support, reloadable converged heads and all owned checks.",
        eval_label_access_count="Evaluation and retention targets stay sealed; interventions will be evaluated later on different source groups.",
        generalized_learning_benefit_score="Fitting and exposed-development evidence earn zero generalized learning credit.",
    )
    return value


def replay(value: Json) -> None:
    """A fresh reader verifies checkpoints and recomputes all measurement fields."""
    for ref in (
        value["raw_shard_hashes"]
        + value["checkpoint_references"]
        + value["code_config_hashes"]
        + value["cited_upstream_artifacts"]
    ):
        checked(ref)
    protocol.verify_references([value["validation_receipts"], value["repository_health"]])
    raw = Path(value["terminal_validation_sidecar_path"]).parent
    plan, measured, fitted, receipts = [
        json.loads((raw / p).read_text())
        for p in ["plan.json", "capture.json", "fitted.json", "validation_receipts.json"]
    ]
    durable_rows = [json.loads(p.read_text()) for p in sorted((raw / "forwards").glob("*.json"))]
    if durable_rows != measured["qualification"]["rows"]:
        raise ValueError("per_view_checkpoint_drift")
    c.reduce(plan["panel"], durable_rows)
    if (raw / "joined.json").is_file():
        joined = json.loads((raw / "joined.json").read_text())
        if joined != join(plan, c.reduce(plan["panel"], measured["qualification"]["rows"])):
            raise ValueError("target_join_drift")
        for head, ref in zip(fitted["heads"], fitted["checkpoints"], strict=True):
            reloaded = json.loads(checked(ref).read_text())
            if reloaded != head:
                raise ValueError("head_reload")
            c.predict(reloaded, joined)
        c.check_fit(joined, fitted)
    expected = build(
        plan,
        measured,
        fitted,
        raw,
        receipts["receipts"],
        value["coverage_statement_counts"],
        value["duration_s"],
    )
    if value != expected:
        raise ValueError("cold_reduction_drift")


def terminal(path: Path) -> Json:
    """Unchanged terminal consumers inspect the exact candidate and primary."""
    replay(json.loads(path.read_text()))
    return terminal_readers(path)


def terminal_readers(path: Path) -> Json:
    """Reuse command supervision so terminal assertions retain argv and log bytes."""
    specs = [
        CommandSpec(
            name, (str(ROOT / ".venv/bin/python"), "-u", script, flag, str(path)), "terminal", 60
        )
        for name, script, flag in [
            ("adversarial", "scripts/adversarial_verify.py", "--json"),
            ("strict_rows", "scripts/verdict_row_consistency_lint.py", "--strict"),
        ]
    ]
    receipts = run_commands(
        ROOT,
        specs,
        log_dir=Path(json.loads(path.read_text())["terminal_validation_sidecar_path"]).parent
        / "terminal_logs"
        / path.name,
    )
    return dict(passed=all(r["passed"] for r in receipts), receipts=receipts)


def main(argv: list[str] | None = None) -> int:
    """Freeze work once, run a bounded child, then publish validated final bytes."""
    started = time.monotonic()
    protocol.s.progress("8023_start", started)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261002"], default="20261002")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    routes = parser.add_mutually_exclusive_group()
    routes.add_argument("--cold-replay", type=Path)
    routes.add_argument("--runtime-child", type=Path)
    routes.add_argument("--reduce-existing", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.cold_replay:
            replay(json.loads(args.cold_replay.read_text()))
            return 0
        if args.runtime_child:
            runtime.worker(args.runtime_child, args.output)
            return 0
        output = args.output.absolute()
        raw = output.parent / "raw" / output.stem
        raw.mkdir(parents=True, exist_ok=True)
        if (raw / "capture.json").exists() and not args.reduce_existing:
            raise ValueError("existing_capture_preserved_use_cold_replay")
        plan = authenticate(args.root)
        measured: Json = {}
        retained: list[Json] = []
        if args.reduce_existing:
            frozen = json.loads((raw / "validation_commands.json").read_text())
            for ref in frozen["code_config_hashes"]:
                if Path(ref["path"]).name in {
                    "likelihood_calibration_8023.py",
                    "likelihood_capture_8023.py",
                    "fixed_answer_likelihood_8022.py",
                    "likelihood_runtime_8022.py",
                }:
                    checked(ref)
            if json.loads((raw / "plan.json").read_text()) != plan:
                raise ValueError("scoring_inputs_changed")
            measured = json.loads((raw / "capture.json").read_text())
            old_receipts = json.loads((raw / "validation_receipts.json").read_text())["receipts"]
            retained = [r for r in old_receipts if r["scope"] == "repository_health"]
            archive = raw / "prior_reporting" / canonical_hash(frozen).split(":")[1]
            shutil.copytree(raw, archive, ignore=shutil.ignore_patterns("prior_reporting"))
            started -= max(
                (r["end_s"] for r in measured.get("phase_spans", [])),
                default=measured.get("duration_s", 0),
            )
            started -= sum(r.get("duration_s", 0) for r in old_receipts)
        atomic_json(raw / "plan.json", plan)
        with TemporaryDirectory(prefix="carnot-8023-") as directory:
            scratch = Path(directory)
            specs = commands(scratch)
            if args.reduce_existing:
                specs = [r for r in specs if r.scope != "repository_health"]
            atomic_json(
                raw / "validation_commands.json",
                dict(
                    methods=METHODS,
                    commands=[asdict(x) for x in specs],
                    code_config_hashes=[
                        reference(ROOT / p)
                        for p in OWNED
                        + [
                            TEST,
                            "python/carnot/inference/fixed_answer_likelihood_8022.py",
                            "python/carnot/inference/likelihood_runtime_8022.py",
                        ]
                    ],
                ),
            )
            phase_start = time.monotonic()
            protocol.s.progress("8023_before_model_work_subprocess", started, 0, 384)
            if not args.reduce_existing and all(r["passed"] for r in plan["checks"]):
                child = run_commands(
                    ROOT,
                    [
                        CommandSpec(
                            "current_fit_tune_capture",
                            (
                                str(ROOT / ".venv/bin/python"),
                                "-u",
                                CLI,
                                "--runtime-child",
                                str(raw / "plan.json"),
                                "--output",
                                str(raw / "capture.json"),
                            ),
                            "runtime",
                            2460,
                        )
                    ],
                    log_dir=raw / "runtime_logs",
                    heartbeat_s=60,
                    extra_env=dict(
                        PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu", CARNOT_FORCE_LIVE="1"
                    ),
                )[0]
                measured = (
                    json.loads((raw / "capture.json").read_text())
                    if (raw / "capture.json").is_file()
                    else {}
                )
                measured["child_receipt"] = child
                if not child["passed"]:
                    measured.setdefault("checks", []).append(
                        dict(
                            upstream_id="exp8023_runtime",
                            path=child["log_path"],
                            hash=child["log_sha256"],
                            artifact_field="exit_code",
                            expected=0,
                            observed=child["exit_code"],
                            passed=False,
                        )
                    )
            rows = measured.get("qualification", {}).get("rows")
            if rows is None:
                rows = c.capture(plan["panel"], None, raw / "forwards", deadline=0)
                measured["qualification"] = dict(rows=rows)
            if not args.reduce_existing:
                measured["phase_spans"] = [
                    dict(
                        phase="current_teacher_forced_model_work",
                        start_s=phase_start - started,
                        end_s=time.monotonic() - started,
                    )
                ]
            atomic_json(raw / "capture.json", measured)
            reduced = c.reduce(plan["panel"], rows)
            protocol.s.progress(
                "8023_after_scoring_seal_before_target_join",
                started,
                sum(r["status"] == "completed" for r in rows),
            )
            fitted: Json = dict(heads=[], checkpoints=[], trials=[], role_support={}, ready=False)
            if (
                len(reduced["source_feature_rows"]) == len(plan["panel"]["rows"])
                and plan["panel"]["rows"]
                and not any(not x["passed"] for x in plan["checks"] + measured.get("checks", []))
            ):
                joined = join(plan, reduced)
                atomic_json(raw / "joined.json", joined)
                fit_start = time.monotonic()
                fitted = c.train(joined, raw)
                measured["phase_spans"].append(
                    dict(
                        phase="CPU_small_head_training",
                        start_s=fit_start - started,
                        end_s=time.monotonic() - started,
                    )
                )
                atomic_json(raw / "capture.json", measured)
            atomic_json(raw / "fitted.json", fitted)
            protocol.s.progress("8023_before_owned_checks", started)
            receipts = (
                []
                if args.root != ROOT
                else run_commands(
                    ROOT,
                    specs,
                    log_dir=raw
                    / ("validation_logs_reduction" if args.reduce_existing else "validation_logs"),
                    heartbeat_s=60,
                    extra_env=dict(
                        PYTHONUNBUFFERED="1",
                        JAX_PLATFORMS="cpu",
                        COVERAGE_FILE=str(scratch / ".coverage"),
                        CARNOT_8023_COVERAGE_CONFIG=str(scratch / "coverage.ini"),
                    ),
                )
            )
            receipts += retained
            atomic_json(raw / "validation_receipts.json", dict(receipts=receipts))
            coverage = (
                json.loads((scratch / "coverage.json").read_text())["totals"]
                if (scratch / "coverage.json").exists()
                else {}
            )
            value = build(
                plan, measured, fitted, raw, receipts, coverage, time.monotonic() - started
            )
            candidate = raw / (NAME + ".json")
            atomic_json(candidate, value)
            cold = run_commands(
                ROOT,
                [
                    CommandSpec(
                        "cold_reduction",
                        (
                            str(ROOT / ".venv/bin/python"),
                            "-u",
                            CLI,
                            "--cold-replay",
                            str(candidate),
                        ),
                        "terminal",
                        120,
                    )
                ],
                log_dir=raw / "cold_logs",
                heartbeat_s=60,
            )
            if not all(r["passed"] for r in cold):
                raise ValueError("cold_reduction_failed")
            publication = publish_primary(output, value, terminal)
            atomic_json(raw / "terminal_validation.json", dict(publication, cold_receipts=cold))
            report = terminal(output)
            atomic_json(raw / "published_validation.json", report)
            readers = reader_receipt(
                TASK,
                output.parent,
                field="likelihood_calibration_ready_score",
                expected=value["likelihood_calibration_ready_score"],
            )
            atomic_json(raw / "primary_readers.json", readers)
            if not report["passed"] or not readers["passed"]:
                raise ValueError("published_readers_failed")
        protocol.s.progress("8023_complete", started, len(value["rows"]))
        return 0
    except (OSError, RuntimeError, TimeoutError, ValueError, KeyError) as error:
        print(f"[exp8023] rejected={type(error).__name__}:{error}", flush=True)
        return 1
