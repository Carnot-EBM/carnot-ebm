"""REQ-REPORT-7994: seal a bounded development cohort with delayed feedback.

This prepares evidence for later learning. It loads no pretrained model and
makes no claim about benefit, global non-exposure or prospective traffic.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.verify import development_cohort_7994 as d

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[2]
NAME = "experiment_7994_v693_development_cohort"
TASK = "exp7994-development-cohort"
MODEL_SPECS: list[str] = []
OWNED = [
    f"python/carnot/{NAME}.py",
    "python/carnot/verify/development_cohort_7994.py",
    f"scripts/experiments/{NAME}.py",
]
TESTS = ["tests/python/test_development_cohort_7994.py", f"tests/python/test_{NAME}.py"]
CONSUMERS = [
    "tests/python/test_evidence_features_7980.py",
    "tests/python/test_response_targets_7955.py",
    "tests/python/test_primary_publication_7928.py",
]
E2ES = [
    "tests/python/test_source_boundary_7852.py",
    "tests/python/test_experiment_7942_v689_sentence_labels.py",
]


def base(failures: list[Json]) -> Json:
    """All terminal branches expose the same schema and honest absent measurements."""
    return dict(
        experiment_id=7994,
        task_id=TASK,
        milestone="2026.10.693",
        run_date="20261001",
        execution_date="20261001",
        honest_verdict="complete_blocked_development_inputs"
        if failures
        else "complete_null_development_cohort",
        verdict_class="blocked" if failures else "null",
        gate_check_summary=failures,
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_specs=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        trained_head_specs=[],
        duration_s=0.0,
        phase_spans=[],
        random_seed=69394,
        reproducibility_checksum=None,
        cited_upstream_artifacts=[],
        code_config_hashes=[],
        raw_shard_hashes=[],
        rows=[],
        sample_size_budget=dict(
            unit="normalized_source_group",
            intended=384,
            eligible=0,
            started=0,
            completed=0,
            excluded=0,
            failed=0,
            censored=0,
            independent=0,
        ),
        verifier_is_oracle=False,
        claim_scope="Recorded source-disjoint development preparation; no benefit or global non-exposure claim.",
        acceptance_gate_results=dict(validity=False, readiness=False, decision_benefit=False),
        positive_control_results={},
        preconditions_checked=[],
        validation_command_manifest_path=None,
        validation_receipts=[],
        coverage_statement_counts={},
        flagged_adversarial=False,
        terminal_validation_sidecar_path=None,
        cohort_ready_score=0,
        public_role_manifests={},
        evaluator_role_manifests={},
        role_hashes={},
        exposure_scope="bounded_recorded_source_disjoint_development",
        known_exposure_rows=[],
        unknown_history="Unrecorded access and public model pretraining remain unknown.",
        selection_shortfalls={},
        label_access_events=[],
        feedback_schedule=dict(
            stream_slots=256,
            label_delay_slots=20,
            replacement_allowed=False,
            compress_unavailable_time=False,
            calibration_groups=64,
            retention_groups=64,
            retention_access="independent_learning_audit_seal",
            current_learning_updates=0,
        ),
        stream_slot_rows=[],
        feature_schema=list(d.f.FEATURES),
        historical_required_failures=[],
        repository_health={},
    )


def measure(plan: Json, root: Path, raw: Path, *, fixture: bool = True) -> Json:
    """Freeze predictor-visible bytes before an evaluator child receives annotation paths."""
    value = base([])
    ids, hashes, history = d.exclusions(plan, fixture=fixture)
    sources, responses = d.public_training(root)
    roster, public = d.select(sources, responses, ids, hashes)
    seal_path = d.seal(raw, roster, public)
    seal = json.loads(seal_path.read_text())
    argv = [
        sys.executable,
        "-u",
        str(ROOT / OWNED[-1]),
        "--evaluate",
        str(seal_path),
        "--root",
        str(root),
        "--evaluator-output",
        str(raw / "evaluator"),
    ]
    config = os.environ.get("CARNOT_7994_COVERAGE_CONFIG")
    if config:
        argv[1:2] = ["-m", "coverage", "run", "--rcfile=" + config]
    receipt = run_commands(
        ROOT,
        [CommandSpec("evaluator", tuple(argv), "evaluator", 300)],
        log_dir=raw / "children",
        heartbeat_s=30,
    )[0]
    if not receipt["passed"]:
        raise ValueError("evaluator_child_failed")
    summary = json.loads((raw / "evaluator/summary.json").read_text())
    rows = summary["rows"]
    available = {r["family_id"] for r in rows if r["eligibility"]}
    selected_ids = {r["source_id"] for r in roster}
    known = [
        dict(
            path=r["path"],
            sha256=r["sha256"],
            selected_source_ids=sorted(selected_ids & set(r["source_ids"])),
        )
        for r in history
        if selected_ids & set(r["source_ids"])
    ]
    value.update(
        rows=rows,
        public_seal=d.c.reference(seal_path),
        public_role_manifests=seal["roles"],
        evaluator_role_manifests=summary["evaluator_role_manifests"],
        evaluator_summary=d.c.reference(raw / "evaluator/summary.json"),
        known_exposure_rows=known,
        exclusion_manifest=dict(
            source_ids=sorted(ids),
            source_hashes=sorted(hashes),
            history_scope="Exp7980 frozen discovery inventory; explicit evaluation/eval/test/retention paths",
            original_groups=640 if not fixture else 0,
        ),
        stream_slot_rows=d.schedule(roster, available),
        positive_control_results=summary["positive_control_results"],
        genuine_headroom=summary["headroom"],
        child_receipts=[receipt],
        label_access_events=[
            dict(
                actor="separate_evaluator_process",
                public_seal_sha256=sha256_file(seal_path),
                full_labels_exported_to_predictor=False,
                retention_consumed_by_learning=False,
            )
        ],
        role_hashes={
            role: dict(
                public=seal["roles"][role]["sha256"],
                evaluator=summary["evaluator_role_manifests"][role]["sha256"],
            )
            for role in d.ROLES
        },
        selection_shortfalls={
            role: count - seal["roles"][role]["count"] for role, count in d.ROLES.items()
        },
        raw_shard_hashes=[
            d.c.reference(root / "data/ragtruth" / name)
            for name in ("source_info.jsonl", "response.jsonl")
        ],
    )
    complete = len(available)
    value["sample_size_budget"].update(
        eligible=complete,
        started=len(rows),
        completed=complete,
        excluded=len(rows) - complete,
        censored=384 - len(rows),
        independent=complete,
    )
    missing = [
        d.c.operand(
            "official_train",
            root / "data/ragtruth/source_info.jsonl",
            role + "_selected_count",
            count,
            seal["roles"][role]["count"],
        )
        for role, count in d.ROLES.items()
        if seal["roles"][role]["count"] != count
    ]
    value.update(
        cohort_ready_score=int(not missing),
        acceptance_gate_results=dict(validity=True, readiness=not missing, decision_benefit=False),
    )
    if missing:
        value.update(
            honest_verdict="complete_blocked_source_shortfall",
            verdict_class="blocked",
            gate_check_summary=missing,
        )
    atomic_json(
        raw / "primitive_rows.json", dict(rows=rows, stream_slot_rows=value["stream_slot_rows"])
    )
    value["rows_hash"] = canonical_hash(rows)
    return value


def replay(value: Json) -> None:
    """Cold replay checks sealed public features, roles, chronology and terminal rows."""
    for item in (
        value["code_config_hashes"] + value["cited_upstream_artifacts"] + value["raw_shard_hashes"]
    ):
        d.c.checked(item)
    if value.get("public_seal"):
        seal = json.loads(d.c.checked(value["public_seal"]).read_text())
        for key in ("public", "features", "roster", "schedule"):
            d.c.checked(seal[key])
        roster = json.loads(d.c.checked(seal["roster"]).read_text())["rows"]
        d.check_roles(roster)
        d.f.replay_features(d.c.checked(seal["public"]), d.c.checked(seal["features"]))
        for role in d.ROLES:
            d.c.checked(value["public_role_manifests"][role])
            d.c.checked(value["evaluator_role_manifests"][role])
        available = {r["family_id"] for r in value["rows"] if r["eligibility"]}
        if d.schedule(roster, available) != value["stream_slot_rows"]:
            raise ValueError("chronology_drift")
        if canonical_hash(value["rows"]) != value["rows_hash"]:
            raise ValueError("rows_drift")
        if json.loads(d.c.checked(value["evaluator_summary"]).read_text())["rows"] != value["rows"]:
            raise ValueError("evaluator_rows_drift")


def freeze_commands(raw: Path, scratch: Path) -> Json:
    """Freeze owned commands and one separate full-suite health command before work."""
    scratch.mkdir(parents=True, exist_ok=True)
    py = str(ROOT / ".venv/bin/python")
    config = scratch / "coverage.ini"
    config.write_text(
        "[run]\nparallel=True\ndata_file="
        + str(scratch / ".coverage")
        + "\ninclude=\n"
        + "".join("    " + str(ROOT / p) + "\n" for p in OWNED)
    )
    common = ["-n0", "-o", "addopts=", "--no-cov", "-q", "--basetemp=" + str(scratch / "pytest")]
    specs = [
        CommandSpec(
            "changed_coverage",
            (
                py,
                "-m",
                "coverage",
                "run",
                "--rcfile=" + str(config),
                "-m",
                "pytest",
                *common,
                *TESTS,
            ),
            "owned",
            600,
        ),
        CommandSpec(
            "direct_consumers", (str(ROOT / ".venv/bin/pytest"), *common, *CONSUMERS), "owned", 300
        ),
        CommandSpec("E2E_015_019", (str(ROOT / ".venv/bin/pytest"), *common, *E2ES), "owned", 300),
        CommandSpec(
            "coverage_combine",
            (py, "-m", "coverage", "combine", "--rcfile=" + str(config)),
            "owned",
            60,
        ),
        CommandSpec(
            "coverage_json",
            (
                py,
                "-m",
                "coverage",
                "json",
                "--rcfile=" + str(config),
                "-o",
                str(scratch / "coverage.json"),
            ),
            "owned",
            60,
        ),
        CommandSpec(
            "coverage_100",
            (
                py,
                "-m",
                "coverage",
                "report",
                "--rcfile=" + str(config),
                "--fail-under=100",
                "--show-missing",
            ),
            "owned",
            60,
        ),
        CommandSpec(
            "ruff_check", (str(ROOT / ".venv/bin/ruff"), "check", *OWNED, *TESTS), "owned", 60
        ),
        CommandSpec(
            "ruff_format",
            (str(ROOT / ".venv/bin/ruff"), "format", "--check", *OWNED, *TESTS),
            "owned",
            60,
        ),
        CommandSpec(
            "strict_mypy",
            (str(ROOT / ".venv/bin/mypy"), "--strict", "--follow-imports=silent", *OWNED[:-1]),
            "owned",
            120,
        ),
        CommandSpec("spec_coverage", (py, "scripts/check_spec_coverage.py", *TESTS), "owned", 60),
        CommandSpec(
            "repository_health",
            (str(ROOT / ".venv/bin/pytest"), "tests/python", "-q"),
            "repository_health",
            600,
        ),
    ]
    value = dict(
        commands=[asdict(s) for s in specs],
        coverage_config=str(config),
        coverage_json=str(scratch / "coverage.json"),
        artifact_guard_enabled=True,
        frozen_before_measurement=True,
        code_config_hashes=[d.c.reference(ROOT / p) for p in OWNED],
        random_seed=69394,
        salt="V693-development",
        roles=d.ROLES,
        stream_slots=256,
        delay_slots=20,
        response_order="numeric_decimal_ID_then_lexical_ID",
        terminal_commands=[
            "adversarial_verify.py --json",
            "verdict_row_consistency_lint.py --strict",
            "cold-replay",
            "both_conductor_readers",
        ],
    )
    atomic_json(raw / "validation_commands.json", value)
    return value


def apply_validation(value: Json, receipts: list[Json]) -> None:
    """Owned failures disqualify; broad pre-existing health remains diagnostic."""
    value["validation_receipts"] = receipts
    value["repository_health"] = next((r for r in receipts if r["name"] == "repository_health"), {})
    failed = [r for r in receipts if r["name"] != "repository_health" and not r["passed"]]
    if failed:
        value.update(
            honest_verdict="complete_disqualified_owned_checks",
            verdict_class="disqualified",
            cohort_ready_score=0,
        )
        value["acceptance_gate_results"].update(validity=False, readiness=False)
        value["historical_required_failures"].extend(failed)


def terminal_check(path: Path) -> Json:
    """Check private candidate bytes while retaining final-byte validator logs."""
    with TemporaryDirectory(prefix="carnot-7994-terminal-") as workspace:
        candidate = Path(workspace) / "candidate.json"
        candidate.write_bytes(path.read_bytes())
        value = json.loads(candidate.read_text())
        atomic_json(
            Path(workspace) / "primitive_rows.json",
            dict(rows=value["rows"], stream_slot_rows=value["stream_slot_rows"]),
        )
        replay(value)
        checks = [
            CommandSpec(
                name,
                (str(ROOT / ".venv/bin/python"), "-u", str(ROOT / script), flag, str(candidate)),
                "terminal",
                60,
            )
            for name, script, flag in [
                ("adversarial", "scripts/adversarial_verify.py", "--json"),
                ("strict_rows", "scripts/verdict_row_consistency_lint.py", "--strict"),
            ]
        ]
        outside = "import os,sys,tempfile; os.chdir(tempfile.mkdtemp(prefix='carnot-7994-replay-')); os.environ.pop('PYTHONPATH',None); os.execv(sys.argv[1],sys.argv[1:])"
        checks.append(
            CommandSpec(
                "external_cold_replay",
                (
                    str(ROOT / ".venv/bin/python"),
                    "-c",
                    outside,
                    str(ROOT / ".venv/bin/python"),
                    str(ROOT / OWNED[-1]),
                    "--cold-replay",
                    str(candidate),
                ),
                "terminal",
                300,
            )
        )
        receipts = run_commands(ROOT, checks, log_dir=path.parent / "terminal", heartbeat_s=30)
        return dict(
            passed=all(r["passed"] for r in receipts),
            receipts=receipts,
            flagged_adversarial=not receipts[0]["passed"],
        )


def publish(output: Path, value: Json) -> None:
    """Publish only checked bytes and retain final reader and validator receipts."""
    raw = output.parent / "raw" / NAME
    value["terminal_validation_sidecar_path"] = str(raw / "terminal_validation.json")
    value["rows_hash"] = canonical_hash(value["rows"])
    value["field_principles"] = {
        k: "Bind current execution, bounded source disjointness, public bytes or evaluator custody."
        for k in value
    }
    receipt = publish_primary(output, value, terminal_check)
    readers = reader_receipt(
        TASK, output.parent, field="cohort_ready_score", expected=value["cohort_ready_score"]
    )
    if not readers["passed"] or readers["gate_sha256"] != sha256_file(output):
        raise ValueError("primary_reader_failure")
    atomic_json(raw / "terminal_validation.json", dict(receipt, readers=readers))


def main(argv: list[str] | None = None) -> int:
    """The public CLI keeps evaluator workers distinct and measures actual elapsed time."""
    started = time.monotonic()
    d.progress("phase=start no_model_load")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261001"], default="20261001")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--private-worker", action="store_true")
    parser.add_argument("--fixture-root", type=Path)
    parser.add_argument("--evaluate", type=Path)
    parser.add_argument("--evaluator-output", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.evaluate:
            if args.evaluator_output is None:
                raise ValueError("evaluator_output_required")
            d.evaluate(args.root, args.evaluate, args.evaluator_output)
            return 0
        if args.cold_replay:
            replay(json.loads(args.cold_replay.read_text()))
            d.progress("replay_passed")
            return 0
        output = args.output.absolute()
        if output.name != NAME + ".json":
            raise ValueError("primary_name")
        if (args.fixture_root or args.private_worker) and output.parent == ROOT / "results":
            raise ValueError("private_output_required")
        raw = output.parent / "raw" / NAME
        root = args.fixture_root or args.root
        with TemporaryDirectory(prefix="carnot-7994-") as workspace:
            scratch = Path(workspace)
            manifest = (
                freeze_commands(raw, scratch)
                if not args.fixture_root and not args.private_worker
                else None
            )
            d.progress("phase=authenticate")
            plan = d.authenticate(root, fixture=bool(args.fixture_root))
            value = (
                base(plan["failures"])
                if plan["failures"]
                else measure(plan, root, raw, fixture=bool(args.fixture_root))
            )
            if args.fixture_root:
                value.update(
                    honest_verdict="complete_circular_positive_protocol_fixture",
                    verdict_class="circular_positive",
                    cohort_ready_score=0,
                )
                value["sample_size_budget"]["independent"] = 0
                value["claim_scope"] = (
                    "Synthetic protocol fixtures only; zero independent natural evidence."
                )
            value.update(
                cited_upstream_artifacts=plan["references"],
                preconditions_checked=plan["checks"],
                code_config_hashes=[d.c.reference(ROOT / p) for p in OWNED],
            )
            if manifest:
                d.progress("phase=before_validation_benchmark")
                receipts = run_commands(
                    ROOT,
                    [
                        CommandSpec(r["name"], tuple(r["argv"]), r["scope"], r["timeout_s"])
                        for r in manifest["commands"]
                    ],
                    log_dir=raw / "validation",
                    extra_env={
                        "CARNOT_7994_COVERAGE_CONFIG": manifest["coverage_config"],
                        "COVERAGE_FILE": str(scratch / ".coverage"),
                    },
                    heartbeat_s=30,
                )
                apply_validation(value, receipts)
                value["validation_command_manifest_path"] = str(raw / "validation_commands.json")
                report_path = Path(manifest["coverage_json"])
                report = (
                    json.loads(report_path.read_text()) if report_path.is_file() else dict(files={})
                )
                value["coverage_statement_counts"] = {
                    p: r["summary"] for p, r in report["files"].items()
                }
                atomic_json(raw / "coverage.json", report)
                if set(value["coverage_statement_counts"]) != set(OWNED) or any(
                    r["num_statements"] <= 0 or r["missing_lines"]
                    for r in value["coverage_statement_counts"].values()
                ):
                    apply_validation(
                        value,
                        receipts + [dict(name="nonempty_100_coverage", passed=False, exit_code=1)],
                    )
                d.progress("phase=after_validation_benchmark")
            elapsed = time.monotonic() - started
            value.update(
                duration_s=elapsed,
                phase_spans=[dict(phase="owned_cpu_invocation", start_s=0.0, end_s=elapsed)],
            )
            value["reproducibility_checksum"] = canonical_hash(
                dict(
                    code=value["code_config_hashes"],
                    inputs=value["cited_upstream_artifacts"],
                    raw_shards=value["raw_shard_hashes"],
                    seed=69394,
                    salt="V693-development",
                )
            )
            d.progress("phase=checked_publication")
            publish(output, value)
            d.progress("phase=complete")
            return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        d.progress("error=" + str(error))
        return 1
