"""REQ-REPORT-8006: qualify readers without replacing historical science.

The saved issue, rather than a later aggregate, determines an error. Original
early label access remains disqualifying even when today's reader works.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.experiment_7997_v693_typed_development_decisions import disjoint
from carnot.reporting import typed_validation_7997 as validation
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
)
from carnot.reporting.evidence_features_custody_7980 import checked, operand, reference
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v693_capstone_reduction import score

Json = dict[str, Any]
STARTED = time.monotonic()
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8006_v694_independent_replay"
TASK = "exp8006-independent-replay"
OWNED = ["python/carnot/reporting/independent_replay_8006.py", f"scripts/experiments/{NAME}.py"]
TEST = "tests/python/test_independent_replay_8006.py"
SOURCES = {
    7997: "experiment_7997_v693_typed_development_decisions",
    8000: "experiment_8000_v693_delayed_confidence",
    8004: "experiment_8004_v693_capstone",
}
METHOD = dict(
    error="int(target not in issued_set) only when issued eligible and target binary",
    release="issue then release at issue_slot+delay",
    phase="(issue_slot-1)%(delay+1)",
    static="original seal and original access receipt; later seals cannot repair custody",
    budget="all saved issued rows; repeated arms and delays never multiply sources",
    gates=[
        "valid_static_reader",
        "valid_issued_reader",
        "all_owned_checks",
        "100% added statements",
    ],
)


def progress(phase: str, units: int = 0, pending: str = "next_phase") -> None:
    """Flush boundaries so supervisors can distinguish work from a stalled process."""
    print(
        f"[exp8006] phase={phase} elapsed_s={time.monotonic() - STARTED:.3f} units={units} pending={pending}",
        flush=True,
    )


def reduce(source: Json) -> Json:
    """Reconstruct two branches separately; fixture success cannot restore history."""
    static, confidence = source["static"], source["confidence"]
    if disjoint(source["roles"]) != static["role_hashes"]:
        raise ValueError("role_hash_drift")
    access = next(e for e in static["label_access_events"] if e["role"] == "stream")
    if (
        access["preceded_by"]["sha256"] != source["stream_seal_sha256"]
        or access["sha256"] != source["target_sha256"]
    ):
        raise ValueError("path_hash_custody_defect")
    public = {r["family_id"]: r for r in source["public"]["stream"]}
    predictions = {(r["family_id"], r["arm"], r["seed"]): r for r in source["predictions"]}
    early = (
        static["prior_exposure_receipt"]["predictions_sealed_before_stream_label_access"] is False
    )
    rows = []
    for r in static["rows"]:
        key = r["family_id"], r["arm"], r["seed"]
        if (
            key not in predictions
            or r["family_id"] not in public
            or public[r["family_id"]]["source_cluster_id"] != r["source_cluster_id"]
        ):
            raise ValueError("static_role_join_defect")
        if predictions[key]["probability"] != r["probability"]:
            raise ValueError("static_prediction_drift")
        rows.append(
            dict(
                r,
                id=f"static:{key}",
                metric="diagnostic_cost",
                numerator=r["actual_cost"] or 0,
                denominator=1,
                exclusion_reason=None,
                censor_reason="unknown_target" if r["y"] is None else None,
                original_exposure_label="known_early_stream_target_access"
                if early
                else "recorded_seal_before_access",
                producer_sha256=source["producer_hashes"]["7997"],
                scientific_eligible=False,
            )
        )
    issued, feedback = {}, {}
    last_release: Json = {}
    for r in confidence["issued_state_rows"]:
        key = r["arm"], r["delay"], r["issue_slot"]
        if (
            key in issued
            or r["phase"] != (r["issue_slot"] - 1) % (r["delay"] + 1)
            or r["due_slot"] != r["issue_slot"] + r["delay"]
        ):
            raise ValueError("issued_identity_or_phase_drift")
        issued[key] = r
    for r in confidence["rows"]:
        key = r["arm"], r["delay"], r["issue_slot"]
        group = str(key[:2])
        if key not in issued or key in feedback or r["release_slot"] <= last_release.get(group, 0):
            raise ValueError("release_identity_or_order_drift")
        last_release[group] = r["release_slot"]
        feedback[key] = r
    errors, failures = [], []
    current: Json = {}
    for index, (key, issue) in enumerate(issued.items()):
        if index % 512 == 0:
            progress("issued_reconstruction", index, str(len(issued) - index))
        r = feedback.get(key)
        y = source["bundle"]["targets"][issue["family_id"]]
        eligible = issue["eligibility"] and type(y) is int and y in (0, 1)
        error = int(y not in issue["prediction_set"]) if eligible and r else None
        if r:
            if any(
                r[f] != issue[f] for f in ("family_id", "prediction_set", "issue_alpha", "phase")
            ) or r["y"] != (y if issue["eligibility"] else None):
                raise ValueError("issued_target_or_set_drift")
            if r["release_slot"] != issue["due_slot"] or r["error"] != error:
                raise ValueError("issued_error_or_release_drift")
            group = str(key[:2])
            base = (
                issue["issue_alpha"] if issue["arm"] == "interleaved" else current.get(group, 0.1)
            )
            after = max(0.01, min(0.50, base + 0.01 * (0.1 - error))) if eligible else base
            after = 0.1 if issue["arm"] == "frozen" else after
            if abs(r["base_alpha"] - base) > 1e-12 or abs(r["alpha_after"] - after) > 1e-12:
                raise ValueError("issued_recurrence_drift")
            current[group] = after
            old_error = int(r["y"] not in r["prediction_set"])
            if old_error != error:
                failures.append(
                    dict(
                        original_assertion="issued_error_drift",
                        original_row=r,
                        expected=error,
                        observed=r["error"],
                        old_reader_expected=old_error,
                        classification="reader_eligibility_defect",
                        producer_sha256=source["producer_hashes"]["8000"],
                    )
                )
        errors.append(
            dict(
                issue,
                id=f"confidence:{key}",
                metric="issued_set_error",
                y=y,
                recomputed_error=error,
                recorded_error=r["error"] if r else None,
                numerator=error or 0,
                denominator=int(bool(r) and eligible),
                status="released" if r else "censored",
                exclusion_reason=None if eligible else "excluded_or_unknown_target",
                censor_reason="pending_at_stream_end"
                if not r
                else (None if eligible else "unknown_or_excluded"),
                original_exposure_label="issued_before_due_release",
                producer_sha256=source["producer_hashes"]["8000"],
            )
        )
    return dict(
        static_reader_ready_score=1,
        historical_static_recovered_score=int(not early and source["original_stream_seal_present"]),
        confidence_recovered_score=1,
        rows=rows,
        issued_error_rows=errors,
        failed_operand_rows=failures,
        independent_reduction_rows=[
            dict(branch="static", qualified=False, **score(static["rows"], {})),
            dict(
                branch="confidence",
                issued=len(errors),
                released=len(feedback),
                scored=sum(r["denominator"] for r in errors),
                errors=sum(r["numerator"] for r in errors),
                excluded_reader_discrepancies=len(failures),
                benefit_claim=False,
            ),
        ],
    )


def load_sources(root: Path, raw: Path) -> tuple[Json, list[Json], list[Json], list[Json]]:
    """Copy authenticated original bytes before reduction so temporary paths can expire."""
    refs: list[Json] = []
    cites, gates, upstream = [], [], {}

    def save(ref: Json, parse: bool = True) -> Json:
        path = checked(ref)
        destination = raw / "snapshots" / (ref["sha256"].split(":")[1] + path.suffix)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
        saved = dict(reference(destination), original_path=str(path))
        refs.append(saved)
        return dict(json.loads(destination.read_text())) if parse else {}

    for eid, name in SOURCES.items():
        path = root / "results" / (name + ".json")
        if not path.is_file():
            gates.append(
                dict(
                    operand(eid, path, "primary_exists", True, False),
                    artifact_field="primary_exists",
                )
            )
            continue
        ref = reference(path)
        upstream[eid] = save(ref)
        cites.append(
            dict(
                ref,
                producer_id=eid,
                snapshot=refs[-1],
                imported_fields=["honest_verdict", "verdict_class"]
                + {
                    7997: [
                        "rows",
                        "checkpoints",
                        "prior_exposure_receipt",
                        "label_access_events",
                        "role_hashes",
                    ],
                    8000: ["rows", "issued_state_rows", "primitive_bundle"],
                    8004: ["independent_reduction_rows"],
                }[eid],
            )
        )
    if gates:
        return {}, refs, cites, gates
    static, confidence = upstream[7997], upstream[8000]
    for ref in static["code_config_hashes"]:
        save(ref, parse=False)
    for name, digest in confidence["code_config_hashes"].items():
        if name != "config":
            save(dict(path=str(ROOT / name), sha256=digest), parse=False)
    atomic_json(
        raw / "role_budget_freeze.json",
        dict(
            role_hashes=static["role_hashes"],
            intended=len(confidence["issued_state_rows"]),
            method=METHOD,
        ),
    )
    checkpoints = {k: save(v) for k, v in static["checkpoints"].items()}
    prior = static["prior_exposure_receipt"]
    save(static["prior_exposure_reference"])
    save(prior["original_task_configuration"])
    save(prior["original_calibration_predictions"])
    original = Path(prior["original_task_configuration"]["path"]).parent
    for name in ("public.json", "roles.json", "heads.json", "validation_manifest.json"):
        save(reference(original / name))
    failed = next(
        r for r in static["validation_receipts"] if r["name"] == "stream_target_exposure_order"
    )
    log = checked(dict(path=failed["log_path"], sha256=failed["log_sha256"]))
    destination = raw / "original_stream_target_exposure_order.log"
    destination.write_bytes(log.read_bytes())
    refs.append(dict(reference(destination), original_path=str(log)))
    save(reference(Path(failed["command_argv"][-1])))
    targets = {k: save(v) for k, v in static["evaluator_targets"].items()}
    bundle = save(confidence["primitive_bundle"])
    save(confidence["point_prediction_seal"])
    save(confidence["restart_state_checkpoint"])
    original_seal = Path(prior["missing_original_stream_predictions"])
    original_present = original_seal.is_file()
    if original_present:
        save(reference(original_seal))
    for field, observed in (
        (
            "predictions_sealed_before_stream_label_access",
            prior["predictions_sealed_before_stream_label_access"],
        ),
        ("original_stream_prediction_seal_exists", original_present),
    ):
        if observed is not True:
            gates.append(
                dict(
                    operand(
                        7997,
                        root / "results" / (SOURCES[7997] + ".json")
                        if field == "predictions_sealed_before_stream_label_access"
                        else original_seal,
                        field,
                        True,
                        observed,
                    ),
                    artifact_field="prior_exposure_receipt."
                    + (
                        field
                        if field == "predictions_sealed_before_stream_label_access"
                        else "missing_original_stream_predictions"
                    ),
                )
            )
    target_map = {r["family_id"]: r["y"] for r in targets["stream"]["rows"]}
    if target_map != bundle["targets"]:
        raise ValueError("original_target_drift")
    source = dict(
        static=static,
        confidence=confidence,
        bundle=bundle,
        roles=checkpoints["roles"],
        public=checkpoints["public"],
        predictions=checkpoints["stream_predictions"]["rows"],
        stream_seal_sha256=static["checkpoints"]["stream_predictions"]["sha256"],
        target_sha256=static["evaluator_targets"]["stream"]["sha256"],
        original_stream_seal_present=original_present,
        producer_hashes={str(r["producer_id"]): r["sha256"] for r in cites},
        original_verdicts={
            str(i): dict(honest_verdict=v["honest_verdict"], verdict_class=v["verdict_class"])
            for i, v in upstream.items()
        },
        original_failed_assertion=failed,
        original_capstone_assertion=next(
            r
            for r in upstream[8004]["independent_reduction_rows"]
            if r["task_id"] == "exp8000-delayed-confidence"
        ),
    )
    return source, refs, cites, gates


def freeze(raw: Path, scratch: Path) -> Json:
    """Pin actual commands and private coverage before reading evaluation targets."""
    py, cov = str(ROOT / ".venv/bin/python"), str(ROOT / ".venv/bin/coverage")
    include = ",".join(str(ROOT / p) for p in OWNED)
    prefix = [
        cov,
        "run",
        "--parallel-mode",
        "--data-file=" + str(scratch / ".coverage"),
        "--include=" + include,
    ]
    tests = [
        TEST,
        "tests/python/test_primary_publication_7928.py",
        "tests/python/test_source_boundary_7852.py",
        "tests/python/test_experiment_7942_v689_sentence_labels.py",
    ]
    commands = []
    for name, argv, required, deadline in (
        (
            "unit_consumers_e2e015_e2e019_issued_mutation",
            prefix
            + [
                "-m",
                "pytest",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "--basetemp=" + str(scratch / "pytest"),
                *tests,
                "-q",
            ],
            True,
            180,
        ),
        (
            "coverage_combine",
            [cov, "combine", "--data-file=" + str(scratch / ".coverage"), str(scratch)],
            True,
            60,
        ),
        (
            "coverage_report",
            [
                cov,
                "report",
                "--data-file=" + str(scratch / ".coverage"),
                "--include=" + include,
                "--fail-under=100",
            ],
            True,
            60,
        ),
        (
            "coverage_json",
            [
                cov,
                "json",
                "--data-file=" + str(scratch / ".coverage"),
                "--include=" + include,
                "-o",
                str(scratch / "coverage.json"),
            ],
            True,
            60,
        ),
        ("ruff_check", [str(ROOT / ".venv/bin/ruff"), "check", *OWNED, TEST], True, 60),
        (
            "ruff_format",
            [str(ROOT / ".venv/bin/ruff"), "format", "--check", *OWNED, TEST],
            True,
            60,
        ),
        (
            "strict_mypy",
            [str(ROOT / ".venv/bin/mypy"), "--strict", "--follow-imports=silent", *OWNED],
            True,
            60,
        ),
        ("spec_coverage", [py, "scripts/check_spec_coverage.py", TEST, *OWNED], True, 60),
        ("full_pytest", [str(ROOT / ".venv/bin/pytest"), "tests/python", "-q"], False, 120),
    ):
        commands.append(
            dict(name=name, argv=argv, required=required, cwd=str(ROOT), deadline_s=deadline)
        )
    manifest = dict(
        commands=commands,
        config=METHOD,
        environment=dict(
            PYTHONUNBUFFERED="1",
            JAX_PLATFORMS="cpu",
            OPENBLAS_NUM_THREADS="1",
            COVERAGE_FILE=str(scratch / ".coverage-health"),
            CARNOT_8006_COVERAGE=str(scratch / ".coverage"),
        ),
    )
    atomic_json(raw / "validation_manifest.json", manifest)
    return manifest


def cold_replay(path: Path) -> Json:
    """Recompute reductions from durable snapshots and reject changed summary bytes."""
    value = json.loads(path.read_text())
    for ref in (
        value["checkpoint_references"] + value["raw_shard_hashes"] + value["code_config_hashes"]
    ):
        checked(ref)
    source = json.loads(checked(value["source_checkpoint"]).read_text())
    reconstructed = reduce(source) if source else {}
    if any(value[k] != v for k, v in reconstructed.items()):
        raise ValueError("reduction_drift")
    if (
        canonical_hash({k: v for k, v in value.items() if k != "artifact_checksum"})
        != value["artifact_checksum"]
    ):
        raise ValueError("artifact_drift")
    return dict(passed=True, replay_passed=True)


def terminal_check(path: Path) -> Json:
    """Use unchanged validators and a new process against the exact candidate bytes."""
    commands = []
    for name, argv in (
        (
            "cold_reduce",
            [
                str(ROOT / ".venv/bin/python"),
                "-u",
                str(ROOT / OWNED[1]),
                "--cold-replay",
                str(path),
            ],
        ),
        (
            "adversarial",
            [
                str(ROOT / ".venv/bin/python"),
                "-u",
                str(ROOT / "scripts/adversarial_verify.py"),
                "--json",
                str(path),
            ],
        ),
        (
            "strict_rows",
            [
                str(ROOT / ".venv/bin/python"),
                "-u",
                str(ROOT / "scripts/verdict_row_consistency_lint.py"),
                "--strict",
                str(path),
            ],
        ),
    ):
        commands.append(dict(name=name, argv=argv, required=True, cwd=str(ROOT), deadline_s=60))
    receipts = validation.execute(
        dict(commands=commands, environment=dict(PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")),
        path.parent / "terminal",
    )
    return dict(passed=all(r["passed"] for r in receipts), receipts=receipts)


def main(argv: list[str] | None = None) -> int:
    """Freeze, reconstruct and publish once; private workers cannot claim qualification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261002", choices=["20261002"])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fixture-input", type=Path)
    parser.add_argument("--validation-worker", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if (args.fixture_input or args.validation_worker) and (
        args.output is None or args.output.resolve().is_relative_to(ROOT / "results")
    ):
        parser.error("private workers require an output outside results")
    began = time.monotonic()
    progress("begin_no_model_load_no_generation", pending="freeze")
    try:
        if args.cold_replay:
            print(json.dumps(cold_replay(args.cold_replay)), flush=True)
            return 0
        output = (args.output or args.root / "results" / (NAME + ".json")).absolute()
        raw = output.parent / "raw" / output.stem
        raw.mkdir(parents=True, exist_ok=True)
        scratch = Path(tempfile.mkdtemp(prefix="carnot-8006-"))
        manifest = freeze(raw, scratch)
        code = []
        for name in OWNED + [TEST]:
            path = raw / "code" / name.replace("/", "_")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((ROOT / name).read_bytes())
            code.append(dict(reference(path), original_path=str(ROOT / name)))
        atomic_json(
            raw / "method_freeze.json",
            dict(
                method=METHOD,
                code=code,
                source_pins={
                    str(i): reference(args.root / "results" / (n + ".json"))
                    for i, n in SOURCES.items()
                    if (args.root / "results" / (n + ".json")).is_file()
                },
            ),
        )
        progress("methods_code_budgets_commands_frozen", pending="source_snapshot")
        source, refs, cites, gates = (
            (json.loads(args.fixture_input.read_text()), [], [], [])
            if args.fixture_input
            else load_sources(args.root, raw)
        )
        atomic_json(raw / "sources.json", source)
        source_ref = reference(raw / "sources.json")
        atomic_json(
            raw / "role_budget_freeze.json",
            dict(
                role_hashes=source.get("static", {}).get("role_hashes", {}),
                intended=len(source.get("confidence", {}).get("issued_state_rows", [])),
                method=METHOD,
            ),
        )
        progress("sources_roles_frozen_before_reduction", pending="independent_readers")
        reduced = (
            reduce(source)
            if source
            else dict(
                static_reader_ready_score=0,
                historical_static_recovered_score=0,
                confidence_recovered_score=0,
                rows=[],
                issued_error_rows=[],
                failed_operand_rows=[],
                independent_reduction_rows=[],
            )
        )
        atomic_json(raw / "reduction.json", reduced)
        progress("independent_readers_complete", len(reduced["issued_error_rows"]), "owned_checks")
        receipts = validation.execute(manifest, raw) if not args.validation_worker else []
        counts = (
            {
                k: v["summary"]
                for k, v in json.loads((scratch / "coverage.json").read_text())["files"].items()
            }
            if (scratch / "coverage.json").is_file()
            else {}
        )
        atomic_json(raw / "coverage_statement_counts.json", counts)
        coverage_ok = set(counts) == set(OWNED) and all(
            c["num_statements"] > 0 and c["missing_lines"] == 0 for c in counts.values()
        )
        qualified = (
            bool(receipts) and coverage_ok and all(r["passed"] for r in receipts if r["required"])
        )
        verdict = (
            "blocked"
            if not source
            else (
                "circular_positive"
                if args.fixture_input
                else ("null" if qualified else "disqualified")
            )
        )
        ready = int(bool(source) and qualified and not args.fixture_input)
        all_rows = reduced["rows"] + reduced["issued_error_rows"]
        value = dict(
            reduced,
            experiment_id=8006,
            task_id=TASK,
            milestone="2026.10.694",
            run_date=args.date,
            execution_date=args.date,
            schema="carnot.independent_replay.v1",
            honest_verdict="complete_" + verdict + "_independent_replay",
            verdict_class=verdict,
            claim_scope="Independent reader qualification and historical diagnostic reconstruction only; no scientific benefit or new confidence sweep.",
            inference_substrate="aggregation_from_upstream_artifacts",
            inference_substrate_class="no_model_load",
            MODEL_SPECS=[],
            model_specs=[],
            model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
            trained_head_specs=[],
            verifier_is_oracle=bool(args.fixture_input),
            gate_check_summary=gates,
            acceptance_gate_results=dict(
                reader_ready=bool(ready),
                historical_static_recovered=bool(reduced["historical_static_recovered_score"]),
                confidence_recovered=bool(reduced["confidence_recovered_score"]),
                scientific_benefit=False,
            ),
            genuine_headroom=dict(
                scientific_benefit_measured=False,
                reader_bug_detected=bool(reduced["failed_operand_rows"]),
            ),
            positive_control_results=dict(
                scope="private circular fixtures only",
                independent_natural_evidence=False,
                valid_and_corrupted_replay_checked=qualified,
            ),
            replay_reader_ready_score=ready,
            original_verdicts=source.get("original_verdicts", {}),
            original_failed_assertion=source.get("original_failed_assertion"),
            original_capstone_assertion=source.get("original_capstone_assertion"),
            cited_upstream_artifacts=cites,
            code_config_hashes=code,
            checkpoint_references=refs + [source_ref],
            source_checkpoint=source_ref,
            raw_shard_hashes=[
                reference(raw / n)
                for n in (
                    "method_freeze.json",
                    "role_budget_freeze.json",
                    "reduction.json",
                    "validation_manifest.json",
                )
            ],
            validation_receipts=receipts,
            coverage_statement_counts=counts,
            repository_health=[r for r in receipts if not r["required"]],
            flagged_adversarial=False,
            random_seed=6948006,
            reproducibility_checksum=canonical_hash(
                dict(source=source_ref, code=code, method=METHOD)
            ),
            sample_size_budget=dict(
                intended=len(all_rows),
                eligible=sum(r["denominator"] > 0 for r in all_rows),
                started=len(all_rows),
                completed=sum(
                    r["status"] != "censored" and r.get("probability", 1) is not None
                    for r in all_rows
                ),
                excluded=sum(bool(r["exclusion_reason"]) for r in all_rows),
                failed=sum(bool(r.get("failure_status")) for r in all_rows),
                censored=sum(bool(r["censor_reason"]) for r in all_rows),
                independent=len({r["source_cluster_id"] for r in all_rows}),
                arms_seeds_delays_are_independent=False,
            ),
            duration_s=time.monotonic() - began,
            terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        )
        value["phase_spans"] = [
            dict(phase="freeze_reconstruction_owned_validation", duration_s=value["duration_s"])
        ]
        value["field_principles"] = {
            k: "Immutable original operands and owned checks determine readiness; fixtures do not recover science."
            for k in value
        }
        value["artifact_checksum"] = canonical_hash(value)
        progress("publication_begin", pending="final_bytes")
        publication = publish_primary(output, value, terminal_check)
        atomic_json(raw / "terminal_validation.json", publication)
        selected = reader_receipt(
            TASK, output.parent, field="replay_reader_ready_score", expected=ready
        )
        atomic_json(raw / "primary_resolution.json", selected)
        if not selected["passed"] or selected["gate_sha256"] != publication["primary_sha256"]:
            raise ValueError("primary_reader_drift")
        cold_replay(output)
        progress("published_rechecked", len(all_rows), "none")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"[exp8006] failed={error}", flush=True)
        return 1
