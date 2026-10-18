"""Publish complete-response support targets from exact human annotations.

REQ-REPORT-7955. This host-only aggregation qualifies annotation transport;
it cannot establish a learned detector benefit or a fresh holdout result.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import time
from typing import Any

from carnot import experiment_7942_v689_sentence_labels as prior
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.verify import response_targets_7955 as targets
from carnot.verify.source_projection import read_jsonl, write_jsonl

Json = dict[str, Any]
ROOT = prior.ROOT
NAME = "experiment_7955_v690_response_targets"
TASK = "exp7955-response-targets"
MODEL_SPECS: list[str] = []
OWNED = [
    f"python/carnot/{NAME}.py",
    "python/carnot/verify/response_targets_7955.py",
    f"scripts/experiments/{NAME}.py",
]
TESTS = ["tests/python/test_response_targets_7955.py", f"tests/python/test_{NAME}.py", *prior.TESTS]
INCLUDE = ",".join(str(ROOT / p) for p in OWNED)
HISTORY = "results/experiment_7942_v689_sentence_labels.json"
HISTORY_HASH = "sha256:0b6e5272f2e3f86f419fec22f3419c7a3d586cdc2df624ba2454bc18652a1173"
IMPORTS = [
    *prior.IMPORTS,
    "carnot.experiment_7942_v689_sentence_labels",
    "carnot.verify.response_targets_7955",
]
reference, operand = prior.reference, prior.operand


def progress(phase: str, started: float, units: int = 0) -> None:
    """Keep actual host work visible with a monotonic elapsed counter."""
    print(
        f"[exp7955] phase={phase} completed_units={units} elapsed_s={time.monotonic() - started:.3f}",
        flush=True,
    )


def authenticate(root: Path) -> tuple[list[Json], Json]:
    """Reuse current pinned corpus custody and authenticate the preserved null."""
    failures, upstream = prior.authenticate(root)
    path = root / HISTORY
    history = operand(
        "exp7942", path, "sha256", HISTORY_HASH, sha256_file(path) if path.is_file() else None
    )
    if not history["passed"]:
        failures.append(history)
    if not failures:
        upstream["authenticated_inputs"].append(reference(path))
        upstream["preconditions"].append(history)
    return failures, upstream


def base(failures: list[Json]) -> Json:
    """Use the complete schema while keeping the executing producer explicit."""
    value = prior.base([], failures)
    for key in list(value):
        if key.startswith("sentence_") or key in {
            "label_counts",
            "role_counts",
            "offset_join_rows",
            "annotation_policy",
        }:
            del value[key]
    value.update(targets.reduce_rows([]))
    value.update(
        schema="carnot.exp7955.response_targets.v1",
        experiment_id=7955,
        task_id=TASK,
        milestone="2026.09.690",
        run_date="20261001",
        random_seed=69055,
        honest_verdict="complete_blocked_annotation_custody",
        public_manifest_path=None,
        public_manifest_sha256=None,
        evaluator_manifest_path=None,
        evaluator_manifest_sha256=None,
        annotation_rows=[],
        response_union_rows=[],
        cited_upstream_artifacts=[],
        inherited_sentence_null=dict(
            path=str(ROOT / HISTORY),
            sha256=HISTORY_HASH,
            class_counts={"0": 59, "1": 3},
            eligible=62,
            honest_verdict="complete_null_sentence_annotation_transport",
        ),
        target_definition=dict(
            unit="complete_response",
            primary="any authenticated human source-unsupported span anywhere in the complete response",
            includes_implicit_true=True,
            includes_due_to_null=True,
            sensitivity="implicit_true_excluded_y (descriptive only)",
            negative_requires="complete quality-approved annotations with no spans",
            selection="all fixed evaluation slots; no label-based selection",
        ),
        qwen_admission=dict(
            applied=False,
            owner="later Qwen task",
            input_token_limit=6000,
            rule="pinned tokenizer admission before evaluator access",
        ),
        oracle_distinct_corrigendum="Preserve September 28: source support labels are not formal truth certificates or independent detector benefit.",
    )
    return value


def check_receipts(value: Json) -> None:
    """Bind readiness checks to frozen argv and unchanged subprocess logs."""
    manifest = json.loads(Path(value["validation_command_manifest_path"]).read_text())
    receipts = {r["name"]: r for r in value["validation_receipts"]}
    for spec in manifest["commands"]:
        if not spec.get("required", True):
            continue
        receipt = receipts.get(spec["name"])
        if (
            receipt is None
            or receipt["command_argv"] != spec["argv"]
            or receipt["exit_code"] != spec["expected_exit"]
            or receipt.get("timed_out")
            or not receipt["passed"]
        ):
            raise ValueError("receipt_drift:" + spec["name"])
        log = Path(receipt["log_path"])
        if not log.is_absolute():
            log = ROOT / log
        if sha256_file(log) != receipt["log_sha256"] or (
            spec["failure_reason"] and spec["failure_reason"] not in log.read_text()
        ):
            raise ValueError("receipt_log_drift:" + spec["name"])


def reconstruct(value: Json) -> Json:
    """Recompute all claims from primitive public bytes and original spans."""
    for item in value.get("code_config_hashes", []) + value["source_artifact_hashes"]:
        prior.checked_reference(item)
    if value["public_manifest_path"] is None:
        if value["response_targets_ready_score"]:
            raise ValueError("unsafe_readiness")
        return targets.reduce_rows([])
    public_path = prior.checked_reference(
        dict(path=value["public_manifest_path"], sha256=value["public_manifest_sha256"])
    )
    ev_path = prior.checked_reference(
        dict(path=value["evaluator_manifest_path"], sha256=value["evaluator_manifest_sha256"])
    )
    saved = json.loads(public_path.read_text())
    data = json.loads(
        prior.checked_reference(
            value.get("fixture_input", value["original_input_manifest"])
        ).read_text()
    )
    if not value.get("fixture_input"):
        failures, upstream = authenticate(Path(value["custody_root"]))
        if failures:
            raise ValueError("cold_custody")
        root = Path(value["custody_root"])
        data = dict(
            public=[
                r for item in upstream["public_shards"] for r in read_jsonl(Path(item["path"]))
            ],
            roles=upstream["rows"],
            evaluators=[
                r for item in upstream["evaluator_shards"] for r in read_jsonl(Path(item["path"]))
            ],
            responses=read_jsonl(root / "data/ragtruth/response.jsonl"),
            sources=read_jsonl(root / "data/ragtruth/source_info.jsonl"),
        )
    public, audit = targets.freeze(data["public"])
    rows, annotations = targets.join(public, audit, data)
    reduced = targets.reduce_rows(rows)
    if saved["predictor_inputs"] != public or saved["boundaries"] != audit:
        raise ValueError("public_drift")
    if (
        read_jsonl(ev_path) != rows
        or read_jsonl(prior.checked_reference(value["annotation_manifest"])) != annotations
    ):
        raise ValueError("evaluator_drift")
    for key in (
        "class_counts",
        "sample_size_budget",
        "source_cluster_counts",
        "rows_sha256",
        "exclusion_rows",
        "disagreement_rows",
    ):
        if value[key] != reduced[key]:
            raise ValueError("reduction_drift:" + key)
    if (
        value["rows"] != rows
        or value["response_union_rows"] != rows
        or value["annotation_rows"] != annotations
    ):
        raise ValueError("reduction_drift:primitive_rows")
    if value["response_targets_ready_score"] and (
        value.get("fixture_input")
        or not reduced["response_targets_ready_score"]
        or value["verdict_class"] in {"blocked", "disqualified"}
        or value["flagged_adversarial"]
        or not all(r["passed"] for r in value["mutation_rows"])
        or not value["validation_receipts"]
        or any(not r["passed"] for r in value["validation_receipts"] if r.get("required", True))
    ):
        raise ValueError("unsafe_readiness")
    if value["response_targets_ready_score"]:
        check_receipts(value)
    return reduced


def freeze_commands(raw: Path) -> Json:
    """Parameterize the existing supervised CLI checks before producing results."""
    manifest = prior.freeze_commands(raw)
    replacements = {
        prior.NAME: NAME,
        "sentence_labels_7942": "response_targets_7955",
        prior.INCLUDE: INCLUDE,
        "20260930": "20261001",
    }
    for c in manifest["commands"]:
        for k, v in replacements.items():
            c["argv"] = [arg.replace(k, v) for arg in c["argv"]]
        if c["name"] == "unit_consumer_e2e015":
            c["argv"] = (
                c["argv"][: c["argv"].index("tests/python/test_response_targets_7955.py")]
                + TESTS
                + ["-q"]
            )
        if c["name"] in {"ruff_check", "ruff_format", "strict_mypy"}:
            c["argv"] = (
                c["argv"][: 2 if c["name"] != "ruff_format" else 3]
                + OWNED
                + (TESTS[:2] if c["name"] != "strict_mypy" else [])
            )
            if c["name"] == "strict_mypy":
                c["argv"] = [
                    str(ROOT / ".venv/bin/mypy"),
                    "--strict",
                    "--follow-imports=skip",
                    *OWNED,
                ]
        if c["name"] == "spec_coverage":
            c["argv"] = c["argv"][:2] + TESTS
        if c["name"] == "repository_health":
            c["deadline_s"] = 60
    manifest.update(
        affected_files=OWNED,
        explicit_tests=TESTS,
        transitive_consumers=IMPORTS,
        coverage_includes=INCLUDE,
        current_date="20261001",
        terminal_commands=[
            dict(
                name="adversarial",
                argv=[
                    str(ROOT / ".venv/bin/python"),
                    "-u",
                    "scripts/adversarial_verify.py",
                    "--json",
                    str(raw / "terminal_candidate.json"),
                ],
                expected_exit=0,
                deadline_s=60,
            ),
            dict(
                name="strict_rows",
                argv=[
                    str(ROOT / ".venv/bin/python"),
                    "-u",
                    "scripts/verdict_row_consistency_lint.py",
                    "--strict",
                    str(raw / "terminal_candidate.json"),
                ],
                expected_exit=0,
                deadline_s=60,
            ),
        ],
    )
    atomic_json(raw / "validation_command_manifest.json", manifest)
    return manifest


def execute_commands(manifest: Json, logs: Path) -> list[Json]:
    """Share the heartbeat supervisor and preserve every real child exit."""
    import os

    os.environ["CARNOT_7955_COVERAGE_FILE"] = manifest["coverage_file"]
    return prior.execute_commands(manifest, logs)


def apply_validation(value: Json, receipts: list[Json]) -> None:
    """Separate current owned failures from historical repository health."""
    value["validation_receipts"] = receipts
    value["observed_child_commands"] = [r.get("command_argv", []) for r in receipts]
    value["repository_health"]["current"] = [r for r in receipts if not r.get("required", True)]
    if any(not r["passed"] for r in receipts if r.get("required", True)):
        value.update(
            verdict_class="disqualified",
            honest_verdict="complete_disqualified_required_validation",
            response_targets_ready_score=0,
        )
        value["acceptance_gate_results"].update(validity=False, readiness=False)


def terminal_check(candidate: Path) -> Json:
    """Cold-reduce current response claims before both exact-byte validators."""
    reconstruct(json.loads(candidate.read_text()))
    py = str(ROOT / ".venv/bin/python")
    specs = [
        prior.CommandSpec(
            "adversarial",
            (py, "-u", "scripts/adversarial_verify.py", "--json", str(candidate)),
            "terminal_candidate",
            60,
        ),
        prior.CommandSpec(
            "strict_rows",
            (py, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "terminal_candidate",
            60,
        ),
    ]
    receipts = prior.run_commands(
        ROOT, specs, log_dir=candidate.parent / "terminal_logs", heartbeat_s=10
    )
    return dict(
        passed=all(r["passed"] for r in receipts),
        flagged_adversarial=any(
            r["name"] == "adversarial" and r["exit_code"] != 0 for r in receipts
        ),
        candidate_sha256=sha256_file(candidate),
        receipts=receipts,
    )


def build(data: Json, raw: Path, *, fixture_path: Path | None = None) -> Json:
    """Seal public bytes before the evaluator join and retain all audit rows."""
    started = time.monotonic()
    public, audit = targets.freeze(data["public"])
    atomic_json(
        raw / "public_manifest.json",
        dict(
            predictor_inputs=public,
            boundaries=audit,
            budget=dict(max_answer_units=16, max_source_windows=128),
            selection="complete_response",
        ),
    )
    progress("public_manifest_sealed_before_labels", started, len(public))
    rows, annotations = targets.join(public, audit, data)
    value = base([])
    value.update(targets.reduce_rows(rows))
    mutations = []
    for key in (
        "human_label",
        "response_id",
        "annotation_byte_offsets",
        "labels",
        "annotation_order",
    ):
        changed = [targets.public_only({**r, key: "mutated"}) for r in public]
        mutations.append(
            dict(
                field=key,
                passed=targets.freeze(changed) == (public, audit),
                predictor_sha256=canonical_hash(changed),
                eligibility_sha256=canonical_hash(audit),
            )
        )
    write_jsonl(raw / "evaluators.jsonl", rows)
    write_jsonl(raw / "annotations.jsonl", annotations)
    atomic_json(raw / "validation_input.json", data)
    value.update(
        rows=rows,
        response_union_rows=rows,
        annotation_rows=annotations,
        mutation_rows=mutations,
        public_manifest_path=str(raw / "public_manifest.json"),
        public_manifest_sha256=sha256_file(raw / "public_manifest.json"),
        evaluator_manifest_path=str(raw / "evaluators.jsonl"),
        evaluator_manifest_sha256=sha256_file(raw / "evaluators.jsonl"),
        annotation_manifest=reference(raw / "annotations.jsonl"),
        original_input_manifest=reference(raw / "validation_input.json"),
    )
    value["gate_check_summary"] = [
        operand(TASK, raw / "evaluators.jsonl", o["field"], o["expected"], o["observed"], o["op"])
        for o in value["failed_operands"]
    ]
    value.update(
        honest_verdict="complete_null_response_annotation_transport"
        if value["response_targets_ready_score"]
        else "complete_null_response_target_capacity",
        verdict_class="null",
    )
    value["acceptance_gate_results"].update(
        validity=True, readiness=bool(value["response_targets_ready_score"])
    )
    if fixture_path is not None:
        value.update(
            fixture_input=reference(fixture_path),
            response_targets_ready_score=0,
            honest_verdict="complete_circular_positive_fixture_transport",
            verdict_class="circular_positive",
        )
        value["acceptance_gate_results"]["readiness"] = False
    progress("response_union_complete", started, len(rows))
    return value


def build_live(root: Path, raw: Path) -> Json:
    """Keep all original roles while exposing only the fixed evaluation target."""
    failures, upstream = authenticate(root)
    if failures:
        value = base(failures)
        value.update(
            inference_substrate_class="blocked_no_run",
            planned_inference_substrate_class="no_model_load",
        )
        return value
    public = [r for item in upstream["public_shards"] for r in read_jsonl(Path(item["path"]))]
    if len(public) != 640 or upstream["role_counts"] != prior.EIGHT_ROLES:
        raise ValueError("original_role_roster")
    # Public admission is saved before opening evaluator shards or annotations.
    frozen, audit = targets.freeze(public)
    atomic_json(
        raw / "public_manifest.json",
        dict(
            predictor_inputs=frozen,
            boundaries=audit,
            budget=dict(max_answer_units=16, max_source_windows=128),
            selection="complete_response",
        ),
    )
    data = dict(
        public=public,
        roles=upstream["rows"],
        evaluators=[
            r for item in upstream["evaluator_shards"] for r in read_jsonl(Path(item["path"]))
        ],
        responses=read_jsonl(root / "data/ragtruth/response.jsonl"),
        sources=read_jsonl(root / "data/ragtruth/source_info.jsonl"),
    )
    value = build(data, raw)
    value.update(
        custody_root=str(root),
        source_artifact_hashes=upstream["authenticated_inputs"],
        preconditions_checked=upstream["preconditions"],
        historical_required_failures=upstream["historical_required_failures"],
        cited_upstream_artifacts=[
            dict(experiment_id=e, fields_imported=fields, **reference(root / path))
            for e, path, fields in [
                (7423, list(prior.PINS)[1], ["commit", "asset_receipt"]),
                (7892, list(prior.PINS)[0], ["rows", "public_shards", "evaluator_shards"]),
                (7942, HISTORY, ["label_counts", "honest_verdict", "repository_health"]),
            ]
        ],
    )
    historical = json.loads((root / HISTORY).read_text())
    value["repository_health"] = dict(
        current=[], historical=historical["repository_health"], affects_required_checks=False
    )
    return value


def main(argv: list[str] | None = None) -> int:
    """Execute one bounded route and expose only atomically checked bytes."""
    started = time.monotonic()
    progress("start", started)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261001"], default="20261001")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    route = parser.add_mutually_exclusive_group()
    route.add_argument("--fixture-input", type=Path)
    route.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    raw = args.output.absolute().parent / "raw" / args.output.stem
    try:
        if args.cold_replay:
            value = json.loads(args.cold_replay.read_text())
            reduced = reconstruct(value)
            atomic_json(
                raw / "cold_replay.json",
                dict(
                    primary=reference(args.cold_replay),
                    passed=True,
                    reconstruction=reduced,
                    executing_producer=TASK,
                    run_date=args.date,
                ),
            )
            progress("replay_passed", started, len(value["rows"]))
            return 0
        manifest = freeze_commands(raw)
        progress("authenticate_freeze_join", started)
        if args.fixture_input:
            value = build(
                json.loads(args.fixture_input.read_text()), raw, fixture_path=args.fixture_input
            )
        else:
            value = build_live(args.root, raw)
        joined = time.monotonic() - started
        value["validation_command_manifest_path"] = str(raw / "validation_command_manifest.json")
        value["validation_command_manifest_sha256"] = sha256_file(
            Path(value["validation_command_manifest_path"])
        )
        value["resolved_imports"] = {
            name: str(Path(importlib.import_module(name).__file__).resolve()) for name in IMPORTS
        }
        value["code_config_hashes"] = [reference(ROOT / p) for p in OWNED + TESTS] + [
            reference(Path(value["validation_command_manifest_path"])),
            *[reference(Path(p)) for p in value["resolved_imports"].values()],
        ]
        if not args.fixture_input and value["verdict_class"] != "blocked":
            progress("before_required_validation", started)
            apply_validation(value, execute_commands(manifest, raw / "validation_logs"))
            measured = json.loads((raw / "coverage.json").read_text())
            value["coverage_statement_counts"] = {
                p: info["summary"] for p, info in measured["files"].items()
            }
            progress("after_required_validation", started)
        value["primary_resolution_receipt"] = dict(
            path=str(raw / "primary_resolution.json"),
            readers=["conductor_gates.evaluate_gates", "in_process_doc_reconcile.find_artifact"],
        )
        value["terminal_validation_sidecar_path"] = str(raw / "terminal_validation.json")
        value["reproducibility_checksum"] = canonical_hash(
            dict(
                inputs=value["source_artifact_hashes"],
                config=value["code_config_hashes"],
                seed=69055,
            )
        )[7:23]
        value["methodology"] = (
            "Freeze complete-response public bytes; authenticate original spans; independently union all spans; cold-reduce counts."
        )
        value["title"] = "Complete-response human source-support target transport"
        value["duration_s"] = time.monotonic() - started
        value["phase_spans"] = [
            dict(phase="authenticate_freeze_join", start_s=0.0, end_s=joined),
            dict(phase="required_validation", start_s=joined, end_s=value["duration_s"]),
        ]
        value["duration_scope"] = "aggregation and required validation before terminal seal"
        value["field_principles"] = {
            key: "Bind this current producer field to authenticated original bytes and primitive response rows."
            for key in value
            if key != "field_principles"
        }
        value["field_principles"].update(
            response_targets_ready_score="Readiness qualifies annotation transport, not a learned detector.",
            target_definition="Prediction and evaluation must share the complete-response unit.",
            class_counts="Sentences and spans do not create independent source families.",
            inherited_sentence_null="Changing the target does not revise historical sentence findings.",
            verdict_class="External incompleteness blocks; owned failure disqualifies; transport is not benefit.",
        )
        candidate = raw / "terminal_candidate.json"
        atomic_json(candidate, value)
        progress("before_terminal_validation", started)
        report = terminal_check(candidate)
        if not report["passed"]:
            value.update(
                flagged_adversarial=report["flagged_adversarial"],
                verdict_class="disqualified",
                honest_verdict="complete_disqualified_terminal_validation",
                response_targets_ready_score=0,
            )
            value["acceptance_gate_results"].update(validity=False, readiness=False)
            atomic_json(candidate, value)
            report = terminal_check(candidate)
        value["flagged_adversarial"] = report["flagged_adversarial"]
        publication = publish_primary(args.output, value, terminal_check)
        atomic_json(
            Path(value["terminal_validation_sidecar_path"]),
            json.loads(Path(publication["sidecar_path"]).read_text()),
        )
        selected = reader_receipt(
            TASK,
            args.output.absolute().parent,
            field="response_targets_ready_score",
            expected=value["response_targets_ready_score"],
        )
        atomic_json(raw / "primary_resolution.json", selected)
        if not selected["passed"] or selected["gate_sha256"] != publication["primary_sha256"]:
            raise ValueError("primary_resolution")
        progress("published", started, len(value["rows"]))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"[exp7955] rejected {type(error).__name__}:{error}", flush=True)
        return 1
