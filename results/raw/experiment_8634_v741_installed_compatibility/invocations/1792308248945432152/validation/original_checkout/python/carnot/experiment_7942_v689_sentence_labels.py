"""Publish byte-checked human sentence targets from historically exposed sources.

REQ-REPORT-7942. This process transports independent human annotations; it
does not load a model, fit a detector, or measure fresh generalization.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.source_boundary_7892 import EIGHT_ROLES
from carnot.verify import sentence_labels_7942 as labels
from carnot.verify.source_projection import read_jsonl, write_jsonl

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[2]
NAME = "experiment_7942_v689_sentence_labels"
TASK = "exp7942-sentence-labels"
MODEL_SPECS: list[str] = []
OWNED = [
    f"python/carnot/{NAME}.py",
    "python/carnot/verify/sentence_labels_7942.py",
    f"scripts/experiments/{NAME}.py",
]
TESTS = [
    "tests/python/test_sentence_labels_7942.py",
    f"tests/python/test_{NAME}.py",
    "tests/python/test_source_boundary_7852.py",
    "tests/python/test_primary_publication_7928.py",
    "tests/python/test_source_boundary_7892.py",
    "tests/python/test_experiment_7932_v688_qwen_completion.py",
]
INCLUDE = ",".join(str(ROOT / path) for path in OWNED)
REVISION = "c103204b9ce28d6bbad859304bf30de72b8ed8fe"
PINS = {
    "results/experiment_7892_v685_source_boundary.json": "sha256:528a2243693f2543f43bbf4f0b2ce789f2b01ccf14d795260c7e15029230ca14",
    "results/raw/experiment_7423_v651_annotated_protocol/corpus_manifest.json": "sha256:a065b2a64f2926c5bade1fd3495b1cbff23bd3b3c967bd104b973a6b118e30c0",
    "data/ragtruth/source_info.jsonl": "sha256:0dffc26ea9f3c1c3d7c7e8336b56ef1646e3cec876edffcca3c9c624d12d578b",
    "data/ragtruth/response.jsonl": "sha256:e4c2e4ac24fff676d8984cc61c35d791612fadc58015335d97dd632375e18073",
    "data/ragtruth/LICENSE": "sha256:b7fd7d6bdfe0cbba63c63a310914beb4a4acb8bf08da73849219f45385f5b244",
}
IMPORTS = [
    "carnot.verify.sentence_labels_7942",
    "carnot.verify.source_alignment",
    "carnot.verify.source_projection",
    "carnot.reporting.current_work_receipt",
    "carnot.reporting.primary_publication",
    "carnot.reporting.experiment_7303_validation_scope",
    "scripts.conductor_gates",
    "scripts.in_process_doc_reconcile",
    "carnot.reporting.source_boundary_7892",
    "carnot.reporting.source_boundary_7866",
    "carnot.reporting.source_boundary_7880",
    "carnot.verify.evidence_views",
    "carnot.verify.training_runtime",
]


def progress(phase: str, started: float, units: int = 0) -> None:
    """Expose real monotonic progress while the host joins cached evidence."""
    print(
        f"[exp7942] phase={phase} completed_units={units} elapsed_s={time.monotonic() - started:.3f}",
        flush=True,
    )


def reference(path: Path) -> Json:
    """Bind each input and sidecar to its exact local bytes."""
    return dict(path=str(path.absolute()), sha256=sha256_file(path))


def operand(
    upstream: str, path: Path, field: str, expected: Any, observed: Any, op: str = "=="
) -> Json:
    """Keep a missing input distinguishable from a failed numeric threshold."""
    return dict(
        upstream_id=upstream,
        path=str(path),
        hash=sha256_file(path) if path.is_file() else None,
        field=field,
        op=op,
        expected=expected,
        observed=observed,
        passed=expected == observed if op == "==" else observed >= expected,
    )


def authenticate(root: Path) -> tuple[list[Json], Json]:
    """Check pins and the consumed upstream operands without opening labels."""
    checks = [
        operand(
            "exp7892" if "7892" in path else "exp7423",
            root / path,
            "sha256",
            digest,
            sha256_file(root / path) if (root / path).is_file() else None,
        )
        for path, digest in PINS.items()
    ]
    if any(not row["passed"] for row in checks):
        return [row for row in checks if not row["passed"]], {}
    upstream = json.loads((root / next(iter(PINS))).read_text())
    manifest_path = (
        root / "results/raw/experiment_7423_v651_annotated_protocol/corpus_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    for field, expected in dict(
        task_id="exp7892-source-boundary",
        source_boundary_ready_score=1,
        flagged_adversarial=False,
        verdict_class="circular_positive",
    ).items():
        checks.append(
            operand("exp7892", root / next(iter(PINS)), field, expected, upstream.get(field))
        )
    checks.append(operand("exp7423", manifest_path, "commit", REVISION, manifest.get("commit")))
    for item in manifest["asset_receipt"]["files"]:
        checks.append(
            operand(
                "exp7423",
                manifest_path,
                item["path"] + ".sha256",
                PINS["data/ragtruth/" + Path(item["path"]).name],
                item["sha256"],
            )
        )
    consumed = [
        *upstream["public_shards"],
        *upstream["evaluator_shards"],
        dict(path=upstream["cohort_manifest_path"], sha256=upstream["cohort_manifest_sha256"]),
    ]
    for item in consumed:
        path = Path(item["path"])
        checks.append(
            operand(
                "exp7892",
                path,
                "sha256",
                item["sha256"],
                sha256_file(path) if path.is_file() else None,
            )
        )
        checks.append(
            operand(
                "exp7892",
                path,
                "shard_below_50_MiB",
                True,
                path.is_file() and path.stat().st_size < 50 * 1024**2,
            )
        )
    upstream["authenticated_inputs"] = [reference(root / path) for path in PINS] + consumed
    upstream["preconditions"] = checks
    return [row for row in checks if not row["passed"]], upstream


def base(sources: list[Json], failures: list[Json]) -> Json:
    """Emit the full terminal schema even when external custody is unavailable."""
    return dict(
        schema="carnot.exp7942.sentence_labels.v1",
        experiment_id=7942,
        task_id=TASK,
        milestone="2026.09.689",
        run_date="20260930",
        honest_verdict="complete_blocked_annotation_custody",
        verdict_class="blocked",
        flagged_adversarial=False,
        gate_check_summary=failures,
        rows=[],
        **labels.reduce_rows([]),
        acceptance_gate_results=dict(
            validity=False,
            readiness=False,
            probability_quality=None,
            calibration=None,
            decision_benefit=None,
            retention=None,
            efficiency=None,
        ),
        duration_s=0.0,
        phase_spans=[],
        random_seed=68942,
        reproducibility_checksum=None,
        source_artifact_hashes=sources,
        preconditions_checked=[],
        resolved_imports={},
        validation_receipts=[],
        validation_command_manifest_path=None,
        observed_child_commands=[],
        coverage_statement_counts={},
        historical_required_failures=[],
        repository_health={},
        primary_resolution_receipt=None,
        terminal_validation_sidecar_path=None,
        verifier_is_oracle=True,
        claim_scope="exposed_development_annotation_transport_only",
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        execution_venue="host",
        MODEL_SPECS=[],
        model_specs=[],
        target_model=None,
        model_invocation_counts=dict(
            model_loads_attempted=0,
            model_loads_completed=0,
            generation_calls_attempted=0,
            generation_calls_completed=0,
        ),
        trained_head_specs=[],
        sentence_public_manifest=None,
        sentence_evaluator_manifest=None,
        sentence_cohort_manifest=None,
        offset_join_rows=None,
        mutation_rows=[],
        annotation_policy=dict(
            primary="contains a human-annotated source-unsupported span",
            offset_convention="half-open Unicode character offsets into raw response; converted to UTF-8 bytes",
            overlap="at least one non-whitespace character",
            primary_includes_implicit_true=True,
            primary_includes_due_to_null=True,
            sensitivity="implicit_true_excluded_y",
            zero_requires="good quality, complete original annotations, no overlapping unsupported character",
            excluded_qualities=["incorrect_refusal", "truncated"],
            selection="SHA256(family_id UTF-8 || canonical JSON interval || v689-sentence-1)",
        ),
        exposure_status=dict(
            status="exposed_development",
            historically_exposed=True,
            fresh_independent_generalization=False,
        ),
        field_principles={},
    )


def prepare_public(
    public: list[Json], roles: list[Json], raw: Path
) -> tuple[list[Json], list[Json]]:
    """Publish every complete sentence boundary before any labels are opened."""
    frozen, boundaries = labels.freeze(public)
    write_jsonl(raw / "predictors.jsonl", frozen)
    atomic_json(
        raw / "cohort.json",
        dict(
            boundaries=boundaries,
            roles=roles,
            selection_salt=labels.SALT.decode(),
            predictor_sha256=canonical_hash(frozen),
        ),
    )
    return frozen, boundaries


def finish(frozen: list[Json], boundaries: list[Json], data: Json, raw: Path, value: Json) -> Json:
    """Publish evaluator-only targets and reconstruct label-independent features."""
    started = time.monotonic()
    progress("before_evaluator_join", started)
    rows, offsets = labels.join(
        frozen, boundaries, data["evaluators"], data["roles"], data["responses"], data["sources"]
    )
    write_jsonl(raw / "evaluators.jsonl", rows)
    write_jsonl(raw / "offset_joins.jsonl", offsets)
    feature_rows = [
        dict(family_id=row["family_id"], features=labels.features(row)) for row in frozen
    ]
    write_jsonl(raw / "features.jsonl", feature_rows)
    progress("after_evaluator_join", started, len(rows))
    metadata = {
        key for row in data["evaluators"] + data["responses"] + data["roles"] for key in row
    } - labels.PUBLIC_KEYS
    metadata.update(
        "annotation." + key for row in data["responses"] for span in row["labels"] for key in span
    )
    mutations = []
    for number, key in enumerate(sorted(metadata), 1):
        changed, changed_bounds = labels.freeze(
            [labels.public_only({**labels.public_only(row), key: "mutated"}) for row in frozen]
        )
        changed_features = [
            dict(family_id=row["family_id"], features=labels.features(row)) for row in changed
        ]
        mutations.append(
            dict(
                field=key,
                predictor_sha256=canonical_hash(changed),
                selection_order_sha256=canonical_hash([r["sentence_interval"] for r in changed]),
                features_sha256=canonical_hash(changed_features),
                passed=changed == frozen
                and changed_bounds == boundaries
                and changed_features == feature_rows,
            )
        )
        progress("metadata_mutation", started, number)
    value.update(labels.reduce_rows(rows))
    value.update(
        rows=rows,
        sentence_public_manifest=reference(raw / "predictors.jsonl"),
        sentence_evaluator_manifest=reference(raw / "evaluators.jsonl"),
        sentence_cohort_manifest=reference(raw / "cohort.json"),
        offset_join_rows=reference(raw / "offset_joins.jsonl"),
        feature_manifest=reference(raw / "features.jsonl"),
        mutation_rows=mutations,
    )
    value["gate_check_summary"] = [
        operand(
            TASK,
            raw / "evaluators.jsonl",
            item["field"],
            item["expected"],
            item["observed"],
            item["op"],
        )
        for item in value["failed_operands"]
    ]
    value.update(honest_verdict="complete_null_sentence_annotation_transport", verdict_class="null")
    value["acceptance_gate_results"].update(
        validity=True, readiness=bool(value["sentence_labels_ready_score"])
    )
    intended = {row["family_id"] for row in data["roles"] if row["role"] == "evaluation"}
    private_evaluators = [row for row in data["evaluators"] if row["family_id"] in intended]
    response_ids = {row["response_id"] for row in private_evaluators}
    private_responses = [row for row in data["responses"] if row["id"] in response_ids]
    source_ids = {row["source_id"] for row in private_responses}
    atomic_json(
        raw / "validation_input.json",
        dict(
            public=[labels.public_only(row) for row in frozen if row["family_id"] in intended],
            roles=[row for row in data["roles"] if row["family_id"] in intended],
            evaluators=private_evaluators,
            responses=private_responses,
            sources=[row for row in data["sources"] if row["source_id"] in source_ids],
        ),
    )
    return value


def build_fixture(path: Path, raw: Path) -> Json:
    """Private human-like fixtures qualify mechanics but provide no scientific benefit."""
    data = json.loads(path.read_text())
    frozen, boundaries = prepare_public(data["public"], data["roles"], raw)
    value = finish(frozen, boundaries, data, raw, base([reference(path)], []))
    value.update(
        fixture_input=reference(path),
        sentence_labels_ready_score=0,
        honest_verdict="complete_circular_positive_fixture_transport",
        verdict_class="circular_positive",
    )
    value["acceptance_gate_results"]["readiness"] = False
    return value


def build_live(root: Path, raw: Path) -> Json:
    """Authenticate cached originals and keep all original role exclusions."""
    failures, upstream = authenticate(root)
    value = base(upstream.get("authenticated_inputs", []), failures)
    if failures:
        value["inference_substrate_class"] = "blocked_no_run"
        value["planned_inference_substrate_class"] = "no_model_load"
        return value
    public = [row for item in upstream["public_shards"] for row in read_jsonl(Path(item["path"]))]
    roles = upstream["rows"]
    if len(public) != 640 or upstream["role_counts"] != EIGHT_ROLES:
        raise ValueError("original_role_roster")
    frozen, boundaries = prepare_public(public, roles, raw)
    progress("boundaries_published_before_annotations", time.monotonic(), len(frozen))
    data = dict(
        roles=roles,
        evaluators=[
            row for item in upstream["evaluator_shards"] for row in read_jsonl(Path(item["path"]))
        ],
        responses=read_jsonl(root / "data/ragtruth/response.jsonl"),
        sources=read_jsonl(root / "data/ragtruth/source_info.jsonl"),
    )
    value.update(
        preconditions_checked=upstream["preconditions"],
        historical_required_failures=upstream["historical_required_failures"],
        custody_root=str(root),
    )
    return finish(frozen, boundaries, data, raw, value)


def checked_reference(item: Json) -> Path:
    """A changed primitive shard invalidates all derived claims."""
    path = Path(item["path"])
    if sha256_file(path) != item["sha256"]:
        raise ValueError("hash_drift:" + str(path))
    return path


def reconstruct(value: Json) -> Json:
    """Cold-rebuild targets from original annotations rather than saved labels."""
    for item in value.get("code_config_hashes", []) + value["source_artifact_hashes"]:
        checked_reference(item)
    if value["sentence_public_manifest"] is None:
        if value["sentence_labels_ready_score"]:
            raise ValueError("unsafe_readiness")
        return labels.reduce_rows([])
    frozen = read_jsonl(checked_reference(value["sentence_public_manifest"]))
    cohort = json.loads(checked_reference(value["sentence_cohort_manifest"]).read_text())
    rows = read_jsonl(checked_reference(value["sentence_evaluator_manifest"]))
    offsets = read_jsonl(checked_reference(value["offset_join_rows"]))
    saved_features = read_jsonl(checked_reference(value["feature_manifest"]))
    if value.get("fixture_input"):
        data = json.loads(checked_reference(value["fixture_input"]).read_text())
    else:
        root = Path(value["custody_root"])
        failures, upstream = authenticate(root)
        if failures:
            raise ValueError("cold_custody")
        data = dict(
            roles=upstream["rows"],
            evaluators=[
                row
                for item in upstream["evaluator_shards"]
                for row in read_jsonl(Path(item["path"]))
            ],
            responses=read_jsonl(root / "data/ragtruth/response.jsonl"),
            sources=read_jsonl(root / "data/ragtruth/source_info.jsonl"),
            public=[
                row for item in upstream["public_shards"] for row in read_jsonl(Path(item["path"]))
            ],
        )
    original, boundaries = labels.freeze(data["public"])
    if original != frozen or cohort["boundaries"] != boundaries or cohort["roles"] != data["roles"]:
        raise ValueError("public_reconstruction_drift")
    expected_rows, expected_offsets = labels.join(
        original, boundaries, data["evaluators"], data["roles"], data["responses"], data["sources"]
    )
    if rows != expected_rows or offsets != expected_offsets:
        raise ValueError("evaluator_drift")
    if saved_features != [
        dict(family_id=row["family_id"], features=labels.features(row)) for row in original
    ]:
        raise ValueError("feature_drift")
    reduced = labels.reduce_rows(rows)
    if value["sentence_labels_ready_score"] and (
        value.get("fixture_input")
        or not reduced["sentence_labels_ready_score"]
        or value["verdict_class"] in {"blocked", "disqualified"}
        or value["flagged_adversarial"]
        or not all(row["passed"] for row in value["mutation_rows"])
        or any(
            not row["passed"] for row in value["validation_receipts"] if row.get("required", True)
        )
    ):
        raise ValueError("unsafe_readiness")
    for key in (
        "label_counts",
        "source_cluster_counts",
        "sample_size_budget",
        "role_counts",
        "rows_sha256",
    ):
        if value[key] != reduced[key]:
            raise ValueError("reduction_drift:" + key)
    return reduced


def freeze_commands(raw: Path) -> Json:
    """Declare exact commands and failures before any measured validation starts."""
    raw.mkdir(parents=True, exist_ok=True)
    py, cov = str(ROOT / ".venv/bin/python"), str(ROOT / ".venv/bin/coverage")
    coverage_workspace = Path(tempfile.mkdtemp(prefix="carnot-7942-coverage-"))
    coverage_file = str(coverage_workspace / ".coverage")
    common = ["-n", "0", "-o", "addopts=", "--no-cov"]
    fixture = str(raw / "private-e2e016/fixture.json")
    historical = str(ROOT / "scripts/experiments/experiment_7868_v683_intervention_protocol.py")
    commands: list[Json] = []

    def add(
        name: str,
        argv: list[str],
        expected: int = 0,
        reason: str | None = None,
        required: bool = True,
        deadline: int = 180,
    ) -> None:
        commands.append(
            dict(
                name=name,
                argv=argv,
                expected_exit=expected,
                failure_reason=reason,
                required=required,
                deadline_s=deadline,
            )
        )

    add(
        "unit_consumer_e2e015",
        [
            cov,
            "run",
            "--parallel-mode",
            "--data-file=" + coverage_file,
            "--include=" + INCLUDE,
            "-m",
            "pytest",
            *common,
            *TESTS,
            "-q",
        ],
    )
    cli = str(ROOT / ("scripts/experiments/" + NAME + ".py"))

    def private_cli(
        name: str, arguments: list[str], expected: int = 0, reason: str | None = None
    ) -> None:
        add(
            name,
            [
                cov,
                "run",
                "--parallel-mode",
                "--data-file=" + coverage_file,
                "--include=" + INCLUDE,
                cli,
                *arguments,
            ],
            expected,
            reason,
        )

    success = raw / "private-success" / (NAME + ".json")
    private_cli(
        "real_fixture_cli",
        [
            "--date",
            "20260930",
            "--fixture-input",
            str(raw / "validation_input.json"),
            "--output",
            str(success),
        ],
    )
    private_cli(
        "cold_fixture_cli",
        [
            "--date",
            "20260930",
            "--cold-replay",
            str(success),
            "--output",
            str(raw / "private-replay" / (NAME + ".json")),
        ],
    )
    private_cli(
        "blocked_cli",
        [
            "--date",
            "20260930",
            "--root",
            str(raw / "absent-prerequisite"),
            "--output",
            str(raw / "private-blocked" / (NAME + ".json")),
        ],
    )
    private_cli(
        "negative_date_cli",
        ["--date", "20260929", "--output", str(raw / "private-negative" / (NAME + ".json"))],
        2,
        "invalid choice",
    )
    add(
        "coverage_combine",
        [cov, "combine", "--keep", "--data-file=" + coverage_file, str(coverage_workspace)],
    )
    add(
        "coverage_json",
        [
            cov,
            "json",
            "--data-file=" + coverage_file,
            "--include=" + INCLUDE,
            "-o",
            str(raw / "coverage.json"),
        ],
    )
    add(
        "coverage_report",
        [
            cov,
            "report",
            "--data-file=" + coverage_file,
            "--include=" + INCLUDE,
            "--show-missing",
            "--fail-under=100",
        ],
    )
    add("ruff_check", [str(ROOT / ".venv/bin/ruff"), "check", *OWNED, *TESTS[:2]])
    add("ruff_format", [str(ROOT / ".venv/bin/ruff"), "format", "--check", *OWNED, *TESTS[:2]])
    add("strict_mypy", [str(ROOT / ".venv/bin/mypy"), "--strict", "--follow-imports=skip", *OWNED])
    add("spec_coverage", [py, "scripts/check_spec_coverage.py", *TESTS])
    add("e2e016_fixture", [py, "-u", historical, "--date", "20260929", "--fixture-e2e", fixture])
    add("e2e016_cold", [py, "-u", historical, "--date", "20260929", "--cold-replay", fixture])
    add(
        "repository_health",
        [str(ROOT / ".venv/bin/pytest"), "tests/python", "-q"],
        required=False,
        deadline=180,
    )
    manifest = dict(
        affected_files=OWNED,
        explicit_tests=TESTS,
        transitive_consumers=IMPORTS,
        coverage_includes=INCLUDE,
        commands=commands,
        current_date="20260930",
        historical_e2e016_date="20260929",
        coverage_file=coverage_file,
        private_cli_routes="Each test route owns a separate tmp_path publication directory.",
    )
    atomic_json(raw / "validation_command_manifest.json", manifest)
    return manifest


def execute_commands(manifest: Json, logs: Path) -> list[Json]:
    """Supervise all owned children with a heartbeat and retain actual exits."""
    specs = [
        CommandSpec(item["name"], tuple(item["argv"]), "frozen_exp7942_scope", item["deadline_s"])
        for item in manifest["commands"]
    ]
    receipts = run_commands(
        ROOT,
        specs,
        log_dir=logs,
        heartbeat_s=10,
        extra_env=dict(CARNOT_7942_COVERAGE_FILE=manifest["coverage_file"], JAX_PLATFORMS="cpu"),
    )
    for receipt, spec in zip(receipts, manifest["commands"], strict=True):
        receipt.update(
            expected_exit=spec["expected_exit"],
            required=spec["required"],
            deadline_s=spec["deadline_s"],
        )
        receipt["passed"] = receipt["exit_code"] == spec["expected_exit"] and not receipt.get(
            "timed_out"
        )
        if spec["failure_reason"]:
            receipt["passed"] = (
                receipt["passed"] and spec["failure_reason"] in receipt["output_tail"]
            )
    return receipts


def apply_validation(value: Json, receipts: list[Json]) -> None:
    """Owned failures zero readiness while historical repository debt stays visible."""
    value["validation_receipts"] = receipts
    value["observed_child_commands"] = [r.get("command_argv", []) for r in receipts]
    value["repository_health"] = dict(
        affects_required_checks=False,
        current=[r for r in receipts if not r.get("required", True)],
        historical_required_failures=value["historical_required_failures"],
    )
    if any(not r["passed"] for r in receipts if r.get("required", True)):
        value.update(
            verdict_class="disqualified",
            honest_verdict="complete_disqualified_required_validation",
            sentence_labels_ready_score=0,
        )
        value["acceptance_gate_results"].update(validity=False, readiness=False)


def terminal_check(candidate: Path) -> Json:
    """Run both final validators against exact bytes after independent cold reduction."""
    value = json.loads(candidate.read_text())
    reconstruct(value)
    py = str(ROOT / ".venv/bin/python")
    commands = [
        CommandSpec(
            "adversarial",
            (py, "-u", "scripts/adversarial_verify.py", "--json", str(candidate)),
            "terminal_candidate",
            60,
        ),
        CommandSpec(
            "strict_rows",
            (py, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "terminal_candidate",
            60,
        ),
    ]
    receipts = run_commands(
        ROOT, commands, log_dir=candidate.parent / "terminal_logs", heartbeat_s=10
    )
    flagged = any(r["name"] == "adversarial" and r["exit_code"] != 0 for r in receipts)
    return dict(
        passed=all(r["passed"] for r in receipts),
        flagged_adversarial=flagged,
        candidate_sha256=sha256_file(candidate),
        receipts=receipts,
    )


def main(argv: list[str] | None = None) -> int:
    """Run only the selected owned route and atomically expose checked bytes."""
    started = time.monotonic()
    progress("start", started)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20260930"], default="20260930")
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
        progress("inputs_authentication", started)
        if args.fixture_input:
            value = build_fixture(args.fixture_input, raw)
        else:
            try:
                value = build_live(args.root, raw)
            except (OSError, ValueError, KeyError, TypeError) as error:
                value = base(
                    [],
                    [
                        operand(
                            "exp7423_exp7892",
                            args.root / "data/ragtruth/response.jsonl",
                            "annotation_custody",
                            "exact_complete_join",
                            f"{type(error).__name__}:{error}",
                        )
                    ],
                )
                value.update(
                    inference_substrate_class="blocked_no_run",
                    planned_inference_substrate_class="no_model_load",
                )
        join_end = time.monotonic() - started
        value["validation_command_manifest_path"] = str(raw / "validation_command_manifest.json")
        value["validation_command_manifest_sha256"] = sha256_file(
            Path(value["validation_command_manifest_path"])
        )
        value["resolved_imports"] = {
            name: str(Path(importlib.import_module(name).__file__).resolve()) for name in IMPORTS
        }
        value["code_config_hashes"] = [reference(ROOT / path) for path in OWNED + TESTS] + [
            reference(Path(value["validation_command_manifest_path"])),
            *[reference(Path(path)) for path in value["resolved_imports"].values()],
        ]
        if not args.fixture_input and value["verdict_class"] != "blocked":
            progress("before_required_validation", started)
            receipts = execute_commands(manifest, raw / "validation_logs")
            apply_validation(value, receipts)
            coverage_path = raw / "coverage.json"
            if coverage_path.is_file():
                measured = json.loads(coverage_path.read_text())
                value["coverage_statement_counts"] = {
                    path: info["summary"] for path, info in measured["files"].items()
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
                seed=68942,
            )
        )[7:23]
        value["methodology"] = (
            "Authenticate pinned human spans, freeze public sentence queries, join exact bytes, cold-reduce cohort operands."
        )
        value["title"] = "Human source-support labels at the queried sentence boundary"
        value["duration_s"] = time.monotonic() - started
        value["duration_scope"] = (
            "annotation_transport_and_required_validation_before_terminal_seal"
        )
        value["phase_spans"] = [
            dict(phase="authenticate_freeze_join", start_s=0.0, end_s=join_end),
            dict(phase="required_validation", start_s=join_end, end_s=value["duration_s"]),
        ]
        value["field_principles"] = {
            key: "Bind this current claim to authenticated bytes and primitive rows."
            for key in value
            if key != "field_principles"
        }
        value["field_principles"].update(
            sentence_labels_ready_score="Custody, exact offsets, separation, cohort counts and owned checks are all required.",
            label_counts="Count one independently sourced human target per intended queried evaluation family.",
            exposure_status="Historically exposed groups cannot establish fresh independent generalization.",
            inference_substrate="Aggregation invokes no pretrained model; cited generator names belong to the corpus.",
            verdict_class="External incompleteness blocks; owned failure disqualifies; transport is not detector benefit.",
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
                sentence_labels_ready_score=0,
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
            field="sentence_labels_ready_score",
            expected=value["sentence_labels_ready_score"],
        )
        atomic_json(raw / "primary_resolution.json", selected)
        if not selected["passed"] or selected["gate_sha256"] != publication["primary_sha256"]:
            raise ValueError("primary_resolution")
        progress("published", started, len(value["rows"]))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"[exp7942] rejected {type(error).__name__}:{error}", flush=True)
        return 1
