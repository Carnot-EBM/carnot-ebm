"""Join public lexical features and reserve development sources before labels.

REQ-REPORT-7980. This experiment prepares observable information. It makes
no claim that source alignment improves a decision policy.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import evidence_features_custody_7980 as c
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import reader_receipt
from carnot.reporting.source_boundary_7892 import EIGHT_ROLES
from carnot.verify import evidence_features_7980 as f
from carnot.verify import response_role_targets_7968 as roles
from carnot.verify import response_targets_7955 as targets

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[2]
NAME = "experiment_7980_v692_evidence_features"
TASK = "exp7980-evidence-features"
MODEL_SPECS: list[str] = []
OWNED = [
    f"python/carnot/{NAME}.py",
    "python/carnot/verify/evidence_features_7980.py",
    "python/carnot/reporting/evidence_features_custody_7980.py",
    f"scripts/experiments/{NAME}.py",
]
TESTS = ["tests/python/test_evidence_features_7980.py", f"tests/python/test_{NAME}.py"]
CONSUMERS = [
    "tests/python/test_response_role_targets_7968.py",
    "tests/python/test_response_targets_7955.py",
    "tests/python/test_source_projection_7838.py",
    "tests/python/test_primary_publication_7928.py",
]
E2ES = [
    "tests/python/test_source_boundary_7852.py",
    "tests/python/test_experiment_7942_v689_sentence_labels.py",
]


def progress(phase: str) -> None:
    """Flushed phase boundaries distinguish measured CPU work from silence."""
    print(f"[exp7980] phase={phase}", flush=True)


def child(name: str, argv: list[str], raw: Path) -> Json:
    """Reuse the bounded supervisor and archive logs only after child exit."""
    config = os.environ.get("CARNOT_7980_COVERAGE_CONFIG")
    if config and str(ROOT / f"scripts/experiments/{NAME}.py") in argv:
        argv[1:2] = ["-m", "coverage", "run", "--rcfile=" + config]
    return run_commands(
        ROOT,
        [CommandSpec(name, tuple(argv), "owned_child", 900)],
        log_dir=raw / "children" / name,
        heartbeat_s=30,
    )[0]


def base(failures: list[Json]) -> Json:
    """Declare actual zero calls and keep unavailable measurements explicitly empty."""
    return dict(
        experiment_id=7980,
        task_id=TASK,
        milestone="2026.10.692",
        run_date="20261001",
        execution_date="20261001",
        honest_verdict="complete_blocked_evidence_inputs"
        if failures
        else "complete_null_evidence_preparation",
        verdict_class="blocked" if failures else "null",
        gate_check_summary=failures,
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_specs=[],
        trained_head_specs=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        duration_s=0.0,
        phase_spans=[],
        random_seed=69280,
        reproducibility_checksum=None,
        cited_upstream_artifacts=[],
        code_config_hashes=[],
        raw_shard_hashes=[],
        preconditions_checked=[],
        validation_command_manifest_path=None,
        validation_receipts=[],
        coverage_statement_counts={},
        rows=[],
        sample_size_budget=dict(
            unit="original_source_response",
            intended=640,
            eligible=0,
            started=0,
            completed=0,
            failed=0,
            censored=0,
            excluded=0,
            independent=0,
        ),
        verifier_is_oracle=False,
        claim_scope="Lexical feature preparation and reserved development evidence; no benefit or pretraining noncontamination claim.",
        acceptance_gate_results=dict(validity=False, readiness=False, decision_benefit=False),
        flagged_adversarial=False,
        terminal_validation_sidecar_path=None,
        feature_views_ready_score=0,
        fresh_panel_ready_score=0,
        public_role_manifests={},
        evaluator_role_manifests={},
        fresh_public_manifest=None,
        fresh_evaluator_manifest=None,
        feature_schema=list(f.FEATURES),
        predicate_bank={},
        exposure_audit={},
        label_access_events=[],
        historical_required_failures=[],
        generator_weights_changed=False,
        production_defaults_changed=False,
        repository_health={},
    )


def measure(plan: Json, root: Path, raw: Path) -> Json:
    """Seal every public roster and feature before fitting or annotation access."""
    value = base([])
    public, by_role, original_refs = [], {}, plan["upstream"][7968]
    normalized_roles, source_roles = {}, {}
    for role, count in EIGHT_ROLES.items():
        view = json.loads(c.checked(original_refs["public_role_manifests"][role]).read_text())
        rows = view["request_rows"]
        if (
            view["role"] != role
            or len(rows) != count
            or targets.freeze(rows)[1] != view["boundaries"]
        ):
            raise ValueError("original_role_roster")
        for row in rows:
            key = f.normalized(bytes.fromhex(row["source_bytes"]))
            if key in normalized_roles and normalized_roles[key] != role:
                raise ValueError("cross_role_normalized_source")
            normalized_roles[key] = role
        public.extend(rows)
        by_role[role] = rows
    if len({r["family_id"] for r in public}) != 640:
        raise ValueError("original_family_roster")
    progress("freeze_public_and_reserve")
    ids, hashes, audit = c.exposure(root, raw)
    train_path = raw / "public/train_projection.json"
    train_receipt = child(
        "train_projection",
        [
            sys.executable,
            "-u",
            str(ROOT / f"scripts/experiments/{NAME}.py"),
            "--train-project",
            "--root",
            str(root),
            "--feature-output",
            str(train_path),
        ],
        raw,
    )
    if not train_receipt["passed"]:
        raise ValueError("train_projection_failed")
    projected = json.loads(train_path.read_text())
    sources, responses = projected["sources"], projected["responses"]
    roster, fresh = f.reserve(sources, responses, hashes | set(normalized_roles), ids)
    audit.update(
        reserved_count=len(roster),
        intended=96,
        shortfall=96 - len(roster),
        exposure_status="reserved_development_evidence",
        reservation_seed="seed69280",
    )
    value["exposure_audit"] = audit
    all_path, feature_path = raw / "public/all.json", raw / "public/features.json"
    fresh_path, fresh_features, roster_path = (
        raw / "public/fresh.json",
        raw / "public/fresh_features.json",
        raw / "public/fresh_roster.json",
    )
    for path, data in [
        (all_path, dict(request_rows=public)),
        (fresh_path, dict(request_rows=fresh)),
        (roster_path, dict(rows=roster)),
    ]:
        atomic_json(path, data)
    cli = str(ROOT / f"scripts/experiments/{NAME}.py")
    receipts = [
        child(
            "public_original",
            [
                sys.executable,
                "-u",
                cli,
                "--public-extract",
                str(all_path),
                "--feature-output",
                str(feature_path),
            ],
            raw,
        ),
        child(
            "public_reserved",
            [
                sys.executable,
                "-u",
                cli,
                "--public-extract",
                str(fresh_path),
                "--feature-output",
                str(fresh_features),
            ],
            raw,
        ),
    ]
    if not all(r["passed"] for r in receipts):
        raise ValueError("public_child_failed")
    features = json.loads(feature_path.read_text())["rows"]
    seal_path = raw / "public/seal.json"
    atomic_json(
        seal_path,
        dict(
            public=c.reference(fresh_path),
            features=c.reference(fresh_features),
            roster=c.reference(roster_path),
            original_public=c.reference(all_path),
            original_features=c.reference(feature_path),
        ),
    )
    value["public_seal"] = c.reference(seal_path)
    progress("fitting_labels_after_public_seal")
    fit_labels = []
    for role in EIGHT_ROLES:
        item = original_refs["evaluator_role_manifests"][role]
        value["evaluator_role_manifests"][role] = item
        evaluator = (
            roles.read_view(c.checked(item), item["sha256"], role, "fitting", {})
            if role == "fit"
            else json.loads(c.checked(item).read_text())
        )
        indexed = {r["family_id"]: r for r in evaluator["rows"]}
        if set(indexed) != {r["family_id"] for r in by_role[role]} or evaluator["role"] != role:
            raise ValueError("evaluator_role_roster")
        for row in by_role[role]:
            label = indexed[row["family_id"]]
            sid = label["source_id"]
            if sid in source_roles and source_roles[sid] != role:
                raise ValueError("cross_role_source_id")
            source_roles[sid] = role
        if role == "fit":
            fit_labels = evaluator["rows"]
        feature_ids = {r["family_id"] for r in by_role[role]}
        role_features = [r for r in features if r["family_id"] in feature_ids]
        path = raw / "public" / (role + ".json")
        atomic_json(
            path,
            dict(
                role=role,
                request_rows=by_role[role],
                features=role_features,
                boundaries=view["boundaries"]
                if role == "retention"
                else targets.freeze(by_role[role])[1],
            ),
        )
        value["public_role_manifests"][role] = dict(c.reference(path), count=len(role_features))
        for feature in role_features:
            inherited = indexed[feature["family_id"]]
            value["rows"].append(
                dict(
                    family_id=feature["family_id"],
                    role=role,
                    source_id=inherited["source_id"],
                    source_cluster_id=inherited["source_cluster_id"],
                    feature_hash=feature["feature_hash"],
                    feature_values=feature["values"],
                    abstention=feature["abstention"],
                    status=inherited["status"],
                    exclusion_reason=inherited["exclusion_reason"],
                    arm="public_lexical_features",
                    seed=69280,
                )
            )
    q = c.q_rows(plan["upstream"], public)
    q_path = raw / "public/q.json"
    atomic_json(q_path, dict(rows=q))
    q_index = {r["family_id"]: r["q"] for r in q}
    fit = [
        dict(family_id=r["family_id"], y=r["y"], q=q_index[r["family_id"]], status=r["status"])
        for r in fit_labels
    ]
    heads_ref = plan["upstream"][7972]["heads_seal"]
    heads = json.loads(c.checked(heads_ref).read_text())["heads"]["gibbs"]
    bank = f.predicates(
        [r for r in features if r["family_id"] in {l["family_id"] for l in fit}], fit, heads
    )
    fit_path = raw / "evaluator/fit_selection.json"
    atomic_json(fit_path, dict(rows=fit))
    fit_path.parent.chmod(0o700)
    fit_path.chmod(0o600)
    value["predicate_bank"] = bank
    value["fitting_selection_inputs"] = dict(fit=c.reference(fit_path), heads=heads_ref)
    value["label_access_events"] = [
        dict(
            role="original_custody_and_fit_selection",
            public_seal_sha256=sha256_file(seal_path),
            purpose="Original roles authenticated; only fit labels select predicates.",
        )
    ]
    progress("reserved_evaluator_join_after_seal")
    evaluator_path = raw / "evaluator/fresh.json"
    receipts.append(
        child(
            "reserved_join",
            [
                sys.executable,
                "-u",
                cli,
                "--reserved-join",
                str(seal_path),
                "--root",
                str(root),
                "--feature-output",
                str(evaluator_path),
            ],
            raw,
        )
    )
    if not receipts[-1]["passed"]:
        raise ValueError("reserved_join_failed")
    value["label_access_events"].append(
        dict(
            role="reserved_development",
            actor="evaluator_child",
            public_seal_sha256=sha256_file(seal_path),
            labels_exported_to_parent=False,
            access_policy="exp7983_after_policy_seal_only",
        )
    )
    value.update(
        fresh_public_manifest=c.reference(fresh_path),
        fresh_evaluator_manifest=c.reference(evaluator_path),
        public_features=c.reference(feature_path),
        original_public=c.reference(all_path),
        fresh_features=c.reference(fresh_features),
        fresh_roster=c.reference(roster_path),
        q_manifest=c.reference(q_path),
        child_receipts=[train_receipt, *receipts],
        feature_views_ready_score=1,
        fresh_panel_ready_score=int(len(roster) == 96 and audit["complete_history_custody"]),
        acceptance_gate_results=dict(validity=True, readiness=True, decision_benefit=False),
    )
    complete = sum(r["status"] == "completed" for r in value["rows"])
    value["sample_size_budget"].update(
        eligible=complete,
        started=640,
        completed=complete,
        excluded=640 - complete,
        independent=len(normalized_roles),
    )
    value["fresh_sample_size_budget"] = dict(
        intended=96,
        selected=len(roster),
        started=len(roster),
        completed=len(roster),
        failed=0,
        censored=0,
        excluded=96 - len(roster),
        independent=len(roster),
        eligible=None,
        eligibility_status="sealed_for_exp7983",
    )
    value["raw_shard_hashes"] = [
        c.reference(root / "data/ragtruth" / name)
        for name in ("source_info.jsonl", "response.jsonl")
    ]
    # A second child sees public bytes after changed evaluator metadata is archived.
    changed = [dict(r, y=999, role="changed", annotation_rows=["changed"]) for r in fit]
    metadata_path = raw / "evaluator/changed_metadata.json"
    atomic_json(metadata_path, dict(rows=changed))
    cold_path = raw / "public/cold_features.json"
    cold_receipt = child(
        "metadata_cold_recompute",
        [
            sys.executable,
            "-u",
            cli,
            "--public-extract",
            str(all_path),
            "--feature-output",
            str(cold_path),
        ],
        raw,
    )
    recomputed = json.loads(cold_path.read_text())["rows"] if cold_receipt["passed"] else []
    if recomputed != features or len(changed) != len(fit):
        raise ValueError("metadata_invariance_failed")
    value["metadata_invariance_receipt"] = dict(
        before=canonical_hash(features),
        after=canonical_hash(recomputed),
        changed_fields=["y", "role", "annotation_rows"],
        changed_metadata=c.reference(metadata_path),
        cold_subprocess=True,
        child_receipt=cold_receipt,
        passed=True,
    )
    return value


def replay(value: Json) -> None:
    """Validate current sidecars and cold features without opening fresh labels."""
    for item in (
        value.get("cited_upstream_artifacts", [])
        + value.get("code_config_hashes", [])
        + value.get("raw_shard_hashes", [])
    ):
        c.checked(item)
    if not value.get("public_features"):
        if value["feature_views_ready_score"]:
            raise ValueError("unsafe_readiness")
        return
    f.replay_features(c.checked(value["original_public"]), c.checked(value["public_features"]))
    f.replay_features(c.checked(value["fresh_public_manifest"]), c.checked(value["fresh_features"]))
    c.checked(value["fresh_evaluator_manifest"])
    c.checked(value["fresh_roster"])
    c.checked(value["public_seal"])
    features = json.loads(c.checked(value["public_features"]).read_text())["rows"]
    indexed = {r["family_id"]: r for r in features}
    expected_rows = []
    for role, item in value["evaluator_role_manifests"].items():
        labels = json.loads(c.checked(item).read_text())["rows"]
        for label in labels:
            feature = indexed[label["family_id"]]
            expected_rows.append(
                dict(
                    family_id=label["family_id"],
                    role=role,
                    source_id=label["source_id"],
                    source_cluster_id=label["source_cluster_id"],
                    feature_hash=feature["feature_hash"],
                    feature_values=feature["values"],
                    abstention=feature["abstention"],
                    status=label["status"],
                    exclusion_reason=label["exclusion_reason"],
                    arm="public_lexical_features",
                    seed=69280,
                )
            )
    if {r["family_id"]: r["role"] for r in expected_rows} != {
        r["family_id"]: r["role"] for r in value["rows"]
    }:
        raise ValueError("role_drift")
    for row in value["rows"]:
        if (
            row["feature_hash"] != indexed[row["family_id"]]["feature_hash"]
            or row["feature_values"] != indexed[row["family_id"]]["values"]
        ):
            raise ValueError("rows_drift")
    if sorted(expected_rows, key=lambda r: r["family_id"]) != sorted(
        value["rows"], key=lambda r: r["family_id"]
    ):
        raise ValueError("custody_rows_drift")
    selection = value["fitting_selection_inputs"]
    fit = json.loads(c.checked(selection["fit"]).read_text())["rows"]
    heads = json.loads(c.checked(selection["heads"]).read_text())["heads"]["gibbs"]
    fit_ids = {r["family_id"] for r in fit}
    if (
        f.predicates([r for r in features if r["family_id"] in fit_ids], fit, heads)
        != value["predicate_bank"]
    ):
        raise ValueError("predicate_drift")
    for item in value["public_role_manifests"].values():
        c.checked(item)
    c.checked(value["q_manifest"])


def freeze_commands(raw: Path, scratch: Path) -> Json:
    """Declare owned checks and separate broad health before any measurement."""
    py, pytest = str(ROOT / ".venv/bin/python"), str(ROOT / ".venv/bin/pytest")
    config = scratch / "coverage.ini"
    config.write_text(
        "[run]\nparallel = True\ndata_file = "
        + str(scratch / ".coverage")
        + "\ninclude =\n"
        + "\n".join("    " + str(ROOT / p) for p in OWNED)
        + "\n"
    )
    common = ["-n0", "-o", "addopts=", "--no-cov", "-q", "--basetemp=" + str(scratch / "pytest")]
    commands = [
        CommandSpec(
            "changed_statement_coverage",
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
        CommandSpec("affected_consumers", (pytest, *common, *CONSUMERS), "owned", 300),
        CommandSpec("E2E_015_019", (pytest, *common, *E2ES), "owned", 300),
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
            "coverage_100_percent",
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
            (str(ROOT / ".venv/bin/mypy"), "--strict", "--follow-imports=silent", *OWNED[:3]),
            "owned",
            120,
        ),
        CommandSpec(
            "scoped_spec_coverage",
            (py, "scripts/check_spec_coverage.py", *TESTS, *CONSUMERS, *E2ES),
            "owned",
            60,
        ),
        CommandSpec("repository_health", (pytest, "tests/python", "-q"), "repository_health", 600),
    ]
    manifest = dict(
        commands=[asdict(r) for r in commands],
        coverage_config=str(config),
        coverage_json=str(scratch / "coverage.json"),
        coverage_includes=OWNED,
        artifact_guard_enabled=True,
        frozen_before_science=True,
        terminal_checks=[
            "external_cold_replay_without_PYTHONPATH",
            "adversarial_verify_--json",
            "verdict_row_consistency_--strict",
            "both_live_primary_readers",
        ],
    )
    atomic_json(raw / "validation_commands.json", manifest)
    return manifest


def apply_validation(value: Json, receipts: list[Json]) -> None:
    """Never turn a failed owned check into a qualified science result."""
    value["validation_receipts"] = receipts
    owned = [r for r in receipts if r["name"] != "repository_health"]
    value["repository_health"] = next((r for r in receipts if r["name"] == "repository_health"), {})
    failed = [r for r in owned if not r["passed"]]
    if failed:
        value.update(
            honest_verdict="complete_disqualified_required_validation",
            verdict_class="disqualified",
            feature_views_ready_score=0,
            fresh_panel_ready_score=0,
        )
        value["acceptance_gate_results"].update(validity=False, readiness=False)
        value["historical_required_failures"].extend(failed)


def terminal_check(path: Path) -> Json:
    """The current final bytes, not a self-report, determine validator status."""
    replay(json.loads(path.read_text()))
    checks = [
        CommandSpec(
            name,
            (str(ROOT / ".venv/bin/python"), "-u", str(ROOT / script), flag, str(path)),
            "terminal",
            60,
        )
        for name, script, flag in [
            ("adversarial", "scripts/adversarial_verify.py", "--json"),
            ("strict_rows", "scripts/verdict_row_consistency_lint.py", "--strict"),
        ]
    ]
    py = str(ROOT / ".venv/bin/python")
    outside = "import os,sys,tempfile; d=tempfile.mkdtemp(prefix='carnot-7980-replay-'); os.chdir(d); os.environ.pop('PYTHONPATH',None); os.execv(sys.argv[1],sys.argv[1:])"
    checks.append(
        CommandSpec(
            "external_cold_replay",
            (
                py,
                "-c",
                outside,
                py,
                str(ROOT / f"scripts/experiments/{NAME}.py"),
                "--cold-replay",
                str(path),
            ),
            "terminal",
            300,
        )
    )
    receipts = run_commands(
        ROOT, checks, log_dir=path.parent / "raw" / path.stem / "terminal", heartbeat_s=30
    )
    return dict(
        passed=all(r["passed"] for r in receipts),
        receipts=receipts,
        flagged_adversarial=not receipts[0]["passed"],
    )


def publish(output: Path, value: Json) -> None:
    """One atomic primary and bound sidecars keep consumer resolution unambiguous."""
    raw = output.parent / "raw" / output.stem
    value["terminal_validation_sidecar_path"] = str(raw / "terminal_validation.json")
    value["primary_resolution_receipt"] = str(raw / "primary_resolution.json")
    value["field_principles"] = {
        k: "Bind current work, public information, label custody or independently observed checks."
        for k in value
    }
    atomic_json(output, value)
    report = terminal_check(output)
    if not report["passed"]:
        value.update(
            honest_verdict="complete_disqualified_terminal_validation",
            verdict_class="disqualified",
            feature_views_ready_score=0,
            fresh_panel_ready_score=0,
            flagged_adversarial=report.get("flagged_adversarial", False),
        )
        value["acceptance_gate_results"].update(validity=False, readiness=False)
        value["historical_required_failures"].append(
            dict(scope="owned_terminal_check", report=report)
        )
        atomic_json(output, value)
        report = terminal_check(output)
    atomic_json(
        raw / "terminal_validation.json",
        dict(primary_path=str(output), primary_sha256=sha256_file(output), report=report),
    )
    selected = reader_receipt(
        TASK,
        output.parent,
        field="feature_views_ready_score",
        expected=value["feature_views_ready_score"],
    )
    if not selected["passed"]:
        raise ValueError("primary_resolution")
    atomic_json(raw / "primary_resolution.json", selected)


def main(argv: list[str] | None = None) -> int:
    """Private workers use the real routes while the owner runs frozen checks."""
    started = time.monotonic()
    progress("start")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261001"], default="20261001")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--feature-output", type=Path)
    parser.add_argument("--validation-worker", action="store_true")
    route = parser.add_mutually_exclusive_group()
    route.add_argument("--public-extract", type=Path)
    route.add_argument("--train-project", action="store_true")
    route.add_argument("--reserved-join", type=Path)
    route.add_argument("--cold-replay", type=Path)
    route.add_argument("--fixture-plan", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.public_extract or args.train_project or args.reserved_join:
            if args.feature_output is None:
                raise ValueError("feature_output_required")
            if args.public_extract:
                if "evaluator" in args.public_extract.parts:
                    raise ValueError("predictor_evaluator_access")
                f.extract_file(args.public_extract, args.feature_output)
            elif args.train_project:
                sources, responses = c.train_public(args.root)
                atomic_json(args.feature_output, dict(sources=sources, responses=responses))
            else:
                c.reserved_join(args.root, args.reserved_join, args.feature_output)
            progress("public_or_evaluator_child_complete")
            return 0
        if args.cold_replay:
            replay(json.loads(args.cold_replay.read_text()))
            progress("replay_passed")
            return 0
        if args.output.name != NAME + ".json":
            raise ValueError("primary_name")
        if (
            args.validation_worker or args.fixture_plan
        ) and args.output.absolute().parent == ROOT / "results":
            raise ValueError("private_output_required")
        raw = args.output.absolute().parent / "raw" / NAME
        with TemporaryDirectory(prefix="carnot-7980-") as workspace:
            scratch = Path(workspace)
            manifest = (
                freeze_commands(raw, scratch)
                if not args.validation_worker and not args.fixture_plan
                else None
            )
            progress("authenticate_inputs")
            if args.fixture_plan:
                fixture = json.loads(args.fixture_plan.read_text())
                plan = fixture["plan"]
                plan["upstream"] = {int(k): v for k, v in plan["upstream"].items()}
                failures, root = [], Path(fixture["root"])
            else:
                failures, plan = c.authenticate(args.root)
                root = args.root
            value = base(failures) if failures else measure(plan, root, raw)
            if args.fixture_plan:
                value.update(
                    honest_verdict="complete_circular_positive_evidence_fixture",
                    verdict_class="circular_positive",
                )
                value["sample_size_budget"]["independent"] = 0
                value["fresh_panel_ready_score"] = 0
                value["claim_scope"] = (
                    "Synthetic fixture plumbing only; zero independent natural sources."
                )
            value.update(
                cited_upstream_artifacts=plan["references"],
                preconditions_checked=plan["checks"],
                code_config_hashes=[c.reference(ROOT / p) for p in OWNED],
                historical_required_failures=[
                    r
                    for upstream in plan["upstream"].values()
                    for r in upstream.get("historical_required_failures", [])
                ],
            )
            value["historical_required_failures"].extend(
                dict(scope="earlier_owned_revision", receipt_source=c.reference(p), failure=r)
                for p in (raw / "development_checks").glob("*/receipts.json")
                for r in json.loads(p.read_text())["receipts"]
                if not r["passed"]
            )
            if manifest:
                progress("before_validation_benchmark")
                commands = [
                    CommandSpec(r["name"], tuple(r["argv"]), r["scope"], r["timeout_s"])
                    for r in manifest["commands"]
                ]
                receipts = run_commands(
                    ROOT,
                    commands,
                    log_dir=raw / "validation",
                    extra_env={
                        "CARNOT_7980_COVERAGE_CONFIG": manifest["coverage_config"],
                        "COVERAGE_FILE": str(scratch / ".coverage"),
                        "COVERAGE_RCFILE": manifest["coverage_config"],
                    },
                    heartbeat_s=30,
                )
                apply_validation(value, receipts)
                value["validation_command_manifest_path"] = str(raw / "validation_commands.json")
                coverage = Path(manifest["coverage_json"])
                if coverage.is_file():
                    report = json.loads(coverage.read_text())
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
                            receipts
                            + [dict(name="nonempty_exact_coverage", passed=False, exit_code=1)],
                        )
                else:
                    apply_validation(
                        value, receipts + [dict(name="missing_coverage", passed=False, exit_code=1)]
                    )
                for path in scratch.glob(".coverage*"):
                    destination = raw / "coverage_data" / path.name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, destination)
                progress("after_validation_benchmark")
            value.update(
                duration_s=time.monotonic() - started,
                phase_spans=[
                    dict(phase="owned_cpu_run", start_s=0.0, end_s=time.monotonic() - started)
                ],
                invocation_timestamp=datetime.now(UTC).isoformat(),
            )
            value["reproducibility_checksum"] = canonical_hash(
                dict(
                    code=value["code_config_hashes"],
                    inputs=value["cited_upstream_artifacts"],
                    seed=69280,
                )
            )
            progress("terminal_publication")
            publish(args.output.absolute(), value)
            progress("published")
            return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"[exp7980] rejected={type(error).__name__}:{error}", flush=True)
        return 1
