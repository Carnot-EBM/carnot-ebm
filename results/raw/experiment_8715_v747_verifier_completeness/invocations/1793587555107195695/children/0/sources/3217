"""Publish isolated response-role targets without fitting on exposed evaluation.

REQ-REPORT-7968. Readiness qualifies custody; it does not claim calibration
benefit, fresh evaluation, or class balance.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone, UTC
import importlib
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot import experiment_7955_v690_response_targets as prior
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.verify import response_role_targets_7968 as roles

Json = dict[str, Any]
ROOT = prior.ROOT
NAME = "experiment_7968_v691_response_role_targets"
TASK = "exp7968-response-role-targets"
HISTORY = "results/experiment_7955_v690_response_targets.json"
HISTORY_HASH = "sha256:dbfac2d991450601add8506a00452c9b6d39a594ad91d3c291f742cfe378c62a"
MODEL_SPECS: list[str] = []
OWNED = [
    f"python/carnot/{NAME}.py",
    "python/carnot/verify/response_role_targets_7968.py",
    f"scripts/experiments/{NAME}.py",
]
TESTS = [
    "tests/python/test_response_role_targets_7968.py",
    f"tests/python/test_{NAME}.py",
    *prior.TESTS,
]
INCLUDE = ",".join(str(ROOT / p) for p in OWNED)
IMPORTS = [
    *prior.IMPORTS,
    "carnot.experiment_7955_v690_response_targets",
    "carnot.verify.response_role_targets_7968",
]
reference, operand = prior.reference, prior.operand


def progress(phase: str, started: float, units: int = 0) -> None:
    """Keep each host phase visible with actual elapsed work."""
    print(
        f"[exp7968] phase={phase} completed_units={units} elapsed_s={time.monotonic() - started:.3f}",
        flush=True,
    )


def authenticate(root: Path) -> tuple[list[Json], Json]:
    """Bind the historical producer to its own frozen identity, including date."""
    failures, upstream = prior.authenticate(root)
    path = root / HISTORY
    check = operand(
        "exp7955", path, "sha256", HISTORY_HASH, sha256_file(path) if path.is_file() else None
    )
    checks = [check]
    history = {}
    if check["passed"]:
        history = json.loads(path.read_text())
        checks += [
            operand("exp7955", path, k, v, history.get(k))
            for k, v in dict(
                experiment_id=7955,
                task_id=prior.TASK,
                milestone="2026.09.690",
                run_date="20261001",
                response_targets_ready_score=1,
                flagged_adversarial=False,
                verdict_class="null",
            ).items()
        ]
        import yaml

        exclusions = yaml.safe_load((ROOT / "ops/exclusion_manifest.yaml").read_text())
        retired = any(
            r.get("experiment_id") == 7955
            for key in ("retired", "retired_experiments")
            for r in exclusions.get(key, [])
        )
        checks.append(operand("exp7955", path, "retired", False, retired))
    failures += [r for r in checks if not r["passed"]]
    if not failures:
        try:
            prior.reconstruct(history)
        except (OSError, ValueError, KeyError, TypeError) as error:
            return [operand("exp7955", path, "reconstruction_custody", True, str(error))], upstream
        upstream.update(history=history)
        upstream["authenticated_inputs"].append(reference(path))
        upstream["preconditions"] += checks
    return failures, upstream


def base(failures: list[Json]) -> Json:
    """A blocked result retains the complete contract and zero readiness."""
    value = prior.base(failures)
    value.pop("response_targets_ready_score")
    value.update(
        schema="carnot.exp7968.response_role_targets.v1",
        experiment_id=7968,
        task_id=TASK,
        milestone="2026.10.691",
        execution_date="20261001",
        random_seed=69168,
        response_roles_ready_score=0,
        public_role_manifests={},
        evaluator_role_manifests={},
        role_counts={},
        class_counts_by_role={},
        source_cluster_counts_by_role={},
        cross_role_overlap_count=0,
        scratch_root_receipt=None,
        started_at=None,
        finished_at=None,
        verifier_is_oracle=False,
        claim_scope="exposed_development",
        sample_size_budget=dict(
            unit="complete_response_family",
            intended=640,
            eligible=0,
            started=0,
            completed=0,
            failed=0,
            censored=0,
            excluded=0,
            independent=0,
        ),
        inherited_response_determination=dict(
            path=str(ROOT / HISTORY),
            sha256=HISTORY_HASH,
            honest_verdict="complete_null_response_annotation_transport",
            class_counts={"0": 43, "1": 19},
        ),
        label_access_policy=dict(
            capture="public bytes and role ID only",
            fitting=["fit", "tune"],
            threshold_design=["policy_design"],
            post_seal=[
                "calibration_replay",
                "online_update",
                "online_admission",
                "evaluation",
                "retention",
            ],
            seals=["heads", "policies"],
        ),
        public_manifest_path=None,
    )
    return value


def build_live(root: Path, raw: Path) -> Json:
    """Preserve historical rows before reading annotation inputs again."""
    failures, upstream = authenticate(root)
    if failures:
        value = base(failures)
        value.update(
            inference_substrate_class="blocked_no_run",
            planned_inference_substrate_class="no_model_load",
        )
        return value
    history = upstream["history"]
    original = history["response_union_rows"]
    data = json.loads(prior.prior.checked_reference(history["original_input_manifest"]).read_text())
    value = build(data, raw, original_rows=original)
    value.update(
        custody_root=str(root),
        source_artifact_hashes=upstream["authenticated_inputs"],
        preconditions_checked=upstream["preconditions"],
        historical_required_failures=history["historical_required_failures"],
        cited_upstream_artifacts=[
            *history["cited_upstream_artifacts"],
            dict(
                experiment_id=7955,
                fields_imported=[
                    "response_union_rows",
                    "annotation_rows",
                    "original_input_manifest",
                ],
                **reference(root / HISTORY),
            ),
        ],
    )
    value["repository_health"] = dict(
        current=[], historical=history["repository_health"], affects_required_checks=False
    )
    return value


def reconstruct(value: Json) -> Json:
    """Cold-reduce original bytes and compare every exported view and count."""
    for item in value.get("code_config_hashes", []) + value["source_artifact_hashes"]:
        prior.prior.checked_reference(item)
    if not value["public_role_manifests"]:
        if value["response_roles_ready_score"]:
            raise ValueError("unsafe_readiness")
        return {}
    data = json.loads(
        prior.prior.checked_reference(
            value.get("fixture_input", value["original_input_manifest"])
        ).read_text()
    )
    if not value.get("fixture_input"):
        failures, upstream = authenticate(Path(value["custody_root"]))
        if failures:
            raise ValueError("cold_custody")
        if value["rows"] != upstream["history"]["response_union_rows"]:
            raise ValueError("original_union_drift")
    public, audit = prior.targets.freeze(data["public"])
    rows, annotations = prior.targets.join(public, audit, data)
    if (
        value["rows"] != rows
        or value["response_union_rows"] != rows
        or value["annotation_rows"] != annotations
    ):
        raise ValueError("union_drift")
    views, reduced = roles.partition(public, audit, data["roles"], rows, annotations)
    for role, view in views.items():
        for kind in ("public", "evaluator"):
            item = value[kind + "_role_manifests"][role]
            saved = json.loads(prior.prior.checked_reference(item).read_text())
            if saved != view[kind] or item["count"] != len(
                view[kind]["request_rows" if kind == "public" else "rows"]
            ):
                raise ValueError("role_view_drift")
    for key in reduced.keys() - {"response_roles_ready_score"}:
        if value[key] != reduced[key]:
            raise ValueError("reduction_drift:" + key)
    if value["response_roles_ready_score"] and (
        value.get("fixture_input")
        or value["verdict_class"] in {"blocked", "disqualified"}
        or value["flagged_adversarial"]
    ):
        raise ValueError("unsafe_readiness")
    if value["response_roles_ready_score"] and value["validation_receipts"]:
        prior.check_receipts(value)
    return reduced


def freeze_commands(raw: Path, scratch: Path) -> Json:
    """Freeze explicit files and private child workspaces before publishing rows."""
    manifest = prior.freeze_commands(scratch / "routes")
    coverage_file = str(scratch / ".coverage")
    old_workspace = Path(manifest["coverage_file"]).parent
    replacements = {
        prior.NAME: NAME,
        "response_targets_7955": "response_role_targets_7968",
        prior.INCLUDE: INCLUDE,
        manifest["coverage_file"]: coverage_file,
        str(old_workspace): str(scratch),
    }
    commands = []
    for spec in manifest["commands"]:
        if spec["name"].startswith("e2e016"):
            continue
        spec = dict(spec)
        for old, new in replacements.items():
            spec["argv"] = [arg.replace(old, new) for arg in spec["argv"]]
        if spec["name"] == "unit_consumer_e2e015":
            start = next(i for i, arg in enumerate(spec["argv"]) if arg.startswith("tests/python/"))
            spec["argv"] = spec["argv"][:start] + [
                "--basetemp=" + str(scratch / "pytest"),
                *TESTS,
                "-q",
            ]
        if spec["name"] in {"ruff_check", "ruff_format", "strict_mypy", "spec_coverage"}:
            prefix = spec["argv"][: 3 if spec["name"] in {"ruff_format", "strict_mypy"} else 2]
            spec["argv"] = prefix + (
                TESTS
                if spec["name"] == "spec_coverage"
                else OWNED + ([] if spec["name"] == "strict_mypy" else TESTS[:2])
            )
        if spec["name"] == "repository_health":
            spec["deadline_s"] = 60
        spec["deadline_s"] = min(spec["deadline_s"], 300)
        commands.append(spec)
    shutil.rmtree(old_workspace)
    manifest.update(
        commands=commands,
        affected_files=OWNED,
        explicit_tests=TESTS,
        transitive_consumers=IMPORTS,
        coverage_includes=INCLUDE,
        coverage_file=coverage_file,
        scratch_root=str(scratch),
        frozen_hashes=[reference(ROOT / p) for p in OWNED + TESTS],
        terminal_commands=[
            dict(
                name=name,
                argv=[
                    str(ROOT / ".venv/bin/python"),
                    "-u",
                    script,
                    *flags,
                    str(raw / "terminal_candidate.json"),
                ],
                expected_exit=0,
                deadline_s=60,
            )
            for name, script, flags in [
                ("adversarial", "scripts/adversarial_verify.py", ["--json"]),
                ("strict_rows", "scripts/verdict_row_consistency_lint.py", ["--strict"]),
            ]
        ],
    )
    atomic_json(raw / "validation_command_manifest.json", manifest)
    return manifest


def apply_validation(value: Json, receipts: list[Json]) -> None:
    """Owned failure disqualifies; repository health keeps its real outcome."""
    value["validation_receipts"] = receipts
    value["observed_child_commands"] = [r.get("command_argv", []) for r in receipts]
    value["repository_health"]["current"] = [r for r in receipts if not r.get("required", True)]
    if any(not r["passed"] for r in receipts if r.get("required", True)):
        value.update(
            verdict_class="disqualified",
            honest_verdict="complete_disqualified_required_validation",
            response_roles_ready_score=0,
        )
        value["acceptance_gate_results"].update(validity=False, readiness=False)


def terminal_check(candidate: Path) -> Json:
    """Check cold claims and both terminal reports on the exact candidate bytes."""
    reconstruct(json.loads(candidate.read_text()))
    specs = [
        prior.prior.CommandSpec(
            name,
            (str(ROOT / ".venv/bin/python"), "-u", script, flag, str(candidate)),
            "terminal_candidate",
            60,
        )
        for name, script, flag in [
            ("adversarial", "scripts/adversarial_verify.py", "--json"),
            ("strict_rows", "scripts/verdict_row_consistency_lint.py", "--strict"),
        ]
    ]
    receipts = prior.prior.run_commands(
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


def main(argv: list[str] | None = None) -> int:
    """Run a bounded export or public-only capture and publish checked bytes."""
    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    progress("start", started)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261001"], default="20261001")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    route = parser.add_mutually_exclusive_group()
    route.add_argument("--fixture-input", type=Path)
    route.add_argument("--cold-replay", type=Path)
    route.add_argument("--capture-input", type=Path)
    route.add_argument("--label-input", type=Path)
    parser.add_argument("--role", default="fit")
    parser.add_argument("--expected-sha256", default="")
    parser.add_argument(
        "--purpose", choices=["fitting", "threshold_design", "evaluation"], default="fitting"
    )
    parser.add_argument("--seals", type=Path)
    args = parser.parse_args(argv)
    raw = args.output.absolute().parent / "raw" / args.output.stem
    try:
        if args.capture_input or args.label_input:
            public = args.capture_input is not None
            view = roles.read_view(
                args.capture_input or args.label_input,
                args.expected_sha256,
                args.role,
                "capture" if public else args.purpose,
                json.loads(args.seals.read_text()) if args.seals else {},
            )
            progress(
                "capture_passed" if public else "label_access_passed",
                started,
                len(view["request_rows" if public else "rows"]),
            )
            return 0
        if args.cold_replay:
            value = json.loads(args.cold_replay.read_text())
            reconstruct(value)
            progress("replay_passed", started, len(value["rows"]))
            return 0
        with TemporaryDirectory(prefix="carnot-7968-") as workspace:
            scratch = Path(workspace)
            manifest = freeze_commands(raw, scratch)
            progress("authenticate_export", started)
            value = (
                build(
                    json.loads(args.fixture_input.read_text()), raw, fixture_path=args.fixture_input
                )
                if args.fixture_input
                else build_live(args.root, raw)
            )
            joined = time.monotonic() - started
            value.update(
                started_at=started_at,
                scratch_root_receipt=dict(
                    path=workspace,
                    outside_checkout=True,
                    purpose="pytest, coverage and mutable child publications",
                    cleanup_after_children=True,
                ),
                validation_command_manifest_path=str(raw / "validation_command_manifest.json"),
            )
            value["resolved_imports"] = {
                name: str(Path(importlib.import_module(name).__file__).resolve())
                for name in IMPORTS
            }
            value["code_config_hashes"] = [reference(ROOT / p) for p in OWNED + TESTS] + [
                reference(Path(value["validation_command_manifest_path"])),
                *[reference(Path(p)) for p in value["resolved_imports"].values()],
            ]
            if not args.fixture_input and value["verdict_class"] != "blocked":
                shutil.copyfile(
                    raw / "validation_input.json", scratch / "routes/validation_input.json"
                )
                progress("before_required_validation", started)
                os.environ["CARNOT_7968_COVERAGE_FILE"] = manifest["coverage_file"]
                os.environ["PYTEST_ADDOPTS"] = "--basetemp=" + str(scratch / "health-pytest")
                os.environ["COVERAGE_FILE"] = str(scratch / "repository-health.coverage")
                receipts = prior.execute_commands(manifest, raw / "validation_logs")
                apply_validation(value, receipts)
                measured = json.loads((scratch / "routes/coverage.json").read_text())
                value["coverage_statement_counts"] = {
                    p: info["summary"] for p, info in measured["files"].items()
                }
                if not measured["files"] or any(
                    info["summary"]["missing_lines"] for info in measured["files"].values()
                ):
                    apply_validation(
                        value,
                        receipts
                        + [dict(name="coverage_nonempty_complete", passed=False, required=True)],
                    )
                shutil.copyfile(scratch / "routes/coverage.json", raw / "coverage.json")
                for path in scratch.glob(".coverage*"):
                    shutil.copyfile(path, raw / path.name)
                progress("after_required_validation", started)
            value.update(
                primary_resolution_receipt=dict(
                    path=str(raw / "primary_resolution.json"),
                    readers=[
                        "conductor_gates.evaluate_gates",
                        "in_process_doc_reconcile.find_artifact",
                    ],
                ),
                terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
                methodology="Authenticate existing all-role union; verify exact original spans; export isolated role views; cold-reduce all counts.",
                title="Complete-response calibration role custody",
                duration_s=time.monotonic() - started,
                finished_at=datetime.now(UTC).isoformat(),
            )
            value["phase_spans"] = [
                dict(phase="authenticate_export", start_s=0.0, end_s=joined),
                dict(phase="required_validation", start_s=joined, end_s=value["duration_s"]),
            ]
            value["reproducibility_checksum"] = canonical_hash(
                dict(
                    inputs=value["source_artifact_hashes"],
                    code=value["code_config_hashes"],
                    seed=69168,
                )
            )[7:23]
            value["field_principles"] = {
                key: "Bind this field to exact input bytes, original role membership and owned receipts."
                for key in value
                if key != "field_principles"
            }
            value["field_principles"].update(
                response_roles_ready_score="Custody and role isolation do not imply balanced classes or calibration benefit.",
                class_counts_by_role="Unknown annotations remain unknown; independent units are original source families.",
                label_access_policy="Capture never opens labels; fitting and threshold design have distinct roles; evaluation requires seals.",
                inherited_response_determination="Original evaluation labels and prior null verdicts remain unchanged.",
                verifier_is_oracle="Human source support annotations are independent of model fitting but fallible, never formal truth certificates.",
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
                    response_roles_ready_score=0,
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
                field="response_roles_ready_score",
                expected=value["response_roles_ready_score"],
            )
            atomic_json(raw / "primary_resolution.json", selected)
            if not selected["passed"] or selected["gate_sha256"] != publication["primary_sha256"]:
                raise ValueError("primary_resolution")
            progress("published", started, len(value["rows"]))
            return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"[exp7968] rejected {type(error).__name__}:{error}", flush=True)
        return 1


def build(
    data: Json,
    raw: Path,
    *,
    fixture_path: Path | None = None,
    original_rows: list[Json] | None = None,
) -> Json:
    """Read the original union first, then verify it and export role views."""
    public, audit = prior.targets.freeze(data["public"])
    rows, annotations = prior.targets.join(public, audit, data)
    if original_rows is not None and rows != original_rows:
        raise ValueError("original_union_drift")
    views, reduced = roles.partition(public, audit, data["roles"], rows, annotations)
    value = base([])
    value.update(
        reduced,
        rows=rows,
        response_union_rows=rows,
        annotation_rows=annotations,
        verdict_class="null",
        honest_verdict="complete_null_response_role_transport",
    )
    for role, view in views.items():
        for kind in ("public", "evaluator"):
            path = raw / kind / (role + ".json")
            atomic_json(path, view[kind])
            value[kind + "_role_manifests"][role] = dict(
                **reference(path),
                count=len(view[kind]["request_rows" if kind == "public" else "rows"]),
            )
    atomic_json(raw / "validation_input.json", data)
    value["original_input_manifest"] = reference(raw / "validation_input.json")
    value["acceptance_gate_results"].update(validity=True, readiness=True)
    if fixture_path is not None:
        value.update(
            fixture_input=reference(fixture_path),
            response_roles_ready_score=0,
            verdict_class="circular_positive",
            honest_verdict="complete_circular_positive_fixture_transport",
        )
        value["acceptance_gate_results"]["readiness"] = False
    return value
