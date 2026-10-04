"""REQ-REPORT-8084: publish authenticated fresh-source and local-memory methods.

The protocol can finish with a blocked cohort. That terminal result preserves
independent numerical work and does not ask the conductor to retry missing data.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.primary_publication import publish_primary, read_bound_sidecar
from carnot.reporting.v686_contract_validation import run_check
from carnot.verify import fresh_cohort_8084 as c
from carnot.verify.evidence_features_7980 import FEATURES

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[2]
NAME = "experiment_8084_v700_fresh_cohort_methods"
TASK = "exp8084-fresh-cohort-methods"
MODULE = f"python/carnot/{NAME}.py"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_fresh_cohort_methods_8084.py"
OWNED = [MODULE, "python/carnot/verify/fresh_cohort_8084.py", CLI]
DESIGN = "openspec/change-proposals/research-roadmap-vNEXT.md"
LITERATURE = f"results/raw/{NAME}/literature_access.json"
PINS = {
    7980: "sha256:dc06fadccb5a0bfce0b545256a9e8133e61f002df642b2d71d829753322438ed",
    8072: "sha256:159df4b8b393c5a4feaeb71292b81b7b1c7219de39bdfd34d47a8bd1ba026d9e",
}
INPUTS = [
    "AGENTS.md",
    "CLAUDE.md",
    "CODEX.md",
    "ops/e2e-test-plan.md",
    DESIGN,
    "ops/exclusion_manifest.yaml",
    "research-references.md",
    "research-studying.md",
    "scripts/experiment_template.py",
    "python/carnot/reporting/current_work_receipt.py",
    "python/carnot/reporting/primary_publication.py",
    "python/carnot/verify/development_cohort_7994.py",
    "python/carnot/verify/response_targets_7955.py",
    "python/carnot/verify/evidence_features_7980.py",
    "data/ragtruth/source_info.jsonl",
    "data/ragtruth/response.jsonl",
    LITERATURE,
]


def methods() -> Json:
    """Bind the full declared protocol; concise keys help downstream consumers."""
    protocol = (
        (ROOT / DESIGN)
        .read_text()
        .split("## Exp8084 frozen execution protocol", 1)[1]
        .split("# Carnot Research", 1)[0]
    )
    return dict(
        protocol=protocol,
        protocol_sha256=canonical_hash(protocol),
        features=["bounded_qwen_unsupported_logit", *FEATURES],
        transforms="fit-only mean/std; zero std=1; fit folds recompute geometry",
        roles=c.ROLES,
        costs=dict(accept="5*y", reject="1-y", escalate=0.5, ties="escalate"),
        hypothesis_family=["H1_radial_source_decisions", "H2_feedback_grown_memory"],
        memory=dict(
            initial_centers=16,
            maximum_additions=12,
            additions_per_opportunity=4,
            opportunities=[64, 128, 192],
            newest_update_rows=64,
            pending_capacity=32,
            basis="exp(-||x-c||^2/(2*sigma^2))",
            width="median positive fit pair distance; fallback1",
        ),
        model_call_limits=dict(
            current=0, future_maximum=768, retries=0, timeout_s=120, output_tokens=64
        ),
        statistics=dict(
            alpha=0.05,
            bootstrap_draws=10000,
            primary_block=32,
            sensitivity_blocks=[16, 64],
            seeds=list(range(101, 121)),
            independent_unit="source_group",
            margin=0.02,
        ),
        no_paper_guarantee_transfers=True,
    )


def preconditions(root: Path) -> Json:
    """Observe files, hashes, primary sidecars and environment before data work."""
    checks, references = [], []
    for relative in INPUTS:
        path = root / relative
        checks.append(c.operand(path, "exists", True, path.is_file()))
        if path.is_file():
            references.append(dict(path=str(path), sha256=sha256_file(path)))
    for tool in ("python", "pytest", "coverage", "ruff", "mypy"):
        path = ROOT / ".venv/bin" / tool
        checks.append(c.operand(path, "executable", True, os.access(path, os.X_OK)))
    checks.append(
        c.operand(Path(sys.executable), "python_version>=3.11", True, sys.version_info >= (3, 11))
    )
    checks.extend(
        c.operand(Path(sys.executable), field, expected, os.environ.get(field))
        for field, expected in (("PYTHONUNBUFFERED", "1"), ("JAX_PLATFORMS", "cpu"))
    )
    for number, expected in PINS.items():
        paths = sorted((root / "results").glob(f"experiment_{number}_*.json"))
        path = (
            paths[0] if len(paths) == 1 else root / "results" / f"experiment_{number}_missing.json"
        )
        checks.append(
            c.operand(path, "primary_hash", expected, sha256_file(path) if path.is_file() else None)
        )
        if path.is_file():
            references.append(dict(path=str(path), sha256=sha256_file(path)))
            try:
                primary = json.loads(path.read_text())
                terminal = Path(primary["terminal_validation_sidecar_path"])
                terminal_value = json.loads(terminal.read_text())
                publication = terminal_value.get("publication", terminal_value)
                sidecar = Path(publication.get("sidecar_path", str(terminal)))
                report = read_bound_sidecar(path, sidecar)
                checks.append(
                    c.operand(sidecar, "terminal_report_passed", True, report["report"]["passed"])
                )
                checks.append(
                    c.operand(terminal, "primary_sha256", expected, publication["primary_sha256"])
                )
                references.extend(
                    dict(path=str(p), sha256=sha256_file(p)) for p in (terminal, sidecar)
                )
            except (OSError, ValueError, KeyError, TypeError) as error:
                checks.append(c.operand(path, "authenticated_terminal", "passed", str(error)))
    return dict(
        checks=checks,
        failures=[r for r in checks if not r["passed"]],
        references=references,
        environment=dict(
            PYTHONUNBUFFERED=os.environ.get("PYTHONUNBUFFERED"),
            JAX_PLATFORMS=os.environ.get("JAX_PLATFORMS"),
            python=sys.version,
            current_model_loads=0,
        ),
    )


def counts(rows: list[Json]) -> Json:
    """Count source groups once; absent observations never turn into completed units."""
    eligible = [r for r in rows if r["eligible"]]
    return dict(
        intended_count=768,
        eligible_count=len(eligible),
        independent_count=len({r["source"] for r in eligible}),
        completed_count=len(rows),
        excluded_count=len(rows) - len(eligible),
        censored_count=768 - len(rows),
        failed_count=0,
    )


def build(
    plan: Json, receipts: list[Json], coverage: Json, duration: float, fixture: bool = False
) -> Json:
    """Separate owned qualification from external data blocks and scientific outcomes."""
    required = [r for r in receipts if r.get("classification", "required") == "required"]
    cov_ok = all(
        p in coverage
        and coverage[p]["summary"]["num_statements"] > 0
        and coverage[p]["summary"]["missing_lines"] == 0
        for p in OWNED
    )
    checks = bool(required) and all(r["passed"] for r in required) and cov_ok and not fixture
    failures = deepcopy(plan["failures"])
    if not checks and not fixture:
        failures.append(
            c.operand(Path(plan["raw"]) / "validation.json", "owned_validation", True, False)
        )
    kind = "disqualified" if not checks and not fixture else "blocked" if failures else "null"
    size = counts(plan["rows"])
    value: Json = dict(
        experiment_id=8084,
        task_id=TASK,
        milestone="2026.10.700",
        run_date="20261004",
        schema="carnot.v700.fresh_cohort_methods.v1",
        honest_verdict="complete_blocked_" + failures[0]["field"]
        if kind == "blocked"
        else "complete_" + kind + "_fresh_cohort_methods",
        verdict_class=kind,
        verifier_is_oracle=False,
        claim_scope="Frozen methods and recorded-history-separated public cohort custody; no current model work, verification benefit or general lifelong learning. Private fixtures establish protocol behavior only.",
        flagged_adversarial=False,
        required_checks_passed=checks,
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(Path(plan["raw"]) / "terminal_validation.json"),
        preconditions_checked=plan["preconditions_checked"],
        gate_check_summary=failures,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=deepcopy(ZERO_INVOCATION_COUNTS),
        trained_head_specs=dict(
            current_fits=0, historical_provenance="upstream references only; no head imported"
        ),
        rows=plan["rows"]
        or [
            dict(
                source=r["path"],
                unit=str(i),
                arm="precondition_observation",
                condition=r["field"],
                metric="check_passed",
                numerator=int(r["passed"]),
                denominator=1,
                status="completed",
                exclusion_reason=None if r["passed"] else r["check"],
            )
            for i, r in enumerate(plan["preconditions_checked"])
        ],
        **size,
        sample_size_budget=dict(**size, role_counts=c.ROLES, independent_unit="source_group"),
        random_seed=7008084,
        reproducibility_checksum=canonical_hash(
            dict(plan=plan, receipts=receipts, coverage=coverage)
        ),
        source_artifact_hashes=plan.get("source_artifact_hashes", []),
        raw_shard_hashes=plan.get("raw_shard_hashes", [])
        + [
            dict(path=str(Path(plan["raw"]) / name), sha256=sha256_file(Path(plan["raw"]) / name))
            for name in (
                "seal.json",
                "work.json",
                "validation.json",
                "methods.json",
                "validation_commands.json",
            )
            if (Path(plan["raw"]) / name).is_file()
        ],
        code_config_hashes={p: sha256_file(ROOT / p) for p in [*OWNED, TEST, DESIGN]},
        phase_spans=plan.get("phase_spans", []),
        duration_s=duration,
        exposure_scope="recorded local declarations only; unknown external history and pretraining remain unknown",
        generalized_learning_benefit_score=0,
        cohort_ready_score=int(checks and not failures and plan["selection_sealed_before_labels"]),
        source_protocol_ready_score=int(checks and not failures),
        learning_protocol_ready_score=int(checks and plan.get("methods_frozen", False)),
        role_manifests=plan["role_manifests"],
        evaluator_label_manifests=plan["evaluator_label_manifests"],
        exposure_inventory=plan["exposure_inventory"],
        near_duplicate_clusters=plan["near_duplicate_clusters"],
        exclusion_rows=plan["exclusion_rows"],
        known_excluded_count=plan.get("known_excluded_count", 0),
        unknown_history=plan.get("unknown_history", []),
        class_support=plan.get("class_support", {}),
        method_source_map=json.loads((ROOT / LITERATURE).read_text())["rows"],
        statistical_plan=methods(),
        frozen_hypothesis_family=methods()["hypothesis_family"],
        method_freeze=plan.get("method_freeze"),
        substrate_declaration=dict(
            reduction="aggregation_from_upstream_artifacts", mode="no_model_load", MODEL_SPECS=[]
        ),
        coverage_statement_counts=coverage,
        repository_health=[r for r in receipts if r.get("classification") == "diagnostic"],
        fixture_only=fixture,
        methodology_note="Allocation precedes labels. Unknown labels remain excluded in original slots. Gaussian local memory changes the declared mechanism; no primary hypothesis is measured by this experiment.",
    )
    value["field_principles"] = {
        k: f"Bind {k} to exact current custody; prevent fixture, historical or missing evidence from becoming scientific credit."
        for k in value
    }
    return value


def replay(path: Path) -> bool:
    """Rebuild terminal counts and validate primitive custody without opening labels."""
    try:
        value = json.loads(path.read_text())
        raw = Path(value["terminal_validation_sidecar_path"]).parent
        for ref in value["source_artifact_hashes"] + value["raw_shard_hashes"]:
            if sha256_file(Path(ref.get("snapshot_path", ref["path"]))) != ref["sha256"]:
                return False
        for relative, digest in value["code_config_hashes"].items():
            if sha256_file(ROOT / relative) != digest:
                return False
        for receipt in value["validation_receipts"]:
            if sha256_file(Path(receipt["log_path"])) != receipt["log_sha256"]:
                return False
        plan = json.loads((raw / "seal.json").read_text())
        for role, ref in plan["role_manifests"].items():
            manifest_rows = json.loads(Path(ref["path"]).read_text())["rows"]
            labels = json.loads(Path(plan["evaluator_label_manifests"][role]["path"]).read_text())[
                "rows"
            ]
            for public, label in zip(manifest_rows, labels, strict=True):
                primitive = next(r for r in plan["rows"] if r["unit"] == public["family_id"])
                if primitive["source"] != c.normalized(
                    bytes.fromhex(public["source_bytes"])
                ) or primitive["numerator"] != int(label["status"] == "completed"):
                    return False
        work = json.loads((raw / "work.json").read_text())
        validation = json.loads((raw / "validation.json").read_text())
        return bool(
            build(
                plan,
                validation["receipts"],
                validation["coverage"],
                work["duration_s"],
                work["fixture"],
            )
            == value
        )
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return False


def manifest(private: Path) -> list[Json]:
    """Freeze commands before observations and keep all test output in private scratch."""
    py, cov, pytest, ruff, mypy = [
        str(ROOT / ".venv/bin" / p) for p in ("python", "coverage", "pytest", "ruff", "mypy")
    ]
    config = private / "coverage.ini"
    config.write_text(
        "[run]\nparallel=true\ndata_file="
        + str(private / ".coverage")
        + "\ninclude=\n"
        + "".join("    " + str(ROOT / p) + "\n" for p in OWNED)
    )
    os.environ["CARNOT_8084_COVERAGE_CONFIG"] = str(config)
    typing = private / "mypy.ini"
    typing.write_text("[mypy]\nstrict=True\nfollow_imports=skip\nignore_missing_imports=True\n")
    common = ["-n", "0", "-o", "addopts=", "--no-cov", "-q"]
    commands = [
        (
            "focused_unit_and_cli",
            [
                cov,
                "run",
                "--rcfile=" + str(config),
                "-m",
                "pytest",
                *common,
                TEST,
                "--basetemp=" + str(private / "unit"),
            ],
            240,
            "required",
        ),
        ("coverage_combine", [cov, "combine", "--rcfile=" + str(config)], 30, "required"),
        (
            "coverage_json",
            [cov, "json", "--rcfile=" + str(config), "-o", str(private / "coverage.json")],
            30,
            "required",
        ),
        (
            "changed_statement_coverage",
            [cov, "report", "--rcfile=" + str(config), "--fail-under=100", "-m"],
            30,
            "required",
        ),
        (
            "consumer_and_e2e_015_019",
            [
                pytest,
                *common,
                "tests/python/test_source_boundary_7852.py",
                "tests/python/test_experiment_7942_v689_sentence_labels.py",
                "tests/python/test_conductor_gates.py",
                "tests/python/test_primary_publication_7928.py",
                "--basetemp=" + str(private / "consumer"),
            ],
            240,
            "required",
        ),
        ("ruff_check", [ruff, "check", *OWNED, TEST], 30, "required"),
        ("ruff_format", [ruff, "format", "--check", *OWNED, TEST], 30, "required"),
        ("strict_mypy", [mypy, "--config-file=" + str(typing), *OWNED], 60, "required"),
        (
            "scoped_spec_coverage",
            [
                py,
                "-c",
                "from scripts.check_spec_coverage import check_python_files; from pathlib import Path; assert not check_python_files([Path("
                + repr(TEST)
                + ")])",
            ],
            30,
            "required",
        ),
        (
            "repository_full_suite",
            [pytest, "tests/python", "-q", "--basetemp=" + str(private / "full")],
            900,
            "diagnostic",
        ),
    ]
    return [
        dict(
            name=name,
            argv=argv,
            deadline_s=deadline,
            expected_exit=0,
            classification=classification,
        )
        for name, argv, deadline, classification in commands
    ]


def terminal(path: Path) -> Json:
    """Validate replay and both auditors against the exact candidate before exposure."""
    raw = Path(json.loads(path.read_text())["terminal_validation_sidecar_path"]).parent
    py = str(ROOT / ".venv/bin/python")
    commands = [
        ("cold_replay", [py, "-u", str(ROOT / CLI), "--cold-replay", str(path)]),
        ("adversarial", [py, str(ROOT / "scripts/adversarial_verify.py"), "--json", str(path)]),
        (
            "strict_rows",
            [py, str(ROOT / "scripts/verdict_row_consistency_lint.py"), "--strict", str(path)],
        ),
    ]
    receipts: list[Json] = []
    with tempfile.TemporaryDirectory(prefix="carnot-8084-terminal-") as temporary:
        for name, argv in commands:
            c.progress("subprocess_before_" + name, len(receipts), len(commands) - len(receipts))
            receipts.append(
                run_check(
                    ROOT,
                    dict(name=name, argv=argv, deadline_s=60, expected_exit=0),
                    Path(temporary),
                    raw / "terminal_logs",
                )
            )
            c.progress("subprocess_after_" + name, len(receipts), len(commands) - len(receipts))
    return dict(passed=all(r["passed"] for r in receipts), receipts=receipts)


def main(argv: list[str] | None = None) -> int:
    """Own validation and publication; a completed external block exits normally."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    c.progress("start")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261004"], default="20261004")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--fixture-output", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--seal-output", type=Path)
    parser.add_argument("--health-receipt", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument(
        "--mutate", choices=["contamination", "duplicate", "unavailable_history"], default=""
    )
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = replay(args.cold_replay)
        c.progress("cold_replay_passed" if passed else "cold_replay_rejected")
        return 0 if passed else 1
    start = time.monotonic()
    receipts: list[Json]
    coverage: Json
    output = (args.fixture_output or args.output).absolute()
    raw = output.parent / "raw" / output.stem
    if output.exists() or (raw / "seal.json").exists() and not args.seal_output:
        c.progress("existing_terminal_preserved")
        return 1
    if args.seal_output:
        raw = args.seal_output.parent
    elif not args.fixture_output:
        raw = raw / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, exist_ok=True)
    if args.fixture_output or args.seal_output:
        method_ref = c.immutable(raw / "methods.json", methods())
        before = (
            preconditions(args.root)
            if args.seal_output
            else dict(checks=[], failures=[], references=[])
        )
        plan = (
            c.seal(args.root, raw, args.mutate, args.inventory)
            if not before["failures"]
            else c.seal(args.root / "external-block", raw)
        )
        plan["failures"] = before["failures"] + plan["failures"]
        plan["preconditions_checked"] = before["checks"] + plan["preconditions_checked"]
        plan["source_artifact_hashes"] = before["references"]
        for ref in plan["source_artifact_hashes"]:
            snapshot = raw / "inputs" / (ref["sha256"][7:] + ".bin")
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            if not snapshot.exists():
                snapshot.write_bytes(Path(ref["path"]).read_bytes())
            snapshot.chmod(0o444)
            ref["snapshot_path"] = str(snapshot)
        plan.update(
            raw=str(raw),
            methods_frozen=True,
            method_freeze=method_ref,
        )
        plan["phase_spans"] = [
            dict(
                name="public_seal_then_evaluator",
                duration_s=time.monotonic() - start,
                completed=len(plan["rows"]),
            )
        ]
        c.immutable(raw / "seal.json", plan)
        if args.seal_output:
            c.progress("seal_child_normal_exit")
            return 0
        receipts, coverage = [], {}
    else:
        with tempfile.TemporaryDirectory(prefix="carnot-8084-validation-") as temporary:
            private = Path(temporary)
            specs = manifest(private)
            health = json.loads(args.health_receipt.read_text()) if args.health_receipt else None
            if health is not None:
                if (
                    health.get("classification") != "diagnostic"
                    or sha256_file(Path(health["log_path"])) != health["log_sha256"]
                ):
                    raise ValueError("unauthenticated_health_receipt")
                specs = [s for s in specs if s["name"] != "repository_full_suite"]
            specs.insert(
                0,
                dict(
                    name="seal_child",
                    argv=[
                        str(ROOT / ".venv/bin/python"),
                        "-u",
                        str(ROOT / CLI),
                        "--root",
                        str(args.root),
                        *(["--inventory", str(args.inventory)] if args.inventory else []),
                        "--seal-output",
                        str(raw / "seal.json"),
                    ],
                    deadline_s=600,
                    expected_exit=0,
                ),
            )
            c.immutable(raw / "validation_commands.json", dict(commands=specs))
            receipts = []
            for spec in specs:
                c.progress(
                    "subprocess_before_" + spec["name"], len(receipts), len(specs) - len(receipts)
                )
                receipts.append(run_check(ROOT, spec, private, raw / "validation_logs"))
                c.progress(
                    "subprocess_after_" + spec["name"], len(receipts), len(specs) - len(receipts)
                )
            coverage = (
                json.loads((private / "coverage.json").read_text()).get("files", {})
                if (private / "coverage.json").is_file()
                else {}
            )
            if health is not None:
                receipts.append(health)
        if not (raw / "seal.json").is_file():
            c.progress("owned_seal_failed")
            return 1
        plan = json.loads((raw / "seal.json").read_text())
    work: Json = dict(duration_s=time.monotonic() - start, fixture=bool(args.fixture_output))
    c.immutable(raw / "work.json", work)
    c.immutable(raw / "validation.json", dict(receipts=receipts, coverage=coverage))
    value = build(plan, receipts, coverage, work["duration_s"], work["fixture"])
    publication = publish_primary(output, value, terminal)
    atomic_json(
        raw / "terminal_validation.json", dict(publication=publication, normal_process_exit=True)
    )
    c.progress("terminal_" + value["verdict_class"], len(plan["rows"]), 0)
    return 1 if value["verdict_class"] == "disqualified" else 0
