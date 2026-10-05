"""REQ-REPORT-8098: publish sealed methods with bounded current validation.

No generator or learned head runs here. Readiness only authorizes subsequent
exposed-development capture under the unchanged registered protocol.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
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
from carnot.reporting.v686_contract_validation import coverage_complete, run_check
from carnot.verify import development_methods_8098 as m

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[2]
NAME = "experiment_8098_v701_development_methods"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_development_methods_8098.py"
OWNED = [f"python/carnot/{NAME}.py", "python/carnot/verify/development_methods_8098.py", CLI]
DESIGN = "openspec/change-proposals/research-roadmap-v701-preserved-20261004.md"
HISTORY = "results/experiment_8084_v700_fresh_cohort_methods.json"
LITERATURE = f"results/raw/{NAME}/literature/access.json"
LITERATURE_MAP = f"results/raw/{NAME}/literature/method_source_map.json"
INPUTS = [
    "AGENTS.md",
    "CLAUDE.md",
    "CODEX.md",
    "ops/e2e-test-plan.md",
    DESIGN,
    "ops/exclusion_manifest.yaml",
    "research-references.md",
    "research-studying.md",
    "openspec/capabilities/research-reporting/spec.md",
    "openspec/capabilities/verification/spec.md",
    "scripts/experiment_template.py",
    "python/carnot/reporting/current_work_receipt.py",
    "python/carnot/reporting/primary_publication.py",
    HISTORY,
    LITERATURE,
    LITERATURE_MAP,
    *[
        f"python/carnot/verify/{s}.py"
        for s in (
            "development_cohort_7994",
            "response_targets_7955",
            "evidence_features_7980",
            "radial_memory_8085",
            "fresh_cohort_8084",
            "multivariate_energy_7982",
        )
    ],
    "data/ragtruth/source_info.jsonl",
    "data/ragtruth/response.jsonl",
]


def preconditions(root: Path) -> Json:
    """Observe exact paths and authenticated terminals without repairing history."""
    checks, refs, history = [], [], {}
    for relative in INPUTS:
        path = root / relative
        checks.append(m.operand(path, "exists", True, path.is_file()))
        if path.is_file():
            refs.append(dict(path=str(path), sha256=sha256_file(path)))
    for name in ("python", "pytest", "coverage", "ruff", "mypy"):
        path = ROOT / ".venv/bin" / name
        checks.append(m.operand(path, "executable", True, os.access(path, os.X_OK)))
    if (root / HISTORY).is_file():
        history = json.loads((root / HISTORY).read_text())
        try:
            terminal = Path(history["terminal_validation_sidecar_path"])
            publication = json.loads(terminal.read_text())["publication"]
            sidecar = Path(publication["sidecar_path"])
            bound = read_bound_sidecar(root / HISTORY, sidecar)
            checks.append(
                m.operand(sidecar, "historical_terminal_passed", True, bound["report"]["passed"])
            )
            checks.append(
                m.operand(
                    terminal,
                    "historical_primary_hash",
                    sha256_file(root / HISTORY),
                    publication["primary_sha256"],
                )
            )
            refs.extend(dict(path=str(p), sha256=sha256_file(p)) for p in (terminal, sidecar))
        except (OSError, ValueError, KeyError) as error:
            checks.append(m.operand(root / HISTORY, "authenticated_terminal", "passed", str(error)))
    if (root / LITERATURE).is_file():
        for r in json.loads((root / LITERATURE).read_text())["rows"]:
            path = Path(r.get("path", "missing"))
            checks.append(
                m.operand(
                    path,
                    "primary_literature_hash",
                    r.get("sha256"),
                    sha256_file(path) if path.is_file() else "missing",
                )
            )
            checks.append(
                m.operand(path, "primary_http_status", 200, r.get("status", r.get("error")))
            )
            if path.is_file():
                refs.append(dict(path=str(path), sha256=sha256_file(path)))
    return dict(
        checks=checks,
        failures=[r for r in checks if not r["passed"]],
        references=refs,
        history=history,
    )


def manifest(private: Path) -> list[Json]:
    """Freeze scoped validation argv and private real CLI probes before measurement."""
    py, cov, pytest, ruff, mypy = [
        str(ROOT / ".venv/bin" / n) for n in ("python", "coverage", "pytest", "ruff", "mypy")
    ]
    common = ["-n", "0", "-o", "addopts=", "--no-cov", "-q"]
    included = "--include=" + ",".join(str(ROOT / n) for n in OWNED)
    data = str(private / ".coverage")
    commands = [
        (
            "coverage_unit",
            [
                cov,
                "run",
                "--data-file=" + data,
                included,
                "-m",
                "pytest",
                *common,
                str(ROOT / TEST),
            ],
            0,
        ),
        (
            "consumers_e2e015_019",
            [
                pytest,
                *common,
                *[
                    str(ROOT / "tests/python" / n)
                    for n in (
                        "test_source_boundary_7852.py",
                        "test_experiment_7942_v689_sentence_labels.py",
                        "test_primary_publication_7928.py",
                    )
                ],
            ],
            0,
        ),
        ("ruff", [ruff, "check", *OWNED, TEST], 0),
        ("format", [ruff, "format", "--check", *OWNED, TEST], 0),
        (
            "mypy",
            [mypy, "--config-file=/dev/null", "--strict", "--follow-imports=silent", *OWNED[:2]],
            0,
        ),
        ("spec_coverage", [py, "scripts/check_spec_coverage.py", TEST], 0),
    ]
    for mode in ("success", "blocked", "contamination", "duplicate", "roles"):
        output = private / mode / (NAME + ".json")
        argv = [
            cov,
            "run",
            "--append",
            "--data-file=" + data,
            included,
            str(ROOT / CLI),
            "--fixture-root",
            str(private / ("missing" if mode == "blocked" else "corpus")),
            "--output",
            str(output),
        ]
        if mode not in {"success", "blocked"}:
            argv += ["--mutate", mode]
        commands.append(("cli_" + mode, argv, 0))
        commands.append(
            (
                "replay_" + mode,
                [
                    cov,
                    "run",
                    "--append",
                    "--data-file=" + data,
                    included,
                    str(ROOT / CLI),
                    "--cold-replay",
                    str(output),
                ],
                0,
            )
        )
    commands.extend(
        [
            (
                "coverage_report",
                [cov, "report", "--data-file=" + data, included, "--fail-under=100"],
                0,
            ),
            (
                "coverage_json",
                [
                    cov,
                    "json",
                    "--data-file=" + data,
                    included,
                    "-o",
                    str(private / "coverage.json"),
                ],
                0,
            ),
        ]
    )
    return [
        dict(
            name=n,
            argv=a,
            expected_exit=e,
            deadline_s=300,
            outside_checkout=n.startswith(("cli_", "replay_")),
        )
        for n, a, e in commands
    ]


def validate(raw: Path) -> Json:
    """Reuse the existing heartbeat supervisor and preserve every real check exit."""
    with tempfile.TemporaryDirectory(prefix="carnot-8098-") as temporary:
        private = Path(temporary)
        runpy.run_path(str(ROOT / TEST))["fixture"](private / "corpus")
        specs = manifest(private)
        m.immutable(raw / "validation_manifest.json", dict(commands=specs, changed_code=OWNED))
        receipts = []
        for index, spec in enumerate(specs):
            m.progress("before_" + spec["name"], index, len(specs) - index)
            receipts.append(
                run_check(
                    private if spec["outside_checkout"] else ROOT,
                    spec,
                    private,
                    raw / "validation_logs",
                )
            )
            m.progress("after_" + spec["name"], index + 1, len(specs) - index - 1)
        report = private / "coverage.json"
        coverage = json.loads(report.read_text()) if report.is_file() else {}
        if coverage:
            atomic_json(raw / "coverage.json", coverage)
        complete = coverage_complete(report, includes=OWNED)
        return dict(
            passed=all(r["passed"] for r in receipts) and complete,
            receipts=receipts,
            coverage=coverage,
        )


def reduction(rows: list[Json]) -> Json:
    """Count sources and preserved unknown slots, never seeds or repeated calls."""
    completed = sum(r["numerator"] for r in rows)
    return dict(
        intended_count=640,
        eligible_count=completed,
        independent_count=len(rows),
        completed_count=completed,
        excluded_count=len(rows) - completed,
        censored_count=640 - len(rows),
        failed_count=0,
    )


def produce(root: Path, raw: Path, mutation: str, fixture: bool) -> Json:
    """The child seals methods before labels and exits before parent publication."""
    start = time.monotonic()
    m.progress("preconditions_before")
    before = (
        dict(checks=[], failures=[], references=[], history={}) if fixture else preconditions(root)
    )
    phases = [dict(name="preconditions", duration_s=time.monotonic() - start)]
    m.progress("preconditions_after_methods_before", len(before["checks"]))
    phase_start = time.monotonic()
    methods = m.methods(ROOT / DESIGN)
    method_ref = m.immutable(raw / "methods.json", methods)
    phases.append(dict(name="methods_sealed", duration_s=time.monotonic() - phase_start))
    m.progress("methods_after_public_seal_before")
    phase_start = time.monotonic()
    if before["failures"]:
        plan = m.empty_plan()
    else:
        plan = m.seal(root, raw, mutation)
    plan["failures"] = before["failures"] + plan["failures"]
    plan["preconditions_checked"] = before["checks"] + plan["preconditions_checked"]
    plan.update(
        source_artifact_hashes=before["references"],
        method_freeze=method_ref,
        method_config=methods,
        historical_operands=dict(
            primary_sha256=sha256_file(root / HISTORY) if (root / HISTORY).is_file() else None,
            honest_verdict=before["history"].get("honest_verdict"),
            malformed_history_observations=sum(
                len(r["observed"])
                for r in before["history"].get("gate_check_summary", [])
                if r["check"] == "available_history"
            ),
            known_exclusions=before["history"].get("known_excluded_count"),
            interpretation="historical operands; source data remains available; prior exposure is declared, never filtered",
        ),
        phase_spans=phases
        + [dict(name="public_seal_then_evaluator", duration_s=time.monotonic() - phase_start)],
    )
    plan["raw_shard_hashes"].append(method_ref)
    return plan


def build(plan: Json, validation: Json, raw: Path, duration: float, fixture: bool) -> Json:
    """Methods readiness has no scientific effect or independent corpus credit."""
    owned = plan["owned_failure"] or not validation["passed"]
    klass = "disqualified" if owned else "blocked" if plan["failures"] else "null"
    size = reduction(plan["rows"])
    verdict = (
        "complete_disqualified_owned_validation"
        if owned
        else "complete_blocked_" + plan["failures"][0]["check"]
        if plan["failures"]
        else "complete_null_exposed_development_methods_sealed"
    )
    return dict(
        plan,
        **size,
        experiment_id=8098,
        task_id="exp8098-development-methods",
        run_date="20261004",
        milestone="2026.10.701",
        honest_verdict=verdict,
        verdict_class=klass,
        verifier_is_oracle=False,
        claim_scope="sealed methods and within-run disjoint exposed development sources; no measured decision or learning benefit",
        exposure_scope="exposed_development_within_run_disjoint",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        flagged_adversarial=False,
        required_checks_passed=validation["passed"],
        validation_receipts=validation["receipts"],
        fixture_protocol_only=fixture,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=ZERO_INVOCATION_COUNTS,
        trained_head_specs=[],
        duration_s=duration,
        random_seed=70198,
        reproducibility_checksum=canonical_hash([plan["raw_shard_hashes"], ROLES_HASH]),
        sample_size_budget=dict(
            size, role_counts=m.ROLES, independent_unit="within_run_source_cluster"
        ),
        gate_check_summary=plan["failures"],
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        cohort_ready_score=0 if owned else plan["cohort_ready_score"],
        fit_support_ready_score=0 if owned else plan["fit_support_ready_score"],
        response_selection_rule=plan["method_config"]["response_selection_rule"],
        statistical_plan=plan["method_config"]["statistical_plan"],
        method_source_map=json.loads((ROOT / LITERATURE_MAP).read_text()),
        code_config_hashes={
            n: sha256_file(ROOT / n)
            for n in [
                *OWNED,
                TEST,
                DESIGN,
                LITERATURE_MAP,
                *[p for p in INPUTS if p.endswith(".py")],
            ]
        },
        acceptance_gates=dict(
            public_separation="640 immutable disjoint public clusters",
            fit_support="fit96 >=16/class; tune48 >=8/class",
            validation="all current owned checks; authenticated terminal sidecar",
            other_support="diagnostic until its own outcome audit",
        ),
        field_principles=dict(
            public_metadata="outcomes never steer selection",
            evaluator_label_manifests="original span custody remains evaluator-only",
            cohort_ready_score="separation differs from label support",
            fit_support_ready_score="capture sees only a boolean",
            historical_operands="old blocks do not imply source data disappeared",
            precision_bounds="64 retention groups cannot certify rare-event safety",
            exposure_scope="unknown history and reused public corpus cannot prove independence",
            validation_receipts="readiness requires actual current normal exits",
            rows="every denominator has a preserved source slot",
            model_invocation_counts="historical generators are not current model work",
        ),
        methodology_note="Public selection and method sealing precede human labels; all missing labels keep their slots. This task measures transport and support only.",
    )


ROLES_HASH = canonical_hash(m.ROLES)


def replay(path: Path) -> bool:
    """Recompute denominators and authenticate all sealed bytes on a cold read."""
    try:
        value = json.loads(path.read_text())
        if any(value[k] != v for k, v in reduction(value["rows"]).items()):
            raise ValueError("reduction_drift")
        for ref in value["raw_shard_hashes"] + value["source_artifact_hashes"]:
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                raise ValueError("hash_drift")
        for name, digest in value["code_config_hashes"].items():
            if sha256_file(ROOT / name) != digest:
                raise ValueError("code_config_drift")
        for receipt in value["validation_receipts"]:
            if (
                receipt.get("log_path")
                and sha256_file(Path(receipt["log_path"])) != receipt["log_sha256"]
            ):
                raise ValueError("validation_log_drift")
        if value["required_checks_passed"] and not value["fixture_protocol_only"]:
            required = {
                "seal_child",
                "coverage_unit",
                "consumers_e2e015_019",
                "ruff",
                "format",
                "mypy",
                "spec_coverage",
                "coverage_report",
                "coverage_json",
            }
            if not required.issubset({r.get("name") for r in value["validation_receipts"]}) or any(
                not r["passed"]
                or r.get("actual_exit") != r.get("expected_exit")
                or r.get("timed_out")
                or not r.get("log_path")
                for r in value["validation_receipts"]
            ):
                raise ValueError("owned_validation_receipts")
        ready = (
            value["required_checks_passed"]
            and not value["flagged_adversarial"]
            and not value["owned_failure"]
        )
        expected_cohort = int(
            ready and value["selection_sealed_before_labels"] and not value["failures"]
        )
        if value["cohort_ready_score"] != expected_cohort:
            raise ValueError("cohort_readiness_drift")
        expected_fit = int(
            expected_cohort
            and all(
                value["class_support"][role]["eligible"] >= total
                and min(value["class_support"][role]["classes"].values()) >= each
                for role, total, each in (("fit", 96, 16), ("tune", 48, 8))
            )
        )
        if value["fit_support_ready_score"] != expected_fit:
            raise ValueError("fit_support_drift")
        if value.get("selection"):
            selected = json.loads(Path(value["selection"]["path"]).read_text())
            m.separation(selected["roster"], selected["public"])
            totals: dict[str, Json] = {
                role: dict(eligible=0, classes={"0": 0, "1": 0}) for role in m.ROLES
            }
            for role, ref in value["evaluator_label_manifests"].items():
                data = json.loads(Path(ref["path"]).read_text())
                for row in data["rows"]:
                    if row["y"] is not None:
                        totals[role]["eligible"] += 1
                        totals[role]["classes"][str(row["y"])] += 1
                if any(totals[role][k] != value["class_support"][role][k] for k in totals[role]):
                    raise ValueError("support_drift")
        m.progress("replay_passed")
        return True
    except (OSError, ValueError, KeyError, TypeError) as error:
        m.progress("replay_rejected:" + str(error))
        return False


def terminal(path: Path, raw: Path) -> Json:
    """Auditors and independent cold replay inspect exact candidate bytes."""
    py = str(ROOT / ".venv/bin/python")
    commands = [
        ("cold_replay", [py, str(ROOT / CLI), "--cold-replay", str(path)]),
        ("adversarial", [py, str(ROOT / "scripts/adversarial_verify.py"), "--json", str(path)]),
        (
            "strict_rows",
            [py, str(ROOT / "scripts/verdict_row_consistency_lint.py"), "--strict", str(path)],
        ),
    ]
    with tempfile.TemporaryDirectory(prefix="carnot-8098-terminal-") as temporary:
        receipts = [
            run_check(
                ROOT,
                dict(name=n, argv=a, expected_exit=0, deadline_s=60),
                Path(temporary),
                raw / "terminal_logs",
            )
            for n, a in commands
        ]
    return dict(passed=all(r["passed"] for r in receipts), receipts=receipts)


def main(argv: list[str] | None = None) -> int:
    """Bound every child and preserve completed external blocks without retries."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    m.progress("start")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261004"], default="20261004")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--fixture-root", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--seal-output", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--health-receipt", type=Path)
    parser.add_argument("--mutate", choices=["contamination", "duplicate", "roles"], default="")
    args = parser.parse_args(argv)
    if args.cold_replay:
        return 0 if replay(args.cold_replay) else 1
    started = time.monotonic()
    fixture = args.fixture_root is not None
    root = args.fixture_root or args.root
    output = args.output.absolute()
    raw = args.seal_output.parent if args.seal_output else output.parent / "raw" / output.stem
    if args.seal_output:
        atomic_json(args.seal_output, produce(root, raw, args.mutate, fixture))
        m.progress("seal_child_normal_exit")
        return 0
    if output.exists():
        m.progress("existing_primary_preserved")
        return 0 if replay(output) else 1
    if fixture:
        plan = produce(root, raw, args.mutate, True)
        validation: Json = dict(passed=True, receipts=[])
    else:
        raw = raw / "invocations" / str(time.time_ns())
        m.progress("preconditions_before_validation")
        preconditions(root)
        validation = validate(raw)
        sealed = raw / "child_plan.json"
        child_argv = [
            str(ROOT / ".venv/bin/python"),
            "-u",
            str(ROOT / CLI),
            "--root",
            str(root),
            "--seal-output",
            str(sealed),
        ]
        if args.mutate:
            child_argv += ["--mutate", args.mutate]
        with tempfile.TemporaryDirectory(prefix="carnot-8098-child-") as temporary:
            child = run_check(
                ROOT,
                dict(name="seal_child", argv=child_argv, expected_exit=0, deadline_s=300),
                Path(temporary),
                raw / "child_logs",
            )
        if sealed.is_file():
            plan = json.loads(sealed.read_text())
        else:
            plan = m.empty_plan()
            method_config = m.methods(ROOT / DESIGN)
            method_freeze = m.immutable(raw / "failed_child_methods.json", method_config)
            plan.update(
                owned_failure=True,
                method_config=method_config,
                method_freeze=method_freeze,
                source_artifact_hashes=[],
                historical_operands={},
                phase_spans=[],
            )
            plan["raw_shard_hashes"].append(method_freeze)
            plan["failures"].append(
                m.operand(
                    sealed,
                    "seal_child_normal_exit",
                    0,
                    child.get("actual_exit", "missing_exit_receipt"),
                )
            )
        validation["receipts"].insert(0, child)
        validation["passed"] = validation["passed"] and child["passed"]
    value = build(plan, validation, raw, time.monotonic() - started, fixture)
    value["global_health"] = (
        json.loads(args.health_receipt.read_text())
        if args.health_receipt
        else dict(status="not_run_in_scientific_preconditions")
    )
    atomic_json(raw / "independent_reduction.json", reduction(value["rows"]))
    candidate = raw / "prepublication.json"
    atomic_json(candidate, value)
    checked = dict(passed=replay(candidate), receipts=[]) if fixture else terminal(candidate, raw)
    value["validation_receipts"].extend(checked["receipts"])
    if not checked["passed"]:
        value.update(
            honest_verdict="complete_disqualified_terminal_validation",
            verdict_class="disqualified",
            required_checks_passed=False,
            cohort_ready_score=0,
            fit_support_ready_score=0,
            flagged_adversarial=True,
        )
    publication = publish_primary(
        output, value, lambda p: dict(passed=replay(p), validation=checked)
    )
    atomic_json(
        raw / "terminal_validation.json",
        dict(
            normal_process_exit=True,
            seal_child_normal_exit=fixture or child["passed"],
            publication=publication,
            owned_checks_passed=value["required_checks_passed"],
            terminal_checks=checked,
        ),
    )
    m.progress("published_" + value["honest_verdict"], len(value["rows"]))
    return 0
