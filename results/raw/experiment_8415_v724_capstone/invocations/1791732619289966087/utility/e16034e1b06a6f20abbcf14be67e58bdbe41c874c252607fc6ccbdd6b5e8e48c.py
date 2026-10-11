"""REQ-REPORT-8361: qualify immutable utility evidence while preserving failed history.

Historical authority is a measured operand. The small adapter selects its sealed
bytes explicitly because today's activated roadmap governs a different task.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any, Iterator
from unittest.mock import patch

from carnot.reporting import static_benefit_audit_8350 as static
from carnot.reporting import learning_retention_audit_8351 as learning
from carnot.reporting import sentence_spline_execution_8334 as execution
from carnot.reporting import v721_contract_methods as current
from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    sha256_file,
    ZERO_INVOCATION_COUNTS,
)
from carnot.reporting.primary_publication import read_bound_sidecar, validate_primary
from carnot.reporting.roadmap_contract import parse_design
from carnot.reporting.v685_authority_lifecycle import assess_authorities
from carnot.reporting.v710_contract_replay import require_reference

Json = dict[str, Any]
ROOT = static.ROOT
NAME, TASK = (
    "experiment_8361_v721_utility_audit_qualification",
    "exp8361-utility-audit-qualification",
)
CLI, TEST = (
    "scripts/experiments/" + NAME + ".py",
    "tests/python/test_utility_audit_qualification_8361.py",
)
OWNED = ["python/carnot/reporting/utility_audit_qualification_8361.py", CLI]
MODEL_SPECS: list[Json] = []
SCRATCH = Path.home() / ".cache/carnot-exp8361-private"
PINS = {
    8350: "sha256:57d2803d86f74a2983db76633f0ff426457b1dcba409b653b12cecfcbde91a5d",
    8351: "sha256:563b0becc7f8ed1309c1af8644ec3681c78c9e0ca9e7a48e9384daeefa4feb33",
}
MODULES = {8350: static, 8351: learning}
reference = static.reference


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush measured phase counts so bounded work cannot appear stalled."""
    print(f"[exp8361] phase={phase} completed={completed} pending={pending}", flush=True)


def historical_authority(root: Path, raw: Path, milestone: str = "2026.10.720") -> Json:
    """Reauthenticate the original producer's complete task objects, without swapping a roadmap."""
    primary = root / "results" / (static.NAME + ".json")
    require_reference(dict(path=str(primary), sha256=PINS[8350]))
    value = json.loads(primary.read_bytes())
    require_reference(value["measurement_reference"])
    work = json.loads(Path(value["measurement_reference"]["path"]).read_bytes())
    snapshots = work["authority"]["authority_snapshots"]
    for snap in snapshots.values():
        require_reference(dict(path=snap["snapshot_path"], sha256=snap["sha256"]))
    paths = {role: Path(snap["snapshot_path"]) for role, snap in snapshots.items()}
    result = assess_authorities(
        paths["design"],
        paths["staged"],
        paths["active"],
        raw,
        milestone=milestone,
        first_id=8346,
        count=14,
    )
    result.update(
        tasks=parse_design(paths["design"].read_text(), milestone=milestone)[1],
        staging_disposition=work["authority"]["staging_disposition"],
    )
    return dict(result)


@contextmanager
def historical_inputs() -> Iterator[None]:
    """Select sealed authority through the existing reader; all scientific code stays intact."""
    with (
        patch.object(static.learning.authority, "authority", historical_authority),
        patch.dict(os.environ),
    ):
        # Owned CLI calls start coverage explicitly. The sealed upstream replay
        # has no newly owned statements, so it needs no duplicate instrumentation.
        os.environ.pop("COVERAGE_PROCESS_CONFIG", None)
        yield


def cli() -> list[str]:
    """Include real child statements in the invocation's private coverage shards."""
    prefix = [str(ROOT / ".venv/bin/python"), "-u"]
    config = os.environ.get("COVERAGE_RCFILE")
    if config:
        prefix += ["-m", "coverage", "run", "--rcfile=" + config]
    return prefix + [str(ROOT / CLI)]


def paired_summary(rows: list[Json]) -> Json:
    """Recount source reductions independently so an imported gain cannot stand alone."""
    complete = [r for r in rows if r["qualified"]]
    return dict(
        intended_count=len(rows),
        complete_count=len(complete),
        class_support={str(y): sum(r["y"] == y for r in complete) for y in (0, 1)},
        all_intended_gain_lower=sum(r["gain_lower"] for r in rows) / len(rows),
        all_intended_gain_upper=sum(r["gain_upper"] for r in rows) / len(rows),
        complete_case_gain=sum(r["gain_lower"] for r in complete) / len(complete)
        if complete
        else None,
    )


def rejection_control(number: int, audit: Json, raw: Path) -> Json:
    """Run valid and repaired-hash mutations through real children, retaining statement evidence."""
    module, line = MODULES[number], 599 if number == 8350 else 517
    work = deepcopy(json.loads(Path(audit["measurement_reference"]["path"]).read_bytes()))
    raw.mkdir(parents=True, exist_ok=True)
    for index, operand in enumerate(work["raw_refs"]):
        target = raw / (str(index) + ".json")
        target.write_bytes(Path(operand["path"]).read_bytes())
        work["raw_refs"][index] = reference(target)
    receipts = [dict(name="real_replay_control", passed=True)]
    measurement, candidate = raw / "measurement.json", raw / "candidate.json"
    atomic_json(measurement, work)
    atomic_json(candidate, module.build(work, raw, receipts))
    command = cli() + ["--historical-replay", str(number), "--candidate", str(candidate)]
    valid = execution.check(dict(name="valid", argv=command, deadline_s=180), raw / "logs")
    if number == 8350:
        primitive_path = Path(work["raw_refs"][0]["path"])
        primitive = json.loads(primitive_path.read_bytes())
        primitive["optimizer"]["passed"] = not primitive["optimizer"]["passed"]
        atomic_json(primitive_path, primitive)
        work["raw_refs"][0] = reference(primitive_path)
    else:
        work["gates"][1]["observed"] = "deliberate_rehashed_gate_drift"
    atomic_json(measurement, work)
    atomic_json(candidate, module.build(work, raw, receipts))
    config, data, report = raw / "trace.ini", raw / "trace.coverage", raw / "trace.json"
    config.write_text("[run]\ninclude = " + str(Path(module.__file__)) + "\n")
    traced = [
        str(ROOT / ".venv/bin/python"),
        "-m",
        "coverage",
        "run",
        "--rcfile=" + str(config),
        "--data-file=" + str(data),
        str(ROOT / CLI),
        "--historical-replay",
        str(number),
        "--candidate",
        str(candidate),
    ]
    tamper = execution.check(
        dict(name="tamper", argv=traced, expected_exit=1, deadline_s=180), raw / "logs"
    )
    exported = execution.check(
        dict(
            name="trace_json",
            argv=[
                str(ROOT / ".venv/bin/python"),
                "-m",
                "coverage",
                "json",
                "--rcfile=" + str(config),
                "--data-file=" + str(data),
                "-o",
                str(report),
            ],
            deadline_s=30,
        ),
        raw / "logs",
    )
    observed = json.loads(report.read_bytes())["files"][
        str(Path(module.__file__).relative_to(ROOT))
    ]["executed_lines"]
    return dict(
        passed=valid["passed"] and tamper["passed"] and exported["passed"] and line in observed,
        target_line=line,
        executed_lines=observed,
        self_consistently_rehashed=True,
        valid_replay_receipt=valid,
        tamper_replay_receipt=tamper,
        coverage_receipt=exported,
        coverage_reference=reference(report),
        candidate_reference=reference(candidate),
    )


def measure(root: Path, raw: Path) -> Json:
    """Authenticate current authority and original operands before the frozen reductions."""
    began = time.monotonic()
    progress("before_preconditions")
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw.chmod(0o700)
    work: Json = dict(
        root=str(root),
        gates=[],
        failures=[],
        refs=[],
        authority={},
        audits={},
        controls={},
        historical=[],
        original_dispositions=[],
        owned_failure=False,
    )
    path = root / current.DESIGN
    owned_phase = False
    try:
        static.require(
            work,
            raw,
            "private_resources",
            True,
            raw.stat().st_mode & 0o077 == 0 and shutil.disk_usage(raw).free > 1_000_000_000,
        )
        for tool in ("python", "pytest", "coverage", "ruff", "mypy"):
            static.require(
                work,
                ROOT / ".venv/bin" / tool,
                "executable",
                True,
                os.access(ROOT / ".venv/bin" / tool, os.X_OK),
            )
        work["authority"] = current.authority(root, raw / "authority")
        static.require(
            work,
            path,
            "exact_task_authority",
            True,
            work["authority"]["activated"]
            and any(
                t["id"] == TASK and t["deliverable"] == "results/" + NAME + ".json"
                for t in work["authority"]["tasks"]
            ),
        )
        for snap in work["authority"]["authority_snapshots"].values():
            if snap["exists"]:
                work["refs"].append(dict(path=snap["snapshot_path"], sha256=snap["sha256"]))
        static.bind(work, reference(root / "ops/exclusion_manifest.yaml"), raw, parse=False)
        progress("after_preconditions")
        for number, module in MODULES.items():
            path = root / "results" / (module.NAME + ".json")
            original = static.bind(work, dict(path=str(path), sha256=PINS[number]), raw)
            prior_work = static.bind(work, original["measurement_reference"], raw)
            for ref in prior_work["code_refs"]:
                static.bind(work, ref, raw, parse=False)
            validate_primary(original, path)
            terminal = static.bind(
                work, reference(Path(original["terminal_validation_sidecar_path"])), raw
            )
            sidecar = Path(terminal["publication"]["sidecar_path"])
            static.bind(work, reference(sidecar), raw)
            static.require(
                work,
                path,
                "bound_terminal_passed",
                True,
                read_bound_sidecar(path, sidecar)["report"]["passed"],
            )
            prior_coverage = static.bind(work, original["owned_coverage_reference"], raw)
            line = 599 if number == 8350 else 517
            static.require(
                work,
                path,
                "original_missing_statement",
                [line],
                prior_coverage["files"][module.OWNED[1]]["missing_lines"],
            )
            work["original_dispositions"].append(
                dict(
                    experiment_id=number,
                    primary_reference=reference(path),
                    honest_verdict=original["honest_verdict"],
                    verdict_class=original["verdict_class"],
                    flagged_adversarial=original["flagged_adversarial"],
                    readiness_imported=False,
                    coverage_reference=original["owned_coverage_reference"],
                )
            )
            progress("before_benchmark_frozen_audit", len(work["audits"]), 2 - len(work["audits"]))
            with historical_inputs():
                audit = module.measure(root, raw / str(number))
            work["failures"].extend(audit["failures"])
            work["owned_failure"] |= audit["owned_failure"]
            work["historical"] = audit["historical"]
            candidate = raw / str(number) / "valid.json"
            atomic_json(
                candidate,
                module.build(
                    audit,
                    candidate.parent,
                    [dict(name="frozen_reconstruction", passed=not audit["owned_failure"])],
                ),
            )
            work["audits"][str(number)] = dict(
                measurement_reference=reference(candidate.parent / "measurement.json"),
                candidate_reference=reference(candidate),
            )
            progress("after_benchmark_frozen_audit", len(work["audits"]), 2 - len(work["audits"]))
            if not audit["failures"]:
                owned_phase = True
                path = raw / "controls" / str(number)
                work["controls"][str(number)] = rejection_control(
                    number, work["audits"][str(number)], raw / "controls" / str(number)
                )
                control = work["controls"][str(number)]
                missing = {
                    p: sorted(
                        set(v["missing_lines"])
                        - (set(control["executed_lines"]) if p == module.OWNED[1] else set())
                    )
                    for p, v in prior_coverage["files"].items()
                }
                control["historical_statement_coverage"] = dict(
                    missing_lines_by_file=missing,
                    passed=not any(missing.values()),
                    original_report_reference=original["owned_coverage_reference"],
                    guard_code_unchanged=True,
                )
                work["owned_failure"] |= not control["passed"] or any(missing.values())
                owned_phase = False
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        work["owned_failure"] |= owned_phase
        if not work["failures"]:
            work["failures"].append(
                current.failure(
                    path,
                    "owned_rejection_control" if owned_phase else "authenticated_external_operand",
                    True,
                    str(error) if owned_phase or path.exists() else None,
                )
            )
    work.update(
        preconditions_checked=True,
        duration_s=time.monotonic() - began,
        phase_spans=[
            dict(
                phase="authenticate_reconstruct_and_reject",
                start_s=0,
                duration_s=time.monotonic() - began,
            )
        ],
        code_refs=[reference(ROOT / p) for p in [*OWNED, TEST]],
        requested_invocation_argv=json.loads(os.environ.get("CARNOT_8361_REQUEST_ARGV", "[]")),
    )
    atomic_json(
        raw / "primitive_evidence.json", dict(audits=work["audits"], controls=work["controls"])
    )
    work["raw_refs"] = [reference(raw / "primitive_evidence.json")]
    atomic_json(raw / "measurement.json", work)
    progress("after_measurement", len(work["audits"]), 2 - len(work["audits"]))
    return work


def build(work: Json, raw: Path, receipts: list[Json]) -> Json:
    """A qualified null grants audit readiness while preserving the original scientific boundaries."""
    audits = {
        n: json.loads(Path(a["measurement_reference"]["path"]).read_bytes())
        for n, a in work["audits"].items()
    }
    a, b = audits.get("8350", {}), audits.get("8351", {})
    h1 = (
        static.k.reduce(a["predictions"], a["targets"], a["comparator"], a["optimizer"])
        if a.get("full_h1_measured")
        else {}
    )
    h2 = learning.k.reduce(b["state"], b["targets"], b["checks"]) if b.get("targets") else {}
    checked = bool(receipts) and all(r["passed"] for r in receipts) and not work["owned_failure"]
    coverage = work.get("owned_coverage_reference")
    if coverage:
        report = json.loads(Path(coverage["path"]).read_bytes())
        checked &= all(report["files"][p]["summary"]["missing_lines"] == 0 for p in OWNED)
    ready = int(
        checked
        and not work["failures"]
        and len(work["controls"]) == 2
        and all(c["passed"] for c in work["controls"].values())
    )
    klass = "disqualified" if not checked else "blocked" if not ready else "null"
    labels = {t["slot"]: t["y"] for t in b.get("targets", [])}
    issued = (
        [
            dict(
                p,
                y=labels[p["slot"]],
                qualified=p["p"] is not None and labels[p["slot"]] is not None,
            )
            for p in b.get("state", {}).get("issued", [])
        ]
        if h2
        else []
    )
    rows = [dict(r, audit="H1") for r in h1.get("rows", [])] + [
        dict(r, audit="H2") for r in [*issued, *h2.get("retention_rows", [])]
    ]
    complete = sum(r["qualified"] for r in rows)
    support = {
        name: paired_summary(h["paired_cost_rows"]) for name, h in [("H1", h1), ("H2", h2)] if h
    }
    value: Json = dict(
        experiment_id=8361,
        task_id=TASK,
        milestone="2026.10.721",
        run_date="20261010",
        honest_verdict="complete_" + klass + "_utility_audit_qualification",
        verdict_class=klass,
        gate_check_summary=work["gates"] + work["failures"],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        no_model_load=True,
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        historical_model_provenance=work["historical"],
        rows=rows,
        intended_count=1888,
        completed_count=complete,
        failed_count=len(rows) - complete,
        censored_count=1888 - len(rows),
        excluded_count=0,
        independent_count=h1.get("qualified_count", 0),
        sample_size_budget=dict(
            H1=static.k.CONFIG,
            H2=learning.k.CONFIG,
            retention_windows=[0, 32, 64, 96],
            retention_intended_per_arm_window=32,
            independent_unit="original_source_not_arm_or_window",
        ),
        verifier_is_oracle=False,
        exposure_scope="exposed_cached_development",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=checked,
        flagged_adversarial=not checked,
        acceptance_gates=dict(
            owned=checked, qualification=bool(ready), H1=static.k.CONFIG, H2=learning.k.CONFIG
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        adversarial_findings=work.get("finding_audits", []),
        preconditions_checked=work["preconditions_checked"],
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        random_seed=7178311,
        source_artifact_hashes=work["refs"],
        code_config_hashes=work["code_refs"],
        raw_shard_hashes=work["raw_refs"],
        static_audit_ready_score=ready,
        learning_audit_ready_score=ready,
        h1_development_signal_score=h1.get("h1_development_signal_score", 0) * ready,
        h2_development_signal_score=h2.get("h2_development_signal_score", 0) * ready,
        H1=h1,
        H2=h2,
        retention_windows=h2.get("retention_window_bounds", []),
        source_support=support,
        missing_bounds=dict(H1=h1.get("missing_bounds", []), H2=h2.get("missing_bounds", [])),
        action_headroom=dict(
            H1=h1.get("permissible_action_oracle", {}),
            H2=h2.get("permissible_action_oracle_headroom"),
        ),
        budget_reachability={
            f: b.get("checks", {}).get(f)
            for f in (
                "reachable_count",
                "certified_unreachable_count",
                "feature_qualified_count",
                "rows",
                "method",
                "update_rule",
            )
        },
        qualified_scope_decision=dict(
            H1=h1.get("science_disposition"),
            H2=h2.get("science_disposition"),
            retire_all_joint_reasoning=False,
            downstream_arithmetic_deployment_independent=True,
            H1_retired_scope="frozen spline34 versus RBF34 training procedure; geometry remains unqualified"
            if ready and h1.get("procedure_null_informative")
            else "no qualified procedure retirement",
            H2_retired_scope="frozen deployed delayed update rule and registered budget"
            if ready and h2.get("utility_null_informative")
            else "no broad utility retirement",
        ),
        original_dispositions=work["original_dispositions"],
        rejection_controls=work["controls"],
        authority=work["authority"],
        measurement_reference=reference(raw / "measurement.json"),
        owned_coverage_reference=coverage,
        execution_manifest_reference=work.get("execution_manifest_reference"),
        invocation_argv=work.get("requested_invocation_argv") or work.get("invocation_argv", []),
        methodology_note="Reauthenticate sealed predictions and targets; reconstruct the original delayed trajectory with independent scalar arithmetic. Repaired-hash controls execute both historical rejection statements. Fixed exposed-development bootstrap intervals and frozen support gates establish no generalization or new inference.",
    )
    value["cited_upstream_artifacts"] = [
        dict(
            r,
            fields_imported=[
                "authority, sealed operands or original terminal outcome; readiness not inherited"
            ],
        )
        for r in work["refs"]
    ]
    value["field_principles"] = {
        f: "Bind "
        + f
        + " to authenticated primitives, frozen denominators and separate audit/benefit scope."
        for f in [*value, "field_principles", "reproducibility_checksum"]
    }
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Rebuild summaries only after exact operand, log and historical replay checks."""
    try:
        value = json.loads(path.read_bytes())
        ref = value["measurement_reference"]
        require_reference(ref)
        work = json.loads(Path(ref["path"]).read_bytes())
        for operand in work["refs"] + work["code_refs"] + work["raw_refs"]:
            require_reference(operand)
        for receipt in value["validation_receipts"]:
            for prefix in ("stdout", "stderr", "log"):
                if receipt.get(prefix + "_path"):
                    require_reference(
                        dict(path=receipt[prefix + "_path"], sha256=receipt[prefix + "_sha256"])
                    )
        primitive = json.loads(Path(work["raw_refs"][0]["path"]).read_bytes())
        if primitive != dict(audits=work["audits"], controls=work["controls"]):
            return False
        if value != build(work, Path(ref["path"]).parent, value["validation_receipts"]):
            return False
        with historical_inputs():
            for number, audit in work["audits"].items():
                require_reference(audit["measurement_reference"])
                require_reference(audit["candidate_reference"])
                if not MODULES[int(number)].replay(Path(audit["candidate_reference"]["path"])):
                    return False
        if work["authority"]:
            with TemporaryDirectory(prefix="carnot-8361-authority-") as directory:
                expected = current.authority(Path(work["root"]), Path(directory))
            if any(
                work["authority"].get(f) != expected.get(f)
                for f in ("activated", "canonical_tasks_sha256", "tasks")
            ):
                return False
        for number, control in work["controls"].items():
            require_reference(control["coverage_reference"])
            report = json.loads(Path(control["coverage_reference"]["path"]).read_bytes())
            executed = report["files"][str(Path(MODULES[int(number)].__file__).relative_to(ROOT))][
                "executed_lines"
            ]
            if control["executed_lines"] != executed or control["target_line"] != (
                599 if number == "8350" else 517
            ):
                return False
            for key, expected_exit in [
                ("valid_replay_receipt", 0),
                ("tamper_replay_receipt", 1),
                ("coverage_receipt", 0),
            ]:
                receipt = control[key]
                for prefix in ("stdout", "stderr", "log"):
                    require_reference(
                        dict(path=receipt[prefix + "_path"], sha256=receipt[prefix + "_sha256"])
                    )
                if receipt["expected_exit"] != expected_exit or receipt["passed"] != (
                    receipt["exit_code"] == expected_exit and not receipt["timed_out"]
                ):
                    return False
            passed = control["target_line"] in executed and all(
                control[k]["passed"]
                for k in ("valid_replay_receipt", "tamper_replay_receipt", "coverage_receipt")
            )
            if control["passed"] != passed:
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False


BASE_MANIFEST = execution.manifest


def manifest(private: Path, candidate: Path) -> Json:
    """Freeze scoped checks before opening labels, with subprocess coverage in disk scratch."""
    with patch.object(execution, "e", sys.modules[__name__]):
        plan = BASE_MANIFEST(private, candidate)
    with (private / "coverage.ini").open("a") as stream:
        stream.write("patch = subprocess\n")
    plan["commands"][0]["deadline_s"] = 1200
    plan["commands"][1]["name"] = "consumers_and_private_E2E018_021"
    plan["commands"][1]["argv"].extend(
        [
            "tests/python/test_source_boundary_7852.py",
            "tests/python/test_static_benefit_audit_8350.py::test_independent_costs_and_frozen_bootstrap",
            "tests/python/test_static_benefit_audit_8350.py::test_unknown_targets_keep_joint_bounds_and_support",
            "tests/python/test_learning_retention_audit_8351.py::test_fixed_later_slots_and_all_windows",
            "tests/python/test_learning_retention_audit_8351.py::test_null_support_and_unreachable_budget",
            TEST + "::test_private_e2e018_frozen_authority",
            TEST + "::test_rehashed_real_rejection",
        ]
    )
    typing = private / "strict-mypy.ini"
    typing.write_text(
        "[mypy]\npython_version = 3.12\nstrict = true\nignore_missing_imports = true\nexplicit_package_bases = true\n"
    )
    next(c for c in plan["commands"] if c["name"] == "strict_mypy")["argv"].insert(
        1, "--config-file=" + str(typing)
    )
    return dict(
        plan,
        owned=OWNED,
        no_full_repository_suite=True,
        task_cap_s=4800,
        heartbeat_s=20,
        historical_authority_primary_sha256=PINS[8350],
        historical_science_primary_sha256=PINS,
    )


def main(argv: list[str] | None = None) -> int:
    """Reuse bounded supervision and unchanged publication with the invocation's correct date."""
    args = list(sys.argv[1:] if argv is None else argv)
    requested = [CLI, *args]
    progress("start")
    if "--historical-replay" in args:
        number = int(args[args.index("--historical-replay") + 1])
        candidate = Path(args[args.index("--candidate") + 1])
        with historical_inputs():
            passed = MODULES[number].replay(candidate)
        progress(
            "historical_replay_passed" if passed else "historical_replay_rejected", int(passed), 0
        )
        return int(not passed)
    if "--date" in args:
        index = args.index("--date") + 1
        if index == len(args) or args[index] != "20261010":
            raise SystemExit("date must be 20261010")
        args[index] = "20261009"
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        SCRATCH.mkdir(parents=True, exist_ok=True, mode=0o700)
        SCRATCH.chmod(0o700)
        with (
            patch("tempfile.tempdir", str(SCRATCH)),
            patch.dict(
                os.environ,
                {
                    "TMPDIR": str(SCRATCH),
                    "PYTHONUNBUFFERED": "1",
                    "CARNOT_8361_REQUEST_ARGV": json.dumps(requested),
                },
            ),
            patch.object(execution, "e", sys.modules[__name__]),
            patch.object(execution, "manifest", manifest),
        ):
            return int(execution.main(args))
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)
