"""REQ-REPORT-9013-V770: qualify new custody while preserving failed V769 history."""

from contextlib import contextmanager
from collections.abc import Iterator
import json
from functools import partial
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import yaml

from carnot.reporting import v769_branch_contract as old
from carnot.reporting import v770_branch_evidence as e
from carnot.reporting import v755_branch_runner as runner
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.roadmap_contract import compare_contract

Json = dict[str, Any]
ROOT, DESIGN, ACTIVE, STAGED = old.ROOT, old.DESIGN, old.ACTIVE, old.STAGED
NAME, TASK, MILESTONE = (
    "experiment_9013_v770_branch_contract",
    "exp9013-branch-contract",
    "2026.11.770",
)
PIN = "sha256:995e65dabf7a45803799614b5cfcd8e30c2608a29ed980c6606aca209ac0ae6a"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_v770_branch_contract_9013.py"
OWNED = [
    "python/carnot/reporting/v770_branch_contract.py",
    "python/carnot/reporting/v770_branch_evidence.py",
    CLI,
]
DIRECT = [
    "tests/python/test_primary_publication_7928.py",
    "tests/python/test_experiment_7837_v681_contract_methods.py",
]
E2E = [
    "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
    "tests/python/test_restricted_decision_audit_8210.py",
    "tests/python/test_hard_exit_learning_qualification_8206.py::test_real_child_two_crashes_and_no_crash",
]
CHECKS = [
    "owned_tests",
    "private_e2e018_021",
    "ruff_check",
    "ruff_format",
    "mypy",
    "spec_coverage",
    "owned_statement_coverage",
]
READINESS = [
    "contract_ready_score",
    "frozen_graph_authority_ready_score",
    "online_protocol_ready_score",
    "arc_new_outcomes_ready_score",
]
MODEL_SPECS: list[Json] = []
reference, checksum, seal, verify_measured = (
    old.reference,
    old.checksum,
    old.seal,
    old.verify_measured,
)
_authority, _freeze, _build, _mutations = old.authority, old.freeze, old.build, old.mutations
REUSED = [
    "python/carnot/reporting/v769_branch_contract.py",
    "python/carnot/reporting/v769_branch_evidence.py",
    "python/carnot/reporting/v768_branch_contract.py",
    "python/carnot/reporting/v768_capstone.py",
    "python/carnot/reporting/v767_branch_contract.py",
    "python/carnot/reporting/v766_branch_evidence.py",
    "python/carnot/reporting/arc_supervisor_frontier_8720.py",
    "python/carnot/reporting/v755_branch_runner.py",
    "python/carnot/reporting/v756_branch_qualification.py",
    "python/carnot/reporting/v709_execution.py",
]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Actual counts keep the waiting operator informed without padding runtime."""
    print(f"[exp9013] phase={phase} completed={completed} pending={pending}", flush=True)


@contextmanager
def binding() -> Iterator[None]:
    """Bind existing custody mechanics locally, leaving historical files unchanged."""
    with patch.multiple(
        old,
        NAME=NAME,
        TASK=TASK,
        MILESTONE=MILESTONE,
        PIN=PIN,
        CLI=CLI,
        OWNED=OWNED,
        TEST=TEST,
        e=e,
        authority=authority,
        mutations=mutations,
        progress=progress,
    ):
        yield


def authority(root: Path) -> Json:
    """An independent digest checks all full objects, not just visible table cells."""

    def compare(text: str, staged: Json, active: Json, **kwargs: Any) -> Json:
        return compare_contract(text, staged, active, milestone=MILESTONE, first_id=9013)

    with binding(), patch.object(old, "compare_contract", compare):
        result = _authority(root)
    for row in result["checks"]:
        if row["artifact_field"] == "end_marker":
            row["observed"] = "<!-- V770_TASK_CONTRACT_END -->" in (root / DESIGN).read_text()
            row["passed"] = row["observed"] is True
    result["ready"] = all(row["passed"] for row in result["checks"])
    for i, row in enumerate(result["rows"]):
        if row["unit_id"].endswith("-missing"):
            row["unit_id"] = row["group_id"] = f"exp{9013 + i}-missing"
        matched = result["ready"] and row["matched"]
        row.update(
            value=int(matched),
            numerator=int(matched),
            absolute_metric=int(matched),
            status="completed" if matched else "censored",
            censored=not matched,
        )
    return result


def mutations(source: Path, raw: Path) -> list[Json]:
    """Keep the established seven independent refusal controls and their raw bytes."""
    with binding():
        return _mutations(source, raw)


def scope_manifest() -> Json:
    """Freeze explicit files and validation policy before opening measurement inputs."""
    return dict(
        owned_paths=OWNED,
        owned_tests=[TEST],
        direct_consumer_tests=DIRECT,
        private_e2e_tests=E2E,
        required_check_names=CHECKS,
        code_test_hashes=[
            reference(ROOT / p) for p in [*OWNED, TEST, *DIRECT, *E2E[:2], E2E[2].split("::")[0]]
        ],
        reused_code_hashes=[reference(ROOT / p) for p in REUSED],
        check_commands=dict(
            owned_tests=["coverage run", "pytest", TEST, *DIRECT],
            private_e2e018_021=["pytest", *E2E],
            ruff_check=["ruff", "check", *OWNED, TEST],
            ruff_format=["ruff", "format", "--check", *OWNED, TEST],
            mypy=["mypy", "--strict", *OWNED],
            spec_coverage=["scripts/check_spec_coverage.py", TEST, *DIRECT],
            owned_statement_coverage=[
                "Coverage.combine",
                "Coverage.json_report",
                "coverage report --fail-under=100",
                *OWNED,
            ],
        ),
        required_normal_exit=True,
        statement_percent=100,
        excluded_statements=0,
        repository_health_separate=True,
        learning_restart="E2E-020",
        policy="New scoped acceptance only; original producers, tests and acceptance rules remain unchanged.",
        measurement_limit_s=3300,
        validation_reserve_s=1200,
        hard_cap_s=4800,
    )


def freeze(root: Path, raw: Path, run_date: str) -> Json:
    """Seal policy before measurement and retain complete immutable upstream authority."""
    if (raw / "work.json").exists():
        raise FileExistsError("immutable_invocation_already_exists")
    manifest = scope_manifest()
    atomic_json(raw / "validation_scope_manifest.json", manifest)
    with binding():
        work = _freeze(root, raw, run_date)
    for name in ["research-complete.yaml", "ops/conductor-log.md"]:
        source, target = root / name, raw / "authority" / name
        if source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
            work["snapshots"].append(dict(reference(target), source=reference(source)))
    retained = raw / "authority"
    historical = e.health(retained)
    work["checks"] += historical["checks"]
    work["repository_health"] = historical
    work["validation_scope_manifest"] = manifest
    work["code"] += [dict(seal(ROOT / p, raw), executed_source=True) for p in REUSED]
    archive = retained / "research-complete.yaml"
    archived = yaml.safe_load(archive.read_bytes())["milestones"] if archive.is_file() else []
    work["archive_lag"] = dict(
        plan_time_archive_end="2026.11.768",
        observed_archive_end=archived[-1]["id"] if archived else "missing_file",
        last_qualified_scientific_milestone="2026.11.768",
        scientific_authority="V768 methods; V769 scheduling closure is not scientific qualification",
        scheduler_reference=reference(retained / "ops/conductor-log.md"),
    )
    work.pop("raw_reference")
    atomic_json(raw / "work.json", work)
    return dict(work, raw_reference=reference(raw / "work.json"))


def receipts_pass(work: Json, validation: Json) -> bool:
    """A passing scalar cannot hide missing, stale, abnormal or altered check evidence."""
    receipts = validation["receipts"]
    return bool(
        validation["coverage"]["complete"]
        and sorted(r["name"] for r in receipts) == sorted(CHECKS)
        and all(
            r.get("passed") is True
            and r.get("normal_exit") is True
            and r.get("timed_out") is False
            and r.get("exit_code") == 0
            and r.get("started_monotonic_ns", 0) >= work["end_ns"]
            and r.get("scope_sha256") == canonical_hash(work["validation_scope_manifest"])
            and reference(Path(r.get("log_path", "/nonexistent")))["sha256"] == r.get("log_sha256")
            and r.get("log_sha256") is not None
            for r in receipts
        )
    )


def build(work: Json, validation: Json) -> Json:
    """Administrative readiness grants no scientific benefit and never repairs history."""
    checked = receipts_pass(work, validation)
    with binding():
        value = _build(work, dict(validation, receipts=validation["receipts"] if checked else []))
    value.update(
        experiment_id=9013,
        random_seed=7709013,
        honest_verdict=value["honest_verdict"].replace("v769", "v770"),
        validation_receipts=validation["receipts"],
        validation_scope_manifest=work["validation_scope_manifest"],
        repository_health=work["repository_health"],
        repository_health_passed=False,
        archive_lag=work["archive_lag"],
        methodology_note="Bind fourteen full V770 tasks, current active or staged bytes and immutable V769 history. Authenticate original four development criteria, states and 1024 graph roles through unchanged V768 science checks. Compare a single retained ARC ledger against Exp8999. Current owned and direct-consumer checks are separate from three original failed global-health receipts. No model, fitting, generation or device benchmark is run.",
    )
    value["raw_shard_hashes"].append(
        reference(Path(work["raw"]) / "validation_scope_manifest.json")
    )
    value["field_principles"] = dict.fromkeys(
        [*value, "reproducibility_checksum"],
        "Retained bytes and current normal-exit receipts bind administrative custody only; failed historical science stays disqualified and optional scientific gates remain independent.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> Json:
    """Reconstruct all summaries so rehashing an aggregate cannot invent readiness."""
    try:
        value = json.loads(path.read_bytes())
        if not value["required_checks_passed"] or not verify_measured(
            value["owned_statement_coverage"], OWNED
        ):
            raise ValueError("unqualified_statements")
        for ref in value["raw_shard_hashes"] + value["code_config_hashes"]:
            if reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
                raise ValueError("operand_drift")
            if (
                ref.get("executed_source")
                and reference(Path(ref["source"]["path"])) != ref["source"]
            ):
                raise ValueError("executed_code_drift")
        ref = value["raw_shard_hashes"][0]
        work = json.loads(Path(ref["path"]).read_bytes())
        raw, retained = Path(work["raw"]), Path(work["raw"]) / "authority"
        if (
            work["contract"] != authority(retained)
            or work["science"] != e.science(retained)
            or work["arc"] != e.frontier(retained)
            or work["repository_health"] != e.health(retained)
            or work["validation_scope_manifest"] != scope_manifest()
        ):
            raise ValueError("authority_or_reduction_drift")
        for control in work["controls"]:
            if not control["passed"] or authority(raw / "controls" / control["mutation"])["ready"]:
                raise ValueError("mutation_control_drift")
        validation_path = raw / "validation.json"
        validation = (
            json.loads(validation_path.read_bytes())
            if validation_path.is_file()
            else dict(
                receipts=value["validation_receipts"],
                coverage=value["owned_statement_coverage"],
                global_test_debt=value["global_test_debt"],
            )
        )
        expected = build(dict(work, raw_reference=ref), validation)
        if validation_path.is_file():
            expected["raw_shard_hashes"].append(reference(validation_path))
        expected["reproducibility_checksum"] = checksum(expected)
        return dict(
            passed=value == expected and checksum(value) == value["reproducibility_checksum"],
            reason="primitive_replay",
        )
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return dict(passed=False, reason="invalid_or_changed_evidence")


_validate = runner.validate
_terminal = runner.terminal


def run_check(
    name: str, argv: list[str], raw: Path, deadline: float = 120, scope: str = "owned"
) -> Json:
    """Keep explicit direct consumers and learning restart inside the frozen scope."""
    if name == "owned_tests":
        argv += DIRECT
    if name == "private_e2e018_021":
        argv += [E2E[2]]
    if name == "spec_coverage":
        argv += DIRECT
    result = old.run_check(name, argv, raw, deadline, scope)
    result["scope_sha256"] = canonical_hash(scope_manifest())
    atomic_json(raw / (name + ".bound_receipt.json"), result)
    return result


def validate(raw: Path, tracer: Any) -> Json:
    """Reuse bounded scoped checks and actual SQLite statement measurement."""
    result = _validate(raw, tracer)
    result["receipts"][-1] = run_check(
        "owned_statement_coverage",
        [
            sys.executable,
            "-m",
            "coverage",
            "report",
            "--rcfile",
            str(raw / "coverage.ini"),
            "--data-file",
            str(raw / ".coverage"),
            "--fail-under=100",
        ],
        raw / "checks",
    )
    # Coverage's report command can combine shards; bind the bytes after it exits.
    result["coverage"]["data_reference"] = reference(raw / ".coverage")
    return result


def terminal(candidate: Path, raw: Path) -> Json:
    """A valid cold replay must also reject a rehashed aggregate in a real child."""
    result = _terminal(candidate, raw)
    forged = json.loads(candidate.read_bytes())
    forged["contract_ready_score"] = 99
    forged["reproducibility_checksum"] = checksum(forged)
    tampered = raw / "tampered.json"
    atomic_json(tampered, forged)
    from carnot.reporting.v709_execution import child

    with patch.object(runner, "child", partial(child, expected=1)):
        refusal = run_check(
            "rehashed_tamper_refused",
            [
                "env",
                "-u",
                "PYTHONPATH",
                "-C",
                "/var/tmp",
                sys.executable,
                "-u",
                str(ROOT / CLI),
                "--cold-replay",
                str(tampered),
            ],
            raw,
        )
    result.setdefault("receipts", []).append(refusal)
    result["passed"] = result["passed"] and refusal["passed"]
    atomic_json(raw / "terminal_validation.json", result)
    return result
