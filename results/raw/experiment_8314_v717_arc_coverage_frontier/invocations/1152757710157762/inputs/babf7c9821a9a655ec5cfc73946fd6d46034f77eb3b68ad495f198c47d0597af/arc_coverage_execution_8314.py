"""REQ-VERIFY-8314: qualify coverage before invoking the authenticated reducer.

Private namespaces reuse the checked executor, process supervisor and publisher.
Preflight children run once; their durable receipts are reused during closeout.
"""

from collections.abc import Callable
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
import time
from types import FunctionType, SimpleNamespace
from typing import Any, cast

from carnot.reporting import arc_outcome_execution_8300 as baseline
from carnot.reporting import arc_coverage_frontier_8314 as reader
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.reporting.v686_contract_validation import coverage_complete

ROOT = reader.ROOT
CLI = "scripts/experiments/experiment_8314_v717_arc_coverage_frontier.py"
OUTPUT = ROOT / "results/experiment_8314_v717_arc_coverage_frontier.json"
TEST = "tests/python/test_arc_coverage_frontier_8314.py"
OWNED = [
    "python/carnot/reporting/arc_coverage_frontier_8314.py",
    "python/carnot/reporting/arc_coverage_execution_8314.py",
    CLI,
    baseline.OWNED[1],
]
CONSUMERS = baseline.CONSUMERS
qualified = baseline.qualified
Json = dict[str, Any]
BUDGET = dict(
    minimum_overlapping_games=3,
    minimum_shared_arms=2,
    minimum_firings_per_game_arm=5,
    independent_unit="game",
    current_game_runs=0,
    current_model_runs=0,
)


def bind(name: str) -> Callable[..., Any]:
    """Reuse original checked statements with this invocation's private globals."""
    original = getattr(qualified, name)
    return cast(Callable[..., Any], FunctionType(original.__code__, vars(qualified) | globals()))


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual counts so bounded silent children remain observable."""
    baseline.progress(phase.replace("exp8257", "exp8314"), completed, pending)


child = baseline.child
consumer_summary = baseline.consumer_summary


def historical_gap() -> Json:
    """Authenticate the persisted failure rather than infer it from today's code."""
    value = json.loads(baseline.OUTPUT.read_text())
    digest = sha256_file(baseline.OUTPUT)
    sidecar = (
        baseline.OUTPUT.parent
        / "raw"
        / baseline.OUTPUT.stem
        / "validators"
        / (digest[7:] + ".json")
    )
    report = read_bound_sidecar(baseline.OUTPUT, sidecar)
    label, expected = next(
        (p, h) for p, h in value["raw_shard_hashes"].items() if p.endswith("/coverage.json")
    )
    coverage = Path(label)
    if (
        report["report"]["passed"] is not True
        or sha256_file(coverage) != expected
        or value["verdict_class"] != "disqualified"
    ):
        raise ValueError("historical_coverage_authentication")
    measured = json.loads(coverage.read_text())["files"][baseline.OWNED[1]]
    return dict(
        measured["summary"],
        missing_lines=measured["missing_lines"],
        path=label,
        sha256=expected,
        primary_sha256=digest,
        sidecar_path=str(sidecar),
        sidecar_sha256=sha256_file(sidecar),
        full_suite_health="historical_timeout_not_rerun",
    )


def commands(private: Path) -> list[Json]:
    """Freeze current coverage and the unchanged seven bounded consumer files."""
    plan = baseline.commands(private)
    replacements = dict(zip(baseline.OWNED, OWNED[:3], strict=True)) | {baseline.TEST: TEST}
    plan = [s for s in plan if s["classification"] != "repository_health"]
    for spec in plan:
        spec["argv"] = [replacements.get(a, a) for a in spec["argv"]]
        if spec["name"] == "combine_coverage":
            spec["argv"].insert(4, "--keep")
        if spec["name"] == "unit_child_coverage":
            spec["deadline_s"] = 240
    config = private / "coverage.ini"
    content = config.read_text()
    for old, new in replacements.items():
        content = content.replace(old, new)
    config.write_text(content + "    " + str(ROOT / baseline.OWNED[1]) + "\n")
    plan.append(
        dict(
            name="e2e_023",
            argv=[
                str(ROOT / ".venv/bin/python"),
                "-u",
                "-m",
                "pytest",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                "tests/python/test_arc_authoritative_frontier_8215.py",
                "--basetemp=" + str(private / "e2e023"),
            ],
            deadline_s=180,
            expected_exit=0,
            classification="required",
        )
    )
    return plan


def preconditions(private: Path) -> Json:
    """Check private storage, tools and bound historical failures before measuring."""
    available = baseline.preconditions(private)
    for label in [*OWNED, TEST]:
        path = ROOT / label
        digest = sha256_file(path) if path.is_file() else None
        available["checks"].append(
            dict(
                reader.authority.operand(path, "is_file", True, path.is_file(), digest),
                required=True,
                passed=path.is_file(),
            )
        )
        if digest:
            available["hashes"][str(path)] = digest
    gap: Json = {}
    if baseline.OUTPUT.is_file():
        available["hashes"][str(baseline.OUTPUT)] = sha256_file(baseline.OUTPUT)
    try:
        gap = historical_gap()
        for label in [gap["path"], gap["sidecar_path"]]:
            available["hashes"][label] = sha256_file(Path(label))
        valid: Any = (
            gap["covered_lines"] == 84
            and gap["num_statements"] == 85
            and gap["missing_lines"] == [212]
        )
    except (ValueError, OSError, KeyError, StopIteration) as error:
        valid = str(error)
    available["checks"].append(
        dict(
            reader.authority.operand(
                baseline.OUTPUT,
                "historical_coverage_gap_authenticated",
                True,
                valid,
                sha256_file(baseline.OUTPUT) if baseline.OUTPUT.is_file() else None,
            ),
            required=True,
            passed=valid is True,
        )
    )
    available["failures"] = [r for r in available["checks"] if not r["passed"]]
    available["historical_coverage_gap"] = gap
    return available


def measured_coverage(private: Path, shard_root: Path | None = None) -> bool:
    """A JSON summary cannot replace missing unit and real CLI SQLite shards."""
    shards = private / "coverage" if shard_root is None else shard_root
    return bool(list(shards.glob(".coverage.*"))) and coverage_complete(
        private / "coverage.json", includes=OWNED
    )


def normalize_artifact_for_template_write(value: Json) -> Json:
    """Separate reader mechanics from observational support and historical failures."""
    value = baseline.normalize_artifact_for_template_write(value)
    fixture = bool(value["fixture_claim_scope"])
    ready = bool(
        value["required_checks_passed"]
        and not fixture
        and value["acceptance_gates"]["authority_authenticated"]
    )
    summary = value["consumer_validation"]
    ready = ready and summary["collection_equivalent"] and summary["elapsed_s"] <= 1800
    missing = (
        sum(v["missing_lines"] for v in value["coverage_statement_counts"].values())
        if value["coverage_statement_counts"]
        else 1
    )
    value.update(
        experiment_id=8314,
        experiment=8314,
        task_id=reader.TASK_ID,
        milestone="2026.10.717",
        run_date="20261008",
        schema="arc-coverage-frontier-v717",
        title="Qualified ARC coverage and authenticated supervisor frontier",
        arc_reader_ready_score=int(ready and missing == 0),
        arc_outcome_support_score=int(ready and bool(value["proposed_arm_change"])),
        owned_missing_statement_count=missing,
        qualified_frontier=value["receipt_frontier"],
        sample_size_budget=BUDGET,
        solve_provenance="live_agent_self_discovery",
        historical_disqualifications=[
            dict(
                path=str(ROOT / "results" / f"experiment_{exp}_{suffix}_arc_outcome_frontier.json"),
                sha256=value["source_artifact_hashes"].get(
                    str(ROOT / "results" / f"experiment_{exp}_{suffix}_arc_outcome_frontier.json")
                ),
                verdict_class="disqualified",
            )
            for exp, suffix in [(8286, "v715"), (8300, "v716")]
        ],
        historical_coverage_gap=value.get("historical_coverage_gap", {}),
        methodology="Qualify private unit and real CLI coverage, including the V716 consumer-summary rejection, before authenticating post-Exp8272 supervisor outcomes. Keep exposed observational support separate from reader readiness; no live defaults change.",
    )
    if value["verdict_class"] == "null" and value["new_outcome_count"] == 0:
        value["honest_verdict"] = "complete_null_no_supervisor_outcomes"
    value["field_principles"].update(
        {
            k: "Bind current coverage and imported live receipt provenance without solve or generalization credit."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["cited_upstream_artifacts"].extend(
        dict(
            row,
            fields_imported=["verdict_class", "coverage_statement_counts", "validation_receipts"],
        )
        for row in value["historical_disqualifications"]
    )
    value["field_principles"].update(
        arc_reader_ready_score="Full owned coverage, replayable bytes and passing isolated consumers qualify an empty read.",
        arc_outcome_support_score="Observational cell support is separate from reader execution readiness.",
        owned_missing_statement_count="Count measured missing statements; absent coverage cannot mean zero.",
        qualified_frontier="Only the authenticated Exp8272 clock and receipt IDs bound later outcomes.",
        sample_size_budget="Freeze game, arm and cell floors; repeated firings are not independent games.",
        historical_disqualifications="Keep Exp8286/8300 outside qualified frontier and solve credit.",
        historical_coverage_gap="Persisted byte-bound V716 coverage identifies the exact missed rejection statement.",
        coverage_shards="Retained unit and CLI SQLite shards prevent a summary from substituting for measurement.",
        solve_provenance="Imported live receipts originated in agent self-discovery; this invocation solves no games.",
    )
    return value


def replay(value: Json) -> list[str]:
    """Recompute primitives and current readiness, including forged summary rejection."""
    expected = dict(
        experiment_id=8314,
        experiment=8314,
        task_id=reader.TASK_ID,
        milestone="2026.10.717",
        run_date="20261008",
        schema="arc-coverage-frontier-v717",
        solve_provenance="live_agent_self_discovery",
        sample_size_budget=BUDGET,
    )
    errors = ["execution_claim:" + k for k, v in expected.items() if value.get(k) != v]
    shards = sorted(p for p in value["raw_shard_hashes"] if "/coverage/.coverage." in p)
    if value.get("coverage_shards") != shards:
        errors.append("coverage_shard_drift")
    if value["acceptance_gates"]["owned_coverage_100"] and not value["fixture_claim_scope"]:
        report = Path(value["primitive_path"]).parent / "coverage.json"
        if not shards or not coverage_complete(report, includes=OWNED):
            errors.append("owned_coverage_evidence_missing")
    summary = consumer_summary(value["validation_receipts"])
    missing = (
        sum(v["missing_lines"] for v in value["coverage_statement_counts"].values())
        if value["coverage_statement_counts"]
        else 1
    )
    ready = bool(
        value["required_checks_passed"]
        and not value["fixture_claim_scope"]
        and value["acceptance_gates"]["authority_authenticated"]
        and summary["collection_equivalent"]
        and summary["elapsed_s"] <= 1800
        and missing == 0
    )
    for key, expected_value in dict(
        arc_reader_ready_score=int(ready),
        arc_outcome_support_score=int(ready and bool(value["proposed_arm_change"])),
        owned_missing_statement_count=missing,
        qualified_frontier=value["receipt_frontier"],
    ).items():
        if value.get(key) != expected_value:
            errors.append("execution_claim:" + key)
    translated = dict(
        value,
        experiment_id=8300,
        experiment=8300,
        milestone="2026.10.716",
        schema="arc-outcome-frontier-v716",
        solve_provenance="live_agent_self_discovery" if value["new_outcome_count"] else None,
    )
    scope = (
        globals() | {"reader": SimpleNamespace(**(vars(reader) | {"inspect": reader.unqualified}))}
        if not value["acceptance_gates"]["owned_coverage_100"]
        else globals()
    )
    legacy_bind = lambda name: FunctionType(
        getattr(qualified, name).__code__, vars(qualified) | scope
    )
    legacy = FunctionType(
        baseline.replay.__code__, vars(baseline) | globals() | {"bind": legacy_bind}
    )
    return errors + cast(list[str], legacy(translated))


def terminal(candidate: Path, raw: Path) -> Json:
    """Use unchanged public validators and the current standalone cold-replay CLI."""
    adapted = FunctionType(baseline.terminal.__code__, vars(baseline) | globals())
    return cast(Json, adapted(candidate, raw))


def execute(
    locator: Path, frontier: Path, output: Path, private: Path, *, fixture: bool = False
) -> int:
    """Qualify once before aggregation, then retain every original execution receipt."""
    start = time.monotonic_ns()
    started_at = datetime.now(UTC).isoformat()
    private.mkdir(parents=True, exist_ok=True)
    progress("exp8314_preflight", 0, 1)
    available = preconditions(private)
    plan = [] if fixture else commands(private)
    qualification = output.parent / "raw" / output.stem / "qualification" / str(time.monotonic_ns())
    qualification.mkdir(parents=True, exist_ok=True)
    manifest = qualification / "command_manifest.json"
    atomic_json(
        manifest,
        dict(
            commands=plan,
            private_path=str(private),
            owned_files=OWNED,
            applicable_e2e=["E2E-015", "E2E-017", "E2E-019", "E2E-023"],
            heartbeat_s=30,
        ),
    )
    cached: dict[str, Json] = {}
    retained = {str(manifest): sha256_file(manifest)}
    early = [
        s
        for s in plan
        if s["name"]
        in {"python_environment", "unit_child_coverage", "combine_coverage", "coverage_100"}
    ]
    for index, spec in enumerate(early):
        progress("exp8314_coverage", index, len(early) - index)
        cached[spec["name"]] = child(
            spec["name"],
            spec["argv"],
            qualification / "logs",
            deadline=spec["deadline_s"],
            expected=spec["expected_exit"],
            scope=spec["classification"],
        )
        # Reporting combines and deletes parallel inputs, so copy each real
        # shard as soon as its child exits. A later JSON summary cannot replace it.
        for original in (private / "coverage").glob(".coverage*"):
            saved = qualification / "coverage" / original.name
            saved.parent.mkdir(exist_ok=True)
            shutil.copyfile(original, saved)
            retained[str(saved)] = sha256_file(saved)
    covered = measured_coverage(private, qualification / "coverage")
    progress("exp8314_coverage_complete", int(covered), int(not covered))
    qualified_at = time.monotonic_ns()

    def reuse(name: str, argv: list[str], logs: Path, **kwargs: Any) -> Json:
        """Retain preflight exits without launching the same children twice."""
        if name in cached:
            row = cached[name]
            atomic_json(logs / (name + ".receipt.json"), row)
            return row
        return child(name, argv, logs, **kwargs)

    def normalize(value: Json) -> Json:
        """Bind surviving SQLite shards and the premeasurement command manifest."""
        value["historical_coverage_gap"] = available["historical_coverage_gap"]
        normalized = normalize_artifact_for_template_write(value)
        normalized["raw_shard_hashes"].update(retained)
        normalized["coverage_shards"] = sorted(p for p in retained if "/coverage/.coverage." in p)
        normalized["started_at"] = started_at
        normalized["duration_s"] = (time.monotonic_ns() - start) / 1e9
        normalized["phase_spans"].insert(
            0,
            dict(
                phase="coverage_qualification",
                started_monotonic_ns=start,
                ended_monotonic_ns=qualified_at,
            ),
        )
        return normalized

    gated_reader = (
        reader
        if covered or fixture
        else SimpleNamespace(**(vars(reader) | {"inspect": reader.unqualified}))
    )
    scope = globals() | {
        "reader": gated_reader,
        "commands": lambda _: plan,
        "preconditions": lambda _: available,
        "child": reuse,
        "coverage_complete": lambda *a, **k: covered,
        "normalize_artifact_for_template_write": normalize,
    }
    rebound = lambda name: FunctionType(getattr(qualified, name).__code__, vars(qualified) | scope)
    adapted = FunctionType(
        baseline.execute.__code__, vars(baseline) | globals() | {"bind": rebound}
    )
    return int(adapted(locator, frontier, output, private, fixture=fixture))
