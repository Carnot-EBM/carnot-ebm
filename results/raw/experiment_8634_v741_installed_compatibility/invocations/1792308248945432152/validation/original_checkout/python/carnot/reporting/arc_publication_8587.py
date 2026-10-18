"""REQ-REPORT-8587: qualify publication without granting new scientific credit.

The prior publication function remains intact. Private children exercise its
validator call before the existing runner freezes genuine coverage shards.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import coverage
from coverage.parser import PythonParser
import yaml

from carnot.reporting import arc_panel_replay_8561 as prior
from carnot.reporting import arc_panel_replay_8548 as old
from carnot.reporting import arc_panel_replay_runner_8548 as runner
from carnot.reporting.current_work_receipt import sha256_file
from scripts.conductor_gates import evaluate_gates

Json = dict[str, Any]
NAME = "experiment_8587_v737_arc_publication_qualification"
TASK = "exp8587-arc-publication-qualification"
TASK_PIN = "sha256:aab245ac8b3028b4713e33fcc5a892b20026480cddb644b7da5c85c0582a4205"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_arc_publication_8587.py"
UPSTREAM = old.ROOT / "results/experiment_8561_v735_arc_panel_replay_qualification.json"
UPSTREAM_PIN = prior.PRIOR_PIN
OWNED = [
    *prior.OWNED,
    "python/carnot/reporting/arc_publication_8587.py",
    "python/carnot/reporting/arc_publication_runner_8587.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
BUILD = old.build
REPLAY = old.replay


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush phase counts so a bounded child cannot silently hide a stalled run."""
    print(f"[exp8587] phase={phase} completed={completed} pending={pending}", flush=True)


def historical_receipt() -> Json:
    """Compare original SQL events with the exact source used by that invocation."""
    primary = old.panel.checked(UPSTREAM, UPSTREAM_PIN)
    work = old.panel.checked(
        Path(primary["work_reference"]["path"]), primary["work_reference"]["sha256"]
    )
    receipt = primary["child_coverage_receipts"]
    old.panel.checked_bytes(receipt["data_path"], receipt["data_sha256"])
    data = coverage.CoverageData(basename=receipt["data_path"])
    data.read()
    shards = []
    for ref in receipt["shards"]:
        old.panel.checked_bytes(ref["path"], ref["sha256"])
        shard = coverage.CoverageData(basename=ref["path"])
        shard.read()
        shards.append(shard)
    differences, unions, pins = {}, {}, {}
    statement = ""
    historical_validator = {}
    for path, ref in receipt["files"].items():
        source = Path(path)
        if source.name == "arc_panel_replay_8561.py":
            source = (
                Path(primary["work_reference"]["path"]).parent
                / "prior_execution_sources"
                / source.name
            )
        old.panel.checked_bytes(str(source), ref["sha256"])
        parser = PythonParser(filename=str(source))
        parser.parse_source()
        executed = set(data.lines(path) or [])
        if source.name == "arc_panel_replay_8561.py":
            line = next(
                n.lineno
                for n in ast.walk(ast.parse(source.read_text()))
                if isinstance(n, ast.Assign) and ast.unparse(n) == "report = validator(candidate)"
            )
            historical_validator = dict(line=line, executed=line in executed)
        operands_changed = (
            sorted(parser.statements) != ref["statements"]
            or sorted(executed) != ref["executed_lines"]
        )
        if operands_changed:
            raise ValueError("historical_statement_operands")
        differences[path] = sorted(parser.statements - executed)
        unions[path] = executed == {line for shard in shards for line in (shard.lines(path) or [])}
        pins[str(source)] = ref["sha256"]
        if differences[path]:
            statement = source.read_text().splitlines()[differences[path][0] - 1].strip()
    for check in work["family_receipts"]:
        for stream in ("stdout", "stderr"):
            old.panel.checked_bytes(check[stream + "_path"], check[stream + "_sha256"])
    union = old.exact_union(
        primary["exact_union_receipt"]["collected_node_ids"], work["family_receipts"]
    )
    current = Path(prior.__file__)
    call = next(
        n
        for n in ast.walk(ast.parse(current.read_text()))
        if isinstance(n, ast.Assign) and ast.unparse(n) == "report = validator(candidate)"
    )
    return dict(
        passed=all(unions.values())
        and union == primary["exact_union_receipt"]
        and union["passed"]
        and sorted(differences.values()) == [[], [], [], [], [282]]
        and statement == "with (",
        statement_differences=differences,
        exact_shard_unions=unions,
        source_hashes=pins,
        combined_data_path=receipt["data_path"],
        combined_data_sha256=receipt["data_sha256"],
        original_shards=receipt["shards"],
        original_union=union,
        historical_missing_statement=statement,
        historical_validator_call=historical_validator,
        prompt_diagnosis_matches_authenticated_source=False,
        current_statement_line=call.lineno,
        current_source_sha256=sha256_file(current),
        original_verdict=primary["verdict_class"],
    )


def panel_consumer(candidate: Path) -> Json:
    """Use the future live panel's shipped gate on private copies of exact bytes."""
    roadmap = old.ROOT / "research-roadmap.yaml"
    task = next(
        t
        for t in yaml.safe_load(roadmap.read_bytes())["tasks"]
        if t["id"] == "exp8588-arc-capability-panel"
    )
    gates = [g for g in task["gated_on"] if g["upstream"] == TASK]
    if gates != [
        dict(upstream=TASK, artifact_field="arc_panel_execution_ready_score", op="==", value=1)
    ]:
        raise ValueError("future_panel_gate_drift")
    value = json.loads(candidate.read_bytes())
    with TemporaryDirectory(prefix="exp8587-consumer-", dir="/var/tmp") as directory:
        primary = Path(directory) / (NAME + ".json")
        primary.write_bytes(candidate.read_bytes())
        checked = evaluate_gates(dict(gated_on=gates), primary.parent)
    return dict(
        accepted=checked.passed,
        matched_expectation=checked.passed == bool(value["arc_panel_execution_ready_score"]),
        consumer_task_id=task["id"],
        gates=gates,
        primary_sha256=sha256_file(candidate),
        authority_sha256=sha256_file(roadmap),
        summary=checked.summary,
    )


def build(work_path: Path, output: Path) -> Json:
    """Keep all original readiness predicates and add invocation-specific evidence."""
    value = BUILD(work_path, output)
    work = json.loads(work_path.read_bytes())
    if work.get("publication_cases") and not work["publication_cases"]["passed"]:
        value.update(
            verdict_class="disqualified",
            arc_panel_execution_ready_score=0,
            required_checks_passed=False,
        )
    value.update(
        experiment_id=8587,
        task_id=TASK,
        milestone="2026.10.737",
        run_date="20261016",
        random_seed=7378587,
    )
    value["honest_verdict"] = (
        f"complete_{value['verdict_class']}_v737_arc_publication_qualification"
    )
    value["exact_statement_differences"] = work.get("publication_history", {})
    value["publication_closure_receipts"] = work.get("publication_cases", {})
    value["panel_consumer_receipt"] = (
        work.get("publication_cases", {})
        .get("valid", {})
        .get("terminal", {})
        .get("panel_consumer", {})
    )
    value["acceptance_gates"]["real_publication_closure"] = work.get("publication_cases", {}).get(
        "passed", False
    )
    value["cited_upstream_artifacts"].append(
        dict(path=str(UPSTREAM), sha256=UPSTREAM_PIN, diagnostic_only=True)
    )
    failed_test = (
        old.ROOT
        / "results/raw"
        / NAME
        / "invocations/1792189521729507215/validation/owned_tests.receipt.json"
    )
    failed_pin = "sha256:09242f47553358f1cb73e68a94513480da2e6504d1c50739d31f74630c928879"
    failed_receipt = old.panel.checked(failed_test, failed_pin)
    value["prior_invocation_diagnostics"] = dict(
        path=str(failed_test),
        sha256=failed_pin,
        required_check_failed=not failed_receipt["passed"],
        reason="Owned regression supplied a relative test path to an outside-checkout CLI. Retain the failed attempt; the current test uses an absolute path.",
    )
    value["source_artifact_hashes"][str(failed_test)] = failed_pin
    value["methodology_note"] += (
        " Exp8548 and Exp8561 remain disqualified. Genuine private publication runs before coverage sealing; final publication runs after immutable evidence snapshots. No new level or leaderboard claim."
    )
    value["field_principles"].update(
        {
            k: "Retain actual invocation receipts; publication qualification grants no solve or learning credit."
            for k in value
            if k not in value["field_principles"]
        }
    )
    return value


def replay(value: Json) -> bool:
    """Rehashing cannot replace measured publication or original diagnostic operands."""
    try:
        if not REPLAY(value):
            return False
        work = old.panel.checked(
            Path(value["work_reference"]["path"]), value["work_reference"]["sha256"]
        )
        if work.get("publication_history") and historical_receipt() != work["publication_history"]:
            return False
        if work.get("publication_cases"):
            ref = work["publication_cases_reference"]
            if old.panel.checked(Path(ref["path"]), ref["sha256"]) != work["publication_cases"]:
                return False
            for snapshot in work["publication_cases"]["immutable_snapshots"]:
                old.panel.checked_bytes(snapshot["path"], snapshot["sha256"])
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False
