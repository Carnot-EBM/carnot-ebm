"""REQ-REPORT-8561: keep old evidence sealed while qualifying real trace events.

Small adapters retain the shipped reducers, validators and publication boundary.
The extra history receipt explains why a nominal coverage percentage was unsafe.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Any, Callable, Iterator
from unittest.mock import patch

import coverage
from coverage.parser import PythonParser
import yaml

from carnot.reporting import arc_panel_replay_8548 as old
from carnot.reporting import arc_panel_replay_runner_8548 as runner
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from scripts.conductor_gates import evaluate_gates

Json = dict[str, Any]
NAME = "experiment_8561_v735_arc_panel_replay_qualification"
TASK = "exp8561-arc-panel-replay-qualification"
TASK_PIN = "sha256:eaff5e3d8be18befc9930836b02aec443a2e33dbb9421ea078ba1cd54c59808c"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_arc_panel_replay_8561.py"
UPSTREAM = "results/experiment_8548_v734_arc_panel_replay_qualification.json"
UPSTREAM_PIN = "sha256:ba5a834865487d4f75c427638aefe0beecb5316b5e321d03fe94f5b3c01ad942"
SOURCES = old.ROOT / "results/raw" / NAME / "historical_sources"
OWNED = [*old.OWNED, "python/carnot/reporting/arc_panel_replay_8561.py", CLI]
BUILD = old.build
VALIDATION = runner.validation
PUBLISH = runner.publish
PUBLISH_PRIMARY = runner.publish_primary
HISTORICAL = [*runner.HISTORICAL, old.TEST]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush the current identity while inherited children retain their own labels."""
    print(f"[exp8561] phase={phase} completed={completed} pending={pending}", flush=True)


def historical_receipt() -> Json:
    """Recompute old statement differences from authenticated source and SQL data."""
    primary = old.panel.checked(old.ROOT / UPSTREAM, UPSTREAM_PIN)
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
    for path, ref in receipt["files"].items():
        source = SOURCES / Path(path).name if Path(path).name.endswith("8548.py") else Path(path)
        old.panel.checked_bytes(str(source), ref["sha256"])
        parser = PythonParser(filename=str(source))
        parser.parse_source()
        executed = set(data.lines(path) or [])
        differences[path] = sorted(parser.statements - executed)
        unions[path] = executed == {line for shard in shards for line in (shard.lines(path) or [])}
        pins[str(source)] = ref["sha256"]
        if (
            sorted(parser.statements) != ref["statements"]
            or sorted(executed) != ref["executed_lines"]
        ):
            raise ValueError("historical_statement_operands")
    replay = primary["terminal_replay_receipts"][0]
    for check in primary["validation_receipts"]:
        for stream in ("stdout", "stderr"):
            if stream + "_path" in check:
                old.panel.checked_bytes(check[stream + "_path"], check[stream + "_sha256"])
    union = old.exact_union(
        primary["exact_union_receipt"]["collected_node_ids"], work["family_receipts"]
    )
    return dict(
        passed=all(unions.values()) and union == primary["exact_union_receipt"] and union["passed"],
        statement_differences=differences,
        exact_shard_unions=unions,
        source_hashes=pins,
        combined_data_path=receipt["data_path"],
        combined_data_sha256=receipt["data_sha256"],
        original_terminal_replay=replay,
        original_union=primary["exact_union_receipt"],
        original_verdict=primary["verdict_class"],
        original_shards=receipt["shards"],
    )


def trace_reproduction(raw: Path) -> Json:
    """Use actual sysmon children for both truth values without changing collectors."""
    raw.mkdir(parents=True, exist_ok=True)
    forms = {
        "before": "    if (\n        value\n        or False\n    ):\n",
        "after": "    rejected = (\n        value\n        or False\n    )\n    if rejected:\n",
    }
    result = {}
    for name, condition in forms.items():
        source = raw / (name + ".py")
        source.write_text(
            "def probe(value):\n"
            + condition
            + "        return 1\n    return 0\nassert probe(True) == 1\nassert probe(False) == 0\n"
        )
        config = raw / (name + ".ini")
        config.write_text(
            "[run]\ncore=sysmon\nparallel=true\npatch=subprocess,_exit\ndata_file="
            + str(raw / (".coverage_" + name))
            + "\ninclude="
            + str(source)
            + "\n"
        )
        child = runner.invoke(
            "trace_" + name,
            [sys.executable, "-m", "coverage", "run", "--rcfile=" + str(config), str(source)],
            raw,
            deadline=30,
        )
        cov = coverage.Coverage(config_file=str(config))
        cov.combine(keep=True)
        cov.save()
        _, statements, _, missing, _ = cov.analysis2(str(source))
        executed = sorted(cov.get_data().lines(str(source)) or [])
        result[name] = dict(
            child=child,
            statements=statements,
            executed_lines=executed,
            missing_lines=missing,
            statement_difference=sorted(set(statements) - set(executed)),
            source_sha256=sha256_file(source),
        )
    result["passed"] = (
        all(result[n]["child"]["passed"] for n in forms)
        and bool(result["before"]["statement_difference"])
        and not result["after"]["statement_difference"]
    )
    atomic_json(raw / "trace_receipt.json", result)
    return result


def build(work_path: Path, output: Path) -> Json:
    """Retain the original qualification rules and give this invocation its identity."""
    value = BUILD(work_path, output)
    work = json.loads(work_path.read_bytes())
    value.update(
        experiment_id=8561,
        task_id=TASK,
        milestone="2026.10.735",
        run_date="20261016",
        random_seed=7358561,
    )
    value["honest_verdict"] = (
        f"complete_{value['verdict_class']}_v735_arc_panel_replay_qualification"
    )
    value["conditional_anchor_receipts"] = work.get("anchor_history", {})
    value["cited_upstream_artifacts"].append(
        dict(path=UPSTREAM, sha256=UPSTREAM_PIN, original_date="20261015", diagnostic_only=True)
    )
    value["methodology_note"] += (
        " Exp8548 remains disqualified. Named predicates retain every assertion and exact statement/shard predicate."
    )
    value["field_principles"]["conditional_anchor_receipts"] = (
        "Recompute old discrepancies from original SQL data and independently pinned source bytes; never clear historical flags."
    )
    return value


def measurement(work: Json, root: Path, raw: Path, private: Path) -> None:
    """Replay stored episodes once under the established bound; never repeat the kill."""
    try:
        history = historical_receipt()
        primary, original = old.authenticate(root)
    except (OSError, ValueError, KeyError) as exc:
        work["failures"][0].update(check="original_input_authentication", observed=str(exc))
        return
    work.update(
        anchor_history=history,
        history=old.history_receipts(primary),
        episode_hashes=old.episode_hashes(original),
        sources={str(root / old.ORIGINAL): old.PIN, str(root / UPSTREAM): UPSTREAM_PIN},
        historical_model_provenance=primary["historical_model_provenance"],
    )
    trace = trace_reproduction(raw / "trace")
    work["reproductions"] = [
        dict(check="conditional_trace", passed=trace["passed"], evidence=trace)
    ]
    work["checks"].append(
        dict(name="failure_reproductions", passed=history["passed"] and trace["passed"])
    )
    candidate = old.private_history(primary, original, private / "stored")
    receipt = runner.invoke(
        "original_terminal_replay",
        [sys.executable, "-u", str(root / old.panel.CLI), "--cold-replay", str(candidate)],
        raw / "measurement",
        deadline=240,
    )
    work["checks"].append(receipt)
    work["terminal_replays"].append(receipt)
    for name, flag, filename in [
        ("primitive_replay", "--primitives", "replay_rows.json"),
        ("original_fixtures", "--fixtures", "fixtures.json"),
    ]:
        target = raw / filename
        receipt = runner.invoke(
            name,
            [sys.executable, "-u", str(root / CLI), flag, str(target)],
            raw / "measurement",
            deadline=240,
        )
        work["checks"].append(receipt)
        if name == "primitive_replay":
            work["terminal_replays"].append(receipt)
        if receipt["passed"]:
            work["replay_rows" if name == "primitive_replay" else "fixtures"] = json.loads(
                target.read_bytes()
            )
    work["failures"] = []


def panel_consumer(candidate: Path) -> Json:
    """Run the future panel's actual gate against these bytes in a private directory."""
    roadmap = old.ROOT / "research-roadmap.yaml"
    plan = yaml.safe_load(roadmap.read_bytes())
    task = next(t for t in plan["tasks"] if t["id"] == "exp8562-arc-capability-panel")
    gates = [g for g in task["gated_on"] if g["upstream"] == TASK]
    if gates != [
        dict(upstream=TASK, artifact_field="arc_panel_execution_ready_score", op="==", value=1)
    ]:
        raise ValueError("future_panel_gate_drift")
    value = json.loads(candidate.read_bytes())
    with TemporaryDirectory(prefix="exp8561-consumer-", dir="/var/tmp") as directory:
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


def publish(value: Json, output: Path, raw: Path) -> Json:
    """Bind the real panel gate to the final candidate before atomic publication."""

    def checked_publish(
        path: Path, candidate_value: Any, validator: Callable[[Path], Json]
    ) -> Json:
        def validate(candidate: Path) -> Json:
            report = validator(candidate)
            consumer = panel_consumer(candidate)
            return dict(
                report,
                panel_consumer=consumer,
                passed=report["passed"] and consumer["matched_expectation"],
            )

        return PUBLISH_PRIMARY(path, candidate_value, validate)

    with patch.object(runner, "publish_primary", checked_publish):
        return dict(PUBLISH(value, output, raw))


@contextmanager
def adapters() -> Iterator[None]:
    """Scope identity changes so imported historical tests keep their original authority."""
    with (
        patch.multiple(
            old,
            NAME=NAME,
            TASK=TASK,
            TASK_PIN=TASK_PIN,
            CLI=CLI,
            TEST=TEST,
            OWNED=OWNED,
            RUN_DATE="20261016",
            build=build,
            progress=progress,
        ),
        patch.multiple(runner, measurement=measurement, HISTORICAL=HISTORICAL, publish=publish),
    ):
        yield


def main(argv: list[str] | None = None) -> int:
    """Use the shipped bounded CLI; original suite workers retain original identities."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--suite" in arguments:
        with patch.object(runner, "HISTORICAL", HISTORICAL):
            return runner.main(arguments)
    with adapters():
        return runner.main(arguments)
