"""REQ-REPORT-8328: seal the existing reader rather than repeat historical authority.

The previous reader already joined supervisor counters to environment actions.
This adapter changes only the cutoff identity and final admissible date.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import time
from types import FunctionType
from typing import Any, cast

from carnot.reporting import arc_coverage_frontier_8314 as reader
from carnot.reporting import arc_supervisor_frontier_8243 as native
from carnot.reporting import v718_contract_replay as authority
from carnot.reporting.arc_supervisor_v688_receipts import event_order
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import publish_primary, read_bound_sidecar

Json = dict[str, Any]
ROOT = reader.ROOT
FRONTIER = ROOT / "results/experiment_8314_v717_arc_coverage_frontier.json"
PIN = "sha256:628dda6ce5252338d22dfeb63940c4fd10de452f7e3e9c89121f3c1d0dedb959"
READERS = [
    "python/carnot/reporting/arc_coverage_frontier_8314.py",
    "python/carnot/reporting/arc_coverage_execution_8314.py",
    "python/carnot/reporting/arc_supervisor_frontier_8243.py",
    "python/carnot/reporting/arc_outcome_delta_8229.py",
    "python/carnot/reporting/arc_authoritative_frontier_8215.py",
    "python/carnot/reporting/arc_supervisor_v688_receipts.py",
    "python/carnot/reporting/arc_outcome_frontier_8257.py",
    "python/carnot/reporting/arc_outcome_frontier_8272.py",
    "python/carnot/reporting/arc_outcome_frontier_8286.py",
    "python/carnot/reporting/arc_outcome_frontier_8300.py",
]


def progress(phase: str) -> None:
    """Flush phase changes so the reader's real work remains visible."""
    print(f"[exp8328] phase={phase}", flush=True)


def json_document(path: Path) -> Json:
    """Keep byte parsing identical in authentication and cold replay."""
    return cast(Json, json.loads(path.read_bytes()))


def authenticate(path: Path) -> Json:
    """An exact previous primary qualifies mechanics, including an empty frontier."""
    if sha256_file(path) != PIN:
        raise ValueError("qualification_bytes")
    value = json_document(path)
    sidecar = path.parent / "raw" / path.stem / "validators" / (PIN[7:] + ".json")
    bound = read_bound_sidecar(path, sidecar)
    if bound.get("report", {}).get("passed") is not True or bound.get("primary_path") != str(
        path.absolute()
    ):
        raise ValueError("qualification_terminal")
    if value.get("arc_reader_ready_score") != 1 or value.get("required_checks_passed") is not True:
        raise ValueError("reader_not_qualified")
    for label in READERS:
        expected = value["code_config_hashes"].get(label) or value["source_artifact_hashes"].get(
            str(ROOT / label)
        )
        if expected is None or sha256_file(ROOT / label) != expected:
            raise ValueError("reader_code_changed:" + label)
    terminal = Path(value["terminal_validation_sidecar_path"])
    report = json_document(terminal)
    if report.get("primary_sha256") != PIN or report.get("report") != bound["report"]:
        raise ValueError("terminal_report")
    return value


def inspect(locator: Path, previous: Json, raw: Path) -> Json:
    """Project a checked interface identity; native source and environment gates stay active."""
    projection = raw / "experiment_8229_v711_arc_outcome_delta.json"
    value = dict(
        previous,
        experiment_id=8229,
        task_id="exp8229-arc-outcome-delta",
        schema="arc-outcome-delta-v711",
        arc_delta_ready_score=1,
    )
    publish_primary(projection, value, lambda _: dict(passed=True, source_sha256=PIN))
    scope = dict(
        vars(native),
        summarize=reader.summarize,
        event_order=lambda row, cutoff, current: event_order(row, cutoff, "20261009"),
    )
    adapted = FunctionType(native.inspect.__code__, scope)
    return cast(Json, adapted(locator, projection))


def recommend(delta: Json) -> list[Json]:
    """Describe supported curated cells using progress first and action cost second."""
    if not delta.get("proposed_arm_change"):
        return []
    cells = delta["per_game_arm_rows"]
    ordered = sorted(
        delta["shared_arms"],
        key=lambda arm: (
            -sum(c["helped"] / c["fired"] for c in cells if c["arm"] == arm),
            sum(
                c["actions_to_levelup"]["numerator"] / c["helped"] if c["helped"] else float("inf")
                for c in cells
                if c["arm"] == arm
            ),
            arm,
        ),
    )
    return [
        dict(
            arm_order=ordered,
            claim="descriptive_only",
            causal_superiority=False,
            live_priority_modified=False,
            per_game_arm_rows=cells,
            future_trial=dict(
                per_game_adapters=False, solve_provenance="live_agent_self_discovery"
            ),
        )
    ]


def measure(raw: Path, private: Path, precondition_failures: list[Json] | None = None) -> Json:
    """Authenticate real operands before inspecting any later supervisor observations."""
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True)
    progress("preconditions_before")
    hashes: dict[str, str] = {}
    checks: list[Json] = list(precondition_failures or [])
    previous: Json = {}
    delta = reader.summarize([], {})
    operand_path, field = private, "private_scratch"
    try:
        if checks:
            raise ValueError("missing_tools")
        hashes[str(native.authority.REGISTRY)] = sha256_file(native.authority.REGISTRY)
        if (
            not private.resolve().is_relative_to(Path("/tmp"))
            or shutil.disk_usage(private).free < 10_000_000
        ):
            raise ValueError("private_scratch")
        operand_path, field = FRONTIER, "arc_reader_ready_score"
        previous = authenticate(FRONTIER)
        hashes[str(FRONTIER)] = PIN
        operand_path, field = ROOT / authority.ACTIVE, "activated"
        contract = authority.authority(ROOT, raw / "authority")
        checks.extend(contract["gate_check_summary"])
        if not contract["activated"]:
            raise ValueError("current_authority")
        for label in [
            *READERS,
            authority.PROTOCOL,
            authority.DESIGN,
            authority.ACTIVE,
            "ops/exclusion_manifest.yaml",
            "python/carnot/reporting/v718_replay_history.py",
            "python/carnot/reporting/v718_replay_runner.py",
            "python/carnot/reporting/primary_publication.py",
            "scripts/adversarial_verify.py",
            "scripts/verdict_row_consistency_lint.py",
        ]:
            hashes[str(ROOT / label)] = sha256_file(ROOT / label)
        hashes[str(Path(previous["terminal_validation_sidecar_path"]))] = sha256_file(
            Path(previous["terminal_validation_sidecar_path"])
        )
        locator = raw / "authority_locator.v1.json"
        atomic_json(locator, reader.authority.discover())
        progress("preconditions_after_measurement_before")
        delta = inspect(locator, previous, raw / "adapter")
        checks.extend(delta["failures"])
        hashes.update(delta["source_artifact_hashes"])
    except (OSError, ValueError, KeyError) as error:
        if isinstance(error, OSError) and error.filename:
            operand_path, field = Path(error.filename), "is_file"
        checks.append(
            dict(
                native.authority.operand(
                    operand_path,
                    field,
                    True,
                    str(error) if operand_path.exists() else None,
                    sha256_file(operand_path) if operand_path.is_file() else None,
                ),
                passed=False,
            )
        )
    retained = {}
    for index, (label, digest) in enumerate(hashes.items()):
        saved = raw / "inputs" / (str(index) + "-" + digest[7:] + ".bin")
        saved.parent.mkdir(exist_ok=True)
        shutil.copyfile(label, saved)
        retained[str(saved)] = digest
    end = time.monotonic_ns()
    progress("measurement_after")
    return dict(
        delta=delta,
        failures=checks,
        source_artifact_hashes=hashes,
        snapshots=retained,
        prior_frontier=previous.get("current_frontier", {}),
        frontier_finished_at=previous.get("finished_at"),
        historical_model_provenance=previous.get("historical_model_provenance", []),
        phase_spans=[
            dict(
                phase="authenticate_and_inspect", started_monotonic_ns=start, ended_monotonic_ns=end
            )
        ],
        duration_s=(end - start) / 1e9,
        private_scratch=str(private),
    )
