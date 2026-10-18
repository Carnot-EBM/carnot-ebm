"""REQ-REPORT-8548: qualify infrastructure while preserving historical quarantine.

Stored episodes are diagnostic operands. The shipped reducers verify actions
and supervisor effects; this reader never steps a game or invokes a model.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
from typing import Any

import coverage
import yaml

from carnot.reporting import arc_capability_panel_8537 as panel
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.historical_consumer_roots_8402 import parse_outcomes
from carnot.reporting.primary_publication import read_bound_sidecar

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8548_v734_arc_panel_replay_qualification"
TASK = "exp8548-arc-panel-replay-qualification"
TASK_PIN = "sha256:78e9097d818d3576561a665a616d8ae86256744af1a28d4bcd34e5e16df3267c"
ORIGINAL = "results/experiment_8537_v733_arc_capability_panel.json"
PIN = "sha256:a6f2f35bd7f96e046eaa2f08c1b141e149cb19c0bb4494da40c959c0ae1f83f3"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_arc_panel_replay_8548.py"
OWNED = [
    "python/carnot/reporting/arc_panel_replay_8548.py",
    "python/carnot/reporting/arc_panel_replay_runner_8548.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
RUN_DATE = "20261015"
REQUIRED = {
    "owned_tests",
    "coverage_report",
    "ruff_check",
    "ruff_format",
    "strict_mypy",
    "spec_coverage",
    "original_terminal_replay",
    "primitive_replay",
    "original_fixtures",
    "missing",
    "intentional",
    "rehashed_tamper",
    "failure_reproductions",
    "original_union",
    "exact_owned_coverage",
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush counts because a quiet reducer still needs a visible supervisor."""
    print(f"[exp8548] phase={phase} completed={completed} pending={pending}", flush=True)


def authenticate(root: Path) -> tuple[Json, Json]:
    """Pin original authority before any historical observation can be used."""
    primary = panel.checked(root / ORIGINAL, PIN)
    work = panel.checked(
        Path(primary["work_reference"]["path"]), primary["work_reference"]["sha256"]
    )
    side = json.loads(Path(primary["terminal_validation_sidecar_path"]).read_bytes())
    bound = read_bound_sidecar(root / ORIGINAL, Path(side["publication"]["sidecar_path"]))
    authority = work["activation_reference"]
    plan = yaml.safe_load(panel.checked_bytes(authority["snapshot_path"], authority["sha256"]))
    task = next(t for t in plan["tasks"] if t["id"] == panel.TASK)
    original_authority_invalid = (
        canonical_hash(task) != panel.TASK_PIN
        or primary["run_date"] != "20261015"
        or primary["verdict_class"] != "disqualified"
        or not bound["report"]["passed"]
    )
    if original_authority_invalid:
        raise ValueError("original_authority_or_quarantine")
    for path, digest in work["code_config_hashes"].items():
        panel.checked_bytes(path, digest)
    episode_hashes(work)
    return primary, work


def episode_hashes(work: Json) -> Json:
    """Keep attempts, steps and terminal seals distinct from a summary checksum."""
    hashes = {}
    for ref in work["references"]:
        directory = Path(ref["directory"])
        for filename, digest in [
            ("episode.json", ref["terminal_sha256"]),
            ("steps.jsonl", ref["steps_sha256"]),
            ("attempts.jsonl", sha256_file(directory / "attempts.jsonl")),
        ]:
            if sha256_file(directory / filename) != digest:
                raise ValueError("original_episode_seal")
            hashes[str(directory / filename)] = digest
    return hashes


def history_receipts(primary: Json) -> Json:
    """Later successful checks cannot erase the original killed replay or24 percent."""
    receipts = primary["validation_receipts"]
    old = next(r for r in receipts if r["name"] == "failed_initial_terminal_replay")
    budget = next(r for r in receipts if r["name"] == "budget_coverage_report")
    for receipt in (old, budget):
        for stream in ("stdout", "stderr"):
            panel.checked_bytes(receipt[stream + "_path"], receipt[stream + "_sha256"])
    return dict(
        old_timeout=old,
        resumed_coverage_receipt=budget,
        resumed_coverage_percent=24,
        original_verdict=primary["verdict_class"],
        diagnostic_only=True,
    )


def private_history(primary: Json, work: Json, directory: Path) -> Path:
    """Copy original episode bytes to disk scratch without collecting fresh actions."""
    directory.mkdir(parents=True, exist_ok=True)
    copied = deepcopy(work)
    for index, ref in enumerate(copied["references"]):
        target = directory / str(index)
        shutil.copytree(ref["directory"], target)
        ref["directory"] = str(target)
    path = directory / "work.json"
    atomic_json(path, copied)
    value = deepcopy(primary)
    value["work_reference"] = dict(path=str(path), sha256=sha256_file(path))
    value["reproducibility_checksum"] = canonical_hash(copied)
    value["raw_shard_hashes"] = {
        str(Path(r["directory"]) / f): r[k]
        for r in copied["references"]
        for f, k in [("episode.json", "terminal_sha256"), ("steps.jsonl", "steps_sha256")]
    }
    relocated = {r["unit"]["episode_id"]: r["directory"] for r in copied["references"]}
    for field in ("rows", "per_episode_rows"):
        for row in value[field]:
            row["primitive_directory"] = relocated[row["episode_id"]]
    candidate = directory / "candidate.json"
    atomic_json(candidate, value)
    return candidate


def reduce_history(work: Json) -> list[Json]:
    """Replay each selection and effect once; report no new policy estimate."""
    rows = []
    for index, ref in enumerate(work["references"]):
        progress("episode_replay_before", index, len(work["references"]) - index)
        row, effects, exclusions = panel.reduce_episode(ref)
        rows.append(
            dict(
                episode_id=ref["unit"]["episode_id"],
                status="replayed",
                missing_reason=None,
                observation=row,
                effects=effects,
                exclusions=exclusions,
                diagnostic_history_only=True,
            )
        )
        progress("episode_replay_after", index + 1, len(work["references"]) - index - 1)
    return rows


def exact_union(nodes: list[str], receipts: list[Json]) -> Json:
    """An exit zero needs a terminal pass for every original node, exactly once."""
    planned = [n for r in receipts for n in r["collected_node_ids"]]
    outcomes = [
        dict(node_id=n, outcome=status)
        for r in receipts
        for status, group in parse_outcomes(Path(r["stdout_path"])).items()
        if status != "summary"
        for n in group
    ]
    terminal = [r["node_id"] for r in outcomes]
    passed = (
        bool(nodes)
        and len(nodes) == len(set(nodes))
        and sorted(nodes) == sorted(planned) == sorted(terminal)
        and all(r["outcome"] == "passed" for r in outcomes)
        and all(r["passed"] and not r["timed_out"] and r["exit_code"] == 0 for r in receipts)
    )
    return dict(
        passed=passed,
        collected_node_ids=nodes,
        planned_node_ids=planned,
        terminal_node_outcomes=outcomes,
        disjoint=len(planned) == len(set(planned)),
    )


def coverage_receipt(config: Path, pins: Json) -> Json:
    """Use independently analysed statements and actual executed lines for each source."""
    cov = coverage.Coverage(config_file=str(config))
    cov.load()
    files = {}
    for name in OWNED:
        path = ROOT / name
        _, statements, _, missing, _ = cov.analysis2(str(path))
        executed = sorted(cov.get_data().lines(str(path)) or [])
        files[str(path)] = dict(
            sha256=sha256_file(path),
            statements=statements,
            executed_lines=executed,
            missing_lines=missing,
        )
    passed = all(
        r["sha256"] == pins.get(p)
        and not r["missing_lines"]
        and set(r["statements"]) <= set(r["executed_lines"])
        for p, r in files.items()
    )
    shards = []
    for shard in sorted(
        Path(cov.config.data_file).parent.glob(Path(cov.config.data_file).name + ".*")
    ):
        data = coverage.CoverageData(basename=str(shard))
        data.read()
        shards.append(
            dict(
                path=str(shard),
                sha256=sha256_file(shard),
                executed_lines={p: sorted(data.lines(p) or []) for p in files if data.lines(p)},
            )
        )
    passed = (
        passed
        and bool(shards)
        and all(
            set(r["executed_lines"])
            == {line for shard in shards for line in shard["executed_lines"].get(path, [])}
            for path, r in files.items()
        )
    )
    return dict(
        passed=passed,
        shards=shards,
        files=files,
        data_path=str(cov.config.data_file),
        data_sha256=sha256_file(Path(cov.config.data_file)),
    )


def build(work_path: Path, output: Path) -> Json:
    """Separate execution qualification from every scientific benefit claim."""
    work = json.loads(work_path.read_bytes())
    names = {r["name"] for r in work["checks"]}
    qualified = (
        REQUIRED <= names
        and all(r["passed"] for r in work["checks"])
        and work["coverage"].get("passed", False)
        and work["union"].get("passed", False)
    )
    failed = any(not r["passed"] for r in work["checks"])
    ready = int(
        qualified
        and not work["failures"]
        and not work["findings"]
        and len(work["replay_rows"]) == 16
    )
    verdict = (
        "disqualified"
        if failed
        else "blocked"
        if work["failures"]
        else "circular_positive"
        if ready
        else "disqualified"
    )
    value = dict(
        experiment_id=8548,
        task_id=TASK,
        milestone="2026.10.734",
        run_date="20261015",
        honest_verdict=f"complete_{verdict}_v734_arc_panel_replay_qualification",
        verdict_class=verdict,
        gate_check_summary=work["failures"],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        historical_model_provenance=work.get("historical_model_provenance", {}),
        rows=work["replay_rows"],
        replay_rows=work["replay_rows"],
        intended_count=16,
        completed_count=len(work["replay_rows"]),
        failed_count=0,
        censored_count=0,
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(stored_episodes=16, independent_new_observations=0),
        verifier_is_oracle=True,
        exposure_scope="sealed_exposed_history_infrastructure_only",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=qualified,
        flagged_adversarial=bool(work["findings"]),
        acceptance_gates=dict(
            terminal_replay=qualified,
            owned_coverage=work["coverage"].get("passed", False),
            exact_original_union=work["union"].get("passed", False),
            no_scientific_benefit_claim=True,
        ),
        validation_receipts=work["checks"],
        terminal_validation_sidecar_path=str(
            output.parent / "raw" / output.stem / "terminal_validation.json"
        ),
        adversarial_findings=work["findings"],
        preconditions_checked=work["preconditions"],
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        random_seed=7348548,
        reproducibility_checksum=canonical_hash(work),
        source_artifact_hashes=work["sources"],
        code_config_hashes=work["code_hashes"],
        raw_shard_hashes=work["raw_hashes"],
        cited_upstream_artifacts=[
            dict(path=ORIGINAL, sha256=PIN, original_date="20261015", diagnostic_only=True)
        ],
        arc_panel_execution_ready_score=ready,
        terminal_replay_receipts=work["terminal_replays"],
        child_coverage_receipts=work["coverage"],
        original_episode_hashes=work["episode_hashes"],
        exact_union_receipt=work["union"],
        solve_provenance="live_agent_self_discovery",
        headline_solve_credit=0,
        historical_diagnostics=work["history"],
        failure_reproductions=work["reproductions"],
        original_qualification_fixtures=work["fixtures"],
        work_reference=dict(path=str(work_path), sha256=sha256_file(work_path)),
        methodology_note="Authenticate sixteen stored model-disabled episodes. Replay their original selections, executed actions, attempts, latches, outcomes and censor markers through the shipped reducers. Actual child coverage and original qualification controls qualify infrastructure only. No fresh game, model, solve, efficiency estimate or policy null is claimed. Exp8537 remains disqualified. This prerequisite permits a separately authorized fresh application panel.",
    )
    value["field_principles"] = {
        k: "Bind this invocation to immutable operands; infrastructure qualification grants no scientific benefit or solve credit."
        for k in value
    }
    value["field_principles"].update(
        arc_panel_execution_ready_score="Require measured terminal replay, complete original controls, actual CLI and100 percent newly owned statements.",
        independent_count="Zero fresh independent observations; sixteen replays do not multiply evidence.",
        historical_diagnostics="Keep original90-second kill, resumed24 percent and subsequent repairs without clearing quarantine.",
        child_coverage_receipts="Union actual executed acceptance and rejection statements bound to exact source hashes.",
    )
    return value


def replay(value: Json) -> bool:
    """Outer rehashing cannot replace original independent producer seals."""
    try:
        work = panel.checked(
            Path(value["work_reference"]["path"]), value["work_reference"]["sha256"]
        )
        output = Path(value["terminal_validation_sidecar_path"]).parents[2] / (NAME + ".json")
        if build(Path(value["work_reference"]["path"]), output) != value:
            return False
        for path, digest in {**work["code_hashes"], **work["raw_hashes"]}.items():
            panel.checked_bytes(path, digest)
        for receipt in work["checks"]:
            for stream in ("stdout", "stderr"):
                if stream + "_path" in receipt:
                    panel.checked_bytes(receipt[stream + "_path"], receipt[stream + "_sha256"])
        if work["replay_rows"]:
            primary, original = authenticate(ROOT)
            history_mismatch = (
                episode_hashes(original) != work["episode_hashes"]
                or reduce_history(original) != work["replay_rows"]
                or history_receipts(primary) != work["history"]
            )
            if history_mismatch:
                return False
            config = Path(work["coverage_config"])
            receipt_mismatch = (
                coverage_receipt(config, work["code_hashes"]) != work["coverage"]
                or exact_union(work["union"]["collected_node_ids"], work["family_receipts"])
                != work["union"]
            )
            if receipt_mismatch:
                return False
        output = Path(value["terminal_validation_sidecar_path"]).parents[2] / (NAME + ".json")
        return build(Path(value["work_reference"]["path"]), output) == value
    except (OSError, ValueError, KeyError, TypeError):
        return False
