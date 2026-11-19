"""REQ-REPORT-9192-V782: reconcile specialists and orchestration without a game solve."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterator
from unittest.mock import patch

from carnot.agentic import arc_solver_kit as kit
from carnot.reporting import arc_mechanic_routing_9178 as prior
from carnot.reporting import mdh_human_release_runner_9056 as checkers
from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
)
from carnot.reporting.logit_methods_9126 import ref
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v709_execution import child
from carnot.reporting.v756_branch_qualification import measured

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME, TASK = "experiment_9192_v782_arc_mechanic_routing", "exp9192-arc-ordering-compatibility"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_arc_mechanic_routing_9192.py"
CONFIG = "openspec/change-proposals/experiment-9192-inputs.json"
FROZEN = json.loads((ROOT / CONFIG).read_bytes())
OWNED = ["python/carnot/reporting/arc_mechanic_routing_9192.py", CLI]
MODEL_SPECS: list[Json] = []
BASELINE = prior.BASELINE
checksum = prior.checksum
_original_scope = prior.scope


def progress(phase: str) -> None:
    """Flushed boundaries keep the CPU study observable even during child checks."""
    print("[exp9192] phase=" + phase, flush=True)


@contextmanager
def context() -> Iterator[None]:
    """Reuse the established reporting lifecycle with this study's frozen authority."""
    with patch.multiple(
        prior,
        NAME=NAME,
        TASK=TASK,
        CLI=CLI,
        TEST=TEST,
        CONFIG=CONFIG,
        FROZEN=FROZEN,
        OWNED=OWNED,
        routing_rows=routing_rows,
        scope=scope,
        inspect=inspect,
        progress=progress,
    ):
        yield


_original_inspect = prior.inspect
_original_receipts_pass = prior.receipts_pass


def inspect(root: Path) -> Json:
    """Authenticate original bytes and distinguish an absent operand from observed zero."""
    result = _original_inspect(root)
    path = root / "results/experiment_9178_v781_arc_mechanic_routing.json"
    upstream = json.loads(path.read_bytes()) if path.is_file() else {}
    for field, expected in [("verdict_class", "disqualified"), ("mechanic_routing_ready_score", 0)]:
        observed = upstream.get(field, "missing_field")
        result["checks"].append(
            dict(
                check="upstream_terminal",
                upstream="exp9178-arc-mechanic-routing",
                **ref(path),
                artifact_field=field,
                op="==",
                expected=expected,
                observed=observed,
                passed=observed == expected,
            )
        )
    return result


def routing_rows(baseline: list[Json]) -> list[Json]:
    """Freeze full order equality, not just set membership or the first operator."""
    rows = []
    for index, source in enumerate(baseline):
        if index % 32 == 0:
            progress(f"matrix_{index}_of_{len(baseline)}")
        reference = next(
            r["before"]
            for r in baseline
            if r["mechanic"] == source["mechanic"]
            and r["action_model"] == source["action_model"]
            and not r["game_name"]
        )
        prefix = FROZEN["compatibility_prefixes"].get(source["mechanic"], [])
        expected = (
            prefix + [n for n in reference if n not in prefix]
            if source["recognized"]
            else source["before"]
        )
        after = [
            o.operator
            for o in kit.select_primitive_operators(
                mechanic_class=source["mechanic"],
                action_model=source["action_model"],
                game=source["game_name"],
            )
        ]
        passed = after == expected
        rows.append(
            dict(
                source,
                before=source["before"],
                after=after,
                reference=expected,
                passed=passed,
                before_mismatch=source["recognized"] and source["before"] != reference,
                qid=None,
                group=source["mechanic"],
                group_id=source["mechanic"],
                sibling=None,
                arm="explicit_mechanic" if source["recognized"] else "legacy_control",
                seed=7829192,
                metric="operator_list_equality",
                absolute_metric=int(passed),
                numerator=int(passed),
                denominator=1,
                status="completed" if passed else "failed",
                exclusion_reason=None,
                censor_reason=None,
            )
        )
    return rows


def scope(raw: Path) -> Json:
    """Retain all original tests, exact child commands and coverage without exclusions."""
    value = _original_scope(raw)
    value.update(
        random_seed=7829192,
        comparison="52 recognized permutations;72 unchanged legacy controls;8 mechanic groups",
        selector_changed_statements="recognized flag and stable specialist promotion",
        compatibility_matrix=FROZEN["compatibility_prefixes"],
    )
    value["commands"][0]["argv"].extend(["tests/python/test_arc_mechanic_routing_9178.py"])
    return value


def consumers() -> Json:
    """Measure the public action entry separately from offline recommendation reachability."""
    from types import SimpleNamespace
    import numpy as np
    from carnot.agentic import arc_solve_learning as learning
    from carnot.agentic.arc_competition_agent import CarnotAgentPolicy

    with patch.object(
        kit, "select_primitive_operators", wraps=kit.select_primitive_operators
    ) as selector:
        rec = learning.recommend_approach("unseen_9192", mechanic="program_editor")
        recommendation_calls = selector.call_count
        dispatch = kit.approach_dispatcher_operator(
            {"approach": "default_graph_explore"},
            [{"approach": "default_graph_explore", "candidate_generated": True, "action_id": 1}],
        )
        selector.reset_mock()
        policy = CarnotAgentPolicy("unseen_9192", solutions={}, force_explore=True)
        frame = SimpleNamespace(
            frame=np.zeros((8, 8), dtype=np.int16),
            levels_completed=0,
            available_actions=[1, 2, 3, 4],
            state="NOT_FINISHED",
        )
        action = policy.next_move([frame], frame)
        live_calls = selector.call_count
    return dict(
        classification="scripted_public_policy_action",
        actual_callers=[
            "arc_solve_learning.recommend_approach:surveyed",
            "arc_solve_learning.recommend_approach:unsurveyed",
            "scripts.arc_loop_solve.solve_adaptered:metadata",
        ],
        measured_fixture_paths=[
            "recommend_approach -> select_primitive_operators",
            "CarnotAgentPolicy.next_move -> StepwiseExplorer.next_move",
        ],
        recommendation_selector_observed=recommendation_calls > 0,
        recommendation_operators=[r["operator"] for r in rec["selected_generic_operators"]],
        dispatcher_reached=dispatch["candidate_generated"],
        dispatcher_result=dispatch,
        live_path_observed=live_calls > 0,
        live_selector_calls=live_calls,
        reached_action=list(action),
        scope="scripted label-free observation; no environment or level credit",
    )


def measure(root: Path, raw: Path, date: str) -> Json:
    """Reuse resource authentication and add operation-accurate consumer observations."""
    start = time.monotonic_ns()
    with context():
        work = prior.measure(root, raw, date)
    work["consumer_reachability"] = consumers() if work["rows"] else {}
    end = time.monotonic_ns()
    work["duration_s"] = (end - start) / 1e9
    receipt = build_current_work_receipt(
        run_id=raw.name,
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="cpu_deterministic_mechanic_routing",
        inference_substrate_details=dict(
            work="selector matrix, recommendation, scripted public policy action; no model or game"
        ),
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=start,
        ended_monotonic_ns=end,
    )
    atomic_json(raw / "current_work.json", receipt)
    work["current_work_reference"] = ref(raw / "current_work.json")
    atomic_json(raw / "work.json", work)
    return work


def receipts_pass(work: Json, validation: Json) -> bool:
    """Exact command, log and SQLite checks must all qualify the executed revision."""
    with context():
        return _original_receipts_pass(work, validation)


def qualify(work: Json) -> Json:
    """One inherited deadline bounds children; broad health is recorded once separately."""
    raw = Path(work["raw"])
    deadline = float(os.environ.get("CARNOT_9192_DEADLINE", str(time.monotonic() + 4800)))
    checks = []
    for command in work["scope"]["commands"]:
        receipt = child(
            command["name"],
            command["argv"],
            raw / "validation",
            deadline=max(0.001, min(600, deadline - time.monotonic() - 300)),
            heartbeat=30,
        )
        receipt["scope_sha256"] = canonical_hash(work["scope"])
        checks.append(receipt)
    coverage = measured(sorted(raw.glob(".coverage.*")), OWNED, raw / "combined_coverage")
    health = child(
        "global_python_suite",
        [str(ROOT / ".venv/bin/pytest"), "tests/python", "-q"],
        raw / "repository_health",
        deadline=max(0.001, min(600, deadline - time.monotonic() - 300)),
        heartbeat=30,
        scope="repository_health",
    )
    return dict(checks=checks, owned_statement_coverage=coverage, repository_health=health)


def build(work: Json, validation: Json) -> Json:
    """Derive compatibility readiness while retaining development-only claim boundaries."""
    with context(), patch.object(prior, "receipts_pass", receipts_pass):
        value = prior.build(work, validation)
    reach = work["consumer_reachability"]
    reachable = bool(
        reach.get("recommendation_selector_observed") and reach.get("dispatcher_reached")
    )
    if value["verdict_class"] != "blocked" and not reachable:
        value.update(verdict_class="disqualified", mechanic_routing_ready_score=0)
    value.update(
        experiment_id=9192,
        milestone="2026.11.782",
        random_seed=7829192,
        honest_verdict="complete_" + value["verdict_class"] + "_arc_ordering_compatibility",
        consumer_reachability=reach,
        compatibility_matrix=FROZEN["compatibility_prefixes"],
        regression_attribution="All ten original assertions fail on both pre9178 immutable source and pre9192 checkout.",
        methodology="Frozen deterministic operator order controls; stable specialist promotion; no game solve.",
        timing_budget=dict(
            measurement_ceiling_s=3300,
            verification_reserve_s=1200,
            publication_reserve_s=300,
            hard_ceiling_s=4800,
        ),
    )
    value["acceptance_gates"]["specialists"] = dict(
        condition="all frozen specialist prefixes and dispatcher execution pass",
        principle="First-slot compatibility and orchestration reachability are separate obligations.",
    )
    value["source_references"] = [
        dict(
            r,
            imported_fields=[
                "original source bytes, original ten assertion failures, frozen matrix or registry authority"
            ],
        )
        for r in value["source_references"]
    ]
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> Json:
    """Recompute rows and aggregates so rehashed invented outcomes still fail replay."""
    value = json.loads(path.read_bytes())
    for reference in [
        value["work_reference"],
        *value["code_config_hashes"],
        *value["current_work_receipts"],
    ]:
        if ref(Path(reference["path"])) != reference:
            raise ValueError("operand_hash")
    work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
    with context():
        inspected = inspect(Path(work["root"]))
    if inspected != dict(
        checks=work["inspection"]["checks"][:-1], baseline_rows=work["inspection"]["baseline_rows"]
    ):
        raise ValueError("source_drift")
    if work["rows"] != routing_rows(work["inspection"]["baseline_rows"]):
        raise ValueError("routing_drift")
    if work["rows"] and work["consumer_reachability"] != consumers():
        raise ValueError("consumer_drift")
    validation = dict(
        checks=value["validation_receipts"],
        owned_statement_coverage=value["owned_statement_coverage"],
        repository_health=value["repository_health"],
    )
    if build(work, validation) != value:
        raise ValueError("reduction_drift")
    return dict(passed=True, replay_passed=True)


def validate(candidate: Path) -> Json:
    """Use the original real scanners and foreign-directory child replay before publication."""
    with patch.object(checkers, "e", sys.modules[__name__]):
        return dict(checkers.validate(candidate, candidate.parent / "checkers"))


def main(argv: list[str] | None = None) -> int:
    """The public command publishes only bytes accepted by actual terminal checkers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261117")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--checks-only", action="store_true")
    args = parser.parse_args(argv)
    os.environ.setdefault("CARNOT_9192_DEADLINE", str(time.monotonic() + 4800))
    progress("start_no_model_load")
    try:
        if args.date != "20261117":
            raise ValueError("run_date")
        if args.cold_replay:
            print(json.dumps(replay(args.cold_replay)), flush=True)
            return 0
        if args.checks_only and args.output.resolve().is_relative_to(ROOT):
            raise ValueError("private_checks_require_external_output")
        raw = args.output.absolute().parent / "raw" / NAME / "invocations" / str(time.time_ns())
        work = measure(args.root, raw, args.date)
        validation = {} if args.checks_only else qualify(work)
        atomic_json(raw / "validation.json", validation)
        value = build(work, validation)
        progress("publication_before")
        receipt = publish_primary(args.output, value, validate)
        progress("publication_after")
        print(json.dumps(receipt), flush=True)
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        progress("refusal=" + str(exc))
        return 1
