"""REQ-REPORT-9178-V781: qualify mechanic priority without running an ARC solver."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time
from typing import Any
from unittest.mock import patch

from carnot.agentic import arc_solver_kit as kit
from carnot.reporting import logit_methods_runner_9126 as lifecycle
from carnot.reporting import mdh_human_release_runner_9056 as publication_checks
from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
)
from carnot.reporting.logit_methods_9126 import ref as ref
from carnot.reporting.primary_publication import publish_primary

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME, TASK = "experiment_9178_v781_arc_mechanic_routing", "exp9178-arc-mechanic-routing"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_arc_mechanic_routing_9178.py"
CONFIG = "openspec/change-proposals/experiment-9178-inputs.json"
BASELINE = f"results/raw/{NAME}/inputs/baseline.json"
FROZEN = json.loads((ROOT / CONFIG).read_bytes())
OWNED = ["python/carnot/reporting/arc_mechanic_routing_9178.py", CLI]
MODEL_SPECS: list[Json] = []


def progress(phase: str) -> None:
    """Flush each boundary so a stalled checker remains visible to the supervisor."""
    print(f"[exp9178] phase={phase}", flush=True)


def checksum(value: Json) -> str:
    """Replay also rebuilds rows, because a new checksum cannot bless invented results."""
    return canonical_hash({k: v for k, v in value.items() if k != "reproducibility_checksum"})


def inspect(root: Path) -> Json:
    """Authenticate original producer bytes before accessing their routing observations."""
    checks = []
    for name, expected in FROZEN["pins"].items():
        reference = ref(root / name)
        observed = reference["sha256"] or "missing_file"
        checks.append(
            dict(
                check="source_bytes",
                upstream=TASK,
                **reference,
                artifact_field="sha256",
                op="==",
                expected=expected,
                observed=observed,
                passed=observed == expected,
            )
        )
    rows = []
    if all(c["passed"] for c in checks):
        rows = json.loads((root / BASELINE).read_bytes())["rows"]
    return dict(checks=checks, baseline_rows=rows)


def routing_rows(baseline: list[Json]) -> list[Json]:
    """Compare each fixed action/mechanic against the original empty-name reference."""
    rows = []
    for source in baseline:
        mechanic, action = source["mechanic"], source["action_model"]
        reference = next(
            r["before"]
            for r in baseline
            if r["mechanic"] == mechanic and r["action_model"] == action and not r["game_name"]
        )
        after = [
            o.operator
            for o in kit.select_primitive_operators(
                mechanic_class=mechanic, action_model=action, game=source["game_name"]
            )
        ]
        passed = after == (reference if source["recognized"] else source["before"])
        rows.append(
            dict(
                source,
                after=after,
                reference=reference,
                passed=passed,
                before_mismatch=source["recognized"] and source["before"] != reference,
                qid=None,
                group=mechanic,
                group_id=mechanic,
                sibling=None,
                arm="explicit_mechanic" if source["recognized"] else "legacy_control",
                seed=7819178,
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
    """Reuse exact-command qualification and retain coverage for changed selector statements."""
    with patch.object(lifecycle, "d", sys.modules[__name__]):
        value = lifecycle.scope(raw)
    command = value["commands"][0]["argv"]
    command.insert(2, "COVERAGE_RCFILE=" + value["coverage_config_reference"]["path"])
    cfg = Path(value["coverage_config_reference"]["path"])
    cfg.write_text(
        cfg.read_text().replace(
            "[report]",
            "    " + str(ROOT / "python/carnot/agentic/arc_solver_kit.py") + "\n[report]",
        )
    )
    value["coverage_config_reference"] = ref(cfg)
    direct = value["commands"][1]["argv"]
    affected = sorted(
        str(p.relative_to(ROOT))
        for p in (ROOT / "tests/python").glob("test_*.py")
        if "select_primitive_operators(" in p.read_text() and p.name != Path(TEST).name
    )
    direct.extend(
        [
            *affected,
            "tests/python/test_experiment_4582_feature_router_transfer.py",
            "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
            "tests/python/test_restricted_decision_audit_8210.py",
        ]
    )
    value["code_test_hashes"].extend(
        ref(ROOT / p) for p in [*direct[direct.index("-q") + 1 :], CONFIG]
    )
    value["code_test_hashes"].extend(
        ref(ROOT / p)
        for p in [
            "python/carnot/agentic/arc_solver_kit.py",
            "python/carnot/agentic/arc_solve_learning.py",
        ]
    )
    value["commands"][4]["argv"].insert(3, "--config-file=/dev/null")
    value.update(
        exclusions=[],
        random_seed=7819178,
        owned_paths=OWNED,
        selector_changed_statements="mechanic recognition guard and gid assignment only",
        comparison="52 recognized permutations and72 unchanged legacy controls;8 mechanic groups",
        e2e_scope="E2E-018/021 private actual children; E2E-020 inapplicable: no learning/recovery change",
    )
    return value


def measure(root: Path, raw: Path, date: str) -> Json:
    """Freeze resources, validation and inputs before the deterministic routing comparison."""
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    progress("preconditions_before")
    manifest = scope(raw)
    inspected = inspect(root)
    disk = shutil.disk_usage(raw).free
    limit = resource.getrlimit(resource.RLIMIT_AS)[0]
    resources = dict(
        cpu_count=os.cpu_count(),
        cpu_limit=resource.getrlimit(resource.RLIMIT_CPU),
        memory_limit=limit,
        free_disk_bytes=disk,
        private_mode=oct(raw.stat().st_mode & 0o777),
        selector_import=kit.__file__,
        python_prefix=sys.prefix,
    )
    available = (
        bool(os.cpu_count()) and disk >= 256 * 1024**2 and (limit == -1 or limit >= 256 * 1024**2)
    )
    inspected["checks"].append(
        dict(
            check="cpu_memory_disk_import",
            upstream=TASK,
            path=str(raw),
            sha256=None,
            artifact_field="available",
            op="==",
            expected=True,
            observed=available,
            passed=available,
        )
    )
    progress("matrix_before")
    rows = (
        routing_rows(inspected["baseline_rows"])
        if all(c["passed"] for c in inspected["checks"])
        else []
    )
    progress("matrix_after")
    end = time.monotonic_ns()
    receipt = build_current_work_receipt(
        run_id=raw.name,
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="cpu_deterministic_mechanic_routing",
        inference_substrate_details=dict(
            work="selector comparisons; no model, generation, solver or episode"
        ),
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=start,
        ended_monotonic_ns=end,
    )
    atomic_json(raw / "current_work.json", receipt)
    work = dict(
        root=str(root),
        raw=str(raw),
        run_date=date,
        scope=manifest,
        inspection=inspected,
        rows=rows,
        start_ns=start,
        duration_s=(end - start) / 1e9,
        preconditions_checked=[
            dict(
                resource="CPU/memory/private disk/import",
                available=available,
                check_receipt=resources,
            )
        ],
        current_work_reference=ref(raw / "current_work.json"),
    )
    atomic_json(raw / "work.json", work)
    return work


def qualify(work: Json) -> Json:
    """Existing supervisors retain child logs, real SQLite coverage and one broad health run."""
    with patch.object(lifecycle, "d", sys.modules[__name__]):
        return dict(lifecycle.qualify(work))


def receipts_pass(work: Json, validation: Json) -> bool:
    """Use exact commands and real coverage; a passing boolean alone grants no readiness."""
    with patch.object(lifecycle, "d", sys.modules[__name__]):
        return bool(lifecycle.receipts_pass(work, validation))


def build(work: Json, validation: Json) -> Json:
    """Reduce deterministic controls without promoting them to independent ARC success."""
    rows = deepcopy(work["rows"])
    blocked = not all(c["passed"] for c in work["inspection"]["checks"])
    qualified = receipts_pass(work, validation)
    correct = len(rows) == 124 and all(r["passed"] for r in rows)
    cls = "blocked" if blocked else "null" if qualified and correct else "disqualified"
    completed = sum(r["status"] == "completed" for r in rows)
    failed = sum(r["status"] == "failed" for r in rows)
    if blocked:
        rows = [
            dict(
                unit_id=f"routing-{n:03d}",
                qid=None,
                group=None,
                sibling=None,
                arm="not_run",
                condition="missing_prerequisite",
                seed=7819178,
                absolute_metric=None,
                numerator=None,
                denominator=0,
                status="censored",
                exclusion_reason=None,
                censor_reason="failed_authenticated_prerequisite",
            )
            for n in range(124)
        ]
    value: Json = dict(
        schema="carnot.arc_mechanic_routing.v1",
        experiment_id=9178,
        task_id=TASK,
        milestone="2026.11.781",
        run_date=work["run_date"],
        honest_verdict=f"complete_{cls}_arc_mechanic_routing",
        verdict_class=cls,
        required_checks_passed=qualified,
        flagged_adversarial=False,
        mechanic_routing_ready_score=int(qualified and correct and not blocked),
        rows=rows,
        intended_count=124,
        attempted_count=0 if blocked else len(rows),
        completed_count=completed,
        failed_count=failed,
        censored_count=124 if blocked else 0,
        excluded_count=0,
        effective_independent_groups=0,
        independent_generalization_score=0,
        sample_size_budget=dict(
            intended=124,
            attempted=0 if blocked else len(rows),
            completed=completed,
            failed=failed,
            censored=124 if blocked else 0,
            effective_independent_groups=0,
            deterministic_mechanic_groups=8,
        ),
        verifier_is_oracle=False,
        claim_scope=dict(
            exposure_scope="exposed_development",
            independent_generalization_score=0,
            allowed_claims=["deterministic routing invariance", "legacy fallback preservation"],
            scope="Development controls only. No model, SDK episode, level solve, leaderboard score or generalization improvement measured.",
        ),
        solve_provenance="development_proxy",
        consumer_reachability=dict(
            classification="offline_recommendation_helper_test",
            actual_callers=[
                "arc_solve_learning.recommend_approach:unsurveyed",
                "arc_solve_learning.recommend_approach:surveyed",
            ],
            measured_fixture_paths=[TEST + "::test_consumers"],
            live_path_observed=False,
            static_upstream="arc_competition_agent calls recommend_approach; static linkage is not measured live reachability",
        ),
        changed_behavior_rows=[
            {
                k: r[k]
                for k in ["mechanic", "action_model", "game_name", "before", "after", "reference"]
            }
            for r in work["rows"]
            if r["before"] != r["after"]
        ],
        regression_result=dict(
            recognized_total=sum(r["recognized"] for r in work["rows"]),
            before_mismatches=sum(r["before_mismatch"] for r in work["rows"]),
            after_mismatches=sum(not r["passed"] for r in work["rows"]),
            legacy_controls=sum(not r["recognized"] for r in work["rows"]),
        ),
        MODEL_SPECS=[],
        model_invocation_status="not_requested",
        requested_inference_substrate_class="no_model_load",
        inference_substrate_class="blocked_no_run" if blocked else "no_model_load",
        inference_substrate="cpu_deterministic_mechanic_routing",
        current_llm_invocations=0,
        generated_tokens=0,
        model_invocation_counts=dict(
            current_llm_invocations=0,
            loads_attempted=0,
            loads_completed=0,
            loads_failed=0,
            forward_calls=0,
            generated_tokens=0,
        ),
        duration_s=work["duration_s"],
        random_seed=7819178,
        gate_check_summary=work["inspection"]["checks"],
        preconditions_checked=work["preconditions_checked"],
        source_references=[
            dict(
                path=str(Path(work["root"]) / name),
                sha256=digest,
                imported_fields=["before operators", "mechanic/name cases"],
                historical_model_provenance="reference_only_no_current_model_work",
            )
            for name, digest in FROZEN["pins"].items()
        ],
        code_config_hashes=work["scope"]["code_test_hashes"],
        current_work_receipts=[work["current_work_reference"]],
        validation_receipts=validation.get("checks", []),
        owned_statement_coverage=validation.get("owned_statement_coverage", {}),
        repository_health=validation.get("repository_health", {}),
        validation_scope_manifest=work["scope"],
        work_reference=ref(Path(work["raw"]) / "work.json"),
        acceptance_gates={
            name: dict(
                condition=condition,
                principle="Frozen operands and actual owned checks govern readiness; fixtures grant no solve credit.",
            )
            for name, condition in dict(
                matrix="52 recognized rows equal empty-game reference",
                legacy="72 unknown/absent mechanic controls unchanged",
                validation="all affected checks pass and100 percent owned statements; zero exclusions",
                scope="no model or game episode; independent_generalization_score=0",
            ).items()
        },
    )
    value["field_principles"] = {
        k: "Exact retained operands qualify this development control only." for k in value
    }
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> Json:
    """Rebuild rows and aggregates so even rehashed tampering is rejected in a fresh process."""
    value = json.loads(path.read_bytes())
    for reference in [
        value["work_reference"],
        *value["code_config_hashes"],
        *value["current_work_receipts"],
    ]:
        if ref(Path(reference["path"])) != reference:
            raise ValueError("operand_hash")
    work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
    if inspect(Path(work["root"])) != {
        k: v for k, v in work["inspection"].items() if k != "checks"
    } | dict(checks=work["inspection"]["checks"][:-1]):
        raise ValueError("source_drift")
    if work["rows"] != routing_rows(work["inspection"]["baseline_rows"]):
        raise ValueError("routing_drift")
    expected = build(
        work,
        dict(
            checks=value["validation_receipts"],
            owned_statement_coverage=value["owned_statement_coverage"],
            repository_health=value["repository_health"],
        ),
    )
    if expected != value:
        raise ValueError("reduction_drift")
    return dict(passed=True, replay_passed=True)


def validate(candidate: Path) -> Json:
    """Run unchanged adversarial, strict-row and real foreign-directory replay checkers."""
    with patch.object(publication_checks, "e", sys.modules[__name__]):
        return dict(publication_checks.validate(candidate, candidate.parent / "checkers"))


def main(argv: list[str] | None = None) -> int:
    """Keep private fixture qualification separate from the reader-visible checked primary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261117")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--checks-only", action="store_true")
    args = parser.parse_args(argv)
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
        print(f"[exp9178] refusal={exc}", flush=True)
        return 1
