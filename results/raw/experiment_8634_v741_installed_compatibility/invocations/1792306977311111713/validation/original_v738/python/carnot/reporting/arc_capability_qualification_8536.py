"""REQ-REPORT-8536: qualify runtime availability without claiming strategy benefit."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any
from unittest.mock import patch

import yaml

from carnot.reporting import arc_capability_fixtures_8536 as fixtures
from carnot.reporting import arc_effect_panel_8523 as history
from carnot.reporting import arc_redirect_runner_8509 as shipped
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.reporting.v721_capstone_evidence import freeze

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME, TASK = (
    "experiment_8536_v733_arc_capability_qualification",
    "exp8536-arc-capability-qualification",
)
TASK_PIN = "sha256:cabec01ecd9b785a14af7c4545c4955e9e8ffd69ab65d5e866df046ab4584f1a"
UPSTREAM = "results/experiment_8523_v732_arc_effect_panel.json"
UPSTREAM_PIN = "sha256:26d4935b96524fa0ce12af38f8bfae4fb823e848f85bb0afbc68ce97881e73fc"
CLI = f"scripts/experiments/{NAME}.py"
POLICY = "python/carnot/agentic/arc_competition_agent.py"
SUPERVISOR = "python/carnot/agentic/arc_trajectory_supervisor.py"
TESTS = [
    "tests/python/test_arc_capability_8536.py",
    "tests/python/test_arc_capability_qualification_8536.py",
    "tests/python/test_arc_capability_runner_8536.py",
]
OWNED = [
    "python/carnot/agentic/arc_supervisor_capability.py",
    "python/carnot/reporting/arc_capability_fixtures_8536.py",
    "python/carnot/reporting/arc_capability_qualification_8536.py",
    "python/carnot/reporting/arc_capability_runner_8536.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
REQUIRED_CHECKS = {
    "owned_tests",
    "coverage_report",
    "policy_added_coverage",
    "ruff_check",
    "ruff_format",
    "strict_mypy",
    "spec_coverage",
    "missing",
    "intentional",
    "rehashed_tamper",
}
checked = history.checked


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush phase boundaries even when a child has no output of its own."""
    print(f"[exp8536] phase={phase} completed={completed} pending={pending}", flush=True)


def authenticate(root: Path, raw: Path) -> Json:
    """Use original authority and direct operands, not current-code historical replay."""
    path = root / UPSTREAM
    primary = checked(path, UPSTREAM_PIN)
    side = json.loads(Path(primary["terminal_validation_sidecar_path"]).read_bytes())
    bound = read_bound_sidecar(path, Path(side["publication"]["sidecar_path"]))
    if not side["passed"] or not bound["report"]["passed"]:
        raise ValueError("historical_validation")
    work = checked(Path(primary["work_reference"]["path"]), primary["work_reference"]["sha256"])
    authority = work["activation_reference"]
    if sha256_file(Path(authority["snapshot_path"])) != authority["sha256"]:
        raise ValueError("historical_authority_hash")
    plan = yaml.safe_load(Path(authority["snapshot_path"]).read_bytes())
    task = next(t for t in plan["tasks"] if t["id"] == history.TASK)
    if canonical_hash(task) != history.TASK_PIN or primary["run_date"] != "20261014":
        raise ValueError("historical_authority")
    motivation = authenticate_operands(primary, work)
    return dict(
        reference=freeze(path, raw / "custody", UPSTREAM_PIN),
        work_reference=freeze(
            Path(primary["work_reference"]["path"]),
            raw / "custody",
            primary["work_reference"]["sha256"],
        ),
        authority_reference=authority,
        original_date="20261014",
        motivation=motivation,
        historical_model_provenance=primary["historical_model_provenance"],
    )


def authenticate_operands(primary: Json, work: Json) -> Json:
    """Authenticate the qualified producer's direct rows and original primitive seals.

    Original validation owns the episode reduction. Checking its sealed bytes
    preserves that authority without running another full historical validator.
    """
    for index, reference in enumerate(work["references"]):
        for filename, field in [
            ("episode.json", "terminal_sha256"),
            ("steps.jsonl", "steps_sha256"),
        ]:
            if sha256_file(Path(reference["directory"]) / filename) != reference[field]:
                raise ValueError("historical_primitive_hash")
        progress("historical_primitive_seals", index + 1, len(work["references"]) - index - 1)
    effects = primary["redirect_effect_rows"]
    motivation = dict(
        diversity_effects=sum(
            r["arm"] == "force_exploration_diversity" and r["effective_action_policy_change"]
            for r in effects
        ),
        disabled_latch_changes=sum(
            r["arm"] == "allow_reinduction"
            and r["applied"]
            and r["induction_outcome"] == "model_disabled"
            and r["model_action_executed"] is False
            for r in effects
        ),
        censored_redirects=sum(r["right_censored"] for r in [*effects, *primary["shadow_rows"]]),
        resolved_redirects=sum(
            r["resolved_by_levelup"] for r in [*effects, *primary["shadow_rows"]]
        ),
    )
    if motivation != dict(
        diversity_effects=8, disabled_latch_changes=8, censored_redirects=32, resolved_redirects=0
    ):
        raise ValueError("historical_effect_counts")
    return motivation


def protocol() -> Json:
    """Freeze the future live comparison before fixtures can influence its design."""
    games = ["ar25-0c556536", "ft09-0d8bbf25", "lf52-271a04aa", "tn36-ef4dde99"]
    modes = ["baseline_applied", "capability_masked_applied"]
    return dict(
        schema="carnot.arc.v733.capability_panel.v1",
        games=games,
        seeds=[11, 22],
        modes=modes,
        max_actions=512,
        episode_cap_s=120,
        aggregate_cap_s=2400,
        supervisor_window=120,
        stall_threshold=150,
        reinduction_evidence_floor=200,
        reinduction_attempt_cap=3,
        refinement_min_overlap_games=3,
        refinement_min_resolved_per_shared_cell=5,
        stop_rules=["agent.is_done", "512_actions", "120_seconds", "exception"],
        model_calls=0,
        MODEL_SPECS=[],
        llm_disabled=True,
        withhold=["adapters", "stored_routes", "solver_artifacts", "game_source_introspection"],
        qualification_task=TASK,
        measurement_task="exp8537-arc-capability-panel",
        solve_provenance="live_agent_self_discovery",
        headline_solve_credit=0,
        units=[
            dict(
                episode_id=f"{g}:{s}:{m}",
                pair_key=f"{g}:{s}",
                game=g,
                seed=s,
                mode=m,
                arm="on",
                capability_mask=m == modes[1],
                max_actions=512,
                max_seconds=120,
                status="planned",
                missing_reason="future_exp8537_invocation",
            )
            for i, (g, s) in enumerate((g, s) for g in games for s in [11, 22])
            for m in (modes if i % 2 == 0 else list(reversed(modes)))
        ],
    )


def measure(root: Path, raw: Path, private: Path) -> Json:
    """Authenticate independent CPU resources before the real wrapper controls."""
    start = time.monotonic()
    import sys

    with patch.multiple(shipped, e=sys.modules[__name__], TESTS=TESTS):
        failures, resources, hashes = shipped.preconditions(root, private)
    source = None
    try:
        source = authenticate(root, raw / "historical")
    except (OSError, ValueError, KeyError) as exc:
        failures.append(
            dict(
                shipped.gate(
                    "original_history",
                    root / UPSTREAM,
                    "sealed_sha256",
                    UPSTREAM_PIN,
                    sha256_file(root / UPSTREAM) if (root / UPSTREAM).is_file() else None,
                ),
                upstream=history.TASK,
                authentication_error=str(exc),
            )
        )
    registry_path = root / "ops/arc_solve_registry.yaml"
    registry = (
        yaml.safe_load(registry_path.read_bytes())["games"] if registry_path.is_file() else []
    )
    registry_precheck = dict(
        path=str(registry_path),
        sha256=sha256_file(registry_path) if registry_path.is_file() else None,
        read_before_live_outcomes=True,
        policy_received_registry=False,
        duplicate_resolve_authorized=False,
        rows=[
            dict(
                game=g,
                public_solve=next((r for r in registry if r["game"] == g.split("-")[0]), None),
            )
            for g in protocol()["games"]
        ],
    )
    code = {
        str(ROOT / p): sha256_file(ROOT / p)
        for p in [*OWNED, POLICY, SUPERVISOR, "python/carnot/agentic/arc_arm_eligibility.py"]
    }
    frozen = dict(protocol=protocol(), code_config_hashes=code, registry_precheck=registry_precheck)
    atomic_json(raw / "frozen_panel.json", frozen)
    observed = fixtures.collect(raw / "fixtures")
    duration = time.monotonic() - start
    if duration >= 3600:
        raise TimeoutError("measurement_deadline")
    return dict(
        input_root=str(root),
        source=source,
        failures=failures,
        preconditions_checked=resources,
        activation_reference=freeze(root / "research-roadmap.yaml", raw / "authority"),
        fixtures=observed,
        frozen_panel=frozen,
        frozen_reference=dict(
            path=str(raw / "frozen_panel.json"), sha256=sha256_file(raw / "frozen_panel.json")
        ),
        registry_precheck=registry_precheck,
        duration_s=duration,
        phase_spans=[dict(phase="preconditions_and_selection_fixtures", duration_s=duration)],
        source_artifact_hashes=hashes,
        code_config_hashes=code,
    )


def build(work_path: Path, receipts: list[Json], output: Path) -> Json:
    """Readiness certifies code and frozen inputs; all live outcomes remain future."""
    work = json.loads(work_path.read_bytes())
    fixtures_data = work["fixtures"]
    parity = all(r["passed"] for r in fixtures_data["action_parity_rows"])
    passed = bool(receipts) and all(r["passed"] for r in receipts)
    qualified = passed and REQUIRED_CHECKS <= {r["name"] for r in receipts}
    ready = int(
        qualified
        and not work["failures"]
        and parity
        and work["frozen_panel"]["protocol"] == protocol()
    )
    verdict = (
        "disqualified"
        if receipts and not passed or not parity
        else "blocked"
        if work["failures"]
        else "circular_positive"
        if ready
        else "null"
    )
    source = work["source"] or {}
    raw = output.parent / "raw" / output.stem
    rows = [
        dict(
            case=r["case"],
            masked=r["masked"],
            status="completed",
            absolute_metric=int(p["passed"]),
            missing_reason=None,
            fixture_is_live_evidence=False,
            solve_provenance="development_proxy",
        )
        for r, p in zip(
            fixtures_data["selection_fixture_rows"],
            fixtures_data["action_parity_rows"],
            strict=True,
        )
    ]
    predicates = [
        dict(
            predicate,
            case=r["case"],
            masked=r["masked"],
            action_index=event["action_index"],
            selected_arm=event["arm"],
            selected=predicate["arm"] == event["arm"],
            applied=predicate["arm"] == event["arm"] and event["applied"],
            applied_effect=event["application_disposition"],
            baseline_eligible=predicate.get("baseline_eligible", predicate["eligible"]),
            capability_eligible=predicate.get("capability_eligible"),
            capability_reason=predicate.get(
                "capability_reason", "mask_disabled_capability_not_queried"
            ),
        )
        for r in fixtures_data["selection_fixture_rows"]
        for event in r["receipts"]
        for predicate in event["eligibility_predicates"]
    ]
    value = dict(
        experiment_id=8536,
        task_id=TASK,
        milestone="2026.10.733",
        run_date="20261015",
        honest_verdict=f"complete_{verdict}_v733_arc_capability_qualification",
        verdict_class=verdict,
        gate_check_summary=work["failures"],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        llm_invocation_count=0,
        historical_model_provenance=source.get("historical_model_provenance", {}),
        rows=rows,
        intended_count=14,
        completed_count=len(rows),
        failed_count=0,
        censored_count=0,
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            private_controls=14,
            independent_live_games=0,
            future_episodes=16,
            seeds=[11, 22],
            max_actions=512,
            episode_cap_s=120,
        ),
        verifier_is_oracle=True,
        exposure_scope="private_selection_fixtures_and_exposed_public_history",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=qualified,
        flagged_adversarial=False,
        acceptance_gates=dict(
            owned_validation=qualified,
            action_parity=parity,
            original_primitives=bool(source),
            frozen_panel=work["frozen_panel"]["protocol"] == protocol(),
            scientific_benefit_claimed=False,
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        adversarial_findings=[],
        preconditions_checked=work["preconditions_checked"],
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        random_seed=7338536,
        reproducibility_checksum=canonical_hash(work),
        source_artifact_hashes=work["source_artifact_hashes"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes={
            str(work_path): sha256_file(work_path),
            work["frozen_reference"]["path"]: work["frozen_reference"]["sha256"],
        },
        cited_upstream_artifacts=[
            dict(
                producer=history.TASK, path=UPSTREAM, sha256=UPSTREAM_PIN, original_date="20261014"
            )
        ],
        arc_capability_ready_score=ready,
        capability_predicate_rows=predicates,
        unknown_capabilities=[
            r for r in predicates if r["masked"] and r["capability_eligible"] is None
        ],
        **fixtures_data,
        frozen_panel_reference=work["frozen_reference"],
        policy_hash=work["code_config_hashes"][str(ROOT / POLICY)],
        runner_hash=work["code_config_hashes"][str(ROOT / CLI)],
        registry_precheck=work["registry_precheck"],
        solve_provenance="live_agent_self_discovery",
        fixture_solve_provenance="development_proxy",
        headline_solve_credit=0,
        live_episodes_measured=0,
        historical_motivation=source.get("motivation"),
        future_dependencies=[
            dict(task_id="exp8537-arc-capability-panel", status="future_dependency_not_observation")
        ],
        work_reference=dict(path=str(work_path), sha256=sha256_file(work_path)),
        methodology_note="Scripted observations drive actual make_carnot_agent/E3AgentPolicy selection with model tripwires. Fourteen fixture variants are engineering controls, not independent games. The sixteen frozen live episodes belong to future Exp8537. Readiness is not strategy efficacy, efficiency, generalization or solve credit. Unknown capability cannot support mask benefit. Historical unresolved redirects remain censored.",
    )
    value["field_principles"] = {
        k: "Bind to actual invocation and immutable operands; qualification grants no scientific benefit."
        for k in value
    }
    value["field_principles"].update(
        arc_capability_ready_score="Require complete owned validation, original direct evidence, exact frozen panel and receipt parity; fixtures grant selection readiness only.",
        capability_predicate_rows="Keep baseline eligibility, current capability, reason, selection and applied effect distinct for every arm.",
        unknown_capabilities="Missing capability is explicitly unknown, never evidence for mask benefit.",
        solve_provenance="Identifies the reachable live-agent path; private fixture rows remain development_proxy and confer no solve.",
        frozen_panel_reference="Seal original four games, two seeds, both applied modes and original caps for future Exp8537.",
        acceptance_gates="All owned checks must pass independently of any benefit; failed owned validation disqualifies.",
    )
    return value


def replay(value: Json) -> bool:
    """Cold replay rejects altered operands even when outer hashes were recomputed."""
    try:
        work = checked(Path(value["work_reference"]["path"]), value["work_reference"]["sha256"])
        frozen = checked(Path(work["frozen_reference"]["path"]), work["frozen_reference"]["sha256"])
        if frozen != work["frozen_panel"] or frozen["protocol"] != protocol():
            return False
        activation = work["activation_reference"]
        if activation["exists"]:
            if sha256_file(Path(activation["snapshot_path"])) != activation["sha256"]:
                return False
            plan = yaml.safe_load(Path(activation["snapshot_path"]).read_bytes())
            task = next((t for t in plan["tasks"] if t["id"] == TASK), None)
            if not work["failures"] and canonical_hash(task) != TASK_PIN:
                return False
        elif not work["failures"]:
            return False
        if work["source"]:
            ref = work["source"]["reference"]
            original = checked(Path(ref["snapshot_path"]), UPSTREAM_PIN)
            original_work = checked(
                Path(work["source"]["work_reference"]["snapshot_path"]),
                original["work_reference"]["sha256"],
            )
            if (
                work["source"]["original_date"] != original["run_date"]
                or work["source"]["historical_model_provenance"]
                != original["historical_model_provenance"]
                or work["source"]["authority_reference"] != original_work["activation_reference"]
            ):
                return False
            if work["source"]["motivation"] != authenticate_operands(original, original_work):
                return False
        elif not work["failures"]:
            return False
        if any(sha256_file(Path(p)) != h for p, h in work["code_config_hashes"].items()):
            return False
        if not fixtures.replay(work["fixtures"]):
            return False
        for receipt in value["validation_receipts"]:
            for channel in ["stdout", "stderr"]:
                if (
                    channel + "_path" in receipt
                    and sha256_file(Path(receipt[channel + "_path"]))
                    != receipt[channel + "_sha256"]
                ):
                    return False
        output = Path(value["terminal_validation_sidecar_path"]).parents[2] / (NAME + ".json")
        return (
            build(Path(value["work_reference"]["path"]), value["validation_receipts"], output)
            == value
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False
