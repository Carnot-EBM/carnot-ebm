"""REQ-REPORT-8522: qualify direct receipts before a future live effect panel."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any
from unittest.mock import patch

import yaml

from carnot.agentic import arc_competition_agent as policy
from carnot.agentic.arc_effect_receipts import SCHEMA
from carnot.reporting import arc_effect_fixtures_8522 as fixtures
from carnot.reporting import arc_redirect_effect_8509 as historical
from carnot.reporting import arc_redirect_runner_8509 as shipped_runner
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
NAME = "experiment_8522_v732_arc_effect_receipt_qualification"
TASK = "exp8522-arc-effect-receipt-qualification"
TASK_PIN = "sha256:3e4d9fcb03cc5fdc220a0b1a72e5f3bf7859dda0be5a4c3884c0d05d16b34c70"
UPSTREAM = "results/experiment_8509_v731_arc_redirect_effect_audit.json"
UPSTREAM_PIN = "sha256:cf0536bf85f9d22dd26c836e8ef67945f061290a35ecda5b7cbaede65ba8a1da"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_arc_effect_qualification_8522.py"
TESTS = [TEST, "tests/python/test_arc_effect_receipts_8522.py"]
OWNED = [
    "python/carnot/agentic/arc_effect_receipts.py",
    "python/carnot/reporting/arc_effect_fixtures_8522.py",
    "python/carnot/reporting/arc_effect_qualification_8522.py",
    "python/carnot/reporting/arc_effect_runner_8522.py",
    CLI,
]
POLICY = "python/carnot/agentic/arc_competition_agent.py"
RUNTIME = "python/carnot/agentic/arc_live_panel_runtime_8384.py"
MODEL_SPECS: list[Json] = []


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real phase boundaries without padding a short qualification task."""
    print(f"[exp8522] phase={phase} completed={completed} pending={pending}", flush=True)


def checked(path: Path, digest: str) -> Json:
    """An independent producer seal must match before decoding any fields."""
    if sha256_file(path) != digest:
        raise ValueError("input_hash:" + str(path))
    return dict(json.loads(path.read_bytes()))


def authenticate(root: Path, raw: Path) -> Json:
    """Original sealed work authenticates history without adopting current authority."""
    path = root / UPSTREAM
    value = checked(path, UPSTREAM_PIN)
    side = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
    bound = read_bound_sidecar(path, Path(side["publication"]["sidecar_path"]))
    if not side["passed"] or not bound["report"]["passed"]:
        raise ValueError("historical_validation")
    work = checked(Path(value["work_reference"]["path"]), value["work_reference"]["sha256"])
    if work["preconditions_checked"]["task_sha256"] != historical.TASK_PIN:
        raise ValueError("historical_authority")
    source = work["source"]
    original = checked(Path(source["primary_reference"]["snapshot_path"]), historical.UPSTREAM_PIN)
    if original != source["primary"] or historical.reduce(source)["rows"] != value["rows"]:
        raise ValueError("historical_primitives")
    refs = [
        freeze(path, raw / "custody", UPSTREAM_PIN),
        freeze(
            Path(value["work_reference"]["path"]),
            raw / "custody",
            value["work_reference"]["sha256"],
        ),
    ]
    archived_policy = ROOT / "results/raw" / NAME / "historical/arc_competition_agent.py"
    original_policy_pin = work["code_config_hashes"][str(ROOT / POLICY)]
    if sha256_file(archived_policy) != original_policy_pin:
        raise ValueError("historical_policy_pin")
    refs.append(freeze(archived_policy, raw / "custody", original_policy_pin))
    for row, operands in zip(original["rows"], source["operands"], strict=True):
        for kind, filename in [("steps", "steps.jsonl"), ("terminal", "episode.json")]:
            operand = Path(operands[kind]["snapshot_path"])
            pin = original["raw_shard_hashes"][str(Path(row["primitive_directory"]) / filename)]
            if sha256_file(operand) != pin:
                raise ValueError("historical_primitive_hash")
            refs.append(freeze(operand, raw / "custody", pin))
    return dict(
        primary_reference=refs[0],
        references=refs,
        historical_date=value["run_date"],
        original_panel_date=original["run_date"],
        original_authority=work["preconditions_checked"],
        historical_redirect_count=len(value["rows"]),
        historical_applied_missing_count=sum(
            r["mode"] == "applied" and r["application_disposition"] is None for r in value["rows"]
        ),
        historical_model_provenance=value["historical_model_provenance"],
    )


def protocol() -> Json:
    """Freeze the registered panel; qualification fixtures cannot supply live outcomes."""
    games = ["ar25-0c556536", "ft09-0d8bbf25", "lf52-271a04aa", "tn36-ef4dde99"]
    return dict(
        schema="carnot.arc.v732.effect_panel.v1",
        games=games,
        seeds=[11, 22],
        modes=["applied", "shadow"],
        max_actions=512,
        episode_cap_s=120,
        aggregate_cap_s=2400,
        supervisor_window=120,
        reinduction_evidence_floor=200,
        reinduction_attempt_cap=3,
        refinement_min_overlap_games=3,
        refinement_min_resolved_per_shared_cell=5,
        alternating_mode_order=True,
        stop_rules=["agent.is_done", "512_actions", "120_seconds", "exception"],
        model_calls=0,
        MODEL_SPECS=[],
        llm_disabled=True,
        receipt_env={"CARNOT_ARC_EFFECT_RECEIPTS": "1", "CARNOT_ARC_DISABLE_INDUCTION": "1"},
        generator_pin=dict(
            model_id=policy.ARC_LIVE_GENERATOR_MODEL_ID,
            filename=policy.ARC_LIVE_GENERATOR_MODEL_FILENAME,
            invoked=False,
        ),
        arm_table_sha256=sha256_file(ROOT / "python/carnot/agentic/arc_trajectory_supervisor.py"),
        withhold=["adapters", "stored_routes", "solver_artifacts", "game_source_introspection"],
        runtime=RUNTIME,
        runtime_sha256=sha256_file(ROOT / RUNTIME),
        policy_sha256=sha256_file(ROOT / POLICY),
        qualification_task=TASK,
        measurement_task="exp8523-arc-effect-panel",
        live_episodes_measured_here=0,
        headline_solve_credit=0,
    )


def measure(root: Path, raw: Path, private: Path) -> Json:
    """Authenticate resources before fixed wrapper controls; optional hardware is irrelevant."""
    started = time.monotonic()
    with (
        patch.object(shipped_runner, "e", __import__(__name__, fromlist=["*"])),
        patch.object(shipped_runner, "TESTS", TESTS),
    ):
        failures, resources, hashes = shipped_runner.preconditions(root, private)
    source = None
    try:
        source = authenticate(root, raw / "historical")
    except (OSError, ValueError, KeyError) as exc:
        failures.append(
            shipped_runner.gate(
                "historical_input",
                root / UPSTREAM,
                "original_sealed_primitives",
                UPSTREAM_PIN,
                str(exc),
            )
        )
    registry_path = root / "ops/arc_solve_registry.yaml"
    registry = (
        yaml.safe_load(registry_path.read_bytes())["games"] if registry_path.is_file() else []
    )
    panel = protocol()
    atomic_json(raw / "panel_protocol.json", panel)
    active = root / "research-roadmap.yaml"
    activation_reference = freeze(active, raw / "authority")
    observed = fixtures.collect(raw / "fixtures")
    duration = time.monotonic() - started
    if duration >= 3600:
        raise TimeoutError("measurement_deadline")
    return dict(
        input_root=str(root),
        source=source,
        failures=failures,
        preconditions_checked=resources,
        activation_reference=activation_reference,
        fixtures=observed,
        protocol_reference=dict(
            path=str(raw / "panel_protocol.json"), sha256=sha256_file(raw / "panel_protocol.json")
        ),
        registry_precheck=dict(
            path=str(registry_path),
            sha256=sha256_file(registry_path) if registry_path.is_file() else None,
            read_before_live_outcomes=True,
            policy_received_registry=False,
            rows=[
                dict(
                    game=g,
                    public_solve=next((r for r in registry if r["game"] == g.split("-")[0]), None),
                )
                for g in panel["games"]
            ],
        ),
        duration_s=duration,
        phase_spans=[dict(phase="preconditions_and_wrapper_qualification", duration_s=duration)],
        source_artifact_hashes=hashes,
        code_config_hashes={
            str(ROOT / p): sha256_file(ROOT / p) for p in [*OWNED, POLICY, RUNTIME]
        },
    )


def build(work_path: Path, receipts: list[Json], output: Path) -> Json:
    """Qualification has no solve or strategy-benefit interpretation."""
    work = json.loads(work_path.read_bytes())
    data = work["fixtures"]
    parity = all(r["passed"] for r in data["action_parity_rows"])
    expected = {
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
    passed = bool(receipts) and all(r["passed"] for r in receipts)
    qualified = passed and expected <= {r["name"] for r in receipts}
    ready = int(qualified and not work["failures"] and parity)
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
    rows = [
        dict(
            case=row["case"],
            status="completed",
            absolute_metric=int(p["passed"]),
            missing_reason=None,
            fixture_is_live_evidence=False,
            solve_provenance="development_proxy",
        )
        for row, p in zip(data["mutation_fixture_rows"], data["action_parity_rows"], strict=True)
    ]
    raw = output.parent / "raw" / output.stem
    references = source.get("references", [])
    value = dict(
        experiment_id=8522,
        task_id=TASK,
        milestone="2026.10.732",
        run_date="20261014",
        honest_verdict=f"complete_{verdict}_v732_arc_effect_receipt_qualification",
        verdict_class=verdict,
        gate_check_summary=work["failures"],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        llm_invocation_count=0,
        historical_model_provenance=source.get("historical_model_provenance", {}),
        rows=rows,
        intended_count=len(fixtures.CASES),
        completed_count=len(rows),
        failed_count=0,
        censored_count=0,
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            private_controls=9,
            independent_live_games=0,
            future_episodes=16,
            seeds=[11, 22],
            max_actions=512,
            episode_cap_s=120,
        ),
        verifier_is_oracle=True,
        exposure_scope="private_instrumentation_fixtures_and_exposed_history",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=qualified,
        flagged_adversarial=False,
        acceptance_gates=dict(
            owned_validation=qualified,
            action_parity=parity,
            original_primitives=bool(source),
            scientific_benefit_claimed=False,
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        adversarial_findings=[],
        preconditions_checked=work["preconditions_checked"],
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        random_seed=7328522,
        reproducibility_checksum=canonical_hash(work),
        source_artifact_hashes=work["source_artifact_hashes"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes={ref["snapshot_path"]: ref["sha256"] for ref in references},
        cited_upstream_artifacts=[
            dict(
                producer="exp8509-arc-redirect-effect-audit",
                path=UPSTREAM,
                sha256=UPSTREAM_PIN,
                original_date="20261014",
            )
        ],
        arc_effect_receipts_ready_score=ready,
        **data,
        policy_hash=work["code_config_hashes"][str(ROOT / POLICY)],
        runner_hash=work["code_config_hashes"][str(ROOT / RUNTIME)],
        panel_protocol_reference=work["protocol_reference"],
        receipt_schema=SCHEMA,
        missing_operands=work["failures"],
        registry_precheck=work["registry_precheck"],
        headline_solve_credit=0,
        solve_provenance="development_proxy",
        live_episodes_measured=0,
        future_dependencies=[
            dict(
                task_id="exp8523-arc-effect-panel",
                status="future_dependency_not_observation",
                solve_provenance="live_agent_self_discovery",
            )
        ],
        work_reference=dict(path=str(work_path), sha256=sha256_file(work_path)),
    )
    value["field_principles"] = {
        key: "Bind this field to exact task-owned operands; private qualification grants no scientific benefit."
        for key in value
    }
    value["field_principles"].update(
        arc_effect_receipts_ready_score="Require complete owned coverage, parity, cold replay and unchanged exact field consumers.",
        mutation_fixture_rows="Real wrapper execution with scripted observations qualifies instrumentation only.",
        action_parity_rows="Same observations and RNG preserve actions, relevant policy state and provenance.",
        panel_protocol_reference="Freeze original game panel, thresholds and budgets for the future Exp8523 invocation.",
    )
    return value


def replay(value: Json) -> bool:
    """Fresh wrapper execution rejects invented effects even with recomputed outer hashes."""
    from tempfile import TemporaryDirectory

    try:
        work = checked(Path(value["work_reference"]["path"]), value["work_reference"]["sha256"])
        activation = work["activation_reference"]
        if activation["exists"]:
            plan = yaml.safe_load(Path(activation["snapshot_path"]).read_bytes())
            task = next((t for t in plan["tasks"] if t["id"] == TASK), None)
            if not work["failures"] and canonical_hash(task) != TASK_PIN:
                return False
        if (
            checked(Path(work["protocol_reference"]["path"]), work["protocol_reference"]["sha256"])
            != protocol()
        ):
            return False
        if work["source"]:
            with TemporaryDirectory(prefix="exp8522-history-", dir="/var/tmp") as directory:
                original = authenticate(Path(work["input_root"]), Path(directory))
            for field in [
                "historical_date",
                "original_panel_date",
                "original_authority",
                "historical_redirect_count",
                "historical_applied_missing_count",
            ]:
                if work["source"][field] != original[field]:
                    return False
            for ref in work["source"]["references"]:
                if sha256_file(Path(ref["snapshot_path"])) != ref["expected_sha256"]:
                    return False
        elif not work["failures"]:
            return False
        for path, digest in work["code_config_hashes"].items():
            if sha256_file(Path(path)) != digest:
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
