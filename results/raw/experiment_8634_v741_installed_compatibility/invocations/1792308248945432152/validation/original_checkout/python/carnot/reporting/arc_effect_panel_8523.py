"""REQ-REPORT-8523: fresh receipt collection does not establish transferable benefit.

The qualified wrapper owns action choice. This module freezes its inputs and
reduces visible operands so a mode label cannot masquerade as an application.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
from unittest.mock import patch

import yaml

from carnot.agentic import arc_live_panel_runtime_8384 as runtime
from carnot.agentic.arc_effect_receipts import SCHEMA, action_policy
from carnot.reporting import arc_effect_qualification_8522 as qualification
from carnot.reporting import arc_redirect_runner_8509 as shipped
from carnot.reporting import arc_supervisor_live_panel_8384 as panel_runtime
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.primary_publication import read_bound_sidecar, reader_receipt
from carnot.reporting.v709_execution import child
from carnot.reporting.v721_capstone_evidence import freeze

Json = dict[str, Any]
ROOT = qualification.ROOT
NAME, TASK = "experiment_8523_v732_arc_effect_panel", "exp8523-arc-effect-panel"
TASK_PIN = "sha256:22de089ab9912c11259ee87ae142ab78cefbe0b0fcf4322267147d44528db4d3"
UPSTREAM = "results/experiment_8522_v732_arc_effect_receipt_qualification.json"
UPSTREAM_PIN = "sha256:4a82097fa0faf69b8d5d3b8da81e5fd9b2acb3bc92ff722f56095cacd9e7356d"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_arc_effect_panel_8523.py"
TESTS = [TEST, "tests/python/test_arc_effect_panel_runner_8523.py"]
OWNED = [
    "python/carnot/reporting/arc_effect_panel_8523.py",
    "python/carnot/reporting/arc_effect_panel_runner_8523.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
checked = qualification.checked


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed boundaries keep both measurement and quiet subprocesses visible."""
    print(f"[exp8523] phase={phase} completed={completed} pending={pending}", flush=True)


def authenticate(root: Path, raw: Path) -> Json:
    """An independent producer pin and both unchanged consumers precede collection."""
    path = root / UPSTREAM
    primary = checked(path, UPSTREAM_PIN)
    side = json.loads(Path(primary["terminal_validation_sidecar_path"]).read_bytes())
    bound = read_bound_sidecar(path, Path(side["publication"]["sidecar_path"]))
    consumer = reader_receipt(
        qualification.TASK, path.parent, field="arc_effect_receipts_ready_score"
    )
    for field, expected in [
        ("task_id", qualification.TASK),
        ("run_date", "20261014"),
        ("arc_effect_receipts_ready_score", 1),
        ("required_checks_passed", True),
        ("flagged_adversarial", False),
    ]:
        if primary.get(field) != expected:
            raise ValueError("upstream_field:" + field)
    if (
        not side["passed"]
        or not bound["report"]["passed"]
        or not consumer["passed"]
        or consumer["gate_sha256"] != UPSTREAM_PIN
    ):
        raise ValueError("upstream_bound_consumer")
    protocol = checked(
        Path(primary["panel_protocol_reference"]["path"]),
        primary["panel_protocol_reference"]["sha256"],
    )
    if protocol != qualification.protocol() or not qualification.replay(primary):
        raise ValueError("upstream_protocol_or_replay")
    return dict(
        primary=primary,
        protocol=protocol,
        consumer=consumer,
        reference=freeze(path, raw / "custody", UPSTREAM_PIN),
    )


def units(protocol: Json) -> list[Json]:
    """Alternate mode order by pair before outcomes; every episode gets a fresh start."""
    pairs = [(g, s) for g in protocol["games"] for s in protocol["seeds"]]
    return [
        dict(
            episode_id=f"{g}:{s}:{m}",
            pair_key=f"{g}:{s}",
            game=g,
            seed=s,
            mode=m,
            arm="on" if m == "applied" else "off",
            max_actions=512,
            max_seconds=120,
        )
        for index, (g, s) in enumerate(pairs)
        for m in (["applied", "shadow"] if index % 2 == 0 else ["shadow", "applied"])
    ]


def episode(unit_path: Path) -> int:
    """Reuse the shipped real wrapper; receipt observation cannot replace its actions."""
    unit = json.loads(unit_path.read_bytes())
    with patch.dict(
        os.environ, {"CARNOT_ARC_EFFECT_RECEIPTS": "1", "CARNOT_ARC_DISABLE_INDUCTION": "1"}
    ):
        result = runtime.episode(unit, panel_runtime.sdk(), Path(unit["directory"]))
    return int(bool(result["model_tripwire_attempts"]))


def measure(root: Path, raw: Path, private: Path) -> Json:
    """Authenticate resources, freeze hashes and stop collection at its fixed horizon."""
    start = time.monotonic()
    with patch.multiple(shipped, e=sys.modules[__name__], TESTS=TESTS):
        failures, resources, hashes = shipped.preconditions(root, private)
    source = None
    try:
        source = authenticate(root, raw / "upstream")
    except (OSError, ValueError, KeyError) as exc:
        observed = {}
        if (root / UPSTREAM).is_file():
            try:
                observed = json.loads((root / UPSTREAM).read_bytes())
            except ValueError:
                pass
        failures.append(
            dict(
                shipped.gate(
                    "qualified_upstream",
                    root / UPSTREAM,
                    "arc_effect_receipts_ready_score",
                    1,
                    observed.get("arc_effect_receipts_ready_score"),
                ),
                upstream=qualification.TASK,
                missing=(root / UPSTREAM).is_file() is False,
                authentication_error=str(exc),
                observed_gate_fields={
                    key: observed.get(key)
                    for key in [
                        "task_id",
                        "run_date",
                        "arc_effect_receipts_ready_score",
                        "required_checks_passed",
                        "flagged_adversarial",
                    ]
                },
            )
        )
    protocol = source["protocol"] if source else qualification.protocol()
    frozen_units = units(protocol)
    roster = []
    if not failures:
        try:
            roster = [str(g.game_id) for g in panel_runtime.sdk().available_environments]
        except (ImportError, OSError, ValueError) as exc:
            failures.append(
                shipped.gate(
                    "sdk_access",
                    ROOT / "environment_files",
                    "installed_public_roster",
                    protocol["games"],
                    str(exc),
                )
            )
        for game in protocol["games"]:
            if game not in roster:
                failures.append(
                    shipped.gate(
                        "installed_game", ROOT / "environment_files", "game_id", game, None
                    )
                )
    registry = ROOT / "ops/arc_solve_registry.yaml"
    registry_value = yaml.safe_load(registry.read_bytes())["games"]
    registry_precheck = dict(
        path=str(registry),
        sha256=sha256_file(registry),
        read_before_live_outcomes=True,
        policy_received_registry=False,
        installed_roster=roster,
        rows=[
            dict(
                game=g,
                public_solve=next(
                    (r for r in registry_value if r["game"] == g.split("-")[0]), None
                ),
            )
            for g in protocol["games"]
        ],
    )
    code = {
        str(ROOT / p): sha256_file(ROOT / p)
        for p in [
            *OWNED,
            qualification.POLICY,
            qualification.RUNTIME,
            "python/carnot/agentic/arc_effect_receipts.py",
            "python/carnot/agentic/arc_trajectory_supervisor.py",
            "python/carnot/agentic/arc_arm_eligibility.py",
            "python/carnot/agentic/arc_generalization_runtime.py",
        ]
    }
    frozen = dict(
        protocol=protocol,
        units=frozen_units,
        code_config_hashes=code,
        registry_precheck=registry_precheck,
    )
    atomic_json(raw / "frozen_panel.json", frozen)
    activation = freeze(root / "research-roadmap.yaml", raw / "authority")
    resources.update(
        aggregate_panel_cap_s=2400, full_V731_design_required_for_this_ungated_task=False
    )
    references, measurement_receipts = [], []
    progress("measurement_before", 0, 16)
    if not failures:
        for index, unit in enumerate(frozen_units):
            remaining = 2400 - (time.monotonic() - start)
            if remaining <= 130:
                break
            directory = raw / "episodes" / f"{index:02d}"
            path = directory / "unit.json"
            atomic_json(path, dict(unit, directory=str(directory)))
            receipt = child(
                f"episode_{index:02d}",
                [sys.executable, "-u", str(ROOT / CLI), "--episode", str(path)],
                raw / "measurement_logs",
                deadline=min(130, remaining),
            )
            measurement_receipts.append(receipt)
            references.append(
                dict(
                    unit=dict(unit, directory=str(directory)),
                    directory=str(directory),
                    terminal_sha256=sha256_file(directory / "episode.json")
                    if (directory / "episode.json").is_file()
                    else None,
                    steps_sha256=sha256_file(directory / "steps.jsonl")
                    if (directory / "steps.jsonl").is_file()
                    else None,
                )
            )
            progress("episodes", index + 1, 15 - index)
    duration = time.monotonic() - start
    progress("measurement_after", len(references), 16 - len(references))
    return dict(
        input_root=str(root),
        source=source,
        failures=failures,
        preconditions_checked=resources,
        activation_reference=activation,
        frozen_reference=dict(
            path=str(raw / "frozen_panel.json"), sha256=sha256_file(raw / "frozen_panel.json")
        ),
        panel=frozen,
        references=references,
        measurement_receipts=measurement_receipts,
        duration_s=duration,
        phase_spans=[dict(phase="authentication_freeze_and_live_collection", duration_s=duration)],
        source_artifact_hashes=hashes,
        code_config_hashes=code,
    )


def reduce_episode(reference: Json) -> tuple[Json, list[Json]]:
    """Reconstruct application from exact before/after operands rather than mode."""
    directory = Path(reference["directory"])
    terminal = checked(directory / "episode.json", reference["terminal_sha256"])
    if terminal["unit"] != reference["unit"]:
        raise ValueError("episode_identity")
    if sha256_file(directory / "steps.jsonl") != reference["steps_sha256"]:
        raise ValueError("steps_hash")
    steps = [json.loads(line) for line in (directory / "steps.jsonl").read_text().splitlines()]
    for index, step in enumerate(steps):
        if (
            step["index"] != index + 1
            or step["action_sha256"] != canonical_hash(step["action"])
            or step["observation_sha256"] != canonical_hash(step["observation"])
        ):
            raise ValueError("primitive_hash_or_sequence")
    provenance = terminal.get("solve_provenance", "live_agent_self_discovery")
    if provenance == "live_agent_self_discovery":
        reduced_terminal = deepcopy(terminal)
        reduced_terminal["supervisor_receipt"].pop("direct_effect_receipts", None)
        with patch.dict(os.environ, {"CARNOT_ARC_EFFECT_RECEIPTS": "0"}):
            panel_runtime.reduce_episode(reference["unit"], reduced_terminal, directory)
    ledger = terminal["supervisor_receipt"]
    direct = ledger.get("direct_effect_receipts", [])
    events = []
    mode = reference["unit"]["mode"]
    for receipt in direct:
        before, after = receipt["before_state"], receipt["after_state"]
        index = receipt["wrapper_action_index"]
        if (
            receipt["schema"] != SCHEMA
            or receipt["mode"] != mode
            or receipt["observation_error"]
            or not 2 <= index <= len(steps)
        ):
            raise ValueError("receipt_boundary")
        if receipt["observation"] != steps[index - 2]["observation"] or receipt[
            "observation_sha256"
        ] != canonical_hash(receipt["observation"]):
            raise ValueError("receipt_observation")
        if (
            before is None
            or after is None
            or receipt["before_state_sha256"] != canonical_hash(before)
            or receipt["after_state_sha256"] != canonical_hash(after)
        ):
            raise ValueError("receipt_state_hash")
        selected = receipt["selected"]
        expected_after = dict(before)
        if selected and mode == "applied":
            arm = receipt["arm"]
            if arm == "drop_goal_bias":
                expected_after["goal_bias_present"] = False
            elif arm == "allow_reinduction":
                expected_after["induced"] = False
            elif arm == "force_exploration_diversity":
                expected_after.update(
                    hybrid_diversity=True,
                    steps_since_progress=max(
                        before["steps_since_progress"], before["stall_threshold"] + 1
                    ),
                )
            else:
                raise ValueError("unsupported_arm")
        changed = before != after
        disposition = (
            "not_selected"
            if not selected
            else "selected_shadow"
            if mode == "shadow"
            else "state_changed"
            if changed
            else "already_active_no_op"
        )
        if (
            after != expected_after
            or receipt["applied"] != bool(changed and selected and mode == "applied")
            or receipt["application_disposition"] != disposition
            or receipt["effective_action_policy_change"]
            != (action_policy(before) != action_policy(after))
        ):
            raise ValueError("receipt_application")
        attempt = receipt["induction_attempt"]
        if (
            receipt["model_action_executed"] is not False
            or attempt
            and (
                attempt["skipped"] != "disabled_by_env"
                or receipt["induction_outcome"] != "model_disabled"
            )
        ):
            raise ValueError("substrate_mismatch")
        if selected:
            event = next(
                (
                    r
                    for r in ledger.get("redirects", ledger.get("would_have_redirects", []))
                    if r["action_index"] == receipt["action_index"] and r["arm"] == receipt["arm"]
                ),
                None,
            )
            if event is None or not any(
                p["arm"] == receipt["arm"] and p["eligible"]
                for p in receipt["eligibility_predicates"]
            ):
                raise ValueError("receipt_selection")
            resolved = bool(
                event.get(
                    "resolved_by_levelup", event.get("levelup_followed_without_redirect", False)
                )
            )
            events.append(
                dict(
                    receipt,
                    episode_id=reference["unit"]["episode_id"],
                    pair_key=reference["unit"]["pair_key"],
                    game=reference["unit"]["game"],
                    seed=reference["unit"]["seed"],
                    resolved_by_levelup=resolved,
                    right_censored=not resolved,
                )
            )
    return dict(
        reference["unit"],
        status="failed" if terminal["error"] else "completed",
        absolute_metric=len(steps) if steps else None,
        missing_reason=terminal["error"]
        if terminal["error"]
        else None
        if steps
        else "no_sdk_observation",
        actions_used=len(steps),
        runtime_s=terminal["duration_s"],
        peak_level=max((s["observation"]["level"] for s in steps), default=None),
        unique_observed_states=len({s["observation_sha256"] for s in steps}),
        censoring=terminal["censoring"],
        mutation_count=sum(r["applied"] for r in events),
        no_op_count=sum(r["application_disposition"] == "already_active_no_op" for r in events),
        receipt_count=len(direct),
        model_tripwire_passed=terminal["model_tripwire_passed"],
        solve_provenance=provenance,
        action_hashes=[s["action_sha256"] for s in steps],
        observation_hashes=[s["observation_sha256"] for s in steps],
        primitive_directory=str(directory),
    ), events


def reduce(references: list[Json], panel: list[Json]) -> Json:
    """Pair only the predeclared keys; seeds and redirects do not add independent games."""
    mapping = {r["unit"]["episode_id"]: r for r in references}
    if len(mapping) != len(references) or not set(mapping) <= {u["episode_id"] for u in panel}:
        raise ValueError("duplicate_episode_or_unknown_slot")
    rows, effects, missing = [], [], []
    for unit in panel:
        reference = mapping.get(unit["episode_id"])
        if reference and reference["terminal_sha256"] is not None:
            row, events = reduce_episode(reference)
            effects.extend(events)
        else:
            row = dict(
                unit,
                status="unstarted",
                absolute_metric=None,
                missing_reason="missing_episode_terminal_or_external_precondition",
                censoring="missing",
                actions_used=None,
                runtime_s=None,
                peak_level=None,
                mutation_count=0,
                no_op_count=0,
                receipt_count=0,
                action_hashes=[],
                observation_hashes=[],
                solve_provenance="unmeasured",
            )
            missing.append(dict(episode_id=unit["episode_id"], reason=row["missing_reason"]))
        rows.append(row)
    pairs = []
    for key in dict.fromkeys(u["pair_key"] for u in panel):
        applied, shadow = [
            next(r for r in rows if r["pair_key"] == key and r["mode"] == m)
            for m in ["applied", "shadow"]
        ]
        differences = [
            i + 1
            for i, (a, b) in enumerate(zip(applied["action_hashes"], shadow["action_hashes"]))
            if a != b
        ]
        matched = bool(
            applied["observation_hashes"]
            and shadow["observation_hashes"]
            and applied["observation_hashes"][0] == shadow["observation_hashes"][0]
        )
        pairs.append(
            dict(
                pair_key=key,
                game=applied["game"],
                seed=applied["seed"],
                matched_initial_state=matched,
                compared_actions=min(len(applied["action_hashes"]), len(shadow["action_hashes"])),
                divergent_actions=len(differences),
                first_divergent_action=differences[0] if differences else None,
                peak_level_delta=applied["peak_level"] - shadow["peak_level"] if matched else None,
                local_effect_rows=[
                    dict(
                        arm=r["arm"],
                        wrapper_action_index=r["wrapper_action_index"],
                        subsequent_action_divergence=any(
                            i >= r["wrapper_action_index"] for i in differences
                        ),
                    )
                    for r in effects
                    if r["pair_key"] == key and r["mode"] == "applied"
                ],
                missing_reason=None if matched else "unmatched_or_missing_fresh_start",
                causal_benefit_claimed=False,
            )
        )
    support = []
    for arm in ["drop_goal_bias", "allow_reinduction", "force_exploration_diversity"]:
        cells = [
            dict(
                game=g,
                mode=m,
                resolved=sum(
                    r["game"] == g
                    and r["mode"] == m
                    and r["arm"] == arm
                    and r["resolved_by_levelup"]
                    for r in effects
                ),
            )
            for g in dict.fromkeys(u["game"] for u in panel)
            for m in ["applied", "shadow"]
        ]
        overlap = [
            g
            for g in dict.fromkeys(u["game"] for u in panel)
            if all(
                any(r["game"] == g and r["mode"] == m and r["arm"] == arm for r in effects)
                for m in ["applied", "shadow"]
            )
        ]
        authorized = len(overlap) >= 3 and all(
            c["resolved"] >= 5 for c in cells if c["game"] in overlap
        )
        support.append(
            dict(
                arm=arm,
                overlapping_games=overlap,
                overlap_count=len(overlap),
                min_overlap_games=3,
                min_resolved_per_shared_cell=5,
                cells=cells,
                authorized=authorized,
            )
        )
    return dict(
        rows=rows,
        per_game_results=[
            dict(game=g, episodes=[r for r in rows if r["game"] == g])
            for g in dict.fromkeys(u["game"] for u in panel)
        ],
        redirect_effect_rows=[r for r in effects if r["mode"] == "applied"],
        shadow_rows=[r for r in effects if r["mode"] == "shadow"],
        action_divergence_rows=pairs,
        effect_support_by_arm=support,
        arm_refinement_authorized=any(s["authorized"] for s in support),
        censoring_counts=dict(
            episodes=sum(bool(r["censoring"]) for r in rows),
            redirects=sum(r["right_censored"] for r in effects),
            resolved_redirects=sum(r["resolved_by_levelup"] for r in effects),
        ),
        missing_operands=missing,
        independent_count=len({r["game"] for r in rows if r["absolute_metric"] is not None}),
    )


def build(work_path: Path, receipts: list[Json], output: Path) -> Json:
    """Data readiness qualifies collection independently of benefit and resolved arm support."""
    work = json.loads(work_path.read_bytes())
    primitive_failure = None
    try:
        summary = reduce(work["references"], work["panel"]["units"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        primitive_failure = dict(
            shipped.gate(
                "primitive_replay", work_path, "qualified_receipt_operands", "valid", str(exc)
            ),
            upstream=TASK,
        )
        summary = reduce([], work["panel"]["units"])
        for row in summary["rows"]:
            if any(r["unit"]["episode_id"] == row["episode_id"] for r in work["references"]):
                row.update(status="failed", missing_reason="primitive_rejected:" + str(exc))
    expected = {
        "owned_tests",
        "coverage_report",
        "ruff_check",
        "ruff_format",
        "strict_mypy",
        "spec_coverage",
        "missing",
        "intentional",
        "rehashed_tamper",
    }
    qualified = (
        bool(receipts)
        and all(r["passed"] for r in receipts)
        and expected <= {r["name"] for r in receipts}
    )
    rows = summary["rows"]
    execution_passed = primitive_failure is None and all(
        r["passed"] for r in work["measurement_receipts"]
    )
    ready = int(
        qualified
        and not work["failures"]
        and execution_passed
        and len(work["references"]) == 16
        and all(
            r["status"] == "completed"
            and r["receipt_count"] > 0
            and r["model_tripwire_passed"]
            and r["solve_provenance"] == "live_agent_self_discovery"
            for r in rows
        )
        and all(r["matched_initial_state"] for r in summary["action_divergence_rows"])
    )
    verdict = (
        "disqualified"
        if receipts
        and not all(r["passed"] for r in receipts)
        or not execution_passed
        or any(r["status"] == "failed" for r in rows)
        else "blocked"
        if work["failures"]
        else "null"
    )
    raw = output.parent / "raw" / output.stem
    source = work["source"] or {}
    value = dict(
        experiment_id=8523,
        task_id=TASK,
        milestone="2026.10.732",
        run_date="20261014",
        honest_verdict=f"complete_{verdict}_v732_arc_effect_panel",
        verdict_class=verdict,
        gate_check_summary=[*work["failures"], *([primitive_failure] if primitive_failure else [])],
        inference_substrate="offline_arcade_live_agent_runtime_self_discovery_no_llm",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        llm_invocation_count=0,
        historical_model_provenance=dict(
            source.get("primary", {}).get("historical_model_provenance", {}),
            unchanged_live_generator_configuration=work["panel"]["protocol"]["generator_pin"],
        ),
        **summary,
        intended_count=16,
        completed_count=sum(r["status"] == "completed" for r in rows),
        failed_count=sum(r["status"] == "failed" for r in rows),
        censored_count=summary["censoring_counts"]["episodes"],
        excluded_count=0,
        sample_size_budget=dict(
            independent_games=4,
            episodes=16,
            seeds=[11, 22],
            modes=["applied", "shadow"],
            max_actions=512,
            episode_cap_s=120,
            aggregate_cap_s=2400,
        ),
        verifier_is_oracle=False,
        exposure_scope="adapter_withheld_historically_exposed_public_games",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=qualified,
        flagged_adversarial=False,
        acceptance_gates=dict(
            owned_validation=qualified,
            qualified_upstream=bool(source),
            collection_qualified=bool(ready),
            arm_support=summary["arm_refinement_authorized"],
            scientific_benefit_claimed=False,
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        adversarial_findings=[],
        preconditions_checked=work["preconditions_checked"],
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        random_seed=7328523,
        reproducibility_checksum=canonical_hash(work),
        source_artifact_hashes=work["source_artifact_hashes"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes={
            str(Path(r["directory"]) / file): r[k]
            for r in work["references"]
            for file, k in [("episode.json", "terminal_sha256"), ("steps.jsonl", "steps_sha256")]
        },
        cited_upstream_artifacts=[
            dict(
                task_id=qualification.TASK,
                path=UPSTREAM,
                sha256=UPSTREAM_PIN,
                original_date="20261014",
            ),
            dict(
                task_id="exp8509-arc-redirect-effect-audit",
                path=qualification.UPSTREAM,
                sha256=qualification.UPSTREAM_PIN,
                original_date="20261014",
            ),
            dict(
                task_id="exp8496-arc-panel-qualification",
                path="results/experiment_8496_v730_arc_panel_qualification.json",
                sha256="sha256:5804a065a668c89df631687b9277a963643848b61b640e59b8efc1ca72a62ceb",
                original_date="20261013",
            ),
        ],
        arc_effect_panel_ready_score=ready,
        registry_precheck=work["panel"]["registry_precheck"],
        headline_solve_credit=0,
        solve_provenance="live_agent_self_discovery" if work["references"] else "unmeasured",
        frozen_panel_reference=work["frozen_reference"],
        measurement_receipts=work["measurement_receipts"],
        work_reference=dict(path=str(work_path), sha256=sha256_file(work_path)),
        methodology_note="Fresh model-disabled actual make_carnot_agent episodes use matched deterministic starts and direct qualified receipts. Four public games are independent; seeds and redirects are descriptive repeats. State effects and action divergence do not establish game-progress or generalization benefit; unresolved windows remain right-censored. No level solve is claimed.",
    )
    value["missing_operands"].extend(work["failures"])
    value["field_principles"] = {
        k: "Bind to actual task-owned invocation and immutable operands; collection qualification grants no scientific benefit."
        for k in value
    }
    value["field_principles"].update(
        arc_effect_panel_ready_score="Require16 qualified direct-receipt episodes, matched fresh starts and all owned checks independently of progress.",
        effect_support_by_arm="Keep3 overlapping games and5 resolved outcomes per shared game/mode/arm cell; right censoring never becomes a failed arm.",
        redirect_effect_rows="Reconstruct applied state change from before/after fields at the qualified seam.",
        action_divergence_rows="Join only frozen game/seed keys; subsequent divergence is descriptive, not causal progress credit.",
        headline_solve_credit="Registry exposure and this diagnostic cannot create a new solve headline.",
        acceptance_gates="Check each engineering requirement independently of benefit, with fixed original thresholds.",
    )
    return value


def replay(value: Json) -> bool:
    """Cold replay checks independent pins as well as self-consistently rehashed operands."""
    try:
        work = checked(Path(value["work_reference"]["path"]), value["work_reference"]["sha256"])
        frozen = checked(Path(work["frozen_reference"]["path"]), work["frozen_reference"]["sha256"])
        if (
            frozen != work["panel"]
            or frozen["protocol"] != qualification.protocol()
            or frozen["units"] != units(frozen["protocol"])
        ):
            return False
        activation = work["activation_reference"]
        if activation["exists"]:
            if sha256_file(Path(activation["snapshot_path"])) != activation["sha256"]:
                return False
            plan = yaml.safe_load(Path(activation["snapshot_path"]).read_bytes())
            task = next((t for t in plan["tasks"] if t["id"] == TASK), None)
            if not work["failures"] and canonical_hash(task) != TASK_PIN:
                return False
        if work["source"]:
            source = work["source"]
            if checked(Path(source["reference"]["snapshot_path"]), UPSTREAM_PIN) != source[
                "primary"
            ] or not qualification.replay(source["primary"]):
                return False
        elif not work["failures"]:
            return False
        if any(sha256_file(Path(p)) != h for p, h in work["code_config_hashes"].items()):
            return False
        for receipt in [*work["measurement_receipts"], *value["validation_receipts"]]:
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
