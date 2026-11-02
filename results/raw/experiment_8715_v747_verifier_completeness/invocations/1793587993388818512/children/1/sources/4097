"""REQ-REPORT-8537: measure availability gating without claiming better reasoning.

The sealed protocol owns the starts and budgets. Small adapters reuse the live
wrapper and direct-effect reducer; all current model loads remain prohibited.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import yaml

from carnot.agentic.arc_supervisor_capability import CapabilitySnapshot
from carnot.agentic.arc_trajectory_supervisor import TrajectorySnapshot
from carnot.reporting import arc_capability_qualification_8536 as qualification
from carnot.reporting import arc_effect_panel_8523 as history
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar, reader_receipt
from carnot.reporting.v721_capstone_evidence import freeze

Json = dict[str, Any]
ROOT = history.ROOT
NAME, TASK = "experiment_8537_v733_arc_capability_panel", "exp8537-arc-capability-panel"
TASK_PIN = "sha256:40fe7fb9d5cf40ee800caf01218ef5c047eba496281060f8ff359fbb0d0d5174"
UPSTREAM = "results/experiment_8536_v733_arc_capability_qualification.json"
UPSTREAM_PIN = "sha256:a02db3b46baaed4ff39320bee359a8e905dfd9c809eb299592a4dcc90ce0294f"
CLI = f"scripts/experiments/{NAME}.py"
TESTS = [
    "tests/python/test_arc_capability_panel_8537.py",
    "tests/python/test_arc_capability_panel_runner_8537.py",
]
OWNED = [
    "python/carnot/reporting/arc_capability_panel_8537.py",
    "python/carnot/reporting/arc_capability_panel_runner_8537.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
checked = history.checked
MODES = ["baseline_applied", "capability_masked_applied"]
REQUIRED = {
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


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual task identity even when an inherited child is quiet."""
    print(f"[exp8537] phase={phase} completed={completed} pending={pending}", flush=True)


def authenticate(root: Path, raw: Path) -> Json:
    """Independent pins preserve original authority without rerunning historical science."""
    path = root / UPSTREAM
    value = checked(path, UPSTREAM_PIN)
    for field, expected in [
        ("task_id", qualification.TASK),
        ("run_date", "20261015"),
        ("arc_capability_ready_score", 1),
        ("required_checks_passed", True),
        ("flagged_adversarial", False),
    ]:
        if value.get(field) != expected:
            raise ValueError(f"{field}:expected={expected!r}:observed={value.get(field)!r}")
    side = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
    bound = read_bound_sidecar(path, Path(side["publication"]["sidecar_path"]))
    consumer = reader_receipt(qualification.TASK, path.parent, field="arc_capability_ready_score")
    if (
        not side["passed"]
        or not bound["report"]["passed"]
        or not consumer["passed"]
        or consumer["gate_sha256"] != UPSTREAM_PIN
    ):
        raise ValueError("upstream_bound_consumer")
    frozen = checked(
        Path(value["frozen_panel_reference"]["path"]), value["frozen_panel_reference"]["sha256"]
    )
    if frozen["protocol"] != qualification.protocol():
        raise ValueError("upstream_frozen_protocol")
    work = checked(Path(value["work_reference"]["path"]), value["work_reference"]["sha256"])
    authority = work["activation_reference"]
    plan = yaml.safe_load(checked_bytes(authority["snapshot_path"], authority["sha256"]))
    task = next(t for t in plan["tasks"] if t["id"] == qualification.TASK)
    if canonical_hash(task) != qualification.TASK_PIN:
        raise ValueError("original_authority")
    for code, digest in value["code_config_hashes"].items():
        checked_bytes(code, digest)
    return dict(
        primary=value,
        protocol=frozen["protocol"],
        reference=freeze(path, raw / "custody", UPSTREAM_PIN),
        consumer=consumer,
    )


def checked_bytes(path: str, digest: str) -> bytes:
    """Check raw bytes without interpreting historical dates as today's authority."""
    if sha256_file(Path(path)) != digest:
        raise ValueError("sealed_hash:" + path)
    return Path(path).read_bytes()


def units(protocol: Json) -> list[Json]:
    """Keep the qualified order and every fixed action/time budget."""
    return [dict(u, status="planned", missing_reason=None) for u in protocol["units"]]


def measure(root: Path, raw: Path, private: Path) -> Json:
    """Reuse collection and resource gates while binding this invocation's identity."""
    shim = SimpleNamespace(
        protocol=qualification.protocol,
        TASK=qualification.TASK,
        POLICY=qualification.POLICY,
        RUNTIME="python/carnot/agentic/arc_live_panel_runtime_8384.py",
    )
    with patch.multiple(
        history,
        NAME=NAME,
        TASK=TASK,
        TASK_PIN=TASK_PIN,
        UPSTREAM=UPSTREAM,
        UPSTREAM_PIN=UPSTREAM_PIN,
        CLI=CLI,
        OWNED=OWNED,
        TESTS=TESTS,
        authenticate=authenticate,
        units=units,
        qualification=shim,
        progress=progress,
    ):
        work = dict(history.measure(root, raw, private))
    for ref, receipt in zip(work["references"], work["measurement_receipts"], strict=True):
        ref["complete_wall_s"] = receipt["duration_s"]
    work["preconditions_checked"]["tool_hashes"] = {
        name: sha256_file(ROOT / ".venv/bin" / name)
        for name in ["python", "pytest", "ruff", "mypy"]
    }
    work["preconditions_checked"]["writable_disk_backed_scratch"] = private.is_dir()
    for row in work["failures"]:
        if row["check"] == "qualified_upstream":
            row.update(
                upstream=qualification.TASK,
                field="sealed_qualified_upstream_sha256",
                expected=UPSTREAM_PIN,
                observed=sha256_file(root / UPSTREAM) if (root / UPSTREAM).is_file() else None,
            )
    return work


def episode(path: Path) -> int:
    """Observe actual attempts at the factory seam without replacing any action."""
    unit = json.loads(path.read_bytes())
    raw = Path(unit["directory"])
    factory = history.runtime.agent_module.make_carnot_agent

    def recording_factory(*args: Any, **kwargs: Any) -> Any:
        cls = factory(*args, **kwargs)

        class Recording(cls):  # type: ignore[misc, valid-type]
            def choose_action(self, frames: Any, latest: Any) -> Any:
                action = super().choose_action(frames, latest)
                with (raw / "attempts.jsonl").open("a") as stream:
                    stream.write(json.dumps(self._policy.induction_attempts, sort_keys=True) + "\n")
                    stream.flush()
                return action

        return Recording

    flags = {
        "CARNOT_ARC_EFFECT_RECEIPTS": "1",
        "CARNOT_ARC_DISABLE_INDUCTION": "1",
        "CARNOT_ARC_SUPERVISOR_CAPABILITY_MASK": str(int(unit["capability_mask"])),
    }
    with (
        patch.dict(os.environ, flags),
        patch.object(history.runtime.agent_module, "make_carnot_agent", recording_factory),
    ):
        result = history.runtime.episode(unit, history.panel_runtime.sdk(), raw)
    if not (raw / "attempts.jsonl").exists():
        (raw / "attempts.jsonl").touch()
    result["attempts_sha256"] = sha256_file(raw / "attempts.jsonl")
    atomic_json(raw / "episode.json", result)
    return int(bool(result["error"] or result["model_tripwire_attempts"]))


def snapshot(**kwargs: Any) -> TrajectorySnapshot:
    """The qualified extension is required to replay masked selection faithfully."""
    return (
        CapabilitySnapshot(**kwargs)
        if "capability_mask_enabled" in kwargs
        else TrajectorySnapshot(**kwargs)
    )


def reduce_episode(reference: Json) -> tuple[Json, list[Json], list[Json]]:
    """Use the shipped effect checks with both arms normalized to applied mode."""
    directory = Path(reference["directory"])
    terminal = checked(directory / "episode.json", reference["terminal_sha256"])
    if terminal["unit"] != reference["unit"]:
        raise ValueError("episode_identity")
    normalized = deepcopy(reference)
    normalized["unit"]["mode"] = "applied"

    def applied_checked(path: Path, digest: str) -> Json:
        value = checked(path, digest)
        value["unit"]["mode"] = "applied"
        return value

    with (
        patch.object(history, "checked", applied_checked),
        patch.object(history.panel_runtime, "TrajectorySnapshot", snapshot),
    ):
        row, effects = history.reduce_episode(normalized)
    row["mode"] = reference["unit"]["mode"]
    row["episode_runtime_s"] = row["runtime_s"]
    row["runtime_s"] = reference.get("complete_wall_s", row["runtime_s"])
    for event in effects:
        event["mode"] = row["mode"]
    attempts_path = directory / "attempts.jsonl"
    checked_bytes(str(attempts_path), terminal["attempts_sha256"])
    attempts = [json.loads(line) for line in attempts_path.read_text().splitlines()]
    if len(attempts) != row["actions_used"] or any(
        a != b[: len(a)] for a, b in zip(attempts, attempts[1:])
    ):
        raise ValueError("attempt_sequence")
    actual = attempts[-1] if attempts else []
    if any(a.get("skipped") != "disabled_by_env" for a in actual):
        raise ValueError("model_attempt_executed")
    direct = terminal["supervisor_receipt"].get("direct_effect_receipts", [])
    exclusions = [
        dict(
            episode_id=row["episode_id"],
            game=row["game"],
            seed=row["seed"],
            mode=row["mode"],
            action_index=r["action_index"],
            **p,
        )
        for r in direct
        for p in r["eligibility_predicates"]
        if p.get("capability_eligible") is False or p.get("capability_reason") == "runtime_unknown"
    ]
    steps = [json.loads(line) for line in (directory / "steps.jsonl").read_text().splitlines()]
    row.update(
        actual_disabled_attempts=len(actual),
        selected_reinduction_count=sum(r["arm"] == "allow_reinduction" for r in effects),
        selected_disabled_attempts=sum(
            r["arm"] == "allow_reinduction" and r["induction_attempt"] is not None for r in effects
        ),
        actions_to_progress=next((s["index"] for s in steps if s["progress"]), None),
        unique_visible_states=len({canonical_hash(s["observation"]["pixels"]) for s in steps}),
        induction_latch_changes=sum(
            r["arm"] == "allow_reinduction" and r["applied"] for r in effects
        ),
        effective_diversity_changes=sum(
            r["arm"] == "force_exploration_diversity" and r["effective_action_policy_change"]
            for r in effects
        ),
        exhaustion_states=terminal["supervisor_receipt"].get("unredirected_windows", []),
        exhaustion_states_dropped=terminal["supervisor_receipt"].get(
            "unredirected_windows_dropped", 0
        ),
        unknown_capability_count=sum(
            p.get("capability_reason") == "runtime_unknown" for p in exclusions
        ),
        unredirected_stagnations=terminal["supervisor_receipt"]["stagnations_unredirected"],
        resolved_redirects=sum(r["resolved_by_levelup"] for r in effects),
    )
    return row, effects, exclusions


def pair(rows: list[Json]) -> Json:
    """Two seeds describe each game; action rows never become transfer units."""
    pairs = []
    for key in dict.fromkeys(r["pair_key"] for r in rows):
        baseline, masked = [
            next(r for r in rows if r["pair_key"] == key and r["mode"] == m) for m in MODES
        ]
        complete = all(r["status"] == "completed" for r in [baseline, masked])
        matched = bool(
            baseline.get("observation_hashes")
            and masked.get("observation_hashes")
            and baseline["observation_hashes"][0] == masked["observation_hashes"][0]
        )
        deltas = {
            k: masked[k] - baseline[k] if complete else None
            for k in [
                "peak_level",
                "runtime_s",
                "actions_used",
                "unique_visible_states",
                "actual_disabled_attempts",
                "selected_reinduction_count",
                "selected_disabled_attempts",
            ]
        }
        pairs.append(
            dict(
                pair_key=key,
                game=baseline["game"],
                seed=baseline["seed"],
                complete=complete,
                matched_initial_state=matched,
                deltas=deltas,
                actions_to_progress={
                    m: r["actions_to_progress"] for m, r in zip(MODES, [baseline, masked])
                },
                right_censored=any(bool(r["censoring"]) for r in [baseline, masked]),
            )
        )
    games = []
    for game in dict.fromkeys(r["game"] for r in rows):
        grouped = [r for r in pairs if r["game"] == game]
        deltas = {
            k: sum(r["deltas"][k] for r in grouped) / len(grouped)
            if all(r["complete"] for r in grouped)
            else None
            for k in pairs[0]["deltas"]
        }
        games.append(
            dict(game=game, pairs=grouped, seed_average_changes=deltas, independent_unit="game")
        )
    complete = bool(pairs) and all(p["complete"] for p in pairs)
    useful = complete and all(
        p["matched_initial_state"]
        and p["deltas"]["actual_disabled_attempts"] < 0
        and p["deltas"]["peak_level"] >= 0
        and p["deltas"]["runtime_s"] <= 0
        for p in pairs
    )
    uncertainty = (
        {
            k: dict(
                game_mean=sum(g["seed_average_changes"][k] for g in games) / len(games),
                game_range=[
                    min(g["seed_average_changes"][k] for g in games),
                    max(g["seed_average_changes"][k] for g in games),
                ],
            )
            for k in pairs[0]["deltas"]
        }
        if complete
        else None
    )
    return dict(
        per_game_results=games,
        paired_results=pairs,
        descriptive_uncertainty=uncertainty,
        useful_efficiency=useful,
        zero_progress_headroom=complete and all(r["peak_level"] == 0 for r in rows),
    )


def reduce(references: list[Json], panel: list[Json]) -> Json:
    """Retain every intended slot and all unresolved windows without imputing zeros."""
    mapping = {r["unit"]["episode_id"]: r for r in references}
    if len(mapping) != len(references) or not set(mapping) <= {u["episode_id"] for u in panel}:
        raise ValueError("duplicate_or_unknown_episode")
    rows, effects, exclusions = [], [], []
    for unit in panel:
        ref = mapping.get(unit["episode_id"])
        if ref and ref["terminal_sha256"]:
            row, events, excluded = reduce_episode(ref)
            effects.extend(events)
            exclusions.extend(excluded)
        else:
            row = dict(
                unit,
                status="failed" if ref else "unstarted",
                absolute_metric=None,
                missing_reason="owned_child_missing_terminal"
                if ref
                else "missing_episode_terminal_or_external_precondition",
                censoring="missing",
                actual_disabled_attempts=None,
                selected_reinduction_count=None,
                selected_disabled_attempts=None,
                peak_level=None,
                runtime_s=None,
                actions_used=None,
                unique_visible_states=None,
                actions_to_progress=None,
                receipt_count=0,
            )
        rows.append(row)
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
            for m in MODES
        ]
        overlap = [
            g
            for g in dict.fromkeys(u["game"] for u in panel)
            if all(
                any(r["game"] == g and r["mode"] == m and r["arm"] == arm for r in effects)
                for m in MODES
            )
        ]
        support.append(
            dict(
                arm=arm,
                cells=cells,
                overlapping_games=overlap,
                authorized=len(overlap) >= 3
                and all(c["resolved"] >= 5 for c in cells if c["game"] in overlap),
            )
        )
    return dict(
        rows=rows,
        per_episode_rows=rows,
        redirect_effect_rows=effects,
        capability_exclusion_rows=exclusions,
        disabled_attempt_counts=[
            {
                k: r[k]
                for k in [
                    "episode_id",
                    "actual_disabled_attempts",
                    "selected_reinduction_count",
                    "selected_disabled_attempts",
                ]
            }
            for r in rows
        ],
        action_costs=[
            {k: r[k] for k in ["episode_id", "actions_used", "actions_to_progress", "runtime_s"]}
            for r in rows
        ],
        resolved_redirects=sum(r["resolved_by_levelup"] for r in effects),
        censoring_counts=dict(
            episodes=sum(bool(r["censoring"]) for r in rows),
            redirects=sum(r["right_censored"] for r in effects),
        ),
        independent_count=len({r["game"] for r in rows if r["status"] == "completed"}),
        effect_support_by_arm=support,
        arm_refinement_authorized=any(r["authorized"] for r in support),
        **pair(rows),
    )


def build(work_path: Path, receipts: list[Json], output: Path) -> Json:
    """A complete qualified panel may be a terminal null on useful efficiency."""
    work = json.loads(work_path.read_bytes())
    primitive_error = None
    try:
        summary = reduce(work["references"], work["panel"]["units"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        primitive_error = str(exc)
        summary = reduce([], work["panel"]["units"])
    rows = summary["rows"]
    qualified = (
        bool(receipts)
        and all(r["passed"] for r in receipts)
        and REQUIRED <= {r["name"] for r in receipts}
    )
    execution = (
        not primitive_error
        and all(r["passed"] for r in work["measurement_receipts"])
        and not any(r["status"] == "failed" for r in rows)
        and (bool(work["failures"]) or len(work["references"]) == 16)
    )
    ready = int(
        qualified
        and execution
        and not work["failures"]
        and all(p["matched_initial_state"] for p in summary["paired_results"])
        and all(
            r["status"] == "completed"
            and r["receipt_count"] > 0
            and r["model_tripwire_passed"]
            and r["solve_provenance"] == "live_agent_self_discovery"
            for r in rows
        )
    )
    efficiency = int(bool(ready and summary["useful_efficiency"]))
    verdict = (
        "disqualified"
        if not execution or receipts and not all(r["passed"] for r in receipts)
        else "blocked"
        if work["failures"]
        else "circular_positive"
        if efficiency
        else "null"
    )
    source = work["source"] or {}
    failures = list(work["failures"])
    if primitive_error:
        failures.append(
            history.shipped.gate(
                "primitive_replay", work_path, "valid_primitive_operands", True, primitive_error
            )
        )
    value = dict(
        experiment_id=8537,
        task_id=TASK,
        milestone="2026.10.733",
        run_date="20261015",
        honest_verdict=f"complete_{verdict}_v733_arc_capability_panel",
        verdict_class=verdict,
        gate_check_summary=failures,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(history.ZERO_INVOCATION_COUNTS),
        llm_invocation_count=0,
        historical_model_provenance=source.get("primary", {}).get(
            "historical_model_provenance", {}
        ),
        **summary,
        intended_count=16,
        completed_count=sum(r["status"] == "completed" for r in rows),
        failed_count=sum(r["status"] == "failed" for r in rows)
        if not primitive_error
        else len(work["references"]),
        censored_count=summary["censoring_counts"]["episodes"],
        excluded_count=0,
        sample_size_budget=dict(
            independent_games=4,
            episodes=16,
            seeds=[11, 22],
            modes=MODES,
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
            useful_efficiency=bool(efficiency),
            arm_support=summary["arm_refinement_authorized"],
            scientific_reasoning_benefit_claimed=False,
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(
            output.parent / "raw" / output.stem / "terminal_validation.json"
        ),
        adversarial_findings=[],
        preconditions_checked=work["preconditions_checked"],
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        random_seed=7338537,
        reproducibility_checksum=canonical_hash(work),
        source_artifact_hashes=work["source_artifact_hashes"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes={
            str(Path(r["directory"]) / f): r.get(k)
            for r in work["references"]
            for f, k in [("episode.json", "terminal_sha256"), ("steps.jsonl", "steps_sha256")]
        },
        cited_upstream_artifacts=[
            dict(
                task_id=qualification.TASK,
                path=UPSTREAM,
                sha256=UPSTREAM_PIN,
                original_date="20261015",
            )
        ],
        arc_capability_panel_ready_score=ready,
        capability_efficiency_score=efficiency,
        registry_precheck=work["panel"]["registry_precheck"],
        solve_provenance="live_agent_self_discovery" if work["references"] else "unmeasured",
        headline_solve_credit=0,
        official_score=None,
        official_score_missing_reason="offline_public_engineering_panel",
        offline_reproduced=False,
        reproduced_levels=0,
        frozen_panel_reference=work["frozen_reference"],
        measurement_receipts=work["measurement_receipts"],
        work_reference=dict(path=str(work_path), sha256=sha256_file(work_path)),
        next_model_enabled_condition="Qualify a separate mandated Qwen runtime and MODEL_SPECS with live generator receipts, then preregister a fresh model-enabled study; this exposed engineering panel cannot select priorities from censoring.",
        methodology_note="Sixteen fresh model-disabled make_carnot_agent episodes on four historically exposed public games. Seeds average within game. Capability exclusions establish mechanical gating only. Useful efficiency requires fewer actual disabled attempts, nondecreasing paired levels and no complete wall-cost increase. No pooled action-level significance, hidden-game result, improved reasoning, learned priorities or new solve is claimed. Unresolved windows stay right-censored; null is terminal.",
    )
    value["field_principles"] = {
        k: "Bind actual invocation to immutable primitive operands; engineering qualification grants no reasoning, generalization or solve credit."
        for k in value
    }
    value["field_principles"].update(
        arc_capability_panel_ready_score="Require all sixteen authenticated episodes and every owned validation independently of efficiency.",
        capability_efficiency_score="Require fewer actual disabled attempts, nondecreasing levels in every pair and no increase in complete wall cost.",
        disabled_attempt_counts="Keep selected reinduction events, their observed disabled attempts and all actual disabled attempts separate.",
        per_game_results="Average two seeds within each of four game transfer units; uncertainty is descriptive game mean and range.",
        arm_refinement_authorized="Keep three overlapping games and five resolved observations per shared arm/mode cell.",
        acceptance_gates="Owned failure disqualifies; external absence blocks; readiness never substitutes for benefit.",
    )
    return value


def replay(value: Json) -> bool:
    """Check independent seals before recomputing every terminal claim in a fresh child."""
    try:
        work = checked(Path(value["work_reference"]["path"]), value["work_reference"]["sha256"])
        frozen = checked(Path(work["frozen_reference"]["path"]), work["frozen_reference"]["sha256"])
        if (
            frozen != work["panel"]
            or frozen["protocol"] != qualification.protocol()
            or frozen["units"] != units(qualification.protocol())
        ):
            return False
        activation = work["activation_reference"]
        if activation["exists"]:
            plan = yaml.safe_load(checked_bytes(activation["snapshot_path"], activation["sha256"]))
            task = next((t for t in plan["tasks"] if t["id"] == TASK), None)
            if not work["failures"] and canonical_hash(task) != TASK_PIN:
                return False
        elif not work["failures"]:
            return False
        if work["source"]:
            source = work["source"]
            if (
                checked(Path(source["reference"]["snapshot_path"]), UPSTREAM_PIN)
                != source["primary"]
            ):
                return False
        elif not work["failures"]:
            return False
        for path, digest in work["code_config_hashes"].items():
            checked_bytes(path, digest)
        if len(work["references"]) != len(work["measurement_receipts"]):
            return False
        for ref, receipt in zip(work["references"], work["measurement_receipts"], strict=True):
            elapsed = (receipt["ended_monotonic_ns"] - receipt["started_monotonic_ns"]) / 1e9
            if ref["complete_wall_s"] != receipt["duration_s"] or elapsed != receipt["duration_s"]:
                return False
        for receipt in [*work["measurement_receipts"], *value["validation_receipts"]]:
            for channel in ["stdout", "stderr"]:
                if channel + "_path" in receipt:
                    checked_bytes(receipt[channel + "_path"], receipt[channel + "_sha256"])
        reduce(work["references"], frozen["units"])
        output = Path(value["terminal_validation_sidecar_path"]).parents[2] / (NAME + ".json")
        return (
            build(Path(value["work_reference"]["path"]), value["validation_receipts"], output)
            == value
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False
