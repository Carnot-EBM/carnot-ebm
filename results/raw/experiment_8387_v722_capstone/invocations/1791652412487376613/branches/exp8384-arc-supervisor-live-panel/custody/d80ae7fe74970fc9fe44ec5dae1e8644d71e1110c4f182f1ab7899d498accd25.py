"""REQ-REPORT-8384: collect a fixed public panel without granting hidden-game credit."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import yaml

from carnot.agentic.arc_generalization_runtime import freeze_schedule
from carnot.agentic.arc_trajectory_supervisor import TrajectorySnapshot, TrajectorySupervisor
from carnot.agentic import arc_live_panel_runtime_8384 as runtime
from carnot.agentic.arc_competition_agent import E3AgentPolicy
from carnot.reporting import v717_contract_runner as coverage_runner
from carnot.reporting import v722_contract_methods as authority_module
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v709_execution import child, execute
from carnot.reporting.v718_replay_runner import audit

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8384_v722_arc_supervisor_live_panel"
TASK = "exp8384-arc-supervisor-live-panel"
TASK_PIN = "sha256:cb5f603dfd4ee8e089f395ec0dc35dbe20c671f0c189cb1cd2da2d06219e7bc4"
CLI = "scripts/experiments/" + NAME + ".py"
OUTPUT = ROOT / "results" / (NAME + ".json")
TEST = "tests/python/test_arc_supervisor_live_panel_8384.py"
OWNED = [
    "python/carnot/reporting/arc_supervisor_live_panel_8384.py",
    "python/carnot/agentic/arc_live_panel_runtime_8384.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
INPUTS = [
    "AGENTS.md",
    "CODEX.md",
    "CLAUDE.md",
    "ops/e2e-test-plan.md",
    "ops/exclusion_manifest.yaml",
    "ops/arc_solve_registry.yaml",
    "openspec/capabilities/research-reporting/spec.md",
    "openspec/capabilities/verification/spec.md",
    "scripts/experiment_template.py",
    "python/carnot/reporting/primary_publication.py",
    "openspec/change-proposals/research-roadmap-vNEXT.md",
    "openspec/change-proposals/research-roadmap-v721-preserved-20261010.md",
    "python/carnot/agentic/arc_competition_agent.py",
    "python/carnot/agentic/arc_generalization_runtime.py",
    "python/carnot/agentic/arc_trajectory_supervisor.py",
    "python/carnot/agentic/arc_frame_change_predictor.py",
    "results/experiment_4629_live_frame_change_cnn.pt",
    "python/carnot/experiment_7708_v671_arc_generalization_runner.py",
    "tests/python/test_experiment_7708_v671_arc_generalization_runner.py",
    "python/carnot/reporting/arc_outcome_delta_8370.py",
    "results/experiment_8370_v721_arc_outcome_delta.json",
    authority_module.ACTIVE,
    authority_module.PROTOCOL,
    "openspec/change-proposals/v717-local-learning-protocol.json",
    "openspec/change-proposals/v721-deployment-protocol.json",
    "openspec/change-proposals/v721-methods-manifest.json",
    *OWNED,
    TEST,
]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush measured counts so bounded work stays visible even during cold reduction."""
    print(f"[exp8384] phase={phase} completed={completed} pending={pending}", flush=True)


def freeze(roster: list[str], registry: list[Json]) -> Json:
    """Reuse the metadata-only hash order; historical registry entries never supply actions."""
    mapped = [
        dict(row, game=game)
        for game in roster
        for row in registry
        if row["game"] == game.split("-")[0]
    ]
    frozen = freeze_schedule(roster, mapped)
    games = frozen["roster"][:2]
    return dict(
        selected_games=games,
        selection_rule=frozen["selection_rule"],
        selection_salt=frozen["selection_salt"],
        roster=frozen["roster"],
        registry_precheck=frozen["registry_precheck"],
        units=[
            dict(
                episode_id=f"{game}:{seed}:{arm}",
                game=game,
                seed=seed,
                arm=arm,
                max_actions=256,
                max_seconds=180,
            )
            for game in games
            for seed in [11, 22]
            for arm in ["off", "on"]
        ],
    )


def preconditions(private: Path, root: Path) -> tuple[list[Json], Json, Json]:
    """Seal exact authority and inputs; external absence cannot turn into a successful check."""
    checks: list[Json] = []
    hashes: Json = {}
    for name in INPUTS:
        path = root / name
        observed = sha256_file(path) if path.is_file() else None
        hashes[str(path)] = observed
        if observed is None:
            checks.append(
                dict(
                    check="input_present",
                    upstream=name,
                    path=str(path),
                    hash=None,
                    field="is_file",
                    operator="==",
                    expected=True,
                    observed=None,
                    passed=False,
                )
            )
    contract = authority_module.authority(root, private / "authority")
    checks.extend(
        dict(
            check=g["check"],
            upstream=g["upstream_id"],
            path=g["artifact_path"],
            hash=g["artifact_hash"],
            field=g["artifact_field"],
            operator=g["op"],
            expected=g["expected"],
            observed=g["observed"],
            passed=False,
        )
        for g in contract["gate_check_summary"]
    )
    task = next((t for t in contract["tasks"] if t.get("id") == TASK), None)
    digest = canonical_hash(task) if task else None
    if digest != TASK_PIN:
        checks.append(
            dict(
                check="exact_task",
                upstream=TASK,
                path=str(root / authority_module.ACTIVE),
                hash=hashes.get(str(root / authority_module.ACTIVE)),
                field="task_sha256",
                operator="==",
                expected=TASK_PIN,
                observed=digest,
                passed=False,
            )
        )
    for name in ["python", "pytest", "coverage", "ruff", "mypy"]:
        path = root / ".venv/bin" / name
        if not path.is_file():
            checks.append(
                dict(
                    check="tool",
                    upstream=name,
                    path=str(path),
                    hash=None,
                    field="is_file",
                    operator="==",
                    expected=True,
                    observed=None,
                    passed=False,
                )
            )
    memory = (
        int(
            next(
                line.split()[1]
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:")
            )
        )
        * 1024
    )
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    private.chmod(0o700)
    probe = private / "disk-probe"
    probe.write_bytes(b"exp8384-private")
    mounts = [line.split() for line in Path("/proc/mounts").read_text().splitlines()]
    filesystem = max(
        (m for m in mounts if private.resolve().is_relative_to(m[1])), key=lambda m: len(m[1])
    )[2]
    resources: Json = dict(
        private_scratch=str(private),
        filesystem=filesystem,
        free_disk_bytes=shutil.disk_usage(private).free,
        available_memory_bytes=memory,
        task_cap_s=4800,
        child_heartbeat_s=30,
        task_sha256=digest,
        task_authorized_by="current_user_instruction_and_pinned_active_task",
    )
    if (
        filesystem == "tmpfs"
        or private.resolve().is_relative_to(root)
        or resources["free_disk_bytes"] < 128_000_000
        or memory < 256_000_000
        or probe.read_bytes() != b"exp8384-private"
    ):
        checks.append(
            dict(
                check="private_resources",
                upstream="host",
                path=str(private),
                hash=None,
                field="disk_backed_private_scratch_and_memory",
                operator="==",
                expected=True,
                observed=resources,
                passed=False,
            )
        )
    return checks, hashes, resources


def plan(private: Path) -> list[Json]:
    """Reuse real CLI coverage and freeze private E2E and wrapper commands before outcomes."""
    binding = SimpleNamespace(ROOT=ROOT, OWNED=OWNED, TEST=TEST)
    with patch.object(coverage_runner, "m", binding):
        specs = list(coverage_runner.manifest(private))
    specs[0]["deadline"] = 900
    common = [str(ROOT / ".venv/bin/pytest"), "-n", "0", "-o", "addopts=", "--no-cov", "-q"]
    specs.insert(
        0,
        dict(
            name="private_E2E017_and_wrapper",
            argv=[
                *common,
                "tests/python/test_arc_supervisor_delta_7874.py",
                "tests/python/test_experiment_7708_v671_arc_generalization_runner.py",
                "tests/python/test_arc_submitted_agent_parity.py",
                "--basetemp=" + str(private / "wrapper"),
            ],
            deadline=240,
            expected=0,
            scope="owned",
        ),
    )
    specs.append(
        dict(
            name="repository_health_once",
            argv=[
                str(ROOT / ".venv/bin/pytest"),
                "tests/python",
                "-q",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "--basetemp=" + str(private / "global"),
            ],
            deadline=900,
            expected=0,
            scope="global",
        )
    )
    return specs


def reduce_episode(unit: Json, terminal: Json | None, raw: Path) -> Json:
    """Rebuild outcome windows from observations and actual supervisor operands, never a headline."""
    stream = raw / "steps.jsonl"
    steps = (
        [json.loads(line) for line in stream.read_text().splitlines()] if stream.exists() else []
    )
    supervisor = TrajectorySupervisor(window=120)
    previous_level = 0
    for index, step in enumerate(steps):
        if step["index"] != index + 1 or step["action_sha256"] != canonical_hash(step["action"]):
            raise ValueError("action_hash")
        if step["observation_sha256"] != canonical_hash(step["observation"]):
            raise ValueError("observation_hash")
        if step["progress"] != (step["observation"]["level"] > previous_level):
            raise ValueError("progress_drift")
        for event in step["supervisor_events"]:
            if event["snapshot"]["level"] != previous_level:
                raise ValueError("supervisor_level")
            redirect = supervisor.observe(TrajectorySnapshot(**event["snapshot"]))
            if event["redirect"] != (asdict(redirect) if redirect else None):
                raise ValueError("redirect_drift")
        previous_level = step["observation"]["level"]
    for event in (terminal or {}).get("pending_supervisor_events", []):
        redirect = supervisor.observe(TrajectorySnapshot(**event["snapshot"]))
        if event["redirect"] != (asdict(redirect) if redirect else None):
            raise ValueError("pending_redirect_drift")
    policy = SimpleNamespace(
        _trajectory_supervisor=supervisor,
        _trajectory_supervisor_applies=unit["arm"] == "on",
        _trajectory_supervisor_errors=0,
    )
    receipt = E3AgentPolicy.trajectory_supervisor_diagnostics(policy)
    if terminal and terminal["supervisor_receipt"] and terminal["supervisor_receipt"] != receipt:
        raise ValueError("supervisor_receipt_drift")
    events = receipt.get("redirects", receipt.get("would_have_redirects", []))
    outcome_rows = [
        dict(
            event,
            applied=unit["arm"] == "on",
            resolved_by_levelup=event.get(
                "resolved_by_levelup", event.get("levelup_followed_without_redirect", False)
            ),
            actions_to_levelup=event.get(
                "actions_to_levelup", event.get("actions_to_levelup_without_redirect")
            ),
            outcome_window_status="resolved"
            if event.get(
                "resolved_by_levelup", event.get("levelup_followed_without_redirect", False)
            )
            else "pending_censored",
        )
        for event in events
    ]
    status = "failed" if terminal and terminal["error"] else "completed" if terminal else "censored"
    return dict(
        unit,
        status=status,
        action_count=len(steps),
        peak_level=max((s["observation"]["level"] for s in steps), default=None),
        absolute_metric=len(steps) if steps else None,
        choose_action_s=sum(s["choose_action_s"] for s in steps),
        sdk_transition_s=sum(s["sdk_transition_s"] for s in steps),
        progress_count=sum(s["progress"] for s in steps),
        censoring=terminal["censoring"] if terminal else "child_deadline",
        error=terminal["error"] if terminal else None,
        missing_reason=None if steps else "no_sdk_observation",
        supervisor_receipt=receipt,
        supervisor_outcome_rows=outcome_rows,
        redirect_count=len(events),
        resolved_by_levelup=sum(e["resolved_by_levelup"] for e in outcome_rows),
        pending_outcome_count=sum(
            e["outcome_window_status"] == "pending_censored" for e in outcome_rows
        ),
        stagnations_unredirected=receipt["stagnations_unredirected"],
        emitted_receipt_count=int(bool(terminal and terminal["supervisor_receipt"])),
        solve_provenance="live_agent_self_discovery",
        headline_solve_credit=0,
        model_tripwire_passed=bool(terminal and terminal["model_tripwire_passed"]),
        actual_wrapper_path=(terminal or {}).get("actual_wrapper_path", {}),
        primitive_directory=str(raw),
    )


def build(work: Json, receipts: list[Json], raw: Path, output: Path, *, seal: bool = True) -> Json:
    """Execution and evidence authority govern readiness; an exposed panel grants no benefit score."""
    if seal:
        atomic_json(raw / "measurement.json", work)
    intended = (work.get("panel") or {}).get(
        "units",
        [
            dict(episode_id=f"missing:{g}:{s}:{a}", game=None, seed=s, arm=a)
            for g in range(2)
            for s in [11, 22]
            for a in ["off", "on"]
        ],
    )
    observed = {r["episode_id"]: r for r in work["rows"]}
    rows = [
        observed.get(
            u["episode_id"],
            dict(
                u,
                status="unstarted",
                absolute_metric=None,
                action_count=None,
                missing_reason="external_precondition_or_qualification",
                censored=True,
            ),
        )
        for u in intended
    ]
    owned = bool(receipts) and all(r["passed"] for r in receipts if r["scope"] == "owned")
    owned_failed = any(not r["passed"] for r in receipts if r["scope"] == "owned")
    verdict = (
        "disqualified"
        if owned_failed or any(r["status"] == "failed" for r in rows)
        else ("blocked" if work["failures"] else "null")
    )
    ready = int(
        owned
        and verdict == "null"
        and len(observed) == 8
        and all(r.get("model_tripwire_passed") for r in rows)
    )
    value = dict(
        experiment_id=8384,
        task_id=TASK,
        milestone="2026.10.722",
        run_date=work["run_date"],
        honest_verdict=f"complete_{verdict}_fresh_public_supervisor_panel",
        verdict_class=verdict,
        gate_check_summary=work["failures"],
        inference_substrate="offline_arcade_live_agent_runtime_self_discovery_no_llm",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(loads=0, generations=0, llm_calls=0),
        historical_model_provenance=work["historical_model_provenance"],
        rows=rows,
        intended_count=8,
        completed_count=sum(r["status"] == "completed" for r in rows),
        failed_count=sum(r["status"] == "failed" for r in rows),
        censored_count=sum(bool(r.get("censoring") or r.get("censored")) for r in rows),
        excluded_count=0,
        independent_count=len({r["game"] for r in work["rows"] if r["action_count"]}),
        sample_size_budget=dict(
            games=2,
            seeds=[11, 22],
            arms=["off", "on"],
            intended_episodes=8,
            max_actions=256,
            max_seconds=180,
            supervisor_window=120,
            independent_unit="public_game",
            hidden_games=0,
        ),
        verifier_is_oracle=False,
        exposure_scope="historically_exposed_installed_public_SDK_games",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=owned,
        flagged_adversarial=verdict == "disqualified",
        acceptance_gates=dict(
            owned_validation=owned,
            external_evidence=not work["failures"],
            fixed_panel=ready == 1,
            scientific_benefit=False,
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(
            output.parent / "raw" / output.stem / "terminal_validation.json"
        ),
        adversarial_findings=work.get("adversarial_findings", []),
        preconditions_checked=work["preconditions_checked"],
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        random_seed=[11, 22],
        reproducibility_checksum=canonical_hash(work),
        source_artifact_hashes=work["source_artifact_hashes"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes=work["raw_shard_hashes"],
        cited_upstream_artifacts=[
            dict(
                path=path,
                sha256=digest,
                fields_imported=["historical_model_provenance"]
                if "experiment_8370" in path
                else ["authority_or_input_bytes"],
            )
            for path, digest in sorted(work["source_artifact_hashes"].items())
        ],
        arc_panel_ready_score=ready,
        per_game_results=rows,
        supervisor_outcome_rows=[
            dict(event, episode_id=r["episode_id"])
            for r in work["rows"]
            for event in r["supervisor_outcome_rows"]
        ],
        emitted_receipt_count=sum(r["emitted_receipt_count"] for r in work["rows"]),
        solve_provenance="live_agent_self_discovery",
        registry_precheck=(work.get("panel") or {}).get("registry_precheck", {}),
        reproduction_evidence=[
            dict(
                game=r["game"],
                seed=r["seed"],
                arm=r["arm"],
                peak_level=r["peak_level"],
                provenance="fresh_paired_seed_environment",
                headline_solve_credit=0,
            )
            for r in work["rows"]
        ],
        headline_solve_credit=0,
        llm_invocation_count=0,
        model_tripwire_passed=all(r["model_tripwire_passed"] for r in work["rows"])
        and bool(work["rows"]),
        adapter_withheld=True,
        actual_wrapper_path=[r["actual_wrapper_path"] for r in work["rows"]],
        frozen_panel=work.get("panel"),
        work_reference=dict(
            path=str(raw / "measurement.json"), sha256=sha256_file(raw / "measurement.json")
        ),
        global_repository_health=[r for r in receipts if r["scope"] == "global"],
    )
    value["field_principles"] = {
        key: "Bind " + key + " to sealed operands and keep missing observations distinct from zero."
        for key in [*value, "field_principles"]
    }
    value["field_principles"].update(
        dict(
            experiment_id="Identify this producer, rather than a historical experiment.",
            task_id="Bind the receipt to the authenticated task digest.",
            milestone="Separate V722 execution from preserved V721 evidence.",
            run_date="Use the exact authorized date, 20261010.",
            honest_verdict="A complete terminal disposition prevents retries of unchanged absence.",
            verdict_class="Owned failures disqualify; external absence blocks; exposed comparisons cannot be positive.",
            gate_check_summary="Name each missing or incorrect external operand with exact authority, path, hash and comparison.",
            inference_substrate="Only real offline SDK transitions through the live policy supply observations.",
            inference_substrate_class="No LLM loads; the existing small common numeric head remains separate.",
            MODEL_SPECS="An empty model roster prevents a false claim of current LLM inference.",
            model_invocation_counts="Tripwires prevent LLM loading and generation before either occurs.",
            historical_model_provenance="Imported model calls are historical, never current calls.",
            rows="Retain every intended game, paired seed and arm, even without an observation.",
            intended_count="The fixed experiment contains exactly eight intended episodes.",
            completed_count="Count terminal episodes with emitted receipts, including bounded action-limit runs.",
            failed_count="Preserve every episode whose SDK or policy failed.",
            censored_count="Count bounded episodes whose future outcome window remains unobserved; this may overlap completion.",
            excluded_count="No intended episode is dropped after observing outcomes.",
            independent_count="Games are independent units; seeds and arms never enlarge independent n.",
            sample_size_budget="Freeze games, seeds, arm order, action cap, time cap and window before outcomes.",
            verifier_is_oracle="Visible runtime observations supply no hidden evaluator truth.",
            exposure_scope="Both installed public games have historical exposure and prior reproduction records.",
            independent_generalization_score="An exposed two-game panel cannot establish hidden-game generalization.",
            generalized_learning_benefit_score="No causal semantic benefit is established by numerical or wrapper parity.",
            required_checks_passed="Only frozen owned checks qualify execution; global health remains separate.",
            flagged_adversarial="Owned failures and unresolved findings close readiness.",
            acceptance_gates="Readiness, evidence authority and scientific benefit remain separate claims.",
            validation_receipts="Keep actual command vectors, exits, deadlines, durations and log hashes.",
            terminal_validation_sidecar_path="Locate the unchanged-validator report bound to exact primary bytes.",
            adversarial_findings="Retain all findings without changing validators or exemptions.",
            preconditions_checked="Measure authority, input bytes, private disk scratch, tools, memory and time budget before collection.",
            duration_s="Elapsed time is measured without padding, not inferred from action counts.",
            phase_spans="Monotonic boundaries distinguish observation costs from validation and repository health.",
            random_seed="Reset fresh actual environments with paired seeds11 and22.",
            reproducibility_checksum="Seal the complete primitive reduction operands.",
            source_artifact_hashes="Authenticate every imported input without assuming future producers exist.",
            code_config_hashes="Bind wrapper adapters and CLI behavior to exact measured code.",
            raw_shard_hashes="Cold replay rejects changed primitive frames, action hashes, snapshots and logs.",
            cited_upstream_artifacts="Name imported fields and their producer byte hashes.",
            arc_panel_ready_score="Require owned qualification, authenticated external evidence and all eight live receipts.",
            per_game_results="Keep per-game, seed and arm action costs and redirect outcomes.",
            supervisor_outcome_rows="Keep applied and shadow outcomes separate, including pending and censored windows.",
            emitted_receipt_count="Count actual terminal supervisor receipts, never fabricated completions.",
            solve_provenance="Any level transition comes from the actual live policy's self-discovery.",
            registry_precheck="Disclose prior exposed levels and full clears before selecting outcomes.",
            reproduction_evidence="Fresh paired environments disclose incidental transitions without new solve credit.",
            headline_solve_credit="Already exposed public games receive zero headline credit.",
            llm_invocation_count="No current LLM invocation occurs on this guarded substrate.",
            model_tripwire_passed="All recorded episode guards must remain clear of attempted model paths.",
            adapter_withheld="The shared isolation context denies game recipes, routes and cached per-game transitions.",
            actual_wrapper_path="Record real cascade factory, policy class, choose_action and is_done call counts.",
            frozen_panel="Metadata-only hash ordering fixes the complete paired schedule before measurement.",
            work_reference="Bind the primary to the exact primitive reduction manifest.",
            global_repository_health="Run the global suite once; its health cannot manufacture scientific benefit.",
            field_principles="Explain each field's evidence limit and interpretation.",
        )
    )
    return value


def replay(value: Json) -> bool:
    """Rehashing a changed summary cannot replace a fresh reduction of sealed primitive evidence."""
    try:
        reference = value["work_reference"]
        primitive = Path(reference["path"])
        if sha256_file(primitive) != reference["sha256"]:
            return False
        work = json.loads(primitive.read_text())
        if canonical_hash(work) != value["reproducibility_checksum"]:
            return False
        for name, digest in work["raw_shard_hashes"].items():
            if sha256_file(Path(name)) != digest:
                return False
        for row in work["rows"]:
            raw = Path(row["primitive_directory"])
            terminal = (
                json.loads((raw / "episode.json").read_text())
                if (raw / "episode.json").exists()
                else None
            )
            unit = next(u for u in work["panel"]["units"] if u["episode_id"] == row["episode_id"])
            if reduce_episode(unit, terminal, raw) != row:
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
        expected = build(work, value["validation_receipts"], primitive.parent, output, seal=False)
        return value == expected
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return False


def controls(value: Json, raw: Path) -> list[Json]:
    """Fresh CLI children must reject absent primitives, deliberate errors and rehashed summary claims."""
    receipts = []
    for name in ["valid", "missing_input", "deliberate_error", "rehashed_tamper"]:
        changed = deepcopy(value)
        if name == "missing_input":
            changed["work_reference"]["path"] = str(raw / "never-produced.json")
        if name == "deliberate_error":
            changed["completed_count"] += 1
        if name == "rehashed_tamper":
            work = json.loads(Path(changed["work_reference"]["path"]).read_text())
            if work["rows"]:
                work["rows"][0]["action_count"] += 1
            else:
                work["rows"] = [dict(episode_id="invented", action_count=999)]
            altered = raw / "rehashed_measurement.json"
            atomic_json(altered, work)
            if value["rows"] and value["completed_count"]:
                changed = build(
                    work,
                    value["validation_receipts"],
                    raw,
                    Path(value["terminal_validation_sidecar_path"]).parents[2] / (NAME + ".json"),
                    seal=False,
                )
            changed.update(
                work_reference=dict(path=str(altered), sha256=sha256_file(altered)),
                reproducibility_checksum=canonical_hash(work),
            )
        path = raw / (name + ".json")
        atomic_json(path, changed)
        receipts.append(
            child(
                "cold_" + name,
                [sys.executable, "-u", str(ROOT / CLI), "--cold-replay", str(path)],
                raw / "controls",
                expected=int(name != "valid"),
                deadline=60,
            )
        )
    return receipts


def publish(value: Json, work: Json, output: Path, raw: Path) -> None:
    """Keep every unchanged-validator finding; only checked terminal bytes become reader-visible."""
    attempts: list[Json] = []

    def validate(candidate: Path) -> Json:
        logs = raw / "terminal" / str(len(attempts))
        cold = child(
            "terminal_cold",
            [sys.executable, "-u", str(ROOT / CLI), "--cold-replay", str(candidate)],
            logs,
            deadline=60,
        )
        findings = audit(candidate, logs, {})
        rows = child(
            "strict_rows",
            [
                sys.executable,
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ],
            logs,
            deadline=60,
        )
        recorded = value["verdict_class"] == "disqualified" and bool(value["adversarial_findings"])
        report = dict(
            passed=cold["passed"] and (findings["passed"] or recorded) and rows["passed"],
            checks=[cold, findings["receipt"], rows],
            adversarial=findings,
        )
        attempts.append(report)
        return report

    try:
        publication = publish_primary(output, value, validate)
    except ValueError as exc:
        if str(exc) != "candidate_rejected":
            raise
        atomic_json(raw / "rejected_candidate.json", value)
        work["adversarial_findings"] = [attempts[0]["adversarial"]]
        value = build(work, value["validation_receipts"] + attempts[0]["checks"], raw, output)
        publication = publish_primary(output, value, validate)
    atomic_json(
        output.parent / "raw" / output.stem / "terminal_validation.json",
        dict(publication=publication, attempts=attempts, passed=attempts[-1]["passed"]),
    )


def sdk() -> Any:
    """Use installed public SDK access without reading any game implementation."""
    from arc_agi import Arcade, OperationMode

    return Arcade(
        arc_api_key="",
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(ROOT / "environment_files"),
    )


def run(output: Path, private: Path, *, private_e2e: bool = False) -> int:
    """Qualify mechanics first, collect the one frozen panel, then publish a cold-checked terminal."""
    started = time.monotonic_ns()
    raw = output.parent / "raw" / output.stem / "invocations" / str(started)
    raw.mkdir(parents=True, exist_ok=True)
    progress("preconditions_before")
    specs = [] if private_e2e else plan(private)
    atomic_json(
        raw / "command_manifest.json",
        dict(
            commands=specs,
            MODEL_SPECS=[],
            task_cap_s=4800,
            heartbeat_s=30,
            arm_order=["off", "on"],
            seeds=[11, 22],
            window=120,
        ),
    )
    checks, hashes, resources = preconditions(private, ROOT)
    snapshots: Json = {}
    for index, (name, digest) in enumerate(hashes.items()):
        if digest:
            target = raw / "inputs" / (str(index) + "-" + digest[7:] + ".bin")
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(name, target)
            snapshots[str(target)] = digest
    panel = None
    if not private_e2e:
        try:
            arcade = sdk()
            roster = [str(g.game_id) for g in arcade.available_environments]
            registry = yaml.safe_load((ROOT / "ops/arc_solve_registry.yaml").read_text())["games"]
            panel = freeze(roster, registry)
            atomic_json(raw / "frozen_panel.json", panel)
        except (ImportError, OSError, ValueError) as exc:
            checks.append(
                dict(
                    check="sdk_access",
                    upstream="arc_agi",
                    path=str(ROOT / "environment_files"),
                    hash=None,
                    field="two_installed_games_metadata",
                    operator="==",
                    expected=True,
                    observed=str(exc),
                    passed=False,
                )
            )
    else:
        checks.append(
            dict(
                check="private_constructed_control",
                upstream=TASK,
                path=str(private),
                hash=None,
                field="research_observations",
                operator="==",
                expected=True,
                observed=None,
                passed=False,
            )
        )
    progress("preconditions_after_validation_before")
    owned_specs = [s for s in specs if s["scope"] == "owned"]
    receipts = execute(owned_specs, raw / "checks")
    rows = []
    if panel and all(r["passed"] for r in receipts) and resources.get("task_sha256") == TASK_PIN:
        progress("live_panel_before", 0, 8)
        for index, unit in enumerate(panel["units"]):
            episode_raw = raw / "episodes" / str(index)
            operand = raw / f"unit-{index}.json"
            atomic_json(operand, unit)
            receipt = child(
                "episode_" + str(index),
                [
                    sys.executable,
                    "-u",
                    str(ROOT / CLI),
                    "--episode",
                    str(operand),
                    "--raw",
                    str(episode_raw),
                ],
                raw / "episodes_logs",
                deadline=180,
                scope="measurement",
            )
            terminal_path = episode_raw / "episode.json"
            terminal = json.loads(terminal_path.read_text()) if terminal_path.exists() else None
            rows.append(reduce_episode(unit, terminal, episode_raw))
            receipts.append(receipt)
            progress("live_panel_progress", index + 1, 7 - index)
        progress("live_panel_after", 8, 0)
    receipts.extend(execute([s for s in specs if s["scope"] == "global"], raw / "checks"))
    for path in raw.rglob("*"):
        if path.is_file():
            snapshots[str(path)] = sha256_file(path)
    upstream = ROOT / "results/experiment_8370_v721_arc_outcome_delta.json"
    historical = json.loads(upstream.read_text()).get("historical_model_provenance", [])
    work = dict(
        panel=panel,
        rows=rows,
        failures=checks,
        source_artifact_hashes=hashes,
        raw_shard_hashes=snapshots,
        code_config_hashes={p: sha256_file(ROOT / p) for p in OWNED},
        preconditions_checked=resources,
        historical_model_provenance=historical,
        phase_spans=[
            dict(
                phase="bounded_validation_and_live_panel",
                started_monotonic_ns=started,
                ended_monotonic_ns=time.monotonic_ns(),
            )
        ],
        duration_s=(time.monotonic_ns() - started) / 1e9,
        run_date="20261010",
    )
    value = build(work, receipts, raw, output)
    progress("cold_controls_before")
    receipts.extend(controls(value, raw))
    value = build(work, receipts, raw, output)
    progress("terminal_validation_before")
    publish(value, work, output, raw)
    progress("terminal_publication_after")
    return 0
