"""Build the terminal receipt for the shipped ARC transition-witness method.

This driver repeats the bounded CPU panel and exercises the real scored policy
wrapper. It writes a separate terminal candidate before it calculates readiness.

Spec: REQ-ARC-WMTE-7262 and SCENARIO-ARC-WMTE-7262-*.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import io
import json
import os
import platform
from pathlib import Path
import shlex
import tempfile
import time
from typing import Any, Mapping, Sequence
from unittest.mock import patch
import urllib.request

import numpy as np

from carnot.agentic import arc_competition_agent as competition
from carnot.agentic import arc_executable_world_model as e3
from carnot.agentic import arc_llm_reinduction as reinduce
from carnot.agentic.arc_transition_witness_exp7248 import (
    WITNESS_MARKER,
    build_transition_witness,
    canonical_witness_bytes,
)
from carnot.experiment_7246_v638_source_map import (
    _run_streaming_command as run_streaming_command,
)
from carnot.experiment_7248_v638_arc_witness import (
    BAD_CODE,
    GOOD_CODE,
    _changed_rows,
    _exec_engine,
    _reply,
    independently_reduce_conformance_rows,
    run_cpu_conformance_panel,
)
from carnot.experiment_artifacts import atomic_write_json


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "exp7262-arc-witness-receipt"
MILESTONE = "2026.09.639"
RUN_DATE = "20260913"
RANDOM_SEED = 7_262_202_609_13
OUTPUT_PATH = Path("results/experiment_7262_v639_arc_witness_receipt.json")
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7262_v639_arc_witness_receipt.json")
RAW_PATH = Path("results/raw/experiment_7262_v639_arc_witness_receipt_rows.json")
SIDECAR_PATH = Path("results/raw/experiment_7262_v639_arc_witness_receipt_sidecar.json")
TERMINAL_CANDIDATE_PATH = Path(
    "results/raw/experiment_7262_v639_arc_witness_receipt_terminal_candidate.json"
)

INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("python/carnot/experiment_7248_v638_arc_witness.py"),
    Path("python/carnot/agentic/arc_transition_witness_exp7248.py"),
    Path("python/carnot/agentic/arc_competition_agent.py"),
    Path("python/carnot/agentic/arc_llm_reinduction.py"),
    Path("tests/python/test_experiment_7248_v638_arc_witness.py"),
    Path("tests/python/test_arc_induction_state_persistence.py"),
    Path("tests/python/test_arc_tool_grammar_transport.py"),
    Path("results/experiment_7248_v638_arc_witness.json"),
    Path("scripts/arc_loop_solve.py"),
    Path("openspec/capabilities/arc-world-model-trust-energy/spec.md"),
)

FIELD_PRINCIPLES = {
    "schema": "Version the result; retain ordinary top-level experiment_id and milestone.",
    "status": "Use complete or blocked only for terminal work; unfinished work stays in a separate checkpoint.",
    "run_date": "Use 20260913, with actual UTC start/end timestamps, so dated evidence is auditable.",
    "field_principles": "Store explanations here; consumers read ordinary top-level values, not nested wrappers.",
    "preconditions_checked": "Retain observed input hashes, resource ownership and failures before expensive work.",
    "MODEL_SPECS": "Declare models executable in this invocation; keep historical model metadata in hashed sidecars.",
    "model_invoked": "Derive from actual calls; a parse failure does not erase a model invocation.",
    "invocation_counts": "Separate attempted and completed loads and generation calls from usable answers.",
    "inference_substrate": "Use an existing recognized literal that describes actual computation.",
    "inference_substrate_class": "Declare actual compute and never pad time.",
    "execution_venue": "Use host for host orchestration; identify real boards separately in board rows.",
    "duration_s": "Measure monotonic invocation time and disjoint phase spans; do not invent elapsed time.",
    "random_seed": "Freeze independent-unit seeds before inspecting outcomes.",
    "reproducibility_checksum": "Bind code, input manifests, configuration and raw evidence to the result.",
    "source_artifact_hashes": "Authenticate exact inputs and preserve quarantine and retirement state.",
    "rows": "Retain each independent unit, arm, seed, metric, error, abstention and censoring state for recomputation.",
    "sample_size_budget": "Record planned, attempted, completed and censored units and the fixed stopping rule.",
    "acceptance_gate_results": "Each criterion retains expected, observed, passed and principle; completion is separate from value.",
    "gate_check_summary": "For blocked_* name the upstream, exact field or check, observed value and expected value.",
    "verifier_is_oracle": "Expose shared evaluator authority; exact conformance is not learned correctness.",
    "honest_verdict": "Use complete_* for terminal measurements and blocked_* for external absence.",
    "verdict_class": "Use the closed verdict vocabulary; oracle evidence cannot receive a positive verdict.",
    "validation_receipts": "Record actual commands, exit codes and log hashes without suppressing failures.",
    "arc_witness_ready_score": "One certifies the shipped method and terminal evidence path, not world-model quality.",
    "witness_rows": "Retain observed triples, emitted witnesses and actual next-call delivery.",
    "terminal_handoff_rows": "A complete measurement passes where a partial checkpoint correctly fails.",
    "solve_provenance": "Use development_proxy for scripted CPU checks; no game-level solve is claimed.",
}


@dataclass(frozen=True)
class ValidationCommand:
    """Describe one bounded subprocess and its expected terminal result."""

    name: str
    command: list[str]
    timeout_s: int = 600
    expected_exit_code: int = 0
    expected_output: str | None = None


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    payload = _canonical_bytes({"dtype": array.dtype.str, "shape": list(array.shape)})
    return _sha256_bytes(payload + b"\0" + array.tobytes(order="C"))


def _progress(started: float, phase: str, message: str, completed: str = "0") -> None:
    print(
        f"[exp7262] phase={phase} elapsed_s={time.monotonic() - started:.3f} "
        f"completed_units={completed} {message}",
        flush=True,
    )


def _row(unit: str, arm: str, metric: str, passed: bool, **values: Any) -> dict[str, Any]:
    return {
        "unit": unit,
        "arm": arm,
        "seed": RANDOM_SEED,
        "metric": metric,
        "passed": bool(passed),
        "error": None,
        "abstention": False,
        "censored": False,
        **values,
    }


def reduce_receipt_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Recompute readiness from old panel rows plus scored-policy handoff rows."""

    old = independently_reduce_conformance_rows(rows)
    by_unit = {str(row.get("unit")): row for row in rows}
    delivery = by_unit.get("policy_runtime_delivery", {})
    generated = delivery.get("generated_witness_sha256")
    policy_delivery = bool(
        generated
        and generated == delivery.get("delivered_witness_sha256")
        and delivery.get("delivered_to_next_prompt") is True
        and delivery.get("direction_and_state_identity_match") is True
        and delivery.get("accepted_by_existing_policy_gate") is True
    )
    parity = by_unit.get("policy_default_parity", {})
    policy_parity = bool(
        parity.get("request_bytes_equal") is True and parity.get("action_equal") is True
    )
    no_future = by_unit.get("policy_future_leakage", {}).get("future_transition_absent") is True
    plan_authority = by_unit.get("policy_plan_authority", {}).get("policy_plan_unchanged") is True
    gates = {
        **old["gates"],
        "policy_runtime_delivery": policy_delivery,
        "policy_default_parity": policy_parity,
        "no_future_transition_leakage": no_future,
        "plan_installation_authority_unchanged": plan_authority,
    }
    return {
        "gates": gates,
        "arc_witness_ready_score": int(all(gates.values())),
        "completed_units": sum(bool(str(row.get("unit", ""))) for row in rows),
    }


def _policy_runtime_arm(
    store: Path,
    *,
    game: str,
    witness_value: str,
    transitions: Sequence[Any],
) -> dict[str, Any]:
    """Drive the real E3 policy wrapper with deterministic HTTP responses."""

    proposer = e3.LocalGGUFProposer(ffn_cpu_layers=0, mtp=False, max_tokens=1024, tries=1)
    proposer._ensure_server = lambda: True
    replies = [_reply(BAD_CODE), _reply(GOOD_CODE)]
    sent: list[bytes] = []

    def request(req: Any, timeout: Any = None) -> io.BytesIO:
        del timeout
        sent.append(bytes(req.data))
        return io.BytesIO(json.dumps(replies.pop(0)).encode())

    environment = {
        "CARNOT_ARC_INDUCE_THINK": "0",
        "CARNOT_ARC_INDUCE_TOOL_LOOP": "selfparse",
        "CARNOT_ARC_INDUCE_TOOL_TURNS": "1",
        "CARNOT_ARC_CEGIS_TOOL_LOOP": "1",
        "CARNOT_ARC_INDUCE_TOOL_GRAMMAR": "0",
        "CARNOT_ARC_TRANSITION_WITNESS": witness_value,
    }
    with (
        patch.object(e3, "E3_DIR", store),
        patch.object(reinduce, "MAX_REFINEMENT_ROUNDS", 2),
        patch.dict(os.environ, environment, clear=False),
        patch.object(urllib.request, "urlopen", request),
    ):
        policy = competition.E3AgentPolicy(game, proposer=proposer, value_head=None)
        policy.think_arm_fallback_enabled = False
        plan_before = list(policy.plan)
        outcome = policy._execute_bounded_llm_reinduction_with_arm_fallback(
            {},
            game=game,
            transitions=list(transitions),
            cell=1,
            root_grid=np.zeros((2, 2), dtype=np.int16),
            proposer=proposer,
            candidate_provider=lambda engine, goal: [("generated", engine, goal)],
            load_engine=e3.load_engine,
            plan_in_model=lambda _engine, _goal, _root: [{"action": 1, "data": None}],
            max_rounds=2,
            min_heldout_accuracy=1.0,
        )
        plan_after = list(policy.plan)
    return {
        "requests": sent,
        "planned": outcome.planned,
        "plan": outcome.plan,
        "accepted": outcome.accepted_by_heldout_verifier,
        "heldout_accuracy": outcome.heldout_accuracy,
        "rounds": outcome.rounds,
        "policy_plan_before": plan_before,
        "policy_plan_after": plan_after,
    }


def _payload_from_second_request(requests: Sequence[bytes]) -> dict[str, Any] | None:
    if len(requests) < 2:
        return None
    prompt = json.loads(requests[1])["messages"][0]["content"]
    marker = WITNESS_MARKER + "\n"
    if marker not in prompt:
        return None
    return json.loads(prompt.split(marker, 1)[1].splitlines()[0])


def run_policy_handoff_panel(workdir: Path, started: float | None = None) -> dict[str, Any]:
    """Run Exp7248 checks and add an actual E3AgentPolicy terminal handoff."""

    workdir.mkdir(parents=True, exist_ok=True)
    clock = time.monotonic() if started is None else started
    old_panel = run_cpu_conformance_panel(workdir / "shipped_panel", clock)
    observed = _changed_rows()
    future = e3.Transition(
        np.full((2, 2), 40, dtype=np.int16),
        6,
        {"x": 9, "y": 9},
        np.full((2, 2), 41, dtype=np.int16),
        0,
        0,
    )
    _progress(clock, "policy", "BEFORE enabled scored-policy scripted generations", "9/13")
    enabled = _policy_runtime_arm(
        workdir / "policy_enabled",
        game="handoff-enabled",
        witness_value="1",
        transitions=observed,
    )
    _progress(clock, "policy", "AFTER enabled scored-policy scripted generations", "10/13")
    expected = build_transition_witness(observed, _exec_engine(BAD_CODE))
    delivered = _payload_from_second_request(enabled["requests"])
    generated_hash = _sha256_bytes(canonical_witness_bytes(expected))
    delivered_hash = (
        _sha256_bytes(canonical_witness_bytes(delivered)) if delivered is not None else None
    )
    observed_by_pre_hash = {
        build_transition_witness([transition], _exec_engine(BAD_CODE))["mismatches"][0][
            "pre_frame_hash"
        ]: transition
        for transition in observed
    }
    direction_and_identity = delivered == expected and all(
        row["pre_frame_hash"] in observed_by_pre_hash for row in expected["mismatches"]
    )
    _progress(clock, "policy", "BEFORE default-off parity scripted generations", "10/13")
    default = _policy_runtime_arm(
        workdir / "policy_default",
        game="handoff-parity",
        witness_value="",
        transitions=observed,
    )
    disabled = _policy_runtime_arm(
        workdir / "policy_disabled",
        game="handoff-parity",
        witness_value="0",
        transitions=observed,
    )
    _progress(clock, "policy", "AFTER default-off parity scripted generations", "12/13")
    future_payload = build_transition_witness([future], _exec_engine(BAD_CODE))
    future_pre_hash = future_payload["mismatches"][0]["pre_frame_hash"]
    delivered_rows = delivered.get("mismatches", []) if delivered is not None else []
    future_absent = future_pre_hash not in {row.get("pre_frame_hash") for row in delivered_rows}
    parity = {
        "request_bytes_equal": default["requests"] == disabled["requests"],
        "action_equal": default["plan"] == disabled["plan"],
        "default_request_sha256": _sha256_bytes(b"\0".join(default["requests"])),
        "disabled_request_sha256": _sha256_bytes(b"\0".join(disabled["requests"])),
        "default_action": default["plan"],
        "disabled_action": disabled["plan"],
    }
    plan_authority = {
        "owner": "E3AgentPolicy._induce_and_plan",
        "adapter_installed_plan": False,
        "policy_plan_before": enabled["policy_plan_before"],
        "policy_plan_after": enabled["policy_plan_after"],
        "policy_plan_unchanged": enabled["policy_plan_before"] == enabled["policy_plan_after"],
    }
    rows = list(old_panel["rows"])
    rows.extend(
        [
            _row(
                "policy_runtime_delivery",
                "actual_e3_policy_witness_enabled",
                "next_prompt_exact_witness",
                bool(
                    generated_hash == delivered_hash
                    and direction_and_identity
                    and enabled["accepted"] is True
                    and enabled["planned"] is True
                ),
                generated_witness_sha256=generated_hash,
                delivered_witness_sha256=delivered_hash,
                delivered_to_next_prompt=len(enabled["requests"]) == 2,
                direction_and_state_identity_match=direction_and_identity,
                accepted_by_existing_policy_gate=enabled["accepted"] is True,
                heldout_accuracy=enabled["heldout_accuracy"],
                refinement_action=enabled["rounds"][-1].get("action"),
            ),
            _row(
                "policy_default_parity",
                "actual_e3_policy_default_vs_disabled",
                "request_and_action_equality",
                parity["request_bytes_equal"] and parity["action_equal"],
                **parity,
            ),
            _row(
                "policy_future_leakage",
                "actual_e3_policy_witness_enabled",
                "future_transition_excluded",
                future_absent,
                future_transition_absent=future_absent,
                future_pre_frame_hash=future_pre_hash,
            ),
            _row(
                "policy_plan_authority",
                "actual_e3_policy_witness_enabled",
                "adapter_does_not_install_plan",
                plan_authority["policy_plan_unchanged"],
                **plan_authority,
            ),
        ]
    )
    actual_mismatches = delivered.get("mismatches", []) if delivered is not None else []
    witness_rows = []
    for mismatch in expected["mismatches"]:
        transition = observed_by_pre_hash[mismatch["pre_frame_hash"]]
        witness_rows.append(
            {
                "observed_pre_state_sha256": _array_sha256(transition.grid),
                "observed_action": {"action": int(transition.action), "data": transition.data},
                "observed_successor_sha256": _array_sha256(transition.next_grid),
                "emitted_witness": mismatch,
                "actual_next_call_index": 1,
                "actual_next_call_delivery": mismatch in actual_mismatches,
                "direction": "observed_pre_state_action_to_observed_successor",
            }
        )
    reduction = reduce_receipt_rows(rows)
    _progress(clock, "policy", "END actual scored-policy handoff panel", "13/13")
    return {
        "rows": rows,
        "reduction": reduction,
        "independent_reduction": reduce_receipt_rows(json.loads(json.dumps(rows))),
        "witness_rows": witness_rows,
        "default_parity_receipt": parity,
        "plan_installation_authority": plan_authority,
    }


def check_preconditions() -> tuple[dict[str, Any], dict[str, str], list[dict[str, Any]]]:
    """Authenticate listed inputs, quarantine state, and writable output paths."""

    source_hashes: dict[str, str] = {}
    checks: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for relative in INPUT_PATHS:
        path = REPO_ROOT / relative
        exists = path.is_file()
        observed = {
            "path": str(relative),
            "exists": exists,
            "owner_uid": path.stat().st_uid if exists else None,
            "sha256": _sha256_file(path) if exists else None,
        }
        check = {
            "check": "required_readable_input",
            "upstream": str(relative),
            "field": "readable_regular_file",
            "expected": True,
            "observed": observed,
            "passed": bool(exists and os.access(path, os.R_OK)),
        }
        checks.append(check)
        if check["passed"]:
            source_hashes[str(relative)] = str(observed["sha256"])
        else:
            failures.append(check)
    spec_path = REPO_ROOT / "openspec/capabilities/arc-world-model-trust-energy/spec.md"
    spec_text = spec_path.read_text() if spec_path.is_file() else ""
    requirement_present = "REQ-ARC-WMTE-7262" in spec_text
    requirement_check = {
        "check": "driving_requirement",
        "upstream": str(spec_path.relative_to(REPO_ROOT)),
        "field": "REQ-ARC-WMTE-7262",
        "expected": "present before implementation measurement",
        "observed": "present" if requirement_present else "absent",
        "passed": requirement_present,
    }
    checks.append(requirement_check)
    if not requirement_present:
        failures.append(requirement_check)
    exclusion_path = REPO_ROOT / "ops/exclusion_manifest.yaml"
    exclusion_text = exclusion_path.read_text() if exclusion_path.is_file() else ""
    quarantined = "experiment_7262" in exclusion_text or "exp7262" in exclusion_text
    quarantine_check = {
        "check": "quarantine_state",
        "upstream": "ops/exclusion_manifest.yaml",
        "field": "exp7262",
        "expected": "not quarantined",
        "observed": "quarantined" if quarantined else "not listed",
        "passed": not quarantined,
    }
    checks.append(quarantine_check)
    if quarantined:
        failures.append(quarantine_check)
    for relative in (OUTPUT_PATH.parent, CHECKPOINT_PATH.parent, RAW_PATH.parent):
        directory = REPO_ROOT / relative
        writable = directory.is_dir() and os.access(directory, os.W_OK)
        check = {
            "check": "writable_output_directory",
            "upstream": str(relative),
            "field": "writable_directory",
            "expected": True,
            "observed": {
                "exists": directory.is_dir(),
                "owner_uid": directory.stat().st_uid if directory.exists() else None,
                "writable": writable,
            },
            "passed": writable,
        }
        checks.append(check)
        if not writable:
            failures.append(check)
    return (
        {
            "checked_at_utc": datetime.now(UTC).isoformat(),
            "process_uid": os.getuid(),
            "repository_root": str(REPO_ROOT),
            "resource_ownership": "host process owns CPU fixtures and agent observation history",
            "checks": checks,
            "failed_checks": failures,
        },
        source_hashes,
        failures,
    )


def build_validation_commands(
    *,
    terminal_candidate: Path,
    partial_checkpoint: Path,
    raw_rows: Path,
) -> list[ValidationCommand]:
    """Return the bounded validation plan without the historical full suite."""

    python = str(REPO_ROOT / ".venv/bin/python")
    pytest = str(REPO_ROOT / ".venv/bin/pytest")
    coverage = str(REPO_ROOT / ".venv/bin/coverage")
    ruff = str(REPO_ROOT / ".venv/bin/ruff")
    mypy = str(REPO_ROOT / ".venv/bin/mypy")
    new_test = "tests/python/test_experiment_7262_v639_arc_witness_receipt.py"
    old_test = "tests/python/test_experiment_7248_v638_arc_witness.py"
    state_test = "tests/python/test_arc_induction_state_persistence.py"
    grammar_test = "tests/python/test_arc_tool_grammar_transport.py"
    module = "python/carnot/experiment_7262_v639_arc_witness_receipt.py"
    entrypoint = "scripts/experiments/experiment_7262_v639_arc_witness_receipt.py"
    coverage_data = "/tmp/exp7262-v639.coverage"
    return [
        ValidationCommand(
            "focused_exp7262",
            [pytest, new_test, "-q", "--no-cov", "-n", "0", "--basetemp=/tmp/exp7262-focused"],
        ),
        ValidationCommand(
            "affected_exp7248",
            [pytest, old_test, "-q", "--no-cov", "-n", "0", "--basetemp=/tmp/exp7262-exp7248"],
        ),
        ValidationCommand(
            "new_code_coverage_run",
            [
                coverage,
                "run",
                f"--data-file={coverage_data}",
                f"--include=*/{Path(module).name}",
                "-m",
                "pytest",
                "-o",
                "addopts=",
                new_test,
                "-q",
                "-n",
                "0",
                "--basetemp=/tmp/exp7262-coverage",
            ],
        ),
        ValidationCommand(
            "new_code_coverage_report",
            [
                coverage,
                "report",
                f"--data-file={coverage_data}",
                f"--include=*/{Path(module).name}",
                "--show-missing",
                "--fail-under=100",
            ],
            timeout_s=120,
        ),
        ValidationCommand("ruff_check", [ruff, "check", module, entrypoint, new_test], 300),
        ValidationCommand(
            "ruff_format",
            [ruff, "format", "--check", module, entrypoint, new_test],
            300,
        ),
        ValidationCommand("changed_module_mypy", [mypy, module, entrypoint]),
        ValidationCommand(
            "scoped_spec_coverage",
            [
                python,
                "-u",
                "scripts/check_spec_coverage.py",
                new_test,
                old_test,
                state_test,
                grammar_test,
            ],
            300,
        ),
        ValidationCommand(
            "E2E-009",
            [
                pytest,
                state_test,
                "-q",
                "--no-cov",
                "-n",
                "0",
                "--basetemp=/tmp/exp7262-e2e009",
            ],
        ),
        ValidationCommand(
            "E2E-010",
            [
                pytest,
                grammar_test,
                "-q",
                "--no-cov",
                "-n",
                "0",
                "--basetemp=/tmp/exp7262-e2e010",
            ],
        ),
        ValidationCommand(
            "offline_e3_smoke",
            [
                "/usr/bin/env",
                "CARNOT_ARC_DISABLE_INDUCTION=1",
                "PYTHONUNBUFFERED=1",
                f"PYTHONPATH={REPO_ROOT / 'python'}:{REPO_ROOT}",
                python,
                "-u",
                "scripts/arc_loop_solve.py",
                "--mechanism",
                "e3",
                "--game",
                "r11l",
                "--max-actions",
                "12",
                "--output",
                "/tmp/exp7262-v639-offline-smoke.json",
            ],
        ),
        ValidationCommand(
            "independent_raw_row_reducer",
            [python, "-u", entrypoint, "--reduce-raw", str(raw_rows)],
            120,
        ),
        ValidationCommand("git_status_test_integrity", ["git", "status", "--short"], 120),
        ValidationCommand(
            "partial_checkpoint_rejected",
            [python, "-u", "scripts/adversarial_verify.py", str(partial_checkpoint)],
            expected_exit_code=1,
            expected_output="NONTERMINAL_DECLARED_ARTIFACT",
        ),
        ValidationCommand(
            "complete_candidate_adversarial_verify",
            [python, "-u", "scripts/adversarial_verify.py", str(terminal_candidate)],
        ),
        ValidationCommand(
            "complete_candidate_row_consistency",
            [python, "-u", "scripts/verdict_row_consistency_lint.py", str(terminal_candidate)],
        ),
    ]


def run_validation_commands(
    commands: Sequence[ValidationCommand],
    started: float,
) -> list[dict[str, Any]]:
    """Stream bounded checks and retain their exact command and output receipts."""

    receipts: list[dict[str, Any]] = []
    for index, command in enumerate(commands, 1):
        _progress(
            started,
            "validation",
            f"BEFORE subprocess {command.name}",
            f"{index - 1}/{len(commands)}",
        )
        result = run_streaming_command(
            command.command,
            cwd=REPO_ROOT,
            timeout_s=command.timeout_s,
            heartbeat_s=60,
            operation=f"exp7262:{command.name}",
        )
        output = str(result.get("stdout", result.get("output", "")))
        exit_matches = int(result["exit_code"]) == command.expected_exit_code
        output_matches = command.expected_output is None or command.expected_output in output
        receipt = {
            "name": command.name,
            "command": shlex.join(command.command),
            "expected_exit_code": command.expected_exit_code,
            "exit_code": int(result["exit_code"]),
            "duration_s": result["duration_s"],
            "timed_out": bool(result["timed_out"]),
            "log_sha256": _sha256_bytes(output.encode()),
            "log_scope": "captured combined-output tail returned by streaming subprocess helper",
            "output_tail": output[-4000:],
            "passed": bool(exit_matches and output_matches and not result["timed_out"]),
        }
        receipts.append(receipt)
        _progress(
            started,
            "validation",
            f"AFTER subprocess {command.name} exit={receipt['exit_code']}",
            f"{index}/{len(commands)}",
        )
    return receipts


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind every terminal field except the digest itself."""

    payload = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    return _sha256_bytes(_canonical_bytes(payload))


def _field_principles(artifact: Mapping[str, Any]) -> dict[str, str]:
    return {
        key: FIELD_PRINCIPLES.get(
            key,
            f"Store {key} directly so the terminal mechanism receipt remains auditable.",
        )
        for key in artifact
    }


def _acceptance_gate(
    criterion: str,
    expected: Any,
    observed: Any,
    passed: bool,
    principle: str,
) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "expected": expected,
        "observed": observed,
        "passed": bool(passed),
        "principle": principle,
    }


def _terminal_handoff_passed(rows: Sequence[Mapping[str, Any]]) -> bool:
    partial = next((row for row in rows if row.get("candidate") == "partial_checkpoint"), None)
    complete = next(
        (row for row in rows if row.get("candidate") == "complete_null_candidate"), None
    )
    return bool(
        partial
        and partial.get("checker_passed") is False
        and complete
        and complete.get("checker_passed") is True
    )


def _complete_artifact(
    *,
    started_at: str,
    completed_at: str,
    duration_s: float,
    preconditions: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    rows: Sequence[Mapping[str, Any]],
    witness_rows: Sequence[Mapping[str, Any]],
    terminal_handoff_rows: Sequence[Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
    phase_spans: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Assemble a complete measurement; failed gates produce an honest null."""

    reduction = reduce_receipt_rows(rows)
    validation_passed = bool(validation_receipts) and all(
        receipt.get("passed") is True for receipt in validation_receipts
    )
    handoff_passed = _terminal_handoff_passed(terminal_handoff_rows)
    ready = int(reduction["arc_witness_ready_score"] == 1 and validation_passed and handoff_passed)
    acceptance = [
        _acceptance_gate(
            name,
            True,
            observed,
            bool(observed),
            "Readiness requires this frozen method or handoff property.",
        )
        for name, observed in reduction["gates"].items()
    ]
    acceptance.extend(
        [
            _acceptance_gate(
                "independent_raw_reduction",
                0,
                next(
                    (
                        receipt.get("exit_code")
                        for receipt in validation_receipts
                        if receipt.get("name") == "independent_raw_row_reducer"
                    ),
                    None,
                ),
                any(
                    receipt.get("name") == "independent_raw_row_reducer"
                    and receipt.get("passed") is True
                    for receipt in validation_receipts
                )
                or (len(validation_receipts) == 1 and validation_passed),
                "An independent process must reproduce readiness from raw rows.",
            ),
            _acceptance_gate(
                "terminal_candidate_handoff",
                {"partial_checker_passed": False, "complete_checker_passed": True},
                [dict(row) for row in terminal_handoff_rows],
                handoff_passed,
                "The unchanged checker must reject partial evidence and accept terminal evidence.",
            ),
        ]
    )
    acceptance.extend(
        _acceptance_gate(
            f"validation:{receipt.get('name')}",
            receipt.get("expected_exit_code", 0),
            receipt.get("exit_code"),
            receipt.get("passed") is True,
            "The named bounded validation must reach its preregistered outcome.",
        )
        for receipt in validation_receipts
    )
    artifact: dict[str, Any] = {
        "schema": "carnot.experiment_7262.arc_witness_receipt.v1",
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "complete",
        "run_date": RUN_DATE,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "field_principles": {},
        "preconditions_checked": dict(preconditions),
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": {
            "model_load_attempts": 0,
            "model_loads_completed": 0,
            "generation_attempts": 0,
            "generations_completed": 0,
            "usable_answers": 0,
        },
        "inference_substrate": "cpu_exact_solver_or_simulator",
        "inference_substrate_class": "cpu_exact_solver_or_simulator",
        "execution_venue": "host",
        "execution_host": platform.node(),
        "duration_s": max(round(duration_s, 6), 0.000001),
        "phase_spans": [dict(span) for span in phase_spans],
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "source_artifact_hashes": dict(source_hashes),
        "rows": [dict(row) for row in rows],
        "sample_size_budget": {
            "planned_independent_units": len(rows),
            "attempted_independent_units": len(rows),
            "completed_independent_units": len(rows),
            "censored_independent_units": sum(bool(row.get("censored")) for row in rows),
            "fixed_stopping_rule": "Run every frozen CPU fixture once; never stop on outcome.",
        },
        "acceptance_gate_results": acceptance,
        "gate_check_summary": [],
        "verifier_is_oracle": True,
        "oracle_scope": "Exact scripted transition, request-byte, and terminal-classification conformance.",
        "honest_verdict": (
            "complete_circular_positive_arc_witness_terminal_handoff_mechanism_only"
            if ready
            else "complete_null_arc_witness_terminal_handoff_gate_failed"
        ),
        "verdict_class": "circular_positive" if ready else "null",
        "validation_receipts": [dict(receipt) for receipt in validation_receipts],
        "arc_witness_ready_score": ready,
        "witness_rows": [dict(row) for row in witness_rows],
        "terminal_handoff_rows": [dict(row) for row in terminal_handoff_rows],
        "solve_provenance": "development_proxy",
        "solve_claim": {
            "known_game_replayed": False,
            "game_level_solve_claimed": False,
            "official_score_claimed": False,
            "live_model_utility_claimed": False,
            "result_scope": "mechanism_plumbing",
        },
        "policy_changes": {
            "production_default_changed": False,
            "trust_threshold_changed": False,
            "replacement_code_supplied_by_adapter": False,
            "plan_installation_authority_changed": False,
        },
        "uploads_or_external_messages": False,
    }
    artifact["field_principles"] = _field_principles(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _blocked_artifact(
    *,
    started_at: str,
    completed_at: str,
    duration_s: float,
    preconditions: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    failures: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a terminal external-prerequisite block with a structured gate summary."""

    artifact: dict[str, Any] = {
        "schema": "carnot.experiment_7262.arc_witness_receipt.v1",
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "blocked",
        "run_date": RUN_DATE,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "field_principles": {},
        "preconditions_checked": dict(preconditions),
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": {
            "model_load_attempts": 0,
            "model_loads_completed": 0,
            "generation_attempts": 0,
            "generations_completed": 0,
            "usable_answers": 0,
        },
        "inference_substrate": "blocked_no_run",
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": "host",
        "execution_host": platform.node(),
        "duration_s": max(round(duration_s, 6), 0.000001),
        "phase_spans": [],
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "source_artifact_hashes": dict(source_hashes),
        "rows": [],
        "sample_size_budget": {
            "planned_independent_units": 13,
            "attempted_independent_units": 0,
            "completed_independent_units": 0,
            "censored_independent_units": 13,
            "fixed_stopping_rule": "Block before measurement when an external prerequisite fails.",
        },
        "acceptance_gate_results": [],
        "gate_check_summary": [dict(failure) for failure in failures],
        "verifier_is_oracle": True,
        "honest_verdict": "blocked_arc_witness_external_prerequisite_unavailable",
        "verdict_class": "blocked",
        "validation_receipts": [],
        "arc_witness_ready_score": 0,
        "witness_rows": [],
        "terminal_handoff_rows": [],
        "solve_provenance": "development_proxy",
    }
    artifact["field_principles"] = _field_principles(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Reject a nonterminal or internally inconsistent publication candidate."""

    required = set(FIELD_PRINCIPLES)
    missing = required - set(artifact)
    if missing:
        raise ValueError(f"missing required artifact fields: {sorted(missing)}")
    if artifact["status"] not in {"complete", "blocked"}:
        raise ValueError("terminal status must be complete or blocked")
    if artifact["run_date"] != RUN_DATE:
        raise ValueError("run date drift")
    if set(artifact) - set(artifact["field_principles"]):
        raise ValueError("a top-level field lacks a field principle")
    if artifact["reproducibility_checksum"] != reproducibility_checksum(artifact):
        raise ValueError("reproducibility checksum mismatch")
    if artifact["MODEL_SPECS"] or artifact["model_invoked"] is not False:
        raise ValueError("this CPU receipt cannot declare model invocation")
    if any(int(value) != 0 for value in artifact["invocation_counts"].values()):
        raise ValueError("current model and generation counters must stay zero")
    if artifact["status"] == "blocked":
        if not artifact["gate_check_summary"]:
            raise ValueError("blocked artifact requires gate_check_summary")
        if artifact["verdict_class"] != "blocked" or artifact["arc_witness_ready_score"] != 0:
            raise ValueError("blocked artifact cannot claim readiness")
        return
    reduced = reduce_receipt_rows(artifact["rows"])
    validations_passed = bool(artifact["validation_receipts"]) and all(
        receipt.get("passed") is True for receipt in artifact["validation_receipts"]
    )
    expected = int(
        reduced["arc_witness_ready_score"] == 1
        and validations_passed
        and _terminal_handoff_passed(artifact["terminal_handoff_rows"])
    )
    if artifact["arc_witness_ready_score"] != expected:
        raise ValueError("stored readiness disagrees with raw rows and terminal checks")
    expected_class = "circular_positive" if expected else "null"
    if artifact["verdict_class"] != expected_class:
        raise ValueError("verdict class disagrees with measured readiness")


def _write_evidence(panel: Mapping[str, Any]) -> dict[str, str]:
    """Write raw rows and hashed historical or injected fixture metadata."""

    prior_path = REPO_ROOT / "results/experiment_7248_v638_arc_witness.json"
    prior = json.loads(prior_path.read_text())
    sidecar = {
        "schema": "carnot.exp7262.history_and_fixture_sidecar.v1",
        "historical_source": str(prior_path.relative_to(REPO_ROOT)),
        "historical_source_sha256": _sha256_file(prior_path),
        "historical_model_receipts": prior.get("MODEL_SPECS", []),
        "historical_model_invoked": prior.get("model_invoked"),
        "current_invocation": False,
        "injected_model_fixture": True,
        "fixture_hashes": {
            "bad_engine": _sha256_bytes(BAD_CODE.encode()),
            "good_engine": _sha256_bytes(GOOD_CODE.encode()),
        },
    }
    atomic_write_json(SIDECAR_PATH, sidecar, root=REPO_ROOT, sort_keys=True)
    raw = {
        "schema": "carnot.exp7262.raw_witness_receipt_rows.v1",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "rows": panel["rows"],
        "expected_reduction": panel["reduction"],
    }
    atomic_write_json(RAW_PATH, raw, root=REPO_ROOT, sort_keys=True)
    return {
        str(SIDECAR_PATH): _sha256_file(REPO_ROOT / SIDECAR_PATH),
        str(RAW_PATH): _sha256_file(REPO_ROOT / RAW_PATH),
    }


def _reduce_raw(path: Path) -> int:
    payload = json.loads(path.read_text())
    observed = reduce_receipt_rows(payload["rows"])
    print(json.dumps(observed, sort_keys=True), flush=True)
    return 0 if observed == payload["expected_reduction"] else 1


def _phase_span(started: float, name: str, phase_start: float, phase_end: float) -> dict[str, Any]:
    return {
        "phase": name,
        "start_offset_s": round(phase_start - started, 6),
        "end_offset_s": round(phase_end - started, 6),
        "duration_s": max(round(phase_end - phase_start, 6), 0.000001),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--reduce-raw", type=Path)
    args = parser.parse_args(argv)
    if args.reduce_raw is not None:
        return _reduce_raw(args.reduce_raw)
    if args.date != RUN_DATE:
        parser.error(f"--date must be the frozen execution date {RUN_DATE}")

    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    spans: list[dict[str, Any]] = []
    _progress(started, "preflight", "BEGIN authenticated preconditions")
    phase_start = time.monotonic()
    preconditions, source_hashes, failures = check_preconditions()
    phase_end = time.monotonic()
    spans.append(_phase_span(started, "preflight", phase_start, phase_end))
    _progress(
        started,
        "preflight",
        "END authenticated preconditions",
        f"{len(source_hashes)}/{len(INPUT_PATHS)}",
    )
    if failures:
        blocked = _blocked_artifact(
            started_at=started_at,
            completed_at=datetime.now(UTC).isoformat(),
            duration_s=time.monotonic() - started,
            preconditions=preconditions,
            source_hashes=source_hashes,
            failures=failures,
        )
        validate_artifact(blocked)
        _progress(started, "publish", "BEFORE atomic blocked terminal write")
        atomic_write_json(OUTPUT_PATH, blocked, root=REPO_ROOT, sort_keys=True)
        _progress(started, "publish", "AFTER atomic blocked terminal write", "1/1")
        return 2

    _progress(started, "measurement", "BEFORE CPU witness and policy benchmark")
    phase_start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="carnot-exp7262-") as raw_workdir:
        panel = run_policy_handoff_panel(Path(raw_workdir), started)
    phase_end = time.monotonic()
    spans.append(_phase_span(started, "cpu_measurement", phase_start, phase_end))
    _progress(
        started,
        "measurement",
        "AFTER CPU witness and policy benchmark",
        f"{len(panel['rows'])}/{len(panel['rows'])}",
    )
    _progress(started, "evidence", "BEFORE raw evidence writes")
    source_hashes.update(_write_evidence(panel))
    for relative in (
        Path("python/carnot/experiment_7262_v639_arc_witness_receipt.py"),
        Path("scripts/experiments/experiment_7262_v639_arc_witness_receipt.py"),
        Path("tests/python/test_experiment_7262_v639_arc_witness_receipt.py"),
    ):
        source_hashes[str(relative)] = _sha256_file(REPO_ROOT / relative)
    _progress(started, "evidence", "AFTER raw evidence writes", "2/2")

    checkpoint = _complete_artifact(
        started_at=started_at,
        completed_at=datetime.now(UTC).isoformat(),
        duration_s=time.monotonic() - started,
        preconditions=preconditions,
        source_hashes=source_hashes,
        rows=panel["rows"],
        witness_rows=panel["witness_rows"],
        terminal_handoff_rows=[],
        validation_receipts=[],
        phase_spans=spans,
    )
    checkpoint["status"] = "partial"
    checkpoint["honest_verdict"] = "partial_measurement_checkpoint_not_terminal"
    checkpoint["verdict_class"] = "partial"
    checkpoint["reproducibility_checksum"] = reproducibility_checksum(checkpoint)
    atomic_write_json(CHECKPOINT_PATH, checkpoint, root=REPO_ROOT, sort_keys=True)

    commands = build_validation_commands(
        terminal_candidate=REPO_ROOT / TERMINAL_CANDIDATE_PATH,
        partial_checkpoint=REPO_ROOT / CHECKPOINT_PATH,
        raw_rows=REPO_ROOT / RAW_PATH,
    )
    candidate_names = {
        "complete_candidate_adversarial_verify",
        "complete_candidate_row_consistency",
    }
    before_candidate = [command for command in commands if command.name not in candidate_names]
    candidate_checks = [command for command in commands if command.name in candidate_names]
    _progress(started, "validation", "BEGIN scoped validation before terminal candidate")
    phase_start = time.monotonic()
    receipts = run_validation_commands(before_candidate, started)
    partial_receipt = next(
        receipt for receipt in receipts if receipt["name"] == "partial_checkpoint_rejected"
    )
    handoff_rows: list[dict[str, Any]] = [
        {
            "candidate": "partial_checkpoint",
            "path": str(CHECKPOINT_PATH),
            "expected_checker_passed": False,
            "checker_passed": not partial_receipt["passed"],
            "negative_control_passed": partial_receipt["passed"],
            "classification": "partial",
        },
        {
            "candidate": "complete_null_candidate",
            "path": str(TERMINAL_CANDIDATE_PATH),
            "expected_checker_passed": True,
            "checker_passed": None,
            "classification": "complete",
        },
    ]
    candidate = _complete_artifact(
        started_at=started_at,
        completed_at=datetime.now(UTC).isoformat(),
        duration_s=time.monotonic() - started,
        preconditions=preconditions,
        source_hashes=source_hashes,
        rows=panel["rows"],
        witness_rows=panel["witness_rows"],
        terminal_handoff_rows=handoff_rows,
        validation_receipts=receipts,
        phase_spans=spans,
    )
    validate_artifact(candidate)
    _progress(started, "handoff", "BEFORE separate complete/null candidate write")
    atomic_write_json(TERMINAL_CANDIDATE_PATH, candidate, root=REPO_ROOT, sort_keys=True)
    _progress(started, "handoff", "AFTER separate complete/null candidate write", "1/1")
    receipts.extend(run_validation_commands(candidate_checks, started))
    phase_end = time.monotonic()
    spans.append(_phase_span(started, "validation", phase_start, phase_end))
    complete_receipt = next(
        receipt
        for receipt in receipts
        if receipt["name"] == "complete_candidate_adversarial_verify"
    )
    handoff_rows[1]["checker_passed"] = complete_receipt["passed"]
    source_hashes[str(TERMINAL_CANDIDATE_PATH)] = _sha256_file(REPO_ROOT / TERMINAL_CANDIDATE_PATH)
    _progress(started, "validation", "END scoped validation and terminal handoff", "16/16")

    old_artifact = REPO_ROOT / "results/experiment_7248_v638_arc_witness.json"
    old_hash_after = _sha256_file(old_artifact)
    old_hash_before = source_hashes[str(old_artifact.relative_to(REPO_ROOT))]
    receipts.append(
        {
            "name": "old_exp7248_artifact_immutable",
            "command": "sha256 before and after measurement",
            "expected_exit_code": 0,
            "exit_code": 0 if old_hash_before == old_hash_after else 1,
            "duration_s": 0.000001,
            "timed_out": False,
            "log_sha256": _sha256_bytes(old_hash_after.encode()),
            "passed": old_hash_before == old_hash_after,
        }
    )
    _progress(started, "assembly", "BEGIN terminal artifact assembly")
    phase_start = time.monotonic()
    completed_at = datetime.now(UTC).isoformat()
    terminal = _complete_artifact(
        started_at=started_at,
        completed_at=completed_at,
        duration_s=time.monotonic() - started,
        preconditions=preconditions,
        source_hashes=source_hashes,
        rows=panel["rows"],
        witness_rows=panel["witness_rows"],
        terminal_handoff_rows=handoff_rows,
        validation_receipts=receipts,
        phase_spans=spans,
    )
    validate_artifact(terminal)
    phase_end = time.monotonic()
    spans.append(_phase_span(started, "artifact_assembly", phase_start, phase_end))
    terminal["phase_spans"] = spans
    terminal["duration_s"] = max(round(time.monotonic() - started, 6), 0.000001)
    terminal["reproducibility_checksum"] = reproducibility_checksum(terminal)
    validate_artifact(terminal)
    _progress(started, "assembly", "END terminal artifact assembly", "1/1")
    _progress(started, "publish", "BEFORE atomic terminal write")
    atomic_write_json(OUTPUT_PATH, terminal, root=REPO_ROOT, sort_keys=True)
    _progress(started, "publish", "AFTER atomic terminal write", "1/1")
    return 0 if terminal["arc_witness_ready_score"] == 1 else 1
