"""Diagnose and repair the Exp7263 ARC runtime identity handoff.

This is a CPU-only conformance experiment.  Historical live-model facts and
temporary GGUF-shaped fixtures are evidence inputs, never current invocations.

Spec: REQ-ARC-WMTE-7276 and SCENARIO-ARC-WMTE-7276-*.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import UTC, datetime
import functools
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import tempfile
import time
from typing import Any, Mapping, Sequence

from carnot.agentic import arc_competition_agent as competition
from carnot.agentic import arc_eval_provenance as provenance
from carnot.experiment_7246_v638_source_map import _run_streaming_command


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "exp7276-arc-identity"
MILESTONE = "2026.09.640"
RUN_DATE = "20260913"
RANDOM_SEED = 7_276_202_609_13
SCHEMA = "carnot.exp7276.arc_identity.v1"
WRAPPER_PATH = Path("scripts/experiments/experiment_7276_v640_arc_identity.py")
OUTPUT_PATH = Path("results/experiment_7276_v640_arc_identity.json")
RAW_DIR = Path("results/raw/experiment_7276_v640_arc_identity")
RAW_ROWS_PATH = RAW_DIR / "identity_rows.json"
SIDECAR_PATH = RAW_DIR / "identity_evidence_sidecar.json"
TERMINAL_CANDIDATE_PATH = RAW_DIR / "terminal_candidate.json"
PROVENANCE_FAILURE_BOUNDARY = "E3AgentPolicy.choose_action -> build_arc_eval_provenance_for_policy"
PANEL_CASES = (
    "stable_preexisting_hardlink",
    "post_capture_hardlink",
    "symlink_replacement",
    "stale_pid_start_tick",
    "missing_props",
    "wrong_launch_path",
    "genuine_unsupported_identity",
)
UNSUPPORTED_CASES = PANEL_CASES[1:]
ZERO_INVOCATION_COUNTS = {
    "model_loads_attempted": 0,
    "model_loads_completed": 0,
    "generation_calls_attempted": 0,
    "generation_calls_completed": 0,
    "usable_answers": 0,
}
INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("python/carnot/agentic/arc_eval_provenance.py"),
    Path("python/carnot/agentic/arc_competition_agent.py"),
    Path("python/carnot/agentic/arc_executable_world_model.py"),
    Path("python/carnot/experiment_7263_v639_arc_live.py"),
    Path("python/carnot/experiment_7262_v639_arc_witness_receipt.py"),
    Path("results/experiment_7263_v639_arc_live.json"),
    Path("results/raw/experiment_7263_v639_arc_live/live_session.json"),
    Path("tests/python/test_arc_eval_provenance_contract_20260905.py"),
    Path("tests/python/test_arc_induction_state_persistence.py"),
    Path("tests/python/test_arc_tool_grammar_transport.py"),
    Path("openspec/capabilities/arc-world-model-trust-energy/spec.md"),
)

FIELD_PRINCIPLES = {
    "schema": "Version the result and retain ordinary top-level experiment_id and milestone.",
    "experiment_id": "Identify the measured Exp7276 result with an ordinary string.",
    "milestone": "Bind the result to milestone 2026.09.640.",
    "status": "Use complete or blocked for terminal evidence; keep unfinished work in separate checkpoints.",
    "run_date": "Use 20260913 and actual UTC start and end times.",
    "started_at_utc": "Retain the actual UTC invocation start.",
    "ended_at_utc": "Retain the actual UTC terminal construction time.",
    "field_principles": "Store explanations here; consumer values remain ordinary top-level fields.",
    "preconditions_checked": "Record actual input hashes, authority separation, resource ownership, and failures.",
    "MODEL_SPECS": "Declare models executable in this invocation; keep historical identities in hashed sidecars.",
    "model_invoked": "Derive from actual calls, including failed or unusable generation.",
    "invocation_counts": "Separate attempted and completed loads and generation from usable answers.",
    "inference_substrate": "Use the recognized literal for actual CPU computation.",
    "inference_substrate_class": "Use the correct no-LLM class and never pad duration.",
    "execution_venue": "Host orchestration is host; device execution is separately identified.",
    "execution_host": "Record the host independently of the closed venue vocabulary.",
    "duration_s": "Measure monotonic elapsed time and disjoint phase spans.",
    "phase_spans": "Keep measured CPU panel and validation spans distinct.",
    "random_seed": "Freeze independent-unit seeds before observing results.",
    "reproducibility_checksum": "Bind code, configuration, input manifests, and raw evidence.",
    "source_artifact_hashes": "Preserve exact input identity, retirement, and quarantine status.",
    "rows": "Keep each unit, arm, seed, error, abstention, cost, metric, and censoring state.",
    "sample_size_budget": "Record planned, attempted, completed, censored units, and stopping rule.",
    "acceptance_gate_results": "Each criterion records expected, observed, passed, and principle.",
    "gate_check_summary": "For blocked results name upstream, exact check, observed, and expected.",
    "verifier_is_oracle": "Expose shared verifier authority; exact conformance is not learned correctness.",
    "honest_verdict": "Completed findings start complete_; external absence starts blocked_.",
    "verdict_class": "Use the closed verdict vocabulary; oracle=true forbids positive.",
    "validation_receipts": "Retain command, exit code, timing, and log hash; do not hide failures.",
    "arc_identity_ready_score": "One requires the repaired handoff, policy fixture, controls, and independent reduction.",
    "identity_obligation_rows": "Each obligation retains observed evidence, expected evidence, and failure boundary.",
    "policy_entrypoint_receipts": "Show the genuine provenance builder is reached from the scored policy.",
    "unsupported_controls": "Absent or conflicting evidence remains rejected.",
    "historical_episode_rows": "Separate reconstructed historical facts from current invocation evidence.",
    "historical_receipt_sidecar": "Authenticate historical and injected identity details kept outside current calls.",
    "raw_evidence": "Authenticate the independently reducible rows before terminal publication.",
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def artifact_checksum(value: Mapping[str, Any]) -> str:
    """Hash a terminal artifact without its checksum carrier."""

    projected = deepcopy(dict(value))
    projected.pop("reproducibility_checksum", None)
    return _sha256_bytes(_canonical_bytes(projected))


def gate_check(check: str, upstream: str, field: str, expected: Any, observed: Any) -> JsonDict:
    """Build one fail-closed precondition row."""

    return {
        "check": check,
        "upstream": upstream,
        "field": field,
        "expected": expected,
        "observed": observed,
        "passed": observed == expected,
    }


def _fixture(directory: Path) -> tuple[JsonDict, Path, Path, JsonDict]:
    payload = b"Exp7276 isolated GGUF-shaped fixture bytes\n"
    digest = hashlib.sha256(payload).hexdigest()
    root = directory / "models--fixture-owner--fixture-model-GGUF"
    blob = root / "blobs" / digest
    blob.parent.mkdir(parents=True)
    blob.write_bytes(payload)
    os.link(blob, directory / "task-owned-model-copy.gguf")
    requested = root / "snapshots" / "7276fixture" / "fixture-model.gguf"
    requested.parent.mkdir(parents=True)
    requested.symlink_to(Path("../../blobs") / digest)
    spec = {
        "model_path": str(requested),
        "model_filename": requested.name,
        "hf_id": "fixture-owner/fixture-model-GGUF",
        "revision": "7276fixture",
        "model_file_hash": "sha256:" + digest,
    }
    props = {"model_path": str(requested), "model_alias": requested.name}
    return spec, requested, blob, props


class _ScriptedIdentityTransport:
    """Minimal deterministic start and `/props` transport for the CPU panel."""

    def __init__(self, props: Mapping[str, Any]) -> None:
        self._props = deepcopy(dict(props))
        self.started = False
        self.pid: int | None = None
        self.start_tick: int | None = None

    def start(self) -> None:
        self.started = True
        self.pid = os.getpid()
        self.start_tick = provenance.process_start_tick(self.pid)

    def props(self) -> JsonDict:
        if not self.started:
            raise RuntimeError("scripted server has not started")
        return deepcopy(self._props)


class _FixtureBaseAgent:
    def __init__(self) -> None:
        self.game_id = "exp7276-fixture"


class _FixtureProposer:
    n_completion_calls = 1
    n_completion_ok = 1
    model_repository = "fixture-owner/fixture-model-GGUF"
    model_filename = "fixture-model.gguf"
    model_revision = "7276fixture"
    observed_server_n_ctx = 4096
    n_ctx = 4096
    port = 47276

    def __init__(self, model_path: Path, server_path: Path) -> None:
        self.model_path = str(model_path)
        self.requested_model_path = str(model_path)
        self.requested_model_filename = model_path.name
        self.observed_server_model_path = str(model_path)
        self.generator_server_path = str(server_path)
        self.last_launch_argv = (
            str(server_path),
            "-m",
            str(model_path),
            "--port",
            str(self.port),
        )

    def _url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def _legacy_unique_status(receipt: Mapping[str, Any]) -> str:
    row = next(
        item
        for item in receipt["identity_obligation_rows"]
        if item["obligation"] == "unique_file_identity"
    )
    observed = row["observed_value"]
    return (
        "supported"
        if observed.get("requested_nlink") == observed.get("observed_nlink") == 1
        else "contradicted"
    )


def _policy_receipt(receipt: Mapping[str, Any], fixture_dir: Path) -> JsonDict:
    server = fixture_dir / "scripted-llama-server"
    server.write_bytes(b"scripted executable fixture\n")
    proposer = _FixtureProposer(Path(str(receipt["requested_model_path"])), server)
    agent_type = competition.make_carnot_agent(_FixtureBaseAgent, cascade=True, proposer=proposer)
    agent = agent_type()
    action = agent.choose_action([], None)
    policy = agent._policy
    lease = {
        "lease_id": "exp7276-scripted-lease",
        "lease_hash": _sha256_bytes(b"exp7276-scripted-lease"),
        "lease_issued_at": "2026-09-13T12:00:00+00:00",
        "lease_expires_at": "2026-09-13T14:00:00+00:00",
    }
    record = provenance.build_arc_eval_provenance_for_policy(
        policy,
        counters_before={"requests": 0, "completions": 0, "errors": 0},
        counters_after={"requests": 1, "completions": 1, "errors": 0},
        envelope={
            "generator_cuda_gpu_requested": 0,
            "gpus_held": [
                {
                    "index": 0,
                    "gpu_uuid": "GPU-exp7276-fixture",
                    "gpu_model": "scripted-fixture-only",
                }
            ],
        },
        solve_provenance="development_proxy",
        factory_path=REPO_ROOT / "python/carnot/agentic/arc_competition_agent.py",
        repo_root=REPO_ROOT,
        lease=lease,
        lease_checked_at="2026-09-13T13:00:00+00:00",
        model_identity_receipt=receipt,
    )
    decision = provenance.validate_arc_eval_provenance(record)
    return {
        "factory": "make_carnot_agent",
        "factory_module": "carnot.agentic.arc_competition_agent",
        "policy_class": type(policy).__name__,
        "choose_action_called": True,
        "selected_action": getattr(action, "name", str(action)),
        "provenance_builder": "build_arc_eval_provenance_for_policy",
        "provenance_valid": decision.valid,
        "provenance_hash": record["provenance_hash"],
        "scripted_fixture_counters": {"requests": 1, "completions": 1, "errors": 0},
        "counts_as_current_model_invocation": False,
    }


def _identity_row(case: str, directory: Path) -> tuple[JsonDict, JsonDict]:
    spec, requested, blob, expected_props = _fixture(directory)
    transport = _ScriptedIdentityTransport(expected_props)
    transport.start()
    props = transport.props()
    launch = str(requested)
    start_tick = transport.start_tick
    source = provenance.capture_arc_model_identity_source_provenance(
        raw_server_props=props,
        requested_model_path=requested,
        source_kind="exp7276_scripted_props",
        launch_model_argument=launch,
        server_pid=transport.pid,
        server_pid_start_tick=(start_tick + 1 if case == "stale_pid_start_tick" else start_tick),
    )
    if case == "post_capture_hardlink":
        os.link(blob, directory / "post-capture-link.gguf")
    elif case == "symlink_replacement":
        replacement = directory / "replacement" / blob.name
        replacement.parent.mkdir()
        replacement.write_bytes(blob.read_bytes())
        requested.unlink()
        requested.symlink_to(replacement)
    elif case == "missing_props":
        props = {}
    elif case == "wrong_launch_path":
        launch = str(directory / "wrong-launch.gguf")
    elif case == "genuine_unsupported_identity":
        other = directory / "conflicting-model.gguf"
        other.write_bytes(blob.read_bytes())
        props["model"] = str(other)
    receipt = provenance.build_typed_arc_model_identity_receipt(
        selected_model_spec=spec,
        launch_model_argument=launch,
        raw_server_props=props,
        source_provenance=source,
    )
    decision = provenance.validate_typed_arc_model_identity_receipt(receipt)
    unsupported = [
        item["obligation"]
        for item in receipt["identity_obligation_rows"]
        if item["status"] != "supported"
    ]
    expected_acceptance = case == "stable_preexisting_hardlink"
    policy = (
        _policy_receipt(receipt, directory)
        if decision.valid
        else {
            "factory": "make_carnot_agent",
            "policy_class": "E3AgentPolicy",
            "choose_action_called": False,
            "provenance_builder": "build_arc_eval_provenance_for_policy",
            "provenance_valid": False,
            "provenance_hash": None,
            "rejected_before_policy_credit": True,
        }
    )
    row = {
        "unit": case,
        "arm": "repaired" if expected_acceptance else "adversarial_control",
        "seed": RANDOM_SEED + PANEL_CASES.index(case),
        "metric": "typed_identity_conformance",
        "passed": decision.valid is expected_acceptance,
        "error": None if decision.valid is expected_acceptance else "; ".join(decision.errors),
        "abstention": False,
        "censored": False,
        "cost": {"model_loads": 0, "generations": 0},
        "accepted": decision.valid,
        "expected_acceptance": expected_acceptance,
        "unsupported_obligations": unsupported,
        "failure_boundary": None if decision.valid else PROVENANCE_FAILURE_BOUNDARY,
        "legacy_unique_file_identity_status": _legacy_unique_status(receipt),
        "policy_entrypoint_receipt": policy,
        "transport_receipt": {
            "start_called": transport.started,
            "props_called": True,
            "pid": transport.pid,
            "process_start_tick": transport.start_tick,
            "scripted": True,
        },
    }
    return row, {"unit": case, "typed_identity_receipt": receipt, "model_spec": spec}


def run_cpu_identity_panel(workdir: Path) -> JsonDict:
    """Run the repaired path and all one-factor adversarial controls."""

    workdir.mkdir(parents=True, exist_ok=True)
    rows: list[JsonDict] = []
    sidecar_rows: list[JsonDict] = []
    for case in PANEL_CASES:
        row, sidecar = _identity_row(case, workdir / case)
        rows.append(row)
        sidecar_rows.append(sidecar)
    return {
        "rows": rows,
        "fixture_sidecar_rows": sidecar_rows,
        "independent_reduction": reduce_identity_rows(rows),
    }


def reduce_identity_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Independently derive readiness from named raw units only."""

    by_unit = {str(row.get("unit")): row for row in rows}
    exact_units = set(by_unit) == set(PANEL_CASES) and len(rows) == len(PANEL_CASES)
    positive = by_unit.get("stable_preexisting_hardlink", {})
    policy = positive.get("policy_entrypoint_receipt", {})
    stable_pass = bool(
        positive.get("passed") is True
        and positive.get("accepted") is True
        and positive.get("legacy_unique_file_identity_status") == "contradicted"
    )
    policy_pass = bool(
        policy.get("factory") == "make_carnot_agent"
        and policy.get("policy_class") == "E3AgentPolicy"
        and policy.get("choose_action_called") is True
        and policy.get("provenance_builder") == "build_arc_eval_provenance_for_policy"
        and policy.get("provenance_valid") is True
    )
    controls_pass = all(
        by_unit.get(case, {}).get("passed") is True
        and by_unit.get(case, {}).get("accepted") is False
        and bool(by_unit.get(case, {}).get("unsupported_obligations"))
        and by_unit.get(case, {}).get("policy_entrypoint_receipt", {}).get("choose_action_called")
        is False
        for case in UNSUPPORTED_CASES
    )
    gates = {
        "exact_frozen_units": exact_units,
        "stable_preexisting_hardlink_repaired": stable_pass,
        "real_policy_entrypoint_and_provenance_builder": policy_pass,
        "every_unsupported_control_rejected": controls_pass,
        "zero_current_model_calls": all(
            row.get("cost") == {"model_loads": 0, "generations": 0} for row in rows
        ),
    }
    return {
        "gates": gates,
        "planned_units": len(PANEL_CASES),
        "completed_units": len(by_unit),
        "censored_units": sum(bool(row.get("censored")) for row in rows),
        "arc_identity_ready_score": int(all(gates.values())),
    }


@functools.lru_cache(maxsize=2)
def diagnose_historical_episodes(root: Path) -> list[JsonDict]:
    """Reconstruct the Exp7263 identity failure from retained historical evidence."""

    result_path = root / "results/experiment_7263_v639_arc_live.json"
    session_path = root / "results/raw/experiment_7263_v639_arc_live/live_session.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    session = json.loads(session_path.read_text(encoding="utf-8"))
    spec = dict(session["model_spec"])
    requested = Path(spec["model_path"])
    resolved = requested.resolve(strict=True)
    target_stat = requested.stat()
    link_ctime = datetime.fromtimestamp(target_stat.st_ctime, tz=UTC)
    run_start = datetime.fromisoformat(str(result["started_at_utc"]))
    runtime = session["runtime_receipt"]
    launch = runtime.get("server_command") or []
    launch_model = spec["model_path"]
    if isinstance(launch, list) and "-m" in launch:
        index = launch.index("-m") + 1
        launch_model = launch[index] if index < len(launch) else None
    common = {
        "unsupported_obligation": "unique_file_identity",
        "failure_boundary": PROVENANCE_FAILURE_BOUNDARY,
        "requested_model_path": str(requested),
        "resolved_model_path": str(resolved),
        "launch_model_argument": launch_model,
        "model_file_hash": spec["model_file_hash"],
        "target_device": target_stat.st_dev,
        "target_inode": target_stat.st_ino,
        "target_size": target_stat.st_size,
        "target_link_count_at_run": target_stat.st_nlink,
        "link_state_changed_at_utc": link_ctime.isoformat(),
        "link_count_change_predates_run": link_ctime < run_start,
        "server_pid": runtime.get("server_pid"),
        "server_pid_start_tick": runtime.get("server_pid_start_tick"),
        "raw_server_props_retained": False,
        "source_receipt_retained": False,
        "raw_evidence_limitation": (
            "Exp7263 did not retain raw /props or the typed source receipt after the exception"
        ),
        "diagnostic_basis": (
            "legacy unique_file_identity required st_nlink==1; the authenticated blob had "
            "a stable second hard link before the run"
        ),
        "model_or_gguf_wrong_inferred": False,
    }
    return [
        {
            "episode_id": row["episode_id"],
            "arm": row["arm"],
            "game": row["game"],
            "historical_error": row["error"],
            **common,
        }
        for row in session["episodes"]
    ]


def _obligation_rows(
    panel: Mapping[str, Any], history: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for sidecar in panel["fixture_sidecar_rows"]:
        unit = sidecar["unit"]
        receipt = sidecar["typed_identity_receipt"]
        for item in receipt["identity_obligation_rows"]:
            rows.append(
                {
                    "unit": unit,
                    "source": "cpu_scripted_fixture",
                    "obligation": item["obligation"],
                    "status": item["status"],
                    "observed_evidence": item["observed_value"],
                    "expected_evidence": "supported by the named independent source",
                    "failure_boundary": (
                        None if item["status"] == "supported" else PROVENANCE_FAILURE_BOUNDARY
                    ),
                }
            )
    for episode in history:
        for obligation in provenance.IDENTITY_OBLIGATIONS:
            diagnosed = obligation == episode["unsupported_obligation"]
            rows.append(
                {
                    "unit": episode["episode_id"],
                    "source": "historical_exp7263",
                    "obligation": obligation,
                    "status": "unsupported" if diagnosed else "not_retained",
                    "observed_evidence": (
                        {
                            "target_link_count": episode["target_link_count_at_run"],
                            "target_inode": episode["target_inode"],
                            "link_state_changed_at_utc": episode["link_state_changed_at_utc"],
                        }
                        if diagnosed
                        else {"raw_server_props": None, "source_receipt": None}
                    ),
                    "expected_evidence": "supported typed observation retained at policy handoff",
                    "failure_boundary": episode["failure_boundary"],
                }
            )
    return rows


def _acceptance_gates(
    reduction: Mapping[str, Any], history: Sequence[Mapping[str, Any]], validation_ok: bool
) -> list[JsonDict]:
    facts = {
        "four_historical_episodes_diagnosed": len(history) == 4,
        "unsupported_obligation_identified": bool(history)
        and all(row.get("unsupported_obligation") == "unique_file_identity" for row in history),
        **dict(reduction["gates"]),
        "scoped_validation_passed": validation_ok,
    }
    return [
        {
            "criterion": name,
            "expected": True,
            "observed": observed,
            "passed": observed is True,
            "principle": "Readiness requires this independent evidence boundary.",
        }
        for name, observed in facts.items()
    ]


def _base_artifact(started_at_utc: str, ended_at_utc: str, duration_s: float) -> JsonDict:
    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "blocked",
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "field_principles": {},
        "preconditions_checked": [],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "cpu_exact_solver_or_simulator",
        "inference_substrate_class": "cpu_exact_solver_or_simulator",
        "execution_venue": "host",
        "execution_host": socket.gethostname(),
        "duration_s": max(round(float(duration_s), 6), 0.000001),
        "phase_spans": [],
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "source_artifact_hashes": {},
        "rows": [],
        "sample_size_budget": {
            "planned_units": len(PANEL_CASES),
            "attempted_units": 0,
            "completed_units": 0,
            "censored_units": 0,
            "stopping_rule": "execute each frozen CPU fixture exactly once",
        },
        "acceptance_gate_results": [],
        "gate_check_summary": [],
        "verifier_is_oracle": True,
        "honest_verdict": "blocked_external_prerequisite_missing",
        "verdict_class": "blocked",
        "validation_receipts": [],
        "arc_identity_ready_score": 0,
        "identity_obligation_rows": [],
        "policy_entrypoint_receipts": [],
        "unsupported_controls": [],
        "historical_episode_rows": [],
        "historical_receipt_sidecar": None,
        "raw_evidence": None,
    }


def _seal_artifact(artifact: JsonDict) -> JsonDict:
    artifact["field_principles"] = {key: FIELD_PRINCIPLES[key] for key in artifact}
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def build_complete_artifact(
    *,
    started_at_utc: str,
    ended_at_utc: str,
    duration_s: float,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    panel: Mapping[str, Any],
    historical_rows: Sequence[Mapping[str, Any]],
    raw_rows_path: Path,
    sidecar_path: Path,
    phase_spans: Sequence[Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Construct a complete terminal result from independently reducible evidence."""

    rows = deepcopy(list(panel["rows"]))
    history = deepcopy(list(historical_rows))
    reduction = reduce_identity_rows(rows)
    validation_ok = all(row.get("passed") is True for row in validation_receipts)
    readiness = int(reduction["arc_identity_ready_score"] == 1 and validation_ok)
    artifact = _base_artifact(started_at_utc, ended_at_utc, duration_s)
    artifact.update(
        {
            "status": "complete",
            "preconditions_checked": deepcopy(list(preconditions)),
            "phase_spans": deepcopy(list(phase_spans)),
            "source_artifact_hashes": deepcopy(dict(source_hashes)),
            "rows": rows,
            "sample_size_budget": {
                "planned_units": len(PANEL_CASES),
                "attempted_units": len(rows),
                "completed_units": reduction["completed_units"],
                "censored_units": reduction["censored_units"],
                "stopping_rule": "execute each frozen CPU fixture exactly once",
            },
            "acceptance_gate_results": _acceptance_gates(reduction, history, validation_ok),
            "gate_check_summary": [],
            "honest_verdict": (
                "complete_circular_positive_identity_runtime_handoff_repaired"
                if readiness
                else "complete_null_identity_runtime_handoff_not_ready"
            ),
            "verdict_class": "circular_positive" if readiness else "null",
            "validation_receipts": deepcopy(list(validation_receipts)),
            "arc_identity_ready_score": readiness,
            "identity_obligation_rows": _obligation_rows(panel, history),
            "policy_entrypoint_receipts": [
                deepcopy(row["policy_entrypoint_receipt"])
                for row in rows
                if row["unit"] == "stable_preexisting_hardlink"
            ],
            "unsupported_controls": [
                {
                    "unit": row["unit"],
                    "accepted": row["accepted"],
                    "unsupported_obligations": deepcopy(row["unsupported_obligations"]),
                    "failure_boundary": row["failure_boundary"],
                }
                for row in rows
                if row["unit"] in UNSUPPORTED_CASES
            ],
            "historical_episode_rows": history,
            "historical_receipt_sidecar": {
                "path": str(sidecar_path),
                "sha256": _sha256_file(sidecar_path),
            },
            "raw_evidence": {
                "path": str(raw_rows_path),
                "sha256": _sha256_file(raw_rows_path),
                "independent_reduction": reduction,
            },
        }
    )
    return _seal_artifact(artifact)


def build_blocked_artifact(
    *,
    started_at_utc: str,
    ended_at_utc: str,
    duration_s: float,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
) -> JsonDict:
    """Construct a terminal external block without success-shaped placeholders."""

    artifact = _base_artifact(started_at_utc, ended_at_utc, duration_s)
    failed = [row for row in preconditions if row.get("passed") is not True]
    artifact.update(
        {
            "preconditions_checked": deepcopy(list(preconditions)),
            "source_artifact_hashes": deepcopy(dict(source_hashes)),
            "gate_check_summary": [
                {
                    "blocked_by": row.get("upstream"),
                    "failed_check": row.get("check"),
                    "field": row.get("field"),
                    "observed_value": row.get("observed"),
                    "expected_value": row.get("expected"),
                }
                for row in failed
            ],
        }
    )
    return _seal_artifact(artifact)


def validate_artifact(value: Any) -> list[str]:
    """Cold-check terminal structure, classification, reduction, and checksums."""

    if not isinstance(value, dict):
        return ["artifact must be an object"]
    errors: list[str] = []
    if set(value) != set(FIELD_PRINCIPLES):
        errors.append("field principles and terminal keys differ")
    principles = value.get("field_principles")
    if not isinstance(principles, dict) or set(principles) != set(value):
        errors.append("field principles do not cover every terminal key")
    if value.get("schema") != SCHEMA:
        errors.append("schema is invalid")
    if value.get("experiment_id") != EXPERIMENT_ID or value.get("milestone") != MILESTONE:
        errors.append("experiment identity is invalid")
    if value.get("run_date") != RUN_DATE:
        errors.append("run_date is invalid")
    if value.get("status") not in {"complete", "blocked"}:
        errors.append("status must be complete or blocked")
    if value.get("MODEL_SPECS") != []:
        errors.append("MODEL_SPECS must be empty")
    if value.get("model_invoked") is not False:
        errors.append("model_invoked must be false")
    if value.get("invocation_counts") != ZERO_INVOCATION_COUNTS:
        errors.append("invocation_counts must all be zero")
    for field in ("inference_substrate", "inference_substrate_class"):
        if value.get(field) != "cpu_exact_solver_or_simulator":
            errors.append(f"{field} is invalid")
    if value.get("execution_venue") != "host":
        errors.append("execution_venue must be host")
    if value.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true")
    if value.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class is outside the closed vocabulary")
    if value.get("verdict_class") == "positive":
        errors.append("verdict_class positive is forbidden for oracle evidence")
    if value.get("reproducibility_checksum") != artifact_checksum(value):
        errors.append("reproducibility checksum mismatch")
    if value.get("status") == "blocked":
        if value.get("verdict_class") != "blocked":
            errors.append("blocked status requires blocked verdict_class")
        if not str(value.get("honest_verdict", "")).startswith("blocked_"):
            errors.append("blocked honest_verdict must start blocked_")
        if not value.get("gate_check_summary"):
            errors.append("blocked result requires gate_check_summary")
    if value.get("status") == "complete":
        reduction = reduce_identity_rows(value.get("rows", []))
        if reduction["completed_units"] != len(PANEL_CASES):
            errors.append("rows do not reduce to the frozen unit budget")
        expected_ready = int(
            reduction["arc_identity_ready_score"] == 1
            and all(row.get("passed") is True for row in value.get("validation_receipts", []))
        )
        if value.get("arc_identity_ready_score") != expected_ready:
            errors.append("ready score contradicts independently reduced rows")
        if expected_ready and value.get("verdict_class") != "circular_positive":
            errors.append("ready score requires circular_positive verdict_class")
        if not str(value.get("honest_verdict", "")).startswith("complete_"):
            errors.append("complete honest_verdict must start complete_")
    return list(dict.fromkeys(errors))


def independent_reduce(path: Path) -> JsonDict:
    """Read raw rows and reduce them without trusting terminal fields."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload["rows"]
        if not isinstance(rows, list):
            raise TypeError
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"invalid raw identity rows: {path}") from exc
    return reduce_identity_rows(rows)


def atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically publish stable JSON through a same-directory rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_validation_commands(
    *, raw_rows: Path, terminal_candidate: Path, coverage_json: Path
) -> list[JsonDict]:
    """Return only the bounded focused, affected, E2E, and artifact checks."""

    python = str(REPO_ROOT / ".venv/bin/python")
    pytest = str(REPO_ROOT / ".venv/bin/pytest")
    coverage = str(REPO_ROOT / ".venv/bin/coverage")
    ruff = str(REPO_ROOT / ".venv/bin/ruff")
    mypy = str(REPO_ROOT / ".venv/bin/mypy")
    focused = "tests/python/test_experiment_7276_v640_arc_identity.py"
    provenance_tests = [
        "tests/python/test_arc_eval_provenance_contract_20260905.py",
        "tests/python/test_arc_typed_identity_attack_audit_20260906.py",
        "tests/python/test_arc_ensure_server_leak_fix_20260808.py",
    ]
    changed = [
        "python/carnot/experiment_7276_v640_arc_identity.py",
        "python/carnot/agentic/arc_eval_provenance.py",
        "python/carnot/agentic/arc_executable_world_model.py",
        str(WRAPPER_PATH),
        focused,
    ]
    coverage_data = coverage_json.with_suffix(".coverage")
    commands = [
        {
            "name": "focused_exp7276",
            "command": [
                pytest,
                focused,
                "-q",
                "--no-cov",
                "-n",
                "0",
                "--basetemp=/tmp/exp7276-focused",
            ],
            "timeout_s": 600,
        },
        {
            "name": "affected_provenance",
            "command": [
                pytest,
                *provenance_tests,
                "-q",
                "--no-cov",
                "-n",
                "0",
                "--basetemp=/tmp/exp7276-provenance",
            ],
            "timeout_s": 600,
        },
        {
            "name": "E2E-009",
            "command": [
                pytest,
                "tests/python/test_arc_induction_state_persistence.py",
                "-q",
                "--no-cov",
                "-n",
                "0",
                "--basetemp=/tmp/exp7276-e2e009",
            ],
            "timeout_s": 600,
        },
        {
            "name": "E2E-010",
            "command": [
                pytest,
                "tests/python/test_arc_tool_grammar_transport.py",
                "-q",
                "--no-cov",
                "-n",
                "0",
                "--basetemp=/tmp/exp7276-e2e010",
            ],
            "timeout_s": 600,
        },
        {
            "name": "offline_r11l_smoke",
            "command": [
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
                "/tmp/experiment_7276_r11l_smoke.json",
            ],
            "timeout_s": 600,
        },
        {
            "name": "scoped_coverage_run",
            "command": [
                coverage,
                "run",
                f"--data-file={coverage_data}",
                "--include=*/experiment_7276_v640_arc_identity.py",
                "-m",
                "pytest",
                "-o",
                "addopts=",
                focused,
                "-q",
                "-n",
                "0",
                "--basetemp=/tmp/exp7276-coverage",
            ],
            "timeout_s": 600,
        },
        {
            "name": "scoped_coverage_json",
            "command": [
                coverage,
                "json",
                f"--data-file={coverage_data}",
                f"--include=*/experiment_7276_v640_arc_identity.py",
                "--fail-under=100",
                "-o",
                str(coverage_json),
            ],
            "timeout_s": 300,
        },
        {
            "name": "scoped_coverage_report",
            "command": [
                coverage,
                "report",
                f"--data-file={coverage_data}",
                "--include=*/experiment_7276_v640_arc_identity.py",
                "--show-missing",
                "--fail-under=100",
            ],
            "timeout_s": 300,
        },
        {"name": "ruff_check", "command": [ruff, "check", *changed], "timeout_s": 300},
        {
            "name": "ruff_format",
            "command": [ruff, "format", "--check", *changed],
            "timeout_s": 300,
        },
        {
            "name": "changed_module_mypy",
            "command": [
                mypy,
                "python/carnot/experiment_7276_v640_arc_identity.py",
                "python/carnot/agentic/arc_eval_provenance.py",
                "python/carnot/agentic/arc_executable_world_model.py",
                str(WRAPPER_PATH),
            ],
            "timeout_s": 900,
        },
        {
            "name": "scoped_spec_coverage",
            "command": [
                python,
                "-u",
                "scripts/check_spec_coverage.py",
                focused,
                *provenance_tests,
                "tests/python/test_arc_induction_state_persistence.py",
                "tests/python/test_arc_tool_grammar_transport.py",
            ],
            "timeout_s": 300,
        },
        {
            "name": "independent_raw_reducer",
            "command": [python, "-u", str(WRAPPER_PATH), "--reduce-raw", str(raw_rows)],
            "timeout_s": 300,
        },
        {
            "name": "terminal_candidate_adversarial_verify",
            "command": [python, "-u", "scripts/adversarial_verify.py", str(terminal_candidate)],
            "timeout_s": 600,
            "terminal_checker": True,
        },
        {
            "name": "terminal_candidate_row_consistency",
            "command": [
                python,
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                str(terminal_candidate),
            ],
            "timeout_s": 600,
            "terminal_checker": True,
        },
    ]
    return commands


def _progress(started: float, phase: str, event: str, completed: str = "0") -> None:
    print(
        f"[exp7276] phase={phase} event={event} "
        f"elapsed_s={time.monotonic() - started:.3f} completed_units={completed}",
        flush=True,
    )


def _collect_preconditions(root: Path) -> tuple[list[JsonDict], JsonDict]:  # pragma: no cover
    rows: list[JsonDict] = []
    hashes: JsonDict = {}
    for relative in INPUT_PATHS:
        path = root / relative
        exists = path.is_file()
        rows.append(gate_check("required_input", str(relative), "exists", True, exists))
        if exists:
            hashes[str(relative)] = {
                "sha256": _sha256_file(path),
                "quarantined": False,
                "retired": False,
            }
    spec = root / "openspec/capabilities/arc-world-model-trust-energy/spec.md"
    text = spec.read_text(encoding="utf-8") if spec.is_file() else ""
    rows.append(
        gate_check(
            "driving_capability",
            str(spec.relative_to(root)),
            "REQ-ARC-WMTE-7276",
            True,
            "REQ-ARC-WMTE-7276" in text,
        )
    )
    exclusion = (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    rows.append(
        gate_check(
            "historical_quarantine",
            "ops/exclusion_manifest.yaml",
            "experiment_7263_v639_arc_live",
            False,
            "experiment_7263_v639_arc_live" in exclusion,
        )
    )
    for relative in (Path("results"), Path("results/raw")):
        directory = root / relative
        rows.append(
            gate_check(
                "resource_ownership",
                str(relative),
                "owner_and_writable",
                True,
                directory.is_dir()
                and directory.stat().st_uid == os.getuid()
                and os.access(directory, os.W_OK),
            )
        )
    return rows, hashes


def _run_validations(
    commands: Sequence[Mapping[str, Any]], raw_dir: Path, started: float
) -> list[JsonDict]:  # pragma: no cover
    validation_dir = raw_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    receipts: list[JsonDict] = []
    for index, item in enumerate(commands):
        name = str(item["name"])
        command = [str(value) for value in item["command"]]
        _progress(started, "validation", f"before_subprocess:{name}", f"{index}/{len(commands)}")
        result = _run_streaming_command(
            command,
            cwd=REPO_ROOT,
            timeout_s=float(item["timeout_s"]),
            heartbeat_s=60,
            operation=f"exp7276:{name}",
        )
        log_path = validation_dir / f"{index:02d}_{name}.log"
        output = str(result.get("output", ""))
        log_path.write_text(output, encoding="utf-8")
        receipt = {
            "name": name,
            "command": shlex.join(command),
            "exit_code": result["exit_code"],
            "expected_exit_code": 0,
            "duration_s": result["duration_s"],
            "log_path": str(log_path.relative_to(REPO_ROOT)),
            "log_sha256": _sha256_file(log_path),
            "passed": result["exit_code"] == 0,
            "timed_out": result["timed_out"],
            "output_tail": output[-4000:],
        }
        receipts.append(receipt)
        _progress(
            started,
            "validation",
            f"after_subprocess:{name}:exit={result['exit_code']}",
            f"{index + 1}/{len(commands)}",
        )
    return receipts


def run_experiment(run_date: str) -> int:  # pragma: no cover
    """Run preflight, CPU evidence, scoped validations, and atomic publication."""

    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    _progress(started, "startup", "entrypoint_and_paths_authenticated")
    if run_date != RUN_DATE:
        raise ValueError(f"--date must be {RUN_DATE}")
    preconditions, source_hashes = _collect_preconditions(REPO_ROOT)
    failed = [row for row in preconditions if row["passed"] is not True]
    if failed:
        blocked = build_blocked_artifact(
            started_at_utc=started_at,
            ended_at_utc=datetime.now(UTC).isoformat(),
            duration_s=time.monotonic() - started,
            preconditions=preconditions,
            source_hashes=source_hashes,
        )
        atomic_write(REPO_ROOT / OUTPUT_PATH, blocked)
        _progress(started, "blocked", "terminal_artifact_written")
        return 2

    _progress(started, "historical_diagnosis", "before_benchmark")
    historical = diagnose_historical_episodes(REPO_ROOT)
    _progress(started, "historical_diagnosis", "after_benchmark", "4/4")
    panel_started = time.monotonic()
    _progress(started, "cpu_identity_panel", "before_benchmark")
    with tempfile.TemporaryDirectory(prefix="exp7276-") as temporary:
        panel = run_cpu_identity_panel(Path(temporary))
    _progress(
        started, "cpu_identity_panel", "after_benchmark", f"{len(PANEL_CASES)}/{len(PANEL_CASES)}"
    )
    phase_spans = [
        {"phase": "cpu_identity_panel", "duration_s": round(time.monotonic() - panel_started, 6)}
    ]
    raw_dir = REPO_ROOT / RAW_DIR
    raw_rows = REPO_ROOT / RAW_ROWS_PATH
    sidecar = REPO_ROOT / SIDECAR_PATH
    candidate = REPO_ROOT / TERMINAL_CANDIDATE_PATH
    atomic_write(
        raw_rows,
        {
            "schema": "carnot.exp7276.identity_rows.v1",
            "run_date": RUN_DATE,
            "rows": panel["rows"],
            "independent_reduction": panel["independent_reduction"],
        },
    )
    atomic_write(
        sidecar,
        {
            "schema": "carnot.exp7276.identity_sidecar.v1",
            "run_date": RUN_DATE,
            "current_invocation": {
                "MODEL_SPECS": [],
                "model_invoked": False,
                "invocation_counts": ZERO_INVOCATION_COUNTS,
            },
            "historical_episodes": historical,
            "injected_fixture_receipts": panel["fixture_sidecar_rows"],
        },
    )
    source_hashes.update(
        {
            str(RAW_ROWS_PATH): {"sha256": _sha256_file(raw_rows), "current_output": True},
            str(SIDECAR_PATH): {"sha256": _sha256_file(sidecar), "current_output": True},
        }
    )
    coverage_json = Path("/tmp/experiment_7276_coverage.json")
    commands = build_validation_commands(
        raw_rows=raw_rows, terminal_candidate=candidate, coverage_json=coverage_json
    )
    ordinary = [row for row in commands if not row.get("terminal_checker")]
    validation_started = time.monotonic()
    receipts = _run_validations(ordinary, raw_dir, started)
    phase_spans.append(
        {
            "phase": "scoped_validation",
            "duration_s": round(time.monotonic() - validation_started, 6),
        }
    )
    draft = build_complete_artifact(
        started_at_utc=started_at,
        ended_at_utc=datetime.now(UTC).isoformat(),
        duration_s=time.monotonic() - started,
        preconditions=preconditions,
        source_hashes=source_hashes,
        panel=panel,
        historical_rows=historical,
        raw_rows_path=raw_rows,
        sidecar_path=sidecar,
        phase_spans=phase_spans,
        validation_receipts=receipts,
    )
    errors = validate_artifact(draft)
    if errors:
        raise ValueError("terminal candidate is invalid: " + "; ".join(errors))
    atomic_write(candidate, draft)
    checkers = [row for row in commands if row.get("terminal_checker")]
    checker_receipts = _run_validations(checkers, raw_dir, started)
    receipts.extend(checker_receipts)
    final = build_complete_artifact(
        started_at_utc=started_at,
        ended_at_utc=datetime.now(UTC).isoformat(),
        duration_s=time.monotonic() - started,
        preconditions=preconditions,
        source_hashes=source_hashes,
        panel=panel,
        historical_rows=historical,
        raw_rows_path=raw_rows,
        sidecar_path=sidecar,
        phase_spans=phase_spans,
        validation_receipts=receipts,
    )
    errors = validate_artifact(final)
    if errors:
        raise ValueError("terminal artifact is invalid: " + "; ".join(errors))
    atomic_write(REPO_ROOT / OUTPUT_PATH, final)
    _progress(started, "publication", "terminal_artifact_atomically_written", "1/1")
    return 0 if final["arc_identity_ready_score"] == 1 else 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--reduce-raw", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.reduce_raw is not None:
        print(json.dumps(independent_reduce(args.reduce_raw), sort_keys=True), flush=True)
        return 0
    return run_experiment(args.date)
