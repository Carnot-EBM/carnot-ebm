"""Tests for REQ-ARC-7128 and its independent raw-trace audit."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from carnot import experiment_7128_v626_arc_loo_causal_audit as exp


ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "openspec/capabilities/arc-agi/spec.md"


def _hash_bytes(value: bytes) -> str:
    return exp.sha256_bytes(value)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return {
        "path": str(path.resolve()),
        "byte_count": path.stat().st_size,
        "sha256": exp.sha256_path(path),
        "event_count": len(rows),
    }


def _phase(phase: str, pid: int, ticks: int, index: int) -> dict[str, object]:
    started = index * 2_000_000_000
    ended = started + 1_000_000_000
    return {
        "kind": "phase",
        "phase": phase,
        "process_pid": pid,
        "process_start_ticks": ticks,
        "started_monotonic_ns": started,
        "ended_monotonic_ns": ended,
        "duration_s": 1.0,
        "cap_s": 300 if phase not in exp.ARM_NAMES else 1500,
        "exit_state": "completed",
        "timeout_state": "not_timed_out",
        "stop_reason": "completed",
    }


def _arm_events(arm: str, game: str, index: int) -> list[dict[str, object]]:
    action_id = f"{arm}-action-0"
    proposal_id = f"{arm}-proposal-0"
    before = _hash_bytes(f"{arm}-before".encode())
    after = _hash_bytes(f"{arm}-after".encode())
    adapters = ["a1", game, "z9"]
    return [
        {
            "kind": "adapter",
            "arm": arm,
            "adapter_keys_before": adapters,
            "adapter_keys_after": ["a1", "z9"] if arm == "adapter_withheld" else adapters,
            "removed_adapters": [game] if arm == "adapter_withheld" else [],
        },
        {
            "kind": "runtime_access",
            "arm": arm,
            "imports": ["carnot.agentic.arc_competition_agent"],
            "arguments": {"game": game, "factory": "make_carnot_agent"},
            "environment": {"CARNOT_ARC_E3_DIR": f"/tmp/{arm}/e3"},
            "filesystem_reads": [],
        },
        {
            "kind": "proposal",
            "arm": arm,
            "request_id": f"{arm}-request-0",
            "proposal_id": proposal_id,
            "action_id": action_id,
            "action": 6,
            "data": {"x": 10 + index, "y": 20},
            "source": "E3AgentPolicy.next_move",
        },
        {
            "kind": "verifier",
            "arm": arm,
            "proposal_id": proposal_id,
            "verifier": exp.EXACT_VERIFIER,
            "accepted": True,
            "oracle": False,
        },
        {
            "kind": "action",
            "arm": arm,
            "action_id": action_id,
            "proposal_id": proposal_id,
            "executed": True,
            "action": 6,
            "data": {"x": 10 + index, "y": 20},
        },
        {
            "kind": "transition",
            "arm": arm,
            "action_id": action_id,
            "executed": True,
            "before_hash": before,
            "after_hash": after,
            "level_before": 0,
            "level_after": 0,
            "reward": None,
        },
        {"kind": "reward", "arm": arm, "action_id": action_id, "reward": None},
        {
            "kind": "level",
            "arm": arm,
            "action_id": action_id,
            "level_before": 0,
            "level_after": 0,
            "solve_provenance": "development_proxy",
            "solve_claim_made": False,
            "offline_reproduced": False,
        },
    ]


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    raw = tmp_path / "raw"
    producer = repo / exp.PRODUCER_RELATIVE_PATH
    adapter = repo / exp.ADAPTER_RELATIVE_PATH
    registry = repo / exp.REGISTRY_RELATIVE_PATH
    producer.parent.mkdir(parents=True, exist_ok=True)
    adapter.parent.mkdir(parents=True, exist_ok=True)
    registry.parent.mkdir(parents=True, exist_ok=True)
    producer.write_text(
        "def _arm_worker():\n"
        "    from carnot.agentic.arc_competition_agent import E3AgentPolicy, make_carnot_agent\n"
        "    return make_carnot_agent(E3AgentPolicy)\n",
        encoding="utf-8",
    )
    adapter.write_text("def _other_game():\n    return None\n", encoding="utf-8")
    registry.write_text("games: []\n", encoding="utf-8")

    parent_rows: list[dict[str, object]] = []
    process_rows = []
    phase_rows = []
    manifest = []
    for index, arm in enumerate(exp.ARM_NAMES):
        pid = 7101 + index
        ticks = 101 + index
        started = {"kind": "process_started", "role": arm, "pid": pid, "start_ticks": ticks}
        finished = {
            "kind": "process_finished",
            "role": arm,
            "pid": pid,
            "start_ticks": ticks,
            "fresh_process": True,
            "exit_code": 0,
            "timed_out": False,
            "stop_reason": "completed",
            "entrypoint": exp.REAL_ENTRYPOINT,
        }
        phase = _phase(arm, pid, ticks, index + 2)
        parent_rows.extend([started, finished, phase])
        process_rows.append({key: value for key, value in finished.items() if key != "kind"})
        phase_rows.append({key: value for key, value in phase.items() if key != "kind"})
        receipt = _write_jsonl(raw / arm / "events.jsonl", _arm_events(arm, "r11l", index))
        manifest.append({"arm": arm, **receipt})
    parent_receipt = _write_jsonl(raw / "parent-events.jsonl", parent_rows)
    manifest.append({"arm": "parent", **parent_receipt})

    transition_rows = [
        {key: value for key, value in row.items() if key != "kind"}
        for arm in exp.ARM_NAMES
        for row in _arm_events(arm, "r11l", exp.ARM_NAMES.index(arm))
        if row["kind"] == "transition"
    ]
    upstream = {
        "run_date": "20260907",
        "verdict_class": "null",
        "honest_verdict": "complete_null_executed_pair_zero_withheld_levels_no_solve_claim",
        "selected_game": {"game": "r11l", "target_level": 1},
        "raw_trace_manifest": manifest,
        "process_rows": process_rows,
        "phase_receipt_rows": phase_rows,
        "arm_rows": [
            {
                "arm": arm,
                "request_count": 0,
                "proposal_count": 1,
                "action_count": 1,
                "transition_count": 1,
                "levels": 0,
            }
            for arm in exp.ARM_NAMES
        ],
        "transition_rows": transition_rows,
        "rows": [
            {
                "game": "r11l",
                "arm": arm,
                "levels": 0,
                "executed_transition_count": 1,
                "solve_provenance": "development_proxy",
                "solve_claim_made": False,
                "offline_reproduced": False,
            }
            for arm in exp.ARM_NAMES
        ],
        "withheld_levels": 0,
        "control_levels": 0,
        "level_delta": 0,
        "solve_provenance": "development_proxy",
        "solve_claim_made": False,
        "offline_reproduced": False,
        "registry_mutated": False,
        "source_artifact_hashes": {
            exp.REGISTRY_RELATIVE_PATH.as_posix(): exp.sha256_path(registry),
            "registry_hash_after": exp.sha256_path(registry),
            exp.PRODUCER_RELATIVE_PATH.as_posix(): exp.sha256_path(producer),
            exp.ADAPTER_RELATIVE_PATH.as_posix(): exp.sha256_path(adapter),
        },
    }
    upstream_path = repo / exp.UPSTREAM_RELATIVE_PATH
    upstream_path.parent.mkdir(parents=True, exist_ok=True)
    upstream_path.write_text(json.dumps(upstream, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return repo, raw, upstream_path


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_req_arc_7128_spec_precedes_implementation() -> None:
    """REQ-ARC-7128 defines every audit surface and adversarial scenario."""

    text = SPEC.read_text(encoding="utf-8")
    section = text.split("## REQ-ARC-7128", 1)[1]
    for scenario in (
        "RAW-BYTES",
        "PROCESS",
        "LEAKAGE",
        "ACTIONS",
        "REMOVAL",
        "REGISTRY",
        "CLASS",
    ):
        assert f"SCENARIO-ARC-7128-{scenario}" in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_absent_upstream_writes_one_terminal_block(tmp_path: Path) -> None:
    """SCENARIO-ARC-7128-CLASS blocks once when Exp7127 is absent."""

    repo = tmp_path / "repo"
    output = tmp_path / "audit.json"
    artifact = exp.run_audit(
        repo_root=repo,
        upstream_path=repo / exp.UPSTREAM_RELATIVE_PATH,
        output_path=output,
        run_date="20260908",
        raw_root=tmp_path / "raw",
    )

    assert output.is_file()
    assert set(artifact) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["verdict_class"] == "blocked"
    assert str(artifact["honest_verdict"]).startswith("blocked_")
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["gate_check_summary"]["failed_check"] == "upstream_artifact_present"
    assert exp.validate_artifact(artifact) == []


def test_changed_raw_bytes_and_path_traversal_receive_no_credit(tmp_path: Path) -> None:
    """SCENARIO-ARC-7128-RAW-BYTES rejects drift and paths outside the raw root."""

    repo, raw, upstream_path = _fixture(tmp_path)
    upstream = _load(upstream_path)
    changed = Path(upstream["raw_trace_manifest"][0]["path"])
    changed.write_bytes(changed.read_bytes() + b"changed\n")
    artifact = exp.audit_upstream(repo, upstream_path, run_date="20260908", raw_root=raw)
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["raw_hash_rows"][0]["passed"] is False
    assert not any(
        row["credited"]
        for row in artifact["action_provenance_rows"]
        if row["arm"] == "adapter_withheld"
    )

    repo, raw, upstream_path = _fixture(tmp_path / "traversal")
    upstream = _load(upstream_path)
    escape = tmp_path / "escape.jsonl"
    escape.write_text("{}\n", encoding="utf-8")
    upstream["raw_trace_manifest"][0].update(
        path=str(raw / ".." / escape.name),
        byte_count=escape.stat().st_size,
        sha256=exp.sha256_path(escape),
        event_count=1,
    )
    _save(upstream_path, upstream)
    artifact = exp.audit_upstream(repo, upstream_path, run_date="20260908", raw_root=raw)
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["raw_hash_rows"][0]["path_safe"] is False


def test_reused_pid_is_disqualified(tmp_path: Path) -> None:
    """SCENARIO-ARC-7128-PROCESS rejects a PID reused by both arms."""

    repo, raw, upstream_path = _fixture(tmp_path)
    parent = raw / "parent-events.jsonl"
    rows = [json.loads(line) for line in parent.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        if row.get("role") == "adapter_visible_control" or row.get("phase") == "adapter_visible_control":
            row["pid"] = 7101
            row["process_pid"] = 7101
    receipt = _write_jsonl(parent, rows)
    upstream = _load(upstream_path)
    parent_manifest = next(row for row in upstream["raw_trace_manifest"] if row["arm"] == "parent")
    parent_manifest.update(receipt)
    upstream["process_rows"][1]["pid"] = 7101
    _save(upstream_path, upstream)

    artifact = exp.audit_upstream(repo, upstream_path, run_date="20260908", raw_root=raw)
    assert artifact["verdict_class"] == "disqualified"
    assert not all(row["fresh_distinct_pid"] for row in artifact["process_isolation_rows"])


def test_hidden_adapter_import_after_key_removal_is_disqualified(tmp_path: Path) -> None:
    """SCENARIO-ARC-7128-LEAKAGE rejects hidden target recipes left in memory."""

    repo, raw, upstream_path = _fixture(tmp_path)
    producer = repo / exp.PRODUCER_RELATIVE_PATH
    adapter = repo / exp.ADAPTER_RELATIVE_PATH
    producer.write_text(
        "def _arm_worker(game):\n"
        "    from carnot.agentic import arc_game_adapters\n"
        "    arc_game_adapters._BUILDERS.pop(game)\n"
        "    from carnot.agentic.arc_competition_agent import E3AgentPolicy, make_carnot_agent\n"
        "    return make_carnot_agent(E3AgentPolicy)\n",
        encoding="utf-8",
    )
    adapter.write_text(
        "R11L_L1_LABELS = ('known-action',)\n"
        "def _r11l():\n"
        "    return R11L_L1_LABELS\n",
        encoding="utf-8",
    )
    upstream = _load(upstream_path)
    upstream["source_artifact_hashes"][exp.PRODUCER_RELATIVE_PATH.as_posix()] = exp.sha256_path(producer)
    upstream["source_artifact_hashes"][exp.ADAPTER_RELATIVE_PATH.as_posix()] = exp.sha256_path(adapter)
    _save(upstream_path, upstream)

    artifact = exp.audit_upstream(repo, upstream_path, run_date="20260908", raw_root=raw)
    withheld = next(row for row in artifact["adapter_access_rows"] if row["arm"] == "adapter_withheld")
    assert withheld["registry_key_removed"] is True
    assert withheld["target_recipe_symbols_retained"] is True
    assert withheld["passed"] is False
    assert artifact["verdict_class"] == "disqualified"


def test_registry_write_is_disqualified(tmp_path: Path) -> None:
    """SCENARIO-ARC-7128-REGISTRY rejects changed registry bytes."""

    repo, raw, upstream_path = _fixture(tmp_path)
    upstream = _load(upstream_path)
    upstream["source_artifact_hashes"]["registry_hash_after"] = _hash_bytes(b"changed")
    _save(upstream_path, upstream)

    artifact = exp.audit_upstream(repo, upstream_path, run_date="20260908", raw_root=raw)
    assert artifact["registry_mutated"] is True
    assert artifact["registry_hash_before"] != artifact["registry_hash_after"]
    assert artifact["verdict_class"] == "disqualified"


def test_proposed_but_unexecuted_action_gets_no_value_or_level_credit(tmp_path: Path) -> None:
    """SCENARIO-ARC-7128-ACTIONS excludes a proposal with no execution chain."""

    repo, raw, upstream_path = _fixture(tmp_path)
    trace = raw / "adapter_withheld" / "events.jsonl"
    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    rows.append(
        {
            "kind": "proposal",
            "arm": "adapter_withheld",
            "proposal_id": "unexecuted-proposal",
            "action_id": "unexecuted-action",
            "action": 6,
            "data": {"x": 99, "y": 99},
            "source": "E3AgentPolicy.next_move",
        }
    )
    receipt = _write_jsonl(trace, rows)
    upstream = _load(upstream_path)
    manifest = next(row for row in upstream["raw_trace_manifest"] if row["arm"] == "adapter_withheld")
    manifest.update(receipt)
    upstream["arm_rows"][0]["proposal_count"] = 2
    _save(upstream_path, upstream)

    artifact = exp.audit_upstream(repo, upstream_path, run_date="20260908", raw_root=raw)
    row = next(row for row in artifact["action_provenance_rows"] if row["proposal_id"] == "unexecuted-proposal")
    credit = next(row for row in artifact["causal_credit_rows"] if row["proposal_id"] == "unexecuted-proposal")
    assert row["credited"] is False
    assert row["executed_action_count"] == 0
    assert credit["causal_credit"] is False
    assert credit["credited_level_delta"] == 0
    withheld = next(row for row in artifact["rows"] if row["arm"] == "adapter_withheld")
    assert withheld["levels"] == 0


def test_fabricated_removal_effect_stays_unavailable_not_zero(tmp_path: Path) -> None:
    """SCENARIO-ARC-7128-REMOVAL ignores labels without replay inputs."""

    repo, raw, upstream_path = _fixture(tmp_path)
    upstream = _load(upstream_path)
    upstream["removal_effect"] = 1
    upstream["causal_label"] = "adapter removal caused the selected action"
    _save(upstream_path, upstream)

    artifact = exp.audit_upstream(repo, upstream_path, run_date="20260908", raw_root=raw)
    assert artifact["verdict_class"] == "null"
    assert all(row["status"] == "unavailable" for row in artifact["removal_replay_rows"])
    assert all(row["action_changed"] is None for row in artifact["removal_replay_rows"])
    assert not any(row["causal_credit"] for row in artifact["causal_credit_rows"])


def test_complete_removal_input_replays_deterministic_selection(tmp_path: Path) -> None:
    """SCENARIO-ARC-7128-REMOVAL measures a supported signal removal."""

    repo, raw, upstream_path = _fixture(tmp_path)
    trace = raw / "adapter_withheld" / "events.jsonl"
    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    rows.append(
        {
            "kind": "selection_replay_input",
            "arm": "adapter_withheld",
            "action_id": "adapter_withheld-action-0",
            "signal": "forecast",
            "selection_rule": "max_score_then_order",
            "selected_action": {"action": 6, "data": {"x": 10, "y": 20}},
            "candidates": [
                {
                    "action": 6,
                    "data": {"x": 10, "y": 20},
                    "score": 3.0,
                    "signal_contribution": 2.5,
                },
                {
                    "action": 1,
                    "data": None,
                    "score": 2.0,
                    "signal_contribution": 0.0,
                },
            ],
        }
    )
    receipt = _write_jsonl(trace, rows)
    upstream = _load(upstream_path)
    manifest = next(row for row in upstream["raw_trace_manifest"] if row["arm"] == "adapter_withheld")
    manifest.update(receipt)
    _save(upstream_path, upstream)

    artifact = exp.audit_upstream(repo, upstream_path, run_date="20260908", raw_root=raw)
    replay = next(
        row
        for row in artifact["removal_replay_rows"]
        if row["action_id"] == "adapter_withheld-action-0"
    )
    credit = next(
        row
        for row in artifact["causal_credit_rows"]
        if row["action_id"] == "adapter_withheld-action-0"
    )
    assert replay["status"] == "replayed"
    assert replay["action_changed"] is True
    assert replay["selected_action_without_signal"] == {"action": 1, "data": None}
    assert credit["causal_credit"] is True
    assert artifact["verdict_class"] == "positive"


def test_real_exp7127_audit_exposes_leakage_without_solve_credit() -> None:
    """REQ-ARC-7128 audits the real complete-zero evidence and source access."""

    artifact = exp.audit_upstream(
        ROOT,
        ROOT / exp.UPSTREAM_RELATIVE_PATH,
        run_date="20260908",
        raw_root=exp.DEFAULT_RAW_ROOT,
    )

    assert exp.validate_artifact(artifact) == []
    assert artifact["upstream_verdict_class"] == "null"
    assert artifact["arc_causal_audit_complete_score"] == 1
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["registry_mutated"] is False
    assert len(artifact["action_provenance_rows"]) == 6
    assert all(row["credited"] for row in artifact["action_provenance_rows"])
    assert all(row["status"] == "unavailable" for row in artifact["removal_replay_rows"])
    assert all(row["solve_provenance"] == "development_proxy" for row in artifact["level_rows"])
    assert artifact["solve_claim_made"] is False
    assert artifact["offline_reproduced"] is False


def test_artifact_validation_rejects_forged_claims(tmp_path: Path) -> None:
    """SCENARIO-ARC-7128-CLASS rejects checksum and solve-claim drift."""

    repo, raw, upstream_path = _fixture(tmp_path)
    artifact = exp.audit_upstream(repo, upstream_path, run_date="20260908", raw_root=raw)
    changed = deepcopy(artifact)
    changed["solve_claim_made"] = True
    changed["verdict_class"] = "unknown"
    changed["honest_verdict"] = "complete_positive_forged"

    errors = exp.validate_artifact(changed)
    assert "solve_claim_made_must_be_false" in errors
    assert "verdict_class_invalid" in errors
    assert "reproducibility_checksum_mismatch" in errors
