"""REQ-ARC-6843 live ARC evidence stratum freeze tests."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6843_live_arc_evidence_stratum_freeze as exp


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _base_paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "v598_root": _write_json(
            tmp_path / "results/experiment_6835_v598_terminal_evidence_freeze.json",
            {
                "schema": "carnot.experiment_6835.v598_terminal_evidence_freeze.v1",
                "status": "complete",
                "v598_evidence_root_ready_score": 1,
                "honest_verdict": "complete_null_v598_terminal_evidence_freeze_source_null_preserved",
                "verdict_class": "null",
            },
        ),
        "arc_registry": _write_text(
            tmp_path / "ops/arc_solve_registry.yaml", "schema_version: 1\n"
        ),
        "ops_status": _write_text(tmp_path / "ops/status.md", "# status\n"),
        "canonical_path_doc": _write_text(
            tmp_path / "ops/arc-live-agent-canonical-path.md",
            "make_carnot_agent -> E3AgentPolicy -> StepwiseExplorer\n",
        ),
        "arc_competition_agent_source": _write_text(
            tmp_path / "python/carnot/agentic/arc_competition_agent.py",
            "SUBMITTED_AGENT_CONFIG = {}\ndef make_carnot_agent(): pass\nclass E3AgentPolicy: pass\n",
        ),
        "arc_trajectory_supervisor_source": _write_text(
            tmp_path / "python/carnot/agentic/arc_trajectory_supervisor.py",
            "class TrajectorySupervisor: pass\nclass TraceAutomatonSupervisor: pass\n",
        ),
        "arc_leaderboard_eval_source": _write_text(
            tmp_path / "scripts/arc_leaderboard_eval.py",
            "def run_game(): pass\n",
        ),
        "upstream_exp6681": _write_json(
            tmp_path / "results/experiment_6681_arc_post_redirect_outcomes.json",
            {
                "honest_verdict": "complete_redirect_outcome_transport",
                "canonical_path_receipt": {
                    "factory": "carnot.agentic.arc_competition_agent.make_carnot_agent",
                    "policy": "carnot.agentic.arc_competition_agent.E3AgentPolicy",
                },
            },
        ),
        "upstream_exp6776": _write_json(
            tmp_path / "results/experiment_6776_arc_shadow_supervisor_accrual.json",
            {
                "status": "complete_blocked_shadow_supervisor_accrual",
                "honest_verdict": "complete_blocked_shadow_supervisor_accrual",
                "shadow_supervisor_transport_ready": False,
                "rows": [],
            },
        ),
        "upstream_exp6777": _write_json(
            tmp_path / "results/experiment_6777_arc_tool_gap_transport.json",
            {
                "status": "blocked",
                "honest_verdict": "blocked_gate_check_failed",
                "gate_check_summary": "gate-unsat(final): upstream blocked",
            },
        ),
        "upstream_exp6820_missing": tmp_path
        / "results/experiment_6820_arc_tool_gap_obligation_transport_v2.json",
    }


def _leaderboard_payload(
    *,
    complete: bool = True,
    game: str = "sp80",
    policy: str = "e3",
    budget: int = 2500,
    tool_gap: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "experiment": "arc_leaderboard_eval",
        "games_mode": "claimed",
        "policy": policy,
        "budget": budget,
        "random_seed": 20260901,
        "complete": complete,
        "honest_verdict": (
            "complete_leaderboard_eval_1_levels_0_gaps"
            if complete
            else "partial_1_of_2_games_run_in_progress"
        ),
        "inference_substrate": "offline_sim_no_quota_frame_only_live_agent",
        "per_game": [
            {
                "game": game,
                "levels": 1,
                "reached": 1,
                "actions": 123,
                "charged_actions": 130,
                "frame_sequence": [{"grid_hash": "sha256:a"}],
                "policy_diagnostics": {
                    "proposer": {
                        "instantiated": True,
                        "repo_substr": "Qwen3.8-27B",
                        "port": 8919,
                        "mtp": False,
                        "kv_quant": "q8_0",
                    },
                    "induction_attempts": [
                        {
                            "model_specs": "Qwen3.8-27B-Q4_K_M GGUF",
                            "tool_gap": tool_gap,
                        }
                    ],
                },
            }
        ],
    }


def _supervisor_ledger_payload() -> dict[str, Any]:
    return {
        "schema": "carnot.arc.supervisor_refinement_ledger.v1",
        "entries": {
            "sha256:entry": {
                "receipt_id": "sha256:entry",
                "game": "tu93",
                "seed": 20260725,
                "harness_arm": "S_llmon",
                "actions_observed": 399,
                "levels": 1,
                "mode": "applied",
                "window": 120,
                "redirects": [
                    {
                        "arm": "drop_goal_bias",
                        "action_index": 120,
                        "resolved_by_levelup": True,
                    }
                ],
            }
        },
        "recommendation": {"status": "insufficient_evidence"},
    }


def test_req_6843_spec_precedes_implementation() -> None:
    """REQ-ARC-6843 declares scenarios and required artifact fields."""

    text = (REPO_ROOT / exp.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("## REQ-ARC-6843:") :]
    for marker in (
        "SCENARIO-ARC-6843-TERMINAL-DETECTION",
        "SCENARIO-ARC-6843-STRATUM-SEPARATION",
        "SCENARIO-ARC-6843-PROCESS-OBSERVATION",
        "SCENARIO-ARC-6843-CHECKSUMS",
        "SCENARIO-ARC-6843-NO-SOLVE-CLAIM",
    ):
        assert marker in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert field in exp.FIELD_PRINCIPLES


def test_scenario_6843_terminal_detection_and_incomplete_rejection(tmp_path: Path) -> None:
    """SCENARIO-ARC-6843-TERMINAL-DETECTION rejects partial and missing rows."""

    paths = _base_paths(tmp_path)
    complete = _write_json(
        tmp_path / "results/arc_leaderboard_eval_runs/sp80-1.json",
        _leaderboard_payload(complete=True),
    )
    partial = _write_json(
        tmp_path / "results/arc_leaderboard_eval_runs/sp80-2.partial.json",
        _leaderboard_payload(complete=False, game="su15"),
    )
    paths.update({"leaderboard_complete": complete, "leaderboard_partial": partial})

    artifact = exp.build_artifact(
        run_date="20260901",
        duration_s=0.25,
        source_paths=paths,
        process_observations=[],
    )

    assert artifact["status"] == "complete_live_arc_inventory"
    assert artifact["arc_inventory_complete_score"] == 1
    assert [row["game"] for row in artifact["rows"]] == ["sp80"]
    incomplete_paths = {row["path"] for row in artifact["incomplete_artifact_manifest"]}
    assert str(partial.relative_to(tmp_path)) in incomplete_paths
    assert "results/experiment_6820_arc_tool_gap_obligation_transport_v2.json" in incomplete_paths
    assert any(
        row["terminal_class"] == "blocked_terminal"
        for row in artifact["terminal_artifact_manifest"]
    )
    assert exp.validate_artifact(artifact) == []


def test_scenario_6843_separates_run_model_policy_budget_tool_and_supervisor(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-6843-STRATUM-SEPARATION keeps unlike configurations apart."""

    paths = _base_paths(tmp_path)
    paths["leaderboard_tool_on"] = _write_json(
        tmp_path / "results/arc_leaderboard_eval_runs/sp80-selfparse.json",
        _leaderboard_payload(
            complete=True,
            game="sp80",
            policy="e3",
            budget=2500,
            tool_gap={
                "tool_gap_events": [],
                "tool_gap_events_dropped": 0,
                "candidate_tools_enabled": [],
                "candidate_tools_rejected": [],
                "terminated_by": "early_stop_non_improving",
                "tool_calls_total": 4,
            },
        ),
    )
    paths["leaderboard_tool_off"] = _write_json(
        tmp_path / "results/arc_leaderboard_eval_runs/sp80-off.json",
        _leaderboard_payload(complete=True, game="sp80", policy="e3", budget=399),
    )
    paths["supervisor_ledger"] = _write_json(
        tmp_path / "ops/arc_supervisor_refinement_ledger.json",
        _supervisor_ledger_payload(),
    )

    artifact = exp.build_artifact(
        run_date="20260901",
        duration_s=0.25,
        source_paths=paths,
        process_observations=[],
    )

    identities = {row["stratum_identity"] for row in artifact["rows"]}
    assert len(identities) == 3
    assert {row["tool_loop_state"] for row in artifact["rows"]} == {
        "selfparse",
        "off_or_unobserved",
        "unknown",
    }
    assert {row["supervisor_state"] for row in artifact["rows"]} == {
        "applied",
        "unobserved",
    }
    assert artifact["tool_gap_eligible_cells"]["count"] == 1
    assert artifact["tool_gap_cells_ready_score"] == 1
    assert artifact["supervisor_eligible_cells"]["count"] == 1
    assert artifact["supervisor_cells_ready_score"] == 1
    assert artifact["configuration_strata"]["count"] == 3


def test_scenario_6843_duplicate_rows_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-ARC-6843-STRATUM-SEPARATION blocks duplicate complete strata."""

    paths = _base_paths(tmp_path)
    payload = _leaderboard_payload(complete=True, game="sp80")
    payload["per_game"].append(deepcopy(payload["per_game"][0]))
    paths["leaderboard_duplicate"] = _write_json(
        tmp_path / "results/arc_leaderboard_eval_runs/dup.json",
        payload,
    )

    artifact = exp.build_artifact(
        run_date="20260901",
        duration_s=0.1,
        source_paths=paths,
        process_observations=[],
    )

    assert artifact["status"] == "complete_blocked_live_arc_inventory"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["arc_inventory_complete_score"] == 0
    assert artifact["gate_check_summary"]["failed_check"] == "duplicate_row_identities"
    assert artifact["rows"] == []
    assert exp.validate_artifact(artifact) == []


def test_scenario_6843_process_observations_are_read_only_and_configured() -> None:
    """SCENARIO-ARC-6843-PROCESS-OBSERVATION parses ps rows without mutation."""

    line = (
        "375005 374980 Mon Aug 31 20:01:12 2026 SNl 10:20:13 "
        "/repo/.venv/bin/python /repo/scripts/arc_leaderboard_eval.py "
        "--policy e3 --only ls20,wa30 --budget 2500"
    )
    observation = exp.parse_ps_line(line)
    assert observation["pid"] == 375005
    assert observation["ppid"] == 374980
    assert observation["state"] == "SNl"
    assert observation["start_time"] == "Mon Aug 31 20:01:12 2026"
    assert observation["observed_configuration"] == {
        "budget": 2500,
        "games": ["ls20", "wa30"],
        "model_id": None,
        "policy": "e3",
        "supervisor_state": "unobserved",
        "tool_loop_state": "off_or_unobserved",
    }
    assert observation["read_only_commands"] == [
        "ps -eo pid,ppid,lstart,stat,etime,args --sort=pid"
    ]

    wrapper_text = """
env.update({
    "CARNOT_ARC_TRAJECTORY_SUPERVISOR": "1",
    "CARNOT_ARC_INDUCE_TOOL_LOOP": "selfparse",
})
"""
    assert exp.extract_wrapper_configuration(wrapper_text) == {
        "supervisor_state": "applied",
        "tool_loop_state": "selfparse",
    }


def test_scenario_6843_blocked_precondition_reports_failed_gate(tmp_path: Path) -> None:
    """REQ-ARC-6843 writes complete_blocked_live_arc_inventory on a gate stop."""

    paths = _base_paths(tmp_path)
    v598 = json.loads(paths["v598_root"].read_text(encoding="utf-8"))
    v598["v598_evidence_root_ready_score"] = 0
    _write_json(paths["v598_root"], v598)

    artifact = exp.build_artifact(
        run_date="20260901",
        duration_s=0.1,
        source_paths=paths,
        process_observations=[],
    )

    assert artifact["status"] == "complete_blocked_live_arc_inventory"
    assert artifact["honest_verdict"] == "complete_blocked_live_arc_inventory"
    assert artifact["gate_check_summary"]["failed_check"] == "v598_evidence_root_ready_score"
    assert artifact["gate_check_summary"]["observed"] == 0
    assert artifact["rows"] == []
    assert artifact["arc_inventory_complete_score"] == 0
    assert exp.validate_artifact(artifact) == []


def test_scenario_6843_hashes_and_no_solve_claim(tmp_path: Path) -> None:
    """SCENARIO-ARC-6843-CHECKSUMS keeps hashes stable and no-solve terminal."""

    paths = _base_paths(tmp_path)
    paths["leaderboard"] = _write_json(
        tmp_path / "results/arc_leaderboard_eval_runs/sp80.json",
        _leaderboard_payload(complete=True),
    )
    processes = [
        {
            "pid": 375005,
            "ppid": 374980,
            "start_time": "Mon Aug 31 20:01:12 2026",
            "state": "SNl",
            "elapsed": "10:20:13",
            "command": "/repo/scripts/arc_leaderboard_eval.py --policy e3 --only ls20,wa30 --budget 2500",
            "observed_configuration": {
                "policy": "e3",
                "games": ["ls20", "wa30"],
                "budget": 2500,
                "model_id": None,
                "tool_loop_state": "off_or_unobserved",
                "supervisor_state": "unobserved",
            },
            "read_only_commands": exp.READ_ONLY_PROCESS_COMMANDS,
            "observation_role": "in_flight_process",
        }
    ]

    artifact = exp.build_artifact(
        run_date="20260901",
        duration_s=0.25,
        source_paths=paths,
        process_observations=processes,
    )

    assert artifact["solve_claim"] is False
    assert artifact["verifier_is_oracle"] is False
    assert artifact["verdict_class"] == "partial"
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["rows"][0]["row_sha256"].startswith("sha256:")
    assert artifact["source_artifact_hashes"]["leaderboard"]["file_sha256"].startswith("sha256:")
    changed = deepcopy(artifact)
    changed["duration_s"] = 999.0
    assert exp.reproducibility_checksum(changed) == artifact["reproducibility_checksum"]


def test_req_6843_validator_and_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-ARC-6843 validates artifacts and writes JSON through the CLI."""

    paths = _base_paths(tmp_path)
    paths["leaderboard"] = _write_json(
        tmp_path / "results/arc_leaderboard_eval_runs/sp80.json",
        _leaderboard_payload(complete=True),
    )
    artifact = exp.build_artifact(
        run_date="20260901",
        duration_s=0.1,
        source_paths=paths,
        process_observations=[],
    )

    broken = deepcopy(artifact)
    broken["field_principles"].pop("rows")
    broken["reproducibility_checksum"] = exp.reproducibility_checksum(broken)
    assert "field principles do not cover every top-level field" in exp.validate_artifact(broken)

    broken = deepcopy(artifact)
    broken["reproducibility_checksum"] = "bad"
    assert "reproducibility checksum mismatch" in exp.validate_artifact(broken)

    broken = deepcopy(artifact)
    broken["solve_claim"] = True
    broken["reproducibility_checksum"] = exp.reproducibility_checksum(broken)
    assert "solve_claim must be false" in exp.validate_artifact(broken)

    monkeypatch.setattr(exp, "collect_default_source_paths", lambda _root: paths)
    monkeypatch.setattr(exp, "sample_process_observations", lambda: [])
    output = tmp_path / "out.json"
    assert exp.main(["--date", "20260901", "--output", str(output)]) == 0
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["run_date"] == "20260901"
    assert written["arc_inventory_complete_score"] == 1

    monkeypatch.setattr(exp, "validate_artifact", lambda _artifact: ["forced failure"])
    assert exp.main(["--date", "20260901", "--output", str(tmp_path / "bad.json")]) == 1


def test_req_6843_defensive_branches_and_live_sampler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-ARC-6843-PROCESS-OBSERVATION covers defensive inventory edges."""

    paths = _base_paths(tmp_path)
    paths["leaderboard"] = _write_json(
        tmp_path / "results/arc_leaderboard_eval_runs/sp80.json",
        _leaderboard_payload(complete=True),
    )
    assert exp.sha256_file(paths["arc_registry"]).startswith("sha256:")
    assert "leaderboard:sp80.json" in exp.collect_default_source_paths(tmp_path)
    assert exp.collect_default_source_paths(tmp_path / "absent")["v598_root"].name.endswith(".json")
    assert exp._common_root(None) == exp.REPO_ROOT
    assert exp._common_root({"missing": tmp_path / "missing"}) == exp.REPO_ROOT

    assert exp._terminal_class(Path("d.json"), {"verdict_class": "disqualified"}, None) == (
        "disqualified_terminal"
    )
    assert exp._terminal_class(Path("p.json"), {"verdict_class": "partial"}, None) == (
        "partial_terminal"
    )
    assert exp._producer_configuration({"model_specs": [{"id": "m"}]}, "source")[
        "model_specs_sha256"
    ].startswith("sha256:")

    model_from_attempt = {
        "policy_diagnostics": {
            "proposer": {},
            "induction_attempts": [{"model_specs": "Qwen3.6-35B-A3B GGUF"}],
        }
    }
    assert exp._extract_model_id(model_from_attempt) == "unsloth/Qwen3.6-35B-A3B-GGUF"
    assert exp._extract_model_id({"policy_diagnostics": {"proposer": {}}}) == "unknown"
    supervisor = exp._supervisor_summary(
        {
            "policy_diagnostics": {
                "trajectory_supervisor": {
                    "enabled": True,
                    "mode": "applied",
                    "redirects": [],
                }
            }
        }
    )
    assert supervisor["state"] == "applied"
    assert supervisor["receipt_complete"] is True

    assert (
        exp._leaderboard_rows(
            {
                "_payload": {"per_game": [None, {}]},
                "path": "bad.json",
                "file_sha256": "sha256:x",
            }
        )
        == []
    )
    assert (
        exp._supervisor_ledger_rows(
            {
                "_payload": {"entries": {"bad": None}},
                "path": "ledger.json",
                "file_sha256": "sha256:x",
            }
        )
        == []
    )

    assert exp._command_config("'bad quote")["budget"] is None
    assert exp._command_config("llama-server -m /models/unknown.gguf --budget nope") == {
        "budget": "nope",
        "games": [],
        "model_id": "unknown.gguf",
        "policy": None,
        "supervisor_state": "unobserved",
        "tool_loop_state": "off_or_unobserved",
    }
    assert (
        exp._command_config("llama-server -m /models/Qwen3.6-35B-A3B/model.gguf")["model_id"]
        == "unsloth/Qwen3.6-35B-A3B-GGUF"
    )
    with pytest.raises(ValueError, match="unparseable ps line"):
        exp.parse_ps_line("not a ps row")

    original_read_bytes = exp._read_bytes
    monkeypatch.setattr(
        exp,
        "_read_bytes",
        lambda _path: (
            b"CARNOT_ARC_TRAJECTORY_SUPERVISOR=1\0CARNOT_ARC_INDUCE_TOOL_LOOP=selfparse\0",
            None,
        ),
    )
    assert exp._read_proc_environ(123) == [
        "CARNOT_ARC_TRAJECTORY_SUPERVISOR=1",
        "CARNOT_ARC_INDUCE_TOOL_LOOP=selfparse",
    ]
    observed = exp._overlay_env_config(
        {
            "pid": 123,
            "observed_configuration": {
                "budget": None,
                "games": [],
                "model_id": None,
                "policy": None,
                "supervisor_state": "unobserved",
                "tool_loop_state": "off_or_unobserved",
            },
        }
    )
    assert observed["observed_configuration"]["supervisor_state"] == "applied"
    assert observed["observed_configuration"]["tool_loop_state"] == "selfparse"
    assert observed["observed_env_keys"] == [
        "CARNOT_ARC_INDUCE_TOOL_LOOP",
        "CARNOT_ARC_TRAJECTORY_SUPERVISOR",
    ]
    monkeypatch.setattr(exp, "_read_bytes", lambda _path: (b"", "FileNotFoundError"))
    assert exp._read_proc_environ(123) == []

    class Completed:
        stdout = (
            "PID PPID STARTED STAT ELAPSED COMMAND\n"
            "bad arc_leaderboard_eval line\n"
            "375005 374980 Mon Aug 31 20:01:12 2026 SNl 10:20:13 "
            "/repo/scripts/arc_leaderboard_eval.py --policy e3 --only ls20 --budget 2500\n"
            "1 0 Mon Aug 31 20:01:12 2026 S 00:00:01 unrelated\n"
        )

    monkeypatch.setattr(exp.subprocess, "run", lambda *_args, **_kwargs: Completed())
    monkeypatch.setattr(exp, "_read_proc_environ", lambda _pid: [])
    sampled = exp.sample_process_observations()
    assert len(sampled) == 1
    assert sampled[0]["pid"] == 375005
    monkeypatch.setattr(exp, "_read_bytes", original_read_bytes)

    artifact = exp.build_artifact(
        run_date="20260901",
        duration_s=0.1,
        source_paths=paths,
        process_observations=[],
    )

    def errors_with(**changes: Any) -> set[str]:
        changed = deepcopy(artifact)
        changed.update(changes)
        changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
        return set(exp.validate_artifact(changed))

    missing_field = deepcopy(artifact)
    missing_field.pop("rows")
    missing_field["field_principles"].pop("rows")
    missing_field["reproducibility_checksum"] = exp.reproducibility_checksum(missing_field)
    assert "required artifact fields are missing" in exp.validate_artifact(missing_field)
    assert "schema mismatch" in errors_with(schema="wrong")
    assert "inference substrate mismatch" in errors_with(inference_substrate="wrong")
    assert "verifier_is_oracle must be false" in errors_with(verifier_is_oracle=True)
    assert "verdict class is outside the closed set" in errors_with(verdict_class="bad")
    assert "honest verdict lacks complete_ terminal prefix" in errors_with(
        honest_verdict="blocked: old style"
    )
    duplicate = deepcopy(artifact["rows"][0])
    assert "duplicate row identities present" in errors_with(rows=[duplicate, duplicate])
    assert "complete inventory verdict_class mismatch" in errors_with(verdict_class="null")
    assert "inventory complete score mismatch" in errors_with(arc_inventory_complete_score=0)
    assert "complete inventory lacks rows" in errors_with(rows=[])
    assert "status mismatch" in errors_with(status="weird")

    blocked = exp.build_artifact(
        run_date="20260901",
        duration_s=0.1,
        source_paths={**paths, "leaderboard_dup": paths["leaderboard"]},
        process_observations=[],
    )

    def blocked_errors_with(**changes: Any) -> set[str]:
        changed = deepcopy(blocked)
        changed.update(changes)
        changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
        return set(exp.validate_artifact(changed))

    assert "blocked verdict_class mismatch" in blocked_errors_with(verdict_class="partial")
    assert "blocked artifact emitted rows" in blocked_errors_with(rows=[duplicate])
    assert "blocked artifact marked complete" in blocked_errors_with(arc_inventory_complete_score=1)
    assert "blocked artifact lacks failed check" in blocked_errors_with(gate_check_summary={})
