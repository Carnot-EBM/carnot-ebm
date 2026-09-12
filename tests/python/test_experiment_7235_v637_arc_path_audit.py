"""Tests for the independent V637 scored-path receipt audit.

Spec refs: REQ-ARC-WMTE-7235 and SCENARIO-ARC-WMTE-7235-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import runpy

import numpy as np
import pytest

from carnot import experiment_7235_v637_arc_path_audit as exp
from carnot.agentic.arc_executable_world_model import Transition

pytestmark = pytest.mark.memory_watchdog_skip


def _complete_inputs(tmp_path: Path) -> dict[str, object]:
    return {
        "run_date": exp.RUN_DATE,
        "duration_s": 1.25,
        "started_at_utc": "2026-09-12T12:00:00+00:00",
        "ended_at_utc": "2026-09-12T12:00:02+00:00",
        "checks": [
            exp.gate_check("upstream", str(exp.UPSTREAM_PATH), "status", "complete", "complete")
        ],
        "source_hashes": {str(exp.UPSTREAM_PATH): "sha256:" + "a" * 64},
        "audit_rows": [
            {
                "row_type": "future_transition",
                "unit_id": "session:engine:17",
                "cluster_id": "session",
                "arm": "recorded_engine",
                "seed": exp.RANDOM_SEED,
                "transition_index": 17,
                "source_sha256": "sha256:" + "b" * 64,
                "engine_sha256": "sha256:" + "c" * 64,
                "useful_prediction": False,
                "policy_used_engine": False,
                "value": 0.0,
                "error": None,
                "abstention": False,
            }
        ],
        "backend_dispositions": [
            {
                "backend": "llamacpp",
                "upstream_disposition": "complete",
                "actual_backend_identity": "LocalGGUFProposer_llama.cpp",
                "audit_disposition": "complete_replay",
            },
            {
                "backend": "vllm",
                "upstream_disposition": "complete",
                "actual_backend_identity": None,
                "audit_disposition": "complete_no_engine_rows",
            },
        ],
        "sidecars": {
            "historical_source_receipts": {
                "path": str(tmp_path / "historical.json"),
                "sha256": "sha256:" + "d" * 64,
            },
            "synthetic_negative_receipts": {
                "path": str(tmp_path / "negative.json"),
                "sha256": "sha256:" + "e" * 64,
            },
        },
        "reducer_replay": {"matches_upstream": True},
        "factory_receipt": {
            "factory": "make_carnot_agent",
            "policy_class": "E3AgentPolicy",
            "adapter_disabled": True,
        },
        "local_scored_differences": {"gateway": "public_local_not_competition"},
        "sample_size": {
            "planned_independent_units": 1,
            "attempted_units": 1,
            "completed_units": 1,
            "censored_units": 0,
            "cluster_count": 1,
            "stopping_rule": "all authenticated archived engines once",
        },
    }


def test_req_7235_spec_and_frozen_identity() -> None:
    """REQ-ARC-WMTE-7235 fixes the exact upstream, date, and no-LLM contract."""

    spec = (exp.REPO_ROOT / exp.SPEC_PATH).read_text(encoding="utf-8")
    assert "## REQ-ARC-WMTE-7235:" in spec
    assert "SCENARIO-ARC-WMTE-7235-NEGATIVE-RECEIPTS" in spec
    assert exp.EXPERIMENT_ID == 7235
    assert exp.MILESTONE == "2026.09.637"
    assert exp.RUN_DATE == "20260912"
    assert exp.UPSTREAM_PATH == Path("results/experiment_7234_v637_arc_scored_dryrun.json")
    assert exp.MODEL_SPECS == []


def test_scenario_7235_upstream_block_rejects_quarantine_before_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ARC-WMTE-7235-UPSTREAM-BLOCK never consumes a quarantined score."""

    upstream = tmp_path / "upstream.json"
    upstream.write_text(
        json.dumps({"status": "complete", "scored_dryrun_complete_score": 1}),
        encoding="utf-8",
    )
    monkeypatch.setattr(exp, "is_quarantined", lambda _payload: True)
    payload, checks = exp.authenticate_upstream(upstream)
    assert payload is None
    assert checks[-1]["check"] == "upstream_not_quarantined"
    assert checks[-1]["field"] == "flagged_adversarial"
    assert checks[-1]["observed_value"] is True
    assert not any(row["field"] == "scored_dryrun_complete_score" for row in checks)


def test_upstream_and_path_authentication_defensive_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ARC-WMTE-7235-UPSTREAM-BLOCK names absent and malformed evidence."""

    absent, checks = exp.authenticate_upstream(tmp_path / "absent.json")
    assert absent is None
    assert checks[0]["check"] == "exact_upstream_present"

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    value, checks = exp.authenticate_upstream(malformed)
    assert value is None
    assert checks[-1]["observed_value"] == "JSONDecodeError"

    sequence = tmp_path / "sequence.json"
    sequence.write_text("[]", encoding="utf-8")
    value, checks = exp.authenticate_upstream(sequence)
    assert value is None
    assert checks[-1]["observed_value"] == "list"

    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        json.dumps({"schema": exp.producer.SCHEMA, "experiment_id": exp.producer.EXPERIMENT_ID}),
        encoding="utf-8",
    )
    monkeypatch.setattr(exp, "is_quarantined", lambda _payload: False)
    value, checks = exp.authenticate_upstream(candidate)
    assert value is not None
    assert all(row["passed"] for row in checks)
    candidate.write_text(json.dumps({"schema": "wrong", "experiment_id": 0}), encoding="utf-8")
    value, checks = exp.authenticate_upstream(candidate)
    assert value is None
    assert [row["passed"] for row in checks[-2:]] == [False, False]

    source = tmp_path / "source.txt"
    source.write_text("source", encoding="utf-8")
    digest = exp.sha256_file(source)
    assert exp._recorded_hash({}, source, tmp_path) is None
    assert (
        exp._recorded_hash(
            {"source_artifact_hashes": {str(source): digest}}, source, tmp_path / "other"
        )
        == digest
    )
    actual, check = exp._authenticate_path(tmp_path, {}, Path("absent.txt"))
    assert actual == ""
    assert check["observed_value"] == "absent"


def test_scenario_7235_authentic_replay_joins_real_receipts() -> None:
    """SCENARIO-ARC-WMTE-7235-AUTHENTIC-REPLAY exercises the reducer without admitting its source."""

    upstream = json.loads((exp.REPO_ROOT / exp.UPSTREAM_PATH).read_text(encoding="utf-8"))
    assert exp.is_quarantined(upstream) is True
    # This direct helper call covers the reducer as code. The terminal runner
    # must not call it because authenticate_upstream rejects this artifact.
    evidence = exp.replay_saved_receipts(exp.REPO_ROOT, upstream)
    assert evidence["reducer_replay"]["matches_upstream"] is True
    assert evidence["factory_receipt"]["factory"] == "make_carnot_agent"
    assert evidence["factory_receipt"]["policy_class"] == "E3AgentPolicy"
    assert evidence["factory_receipt"]["adapter_disabled"] is True
    assert evidence["game_source_access_receipt"]["game_source_read"] is False
    assert [row["request_sequence"] for row in evidence["engine_receipts"]] == [3, 16]
    assert {row["induction_action_index"] for row in evidence["engine_receipts"]} == {26}
    assert all(row["request_hash_matches"] for row in evidence["engine_receipts"])
    assert all(row["engine_hash_matches"] for row in evidence["engine_receipts"])
    assert all(row["transition_hash_matches"] for row in evidence["engine_receipts"])
    assert all(row["policy_used_engine"] is False for row in evidence["engine_receipts"])


def test_scenario_7235_future_controls_use_current_verifier() -> None:
    """SCENARIO-ARC-WMTE-7235-FUTURE-CONTROLS keeps three arms per held row."""

    grids = [np.array([[0, 0], [0, 0]]), np.array([[1, 0], [0, 0]])]
    transitions = [
        Transition(grids[0], 1, {"x": 0}, grids[1], 0, 0),
        Transition(grids[1], 2, {"x": 1}, grids[0], 0, 0),
    ]

    def engine(grid: np.ndarray, action: int, data: object) -> np.ndarray:
        del data
        out = grid.copy()
        out[0, 0] = 1 if action == 1 else 0
        return out

    rows = exp.score_future_controls(
        engine=engine,
        transitions=transitions,
        transition_indices=[17, 18],
        engine_sha256="sha256:" + "f" * 64,
        transition_source_sha256="sha256:" + "1" * 64,
        session_hash="sha256:" + "2" * 64,
        seed=exp.RANDOM_SEED,
    )
    assert len(rows) == 6
    assert {row["arm"] for row in rows} == {
        "recorded_engine",
        "noop_control",
        "shuffled_action_control",
    }
    assert {row["transition_index"] for row in rows} == {17, 18}
    assert all(row["cluster_id"] == "sha256:" + "2" * 64 for row in rows)
    assert all(row["policy_used_engine"] is False for row in rows)
    assert all(row["error"] is None for row in rows)
    assert (
        sum(row["exact_transition_correct"] for row in rows if row["arm"] == "recorded_engine") == 2
    )


def test_parser_and_engine_import_defensive_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-7235 rejects parser aliases and non-engine modules."""

    monkeypatch.setattr(
        exp,
        "parse_xml_tool_calls",
        lambda _text: (
            [
                {"function": {"name": "other", "arguments": "{}"}},
                {"function": {"name": "run_engine_on_transitions", "arguments": "{"}},
                {
                    "function": {
                        "name": "run_engine_on_transitions",
                        "arguments": {"code": "def engine(grid, action, data): return grid"},
                    }
                },
            ],
            3,
            1,
        ),
    )
    response = {
        "choices": [
            {"content": "plain"},
            {"content": "<tool_call run_engine_on_transitions"},
        ]
    }
    assert exp._engine_calls(response) == ["def engine(grid, action, data): return grid"]
    assert exp._json_strings(3) == []

    no_engine = tmp_path / "no_engine.py"
    no_engine.write_text("value = 1\n", encoding="utf-8")
    with pytest.raises(TypeError, match="engine_not_callable"):
        exp._load_engine(no_engine)
    monkeypatch.setattr(exp.importlib.util, "spec_from_file_location", lambda *_args: None)
    with pytest.raises(ImportError, match="engine_import_spec_unavailable"):
        exp._load_engine(no_engine)


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ({"engine_sha256": None}, "parsed_tool_without_engine"),
        ({"engine_functionally_identity": True}, "identity_engine"),
        ({"policy_used_engine": False}, "engine_never_consumed"),
        ({"session_hash": "sha256:" + "0" * 64}, "stale_session_hash"),
    ],
)
def test_scenario_7235_negative_receipts_fail_closed(
    mutation: dict[str, object], reason: str
) -> None:
    """SCENARIO-ARC-WMTE-7235-NEGATIVE-RECEIPTS rejects each named false proof."""

    expected = {
        "session_hash": "sha256:" + "1" * 64,
        "request_sha256": "sha256:" + "2" * 64,
        "engine_sha256": "sha256:" + "3" * 64,
        "transition_sha256": "sha256:" + "4" * 64,
        "induction_action_index": 26,
    }
    receipt = {
        **expected,
        "parsed_tool": True,
        "engine_functionally_identity": False,
        "policy_used_engine": True,
    }
    receipt.update(mutation)
    result = exp.evaluate_path_receipt(receipt, expected)
    assert result["admitted"] is False
    assert reason in result["rejection_reasons"]


def test_receipt_rejects_unparsed_and_all_join_mismatches() -> None:
    """SCENARIO-ARC-WMTE-7235-NEGATIVE-RECEIPTS preserves every failed join."""

    expected = {
        "session_hash": "session",
        "request_sha256": "request",
        "engine_sha256": "engine",
        "transition_sha256": "transition",
        "induction_action_index": 26,
    }
    result = exp.evaluate_path_receipt(
        {
            "parsed_tool": False,
            "session_hash": "bad",
            "request_sha256": "bad",
            "engine_sha256": "bad",
            "transition_sha256": "bad",
            "induction_action_index": 27,
            "engine_functionally_identity": False,
            "policy_used_engine": True,
        },
        expected,
    )
    assert result["rejection_reasons"] == [
        "tool_not_parsed",
        "stale_session_hash",
        "request_hash_mismatch",
        "engine_hash_mismatch",
        "transition_hash_mismatch",
        "action_index_mismatch",
    ]


def test_replay_fails_closed_when_saved_receipt_is_missing(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7235-AUTHENTIC-REPLAY requires every fixed raw receipt."""

    evidence = exp.replay_saved_receipts(tmp_path, {"source_artifact_hashes": {}})
    assert evidence["error"] == "required_saved_receipt_failed"
    assert all(row["passed"] is False for row in evidence["checks"])


def test_replay_skips_individually_invalid_optional_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-ARC-WMTE-7235 keeps malformed optional rows out of replay evidence."""

    upstream = json.loads((exp.REPO_ROOT / exp.UPSTREAM_PATH).read_text(encoding="utf-8"))
    upstream["per_game_results"].append("not-a-backend-row")
    original_authenticate = exp._authenticate_path
    original_read_json = exp._read_json

    def selective_authenticate(
        root: Path,
        source: dict[str, object],
        path: Path,
        expected_hash: str | None = None,
    ) -> tuple[str, dict[str, object]]:
        if "064005" in str(path) and path.name == "engine.py":
            return "", exp.gate_check(
                "source_artifact_authentic",
                str(path),
                "sha256",
                expected_hash,
                "forced-invalid-engine",
                False,
            )
        if path.name == "000_response.json":
            return "", exp.gate_check(
                "source_artifact_authentic",
                str(path),
                "sha256",
                expected_hash,
                "forced-invalid-response",
                False,
            )
        return original_authenticate(root, source, path, expected_hash=expected_hash)

    def selective_read_json(path: Path) -> object:
        if path.name == "001_response.json":
            return []
        return original_read_json(path)

    monkeypatch.setattr(exp, "_authenticate_path", selective_authenticate)
    monkeypatch.setattr(exp, "_read_json", selective_read_json)
    evidence = exp.replay_saved_receipts(exp.REPO_ROOT, upstream)
    assert len(evidence["engine_receipts"]) == 1
    assert any(row["passed"] is False for row in evidence["checks"])
    assert all(isinstance(row["backend"], str) for row in evidence["backend_dispositions"])


def test_scenario_7235_complete_null_artifact(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7235-NONCLAIM completes the audit without claiming efficacy."""

    artifact = exp.build_terminal_artifact(**_complete_inputs(tmp_path))
    assert artifact["status"] == "complete"
    assert artifact["arc_path_audit_complete_score"] == 1
    assert artifact["verdict_class"] == "null"
    assert artifact["world_model_efficacy_claim"] is None
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["inference_substrate"] == "cpu_exact_solver_or_simulator"
    assert artifact["inference_substrate_class"] == "cpu_exact_solver_or_simulator"
    assert artifact["runner_receipt"]["model_request_count"] == 0
    assert artifact["transport_validation_question_retired"] is True
    assert artifact["independent_game_generalization_claimed"] is False
    assert exp.validate_artifact(artifact) == []


def test_complete_positive_branch_and_sidecar_build(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7235 keeps the positive branch gated on use and progress."""

    inputs = _complete_inputs(tmp_path)
    inputs["audit_rows"][0]["useful_prediction"] = True
    inputs["audit_rows"][0]["policy_used_engine"] = True
    inputs["backend_dispositions"][0]["progress"] = 1
    artifact = exp.build_terminal_artifact(**inputs)
    assert artifact["verdict_class"] == "positive"
    assert artifact["world_model_efficacy_claim"] == {
        "useful_policy_action": True,
        "progress": True,
    }
    assert artifact["next_mechanism_condition"].startswith("Use a fresh session")

    negative_without_source = exp._negative_receipts([], "session")
    assert negative_without_source["expected"]["session_hash"] == "session"
    assert all(not row["result"]["admitted"] for row in negative_without_source["attacks"])
    upstream = {
        "MODEL_SPECS": [{"historical": True}],
        "model_invoked": True,
        "runner_receipt": {},
    }
    sidecars = exp._write_sidecars(tmp_path / "sidecars", upstream, {})
    assert all(Path(row["path"]).is_file() for row in sidecars.values())
    assert all(exp.sha256_file(row["path"]) == row["sha256"] for row in sidecars.values())


def test_scenario_7235_blocked_artifact_has_exact_gate(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7235-UPSTREAM-BLOCK uses blocked_no_run and a terminal gate."""

    inputs = _complete_inputs(tmp_path)
    inputs.update(
        {
            "checks": [
                exp.gate_check(
                    "exact_upstream_present",
                    str(exp.UPSTREAM_PATH),
                    "path",
                    True,
                    False,
                )
            ],
            "audit_rows": [],
            "backend_dispositions": [],
            "reducer_replay": {},
            "factory_receipt": {},
            "sample_size": {
                "planned_independent_units": 1,
                "attempted_units": 0,
                "completed_units": 0,
                "censored_units": 1,
                "cluster_count": 0,
                "stopping_rule": "exact upstream required",
            },
        }
    )
    artifact = exp.build_terminal_artifact(**inputs)
    assert artifact["status"] == "blocked"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == "blocked_exact_upstream_present"
    assert artifact["inference_substrate"] == "blocked_no_run"
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["arc_path_audit_complete_score"] == 0
    assert artifact["gate_check_summary"]["upstream"] == str(exp.UPSTREAM_PATH)
    assert exp.validate_artifact(artifact) == []


def test_run_experiment_blocks_real_quarantined_input_before_replay(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7235-UPSTREAM-BLOCK rejects the real producer before replay."""

    result_path = tmp_path / "result.json"
    raw_dir = tmp_path / "raw"
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.parent.joinpath("validation_receipts.json").write_text(
        json.dumps([{"command": "focused", "exit_code": 0}]), encoding="utf-8"
    )
    artifact = exp.run_experiment(
        exp.parse_args(
            [
                "--date",
                exp.RUN_DATE,
                "--result-path",
                str(result_path),
                "--raw-dir",
                str(raw_dir),
                "--checkpoint-path",
                str(checkpoint),
            ]
        )
    )
    assert result_path.is_file()
    assert artifact == json.loads(result_path.read_text(encoding="utf-8"))
    assert artifact["arc_path_audit_complete_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == "blocked_upstream_not_quarantined"
    assert artifact["sample_size_budget"]["completed_units"] == 0
    assert artifact["sample_size_budget"]["planned_independent_units"] == 1
    assert artifact["sample_size_budget"]["censored_units"] == 1
    assert artifact["sample_size_budget"]["cluster_count"] == 0
    assert artifact["audit_rows"] == []
    assert artifact["sidecar_receipts"] == {}
    assert artifact["validation_receipts"] == [{"command": "focused", "exit_code": 0}]
    assert not raw_dir.exists() or not any(raw_dir.iterdir())
    assert exp.validate_artifact(result_path) == []


def test_validate_artifact_rejects_tampering(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7235 binds principles, rows, invocation fields, and checksum."""

    artifact = exp.build_terminal_artifact(**_complete_inputs(tmp_path))
    broken = deepcopy(artifact)
    broken["field_principles"].pop("status")
    broken["MODEL_SPECS"] = [{"model": "forbidden"}]
    broken["model_invoked"] = True
    broken["runner_receipt"]["model_request_count"] = 1
    broken["rows"][0].pop("abstention")
    errors = exp.validate_artifact(broken)
    assert "field_principles_must_cover_every_top_level_field" in errors
    assert "no_llm_contract_violated" in errors
    assert "row_schema_invalid:0" in errors
    assert "reproducibility_checksum_mismatch" in errors

    assert exp.validate_artifact(tmp_path / "absent.json") == [
        "artifact_unreadable:FileNotFoundError"
    ]
    more = deepcopy(artifact)
    more.update(
        {
            "schema": "wrong",
            "run_date": "wrong",
            "status": "running",
            "verdict_class": "wrong",
            "honest_verdict": "wrong",
            "inference_substrate": "wrong",
            "inference_substrate_class": "wrong",
            "arc_path_audit_complete_score": 1,
        }
    )
    errors = exp.validate_artifact(more)
    assert "schema_or_experiment_identity_mismatch" in errors
    assert "run_date_mismatch" in errors
    assert "status_not_terminal" in errors
    assert "verdict_class_invalid" in errors
    assert "inference_substrate_inconsistent" in errors
    assert "arc_path_audit_complete_score_inconsistent" in errors

    complete_prefix = deepcopy(artifact)
    complete_prefix["honest_verdict"] = "wrong"
    assert "complete_honest_verdict_prefix_invalid" in exp.validate_artifact(complete_prefix)
    blocked_prefix = deepcopy(artifact)
    blocked_prefix.update({"status": "blocked", "honest_verdict": "wrong"})
    assert "blocked_honest_verdict_prefix_invalid" in exp.validate_artifact(blocked_prefix)


def test_entrypoint_is_thin(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-ARC-WMTE-7235 exposes a thin runnable script."""

    monkeypatch.setattr(exp, "main", lambda argv=None: 0)
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(exp.REPO_ROOT / exp.WRAPPER_PATH), run_name="__main__")
    assert stopped.value.code == 0


def test_complete_runner_branch_and_main_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ARC-WMTE-7235-AUTHENTIC-REPLAY writes sidecars after admission."""

    fake_upstream = {
        "MODEL_SPECS": [],
        "model_invoked": False,
        "runner_receipt": {},
        "submission_configuration_diff": {"gateway": "local"},
    }
    monkeypatch.setattr(
        exp,
        "authenticate_upstream",
        lambda _path: (
            fake_upstream,
            [exp.gate_check("upstream", str(exp.UPSTREAM_PATH), "quarantine", False, False)],
        ),
    )
    monkeypatch.setattr(
        exp,
        "replay_saved_receipts",
        lambda _root, _upstream: {
            "checks": [],
            "source_hashes": {},
            "session_hash": "session",
            "reducer_replay": {"matches_upstream": True},
            "factory_receipt": {"reachable": True},
            "game_source_access_receipt": {"game_source_read": False, "adapter_used": False},
            "engine_receipts": [{"session_hash": "session"}],
            "audit_rows": [
                {
                    "row_type": "future_transition",
                    "unit_id": "unit",
                    "arm": "recorded_engine",
                    "seed": exp.RANDOM_SEED,
                    "value": None,
                    "error": "no_saved_held_future_transition",
                    "abstention": True,
                    "useful_prediction": False,
                    "policy_used_engine": False,
                }
            ],
            "backend_dispositions": [],
        },
    )
    args = exp.parse_args(
        [
            "--result-path",
            str(tmp_path / "complete.json"),
            "--raw-dir",
            str(tmp_path / "raw"),
            "--checkpoint-path",
            str(tmp_path / "checkpoint.json"),
        ]
    )
    artifact = exp.run_experiment(args)
    assert artifact["status"] == "complete"
    assert len(artifact["sidecar_receipts"]) == 2

    assert exp.main(["--validate", str(tmp_path / "complete.json")]) == 0
    monkeypatch.setattr(exp, "run_experiment", lambda _args: artifact)
    assert exp.main(["--result-path", str(tmp_path / "unused.json")]) == 0


def test_runner_raises_if_cold_validation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-7235 never publishes a result that fails its cold check."""

    monkeypatch.setattr(exp, "validate_artifact", lambda _artifact: ["forced_failure"])
    args = exp.parse_args(
        [
            "--result-path",
            str(tmp_path / "result.json"),
            "--raw-dir",
            str(tmp_path / "raw"),
            "--checkpoint-path",
            str(tmp_path / "checkpoint.json"),
        ]
    )
    with pytest.raises(ValueError, match="terminal_artifact_invalid:forced_failure"):
        exp.run_experiment(args)
