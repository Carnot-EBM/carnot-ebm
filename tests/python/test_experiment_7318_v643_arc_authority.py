"""Exp7318 owner-issued ARC authority and scored-path conformance tests.

Spec refs: REQ-ARC-WMTE-7318 and SCENARIO-ARC-WMTE-7318-*.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from carnot import experiment_7318_v643_arc_authority as mod


pytestmark = pytest.mark.memory_watchdog_skip


def test_req_7318_owner_issued_denial_matrix_uses_current_kernel_lease(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7318-DENIAL-MATRIX: all grants fail closed."""

    receipt = mod.run_authority_denial_panel(tmp_path / "authority")
    rows = {row["case"]: row for row in receipt["rows"]}
    assert receipt["issuer_kind"] == "current_kernel_backed_gpu_lease_owner"
    assert receipt["issuer_secret_exported"] is False
    assert rows["valid"]["allowed"] is True
    assert rows["valid"]["reason"] == "allowed"
    assert {
        case: rows[case]["reason"]
        for case in (
            "missing",
            "expired",
            "tampered",
            "wrong_owner",
            "wrong_game",
            "wrong_model",
            "replayed",
        )
    } == {
        "missing": "missing_authority",
        "expired": "authority_expired",
        "tampered": "authority_hash_mismatch",
        "wrong_owner": "wrong_owner",
        "wrong_game": "wrong_game",
        "wrong_model": "wrong_model_identity",
        "replayed": "authority_replayed",
    }
    assert all(row["denied_before_model_load"] for case, row in rows.items() if case != "valid")


def test_req_7318_real_launcher_child_proposer_and_caller_handoff(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7318-OWNER-ISSUE-AND-HANDOFF: one hash reaches every hop."""

    receipt = mod.run_owned_child_probe(tmp_path / "child-probe")
    assert receipt["child_exit_code"] == 0
    assert receipt["owned_child_survives"] is False
    assert receipt["model_load_count"] == 0
    assert receipt["generation_count"] == 0
    assert receipt["child_receipt"]["preflight_allowed"] is True
    hashes = [row["authority_hash"] for row in receipt["authority_handoff_rows"]]
    assert len(hashes) == 5
    assert len(set(hashes)) == 1
    assert [row["hop"] for row in receipt["authority_handoff_rows"]] == [
        "gpu_lease_issuer",
        "child_environment",
        "episode_environment",
        "E3AgentPolicy.proposer",
        "arc_eval_provenance_caller",
    ]
    assert all(row["required_fields_present"] for row in receipt["authority_handoff_rows"])
    assert all(not row["field_dropped"] for row in receipt["authority_handoff_rows"])
    assert receipt["former_field_drop_location"].endswith(
        "run_child_with_lease:child_environment_before_subprocess"
    )
    assert receipt["gpu_resource_ownership"]["lease_id"].startswith("lease:")
    assert receipt["episode_authority"]["episode_scope"]["game"] == "r11l"


def test_req_7318_bundle_validation_and_direct_child_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-7318: bundle shape, scope, selection, and child role fail closed."""

    from carnot import gpu_lease_phase_journal as lease_api

    lease = mod._acquire_lease(tmp_path)
    try:
        authority = mod._issue(lease, tmp_path, suffix="direct-child")
        environment = mod.authority_bundle_environment(
            {lease_api.ARC_AUTHORITY_ENV: "stale"}, [authority]
        )
        assert lease_api.ARC_AUTHORITY_ENV not in environment
        schedule = [{"episode_id": "r11l:direct_selfparse", "game": "r11l"}]
        assert mod.validate_authority_bundle_before_model_load(
            environment, schedule, mod.MODEL_IDENTITY
        )["allowed"]
        selected, selected_environment = mod.select_episode_authority(
            environment, schedule[0], mod.MODEL_IDENTITY
        )
        assert selected["authority_hash"] == authority["authority_hash"]
        assert json.loads(selected_environment[lease_api.ARC_AUTHORITY_ENV]) == authority
        proposer = SimpleNamespace()
        assert mod.attach_episode_authority(proposer, authority) == authority

        assert (
            mod.validate_authority_bundle_before_model_load(
                {lease_api.ARC_AUTHORITY_BUNDLE_ENV: "{}"}, schedule, mod.MODEL_IDENTITY
            )["reason"]
            == "missing_authority_bundle"
        )
        assert (
            mod.validate_authority_bundle_before_model_load(
                environment, [*schedule, *schedule], mod.MODEL_IDENTITY
            )["reason"]
            == "authority_schedule_mismatch"
        )
        wrong = [{"episode_id": "r11l:direct_selfparse", "game": "other"}]
        assert (
            mod.validate_authority_bundle_before_model_load(environment, wrong, mod.MODEL_IDENTITY)[
                "reason"
            ]
            == "wrong_game"
        )
        with pytest.raises(RuntimeError, match="missing_authority"):
            mod.select_episode_authority(
                environment,
                {"episode_id": "r11l:absent", "game": "r11l"},
                mod.MODEL_IDENTITY,
            )

        monkeypatch.delenv(lease_api.ARC_AUTHORITY_BUNDLE_ENV, raising=False)
        denied_output = tmp_path / "denied-child.json"
        assert (
            mod.main(
                [
                    "--date",
                    "20260915",
                    "--role",
                    "authority-child",
                    "--authority-output",
                    str(denied_output),
                ]
            )
            == 3
        )
        assert json.loads(denied_output.read_text())["reason"] == "missing_authority_bundle"

        monkeypatch.setenv(
            lease_api.ARC_AUTHORITY_BUNDLE_ENV,
            environment[lease_api.ARC_AUTHORITY_BUNDLE_ENV],
        )
        output = tmp_path / "direct-child.json"
        assert (
            mod.main(
                [
                    "--date",
                    "20260915",
                    "--role",
                    "authority-child",
                    "--authority-output",
                    str(output),
                ]
            )
            == 0
        )
        assert json.loads(output.read_text())["preflight_allowed"] is True
    finally:
        mod._finish_lease(lease)


def test_req_7318_missing_before_launch_and_expiry_during_call(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7318-EXPIRY-DURING-CALL: preflight never refreshes a grant."""

    receipt = mod.run_expiry_controls(tmp_path / "expiry")
    assert receipt == {
        "missing_before_launch": {
            "allowed": False,
            "reason": "missing_authority_bundle",
            "model_load_count": 0,
        },
        "valid_before_call": {"allowed": True, "reason": "allowed"},
        "expired_after_call": {
            "allowed": False,
            "reason": "authority_expired",
            "provenance_accepted": False,
        },
    }


def test_req_7318_real_live_child_rejects_missing_authority_before_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ARC-WMTE-7318-EXPIRY-DURING-CALL: launch fails before model setup."""

    from carnot import experiment_7263_v639_arc_live as live
    from carnot import gpu_lease_phase_journal as lease_api

    schedule = tmp_path / "schedule.json"
    schedule.write_text(
        json.dumps({"rows": [{"episode_id": "r11l:probe", "game": "r11l"}]}),
        encoding="utf-8",
    )
    monkeypatch.delenv(lease_api.ARC_AUTHORITY_BUNDLE_ENV, raising=False)
    args = SimpleNamespace(
        raw_dir=tmp_path / "raw",
        schedule_path=schedule,
        checkpoint_path=tmp_path / "checkpoint.json",
        session_path=tmp_path / "session.json",
        model_path=tmp_path / "must-not-load.gguf",
        model_hash="sha256:" + "1" * 64,
    )
    with pytest.raises(RuntimeError, match="missing_authority_bundle"):
        live.run_live_session(args)


def test_req_7318_scored_provenance_seam_consumes_valid_grant(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7318: the scored provenance module consumes, then rejects replay."""

    from carnot.agentic.arc_eval_provenance import _consume_live_episode_authority

    lease = mod._acquire_lease(tmp_path)
    try:
        authority = mod._issue(lease, tmp_path, suffix="scored-caller")
        checked_at = datetime.fromisoformat(authority["lease_checked_at"]).isoformat()
        policy = SimpleNamespace(short="r11l")
        _consume_live_episode_authority(
            policy,
            authority,
            model_repository=mod.MODEL_IDENTITY["hf_id"],
            model_path=mod.MODEL_IDENTITY["model_path"],
            model_hash=mod.MODEL_IDENTITY["model_hash"],
            lease_checked_at=checked_at,
        )
        with pytest.raises(ValueError, match="authority_replayed"):
            _consume_live_episode_authority(
                policy,
                authority,
                model_repository=mod.MODEL_IDENTITY["hf_id"],
                model_path=mod.MODEL_IDENTITY["model_path"],
                model_hash=mod.MODEL_IDENTITY["model_hash"],
                lease_checked_at=checked_at,
            )
    finally:
        mod._finish_lease(lease)


def test_req_7318_actual_e3_scripted_tool_to_action_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ARC-WMTE-7318-E3-CONFORMANCE: real policy carries result to action."""

    monkeypatch.setenv("CARNOT_ARC_INDUCE_THINK", "preexisting")
    monkeypatch.setenv("CARNOT_ARC_DISABLE_INDUCTION", "1")
    receipt = mod.run_e3_conformance_panel(tmp_path / "e3")
    rows = {row["case"]: row for row in receipt["rows"]}
    success = rows["tool_result_to_action"]
    assert success["policy_class"] == "E3AgentPolicy"
    assert success["http_completion_count"] == 2
    assert success["successful_tool_result_count"] == 2
    assert success["later_request_contains_first_tool_result"] is True
    assert success["plan_installed"] is True
    assert success["subsequent_environment_action"] is True
    assert success["passed"] is True
    assert rows["no_tool_needed"]["http_completion_count"] == 1
    assert rows["no_tool_needed"]["passed"] is True
    assert rows["malformed_tool"]["tool_dispatch_count"] == 0
    assert rows["malformed_tool"]["passed"] is True
    assert receipt["counts_as_current_model_invocation"] is False
    assert Path(receipt["sidecar_path"]).is_file()
    assert receipt["sidecar_sha256"].startswith("sha256:")
    assert os.environ["CARNOT_ARC_INDUCE_THINK"] == "preexisting"
    assert os.environ["CARNOT_ARC_DISABLE_INDUCTION"] == "1"


def test_req_7318_terminal_artifact_is_fail_closed_and_checksum_bound(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7318-TERMINAL-RECEIPT: artifact scalars and gates reconcile."""

    artifact = mod.build_fixture_artifact(tmp_path / "artifact")
    assert mod.validate_artifact(artifact) == []
    assert artifact["schema"] == "carnot.experiment_7318.v643.arc_authority.v1"
    assert artifact["status"] == "complete"
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert set(artifact["invocation_counts"].values()) == {0}
    assert artifact["inference_substrate"] == "cpu_exact_solver_or_simulator"
    assert artifact["inference_substrate_class"] == "cpu_exact_solver_or_simulator"
    assert artifact["execution_venue"] == "host"
    assert artifact["verdict_class"] == "null"
    assert artifact["arc_authority_ready_score"] == 1
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["gate_check_summary"]["first_failure"] is None
    assert set(artifact) <= set(artifact["field_principles"])
    assert (
        artifact["source_artifact_hashes"]["results/experiment_7305_v642_arc_selfparse.json"][
            "authorizes_readiness"
        ]
        is False
    )
    scripted_sources = [
        row
        for row in artifact["source_artifact_hashes"].values()
        if row["role"] == "scripted_model_event_conformance_fixture"
    ]
    assert len(scripted_sources) == 1
    assert scripted_sources[0]["counts_as_current_model_invocation"] is False
    assert all(
        {
            "unit",
            "arm",
            "metrics",
            "costs",
            "failures",
            "abstentions",
            "censored",
        }
        <= set(row)
        for row in artifact["rows"]
    )

    bad = deepcopy(artifact)
    bad["model_invoked"] = True
    assert "current model declaration mismatch" in mod.validate_artifact(bad)
    bad = deepcopy(artifact)
    bad["denial_control_rows"][0]["allowed"] = True
    assert "denial controls mismatch" in mod.validate_artifact(bad)
    bad = deepcopy(artifact)
    bad["reproducibility_checksum"] = "sha256:" + "0" * 64
    assert "reproducibility checksum mismatch" in mod.validate_artifact(bad)

    mutations = [
        (lambda row: row.pop("schema"), "missing required field: schema"),
        (lambda row: row.__setitem__("schema", "wrong"), "schema or experiment identity mismatch"),
        (lambda row: row.__setitem__("run_date", "wrong"), "milestone or run date mismatch"),
        (lambda row: row.__setitem__("status", "partial"), "terminal status mismatch"),
        (lambda row: row.__setitem__("verdict_class", "positive"), "verdict class mismatch"),
        (
            lambda row: row.__setitem__("gate_check_summary", {}),
            "gate check summary mismatch",
        ),
        (
            lambda row: row.__setitem__("arc_authority_ready_score", 0),
            "complete readiness mismatch",
        ),
        (
            lambda row: row.__setitem__("honest_verdict", "wrong"),
            "complete verdict prefix mismatch",
        ),
        (lambda row: row.__setitem__("field_principles", {}), "field principles mismatch"),
    ]
    for mutate, expected_error in mutations:
        bad = deepcopy(artifact)
        mutate(bad)
        assert expected_error in mod.validate_artifact(bad)

    unsafe = deepcopy(artifact)
    unsafe.update(
        {
            "status": "disqualified",
            "verdict_class": "disqualified",
            "arc_authority_ready_score": 1,
        }
    )
    assert "unsafe readiness on non-ready artifact" in mod.validate_artifact(unsafe)


def test_req_7318_exclusion_manifest_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-ARC-WMTE-7318: malformed, unreadable, or matching exclusions reject readiness."""

    original = mod.yaml.safe_load
    monkeypatch.setattr(
        mod.yaml,
        "safe_load",
        lambda text: (_ for _ in ()).throw(mod.yaml.YAMLError("injected")),
    )
    assert mod._manifest_rejects_current_task() is True
    monkeypatch.setattr(mod.yaml, "safe_load", lambda text: None)
    assert mod._manifest_rejects_current_task() is True
    monkeypatch.setattr(mod.yaml, "safe_load", lambda text: {"retired": "wrong"})
    assert mod._manifest_rejects_current_task() is True
    monkeypatch.setattr(
        mod.yaml,
        "safe_load",
        lambda text: {"retired": [{"experiment_id": 7318}]},
    )
    assert mod._manifest_rejects_current_task() is True
    monkeypatch.setattr(mod.yaml, "safe_load", original)
    assert mod._manifest_rejects_current_task() is False
    health = mod._repository_health(
        {
            "passed": False,
            "command": ".venv/bin/pytest tests/python -q",
            "exit_code": -15,
            "timed_out": False,
            "log_path": "results/raw/exp7318/full.log",
            "log_sha256": "sha256:" + "4" * 64,
            "output_tail": "repository-wide failures observed",
        }
    )
    assert health["current_full_suite_passed"] is False
    assert health["historical_failures"][-1]["source_experiment"] == mod.EXPERIMENT_ID


def test_req_7318_blocked_artifact_keeps_exact_gate_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-7318: an unavailable issuer is terminal blocked, not partial."""

    artifact = mod.build_blocked_fixture_artifact(tmp_path / "blocked")
    assert mod.validate_artifact(artifact) == []
    assert artifact["status"] == "blocked"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["arc_authority_ready_score"] == 0
    assert artifact["gate_check_summary"]["first_failure"] == {
        "upstream": "gpu_lease_phase_journal.GpuLease",
        "check": "current_authorized_issuer",
        "artifact_field": "issuer_available",
        "expected": True,
        "observed": False,
    }
    bad = deepcopy(artifact)
    bad["honest_verdict"] = "complete_wrong"
    assert "blocked verdict mismatch" in mod.validate_artifact(bad)

    monkeypatch.setattr(
        mod,
        "_acceptance_gates",
        lambda *args, **kwargs: [
            {
                "check": "forced_affected_failure",
                "expected": True,
                "observed": False,
                "passed": False,
                "principle": "A failing affected check disqualifies readiness.",
            }
        ],
    )
    disqualified = mod.build_fixture_artifact(tmp_path / "disqualified")
    assert disqualified["status"] == "disqualified"
    assert disqualified["arc_authority_ready_score"] == 0
    assert disqualified["honest_verdict"].startswith("complete_disqualified")


def test_req_7318_cli_and_atomic_write(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """REQ-ARC-WMTE-7318: the thin CLI validates dates and writes terminal JSON atomically."""

    args = mod.parse_args(["--date", "20260915", "--output", str(tmp_path / "result.json")])
    assert args.date == "20260915"
    with pytest.raises(SystemExit):
        mod.parse_args(["--date", "20260914"])
    value = {"ordinary": {"nested": True}}
    target = tmp_path / "atomic.json"
    mod.atomic_write(target, value)
    assert json.loads(target.read_text()) == value
    assert mod.main(["--date", "20260915", "--fixture-only", "--output", str(target)]) == 0
    assert json.loads(target.read_text())["experiment_id"] == "exp7318-arc-authority"
    assert "phase" in capsys.readouterr().out


def test_req_7318_atomic_write_cleans_failed_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-7318: a failed replace cannot leave a success-shaped temporary."""

    def reject_replace(source: Path, destination: Path) -> None:
        del source, destination
        raise OSError("injected replace failure")

    monkeypatch.setattr(mod.os, "replace", reject_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        mod.atomic_write(tmp_path / "never-published.json", {"status": "complete"})
    assert list(tmp_path.iterdir()) == []
