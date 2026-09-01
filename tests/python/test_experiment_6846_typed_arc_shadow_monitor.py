"""REQ-ARC-6846 typed ARC shadow monitor tests."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6846_typed_arc_shadow_monitor as exp
from carnot.agentic import arc_solve_artifact_discipline as discipline
from carnot.agentic import arc_typed_obligation_shadow_monitor as shadow
from scripts import arc_artifact_lint as arc_lint


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _source_paths(
    tmp_path: Path,
    *,
    typed_payload: dict[str, Any] | None = None,
    inventory_payload: dict[str, Any] | None = None,
    agent_source: str | None = None,
) -> dict[str, Path]:
    typed = typed_payload or _read_json(REPO_ROOT / exp.TYPED_PROGRAM_PATH)
    inventory = inventory_payload or _read_json(REPO_ROOT / exp.LIVE_INVENTORY_PATH)
    paths = {
        "spec": _write_text(
            tmp_path / exp.SPEC_PATH,
            (REPO_ROOT / exp.SPEC_PATH).read_text(encoding="utf-8"),
        ),
        "typed_program_artifact": _write_json(tmp_path / exp.TYPED_PROGRAM_PATH, typed),
        "live_inventory_artifact": _write_json(tmp_path / exp.LIVE_INVENTORY_PATH, inventory),
        "agent_source": _write_text(
            tmp_path / exp.AGENT_SOURCE_PATH,
            agent_source
            if agent_source is not None
            else (REPO_ROOT / exp.AGENT_SOURCE_PATH).read_text(encoding="utf-8"),
        ),
        "supervisor_source": _write_text(
            tmp_path / exp.SUPERVISOR_SOURCE_PATH,
            (REPO_ROOT / exp.SUPERVISOR_SOURCE_PATH).read_text(encoding="utf-8"),
        ),
        "tool_gap_source": _write_text(
            tmp_path / exp.TOOL_GAP_SOURCE_PATH,
            (REPO_ROOT / exp.TOOL_GAP_SOURCE_PATH).read_text(encoding="utf-8"),
        ),
        "shadow_monitor_source": _write_text(
            tmp_path / exp.SHADOW_MONITOR_SOURCE_PATH,
            (REPO_ROOT / exp.SHADOW_MONITOR_SOURCE_PATH).read_text(encoding="utf-8"),
        ),
        "module_source": _write_text(
            tmp_path / exp.MODULE_PATH,
            (REPO_ROOT / exp.MODULE_PATH).read_text(encoding="utf-8"),
        ),
        "wrapper_source": _write_text(
            tmp_path / exp.WRAPPER_PATH,
            (REPO_ROOT / exp.WRAPPER_PATH).read_text(encoding="utf-8"),
        ),
    }
    return paths


def test_req_6846_spec_precedes_implementation() -> None:
    """REQ-ARC-6846 declares scenarios and required artifact fields."""

    text = (REPO_ROOT / exp.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("## REQ-ARC-6846:") :]
    for marker in (
        "SCENARIO-ARC-6846-DEFAULT-OFF-NO-ACTION-MUTATION",
        "SCENARIO-ARC-6846-CANONICAL-REACHABILITY",
        "SCENARIO-ARC-6846-ATOM-MAPPING",
        "SCENARIO-ARC-6846-FAIL-CLOSED-DIAGNOSTICS",
        "SCENARIO-ARC-6846-REPLAY-DETERMINISM",
        "SCENARIO-ARC-6846-LATENCY",
        "SCENARIO-ARC-6846-MISSING-FIELDS",
        "SCENARIO-ARC-6846-ARTIFACT-NO-SOLVE",
    ):
        assert marker in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert field in exp.FIELD_PRINCIPLES


def test_scenario_6846_default_off_monitor_preserves_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-ARC-6846-DEFAULT-OFF-NO-ACTION-MUTATION checks the seam helper."""

    move = (6, {"x": 2, "y": 3})
    monkeypatch.delenv(exp.FLAG_ENV, raising=False)

    assert shadow.typed_arc_shadow_monitor_enabled() is False
    assert shadow.maybe_make_typed_arc_shadow_monitor(game_id="tu93") is None

    disabled = shadow.TypedArcShadowMonitor(enabled=False, game_id="tu93", run_label="unit")
    assert disabled.observe(move, seam="tool_gap_action") is move
    assert disabled.receipt()["row_count"] == 0

    enabled = shadow.TypedArcShadowMonitor(enabled=True, game_id="tu93", run_label="unit")
    returned = enabled.observe(move, seam="tool_gap_action", context={"source": "unit"})
    assert returned is move
    receipt = enabled.receipt()
    assert receipt["enabled"] is True
    assert receipt["row_count"] == 1
    assert receipt["rows"][0]["guard_decision"] is True
    assert receipt["rows"][0]["energy"] == 0
    assert receipt["rows"][0]["action_byte_identity"] is True
    assert receipt["rows"][0]["diagnostics"][0]["atom"] == "shadow_observation_only"
    assert shadow.action_payload(["list-action"]) == {"repr": "['list-action']"}
    assert shadow.action_payload((6, {"x": [object()]}))["data"]["x"][0].startswith("<object")
    assert shadow.action_sha256(move).startswith("sha256:")

    poisoned = shadow.TypedArcShadowMonitor(enabled=True, game_id="tu93", run_label="unit")

    def _raise(_action: Any) -> bytes:
        raise RuntimeError("poisoned")

    monkeypatch.setattr(shadow, "canonical_action_bytes", _raise)
    assert poisoned.observe(move, seam="tool_gap_action") is move
    assert poisoned.receipt()["error_count"] == 1


def test_scenario_6846_replays_real_rows_with_complete_diagnostics() -> None:
    """SCENARIO-ARC-6846-ATOM-MAPPING replays frozen rows without a solve claim."""

    artifact = exp.build_artifact(run_date="20260901", duration_s=0.02, environ={})

    assert exp.validate_artifact(artifact) == []
    assert artifact["typed_arc_shadow_monitor_ready_score"] == 1
    assert artifact["solve_claim"] is False
    assert artifact["verifier_is_oracle"] is False
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert (
        discipline.duration_floor_s(discipline.ARC_TYPED_OBLIGATION_SHADOW_REPLAY_SUBSTRATE)
        == 0.0001
    )
    assert arc_lint.lint_artifact(exp.OUTPUT_PATH, artifact) == []

    inventory = _read_json(REPO_ROOT / exp.LIVE_INVENTORY_PATH)
    rows = artifact["per_game_results"]
    assert len(rows) == len(inventory["rows"])
    assert artifact["default_off_receipt"]["default_enabled"] is False
    assert artifact["default_off_receipt"]["action_byte_identity_preserved"] is True
    assert artifact["canonical_reachability_receipt"]["reachable"] is True
    assert artifact["canonical_reachability_receipt"]["tool_gap_action_seam_hook_present"] is True
    assert artifact["atom_mapping_manifest"]["per_game_adapter_used"] is False
    assert artifact["atom_mapping_manifest"]["offline_search_path_used"] is False
    assert artifact["exact_agreement_results"]["agreement_rate"] == 1.0
    assert artifact["false_intervention_results"]["count"] == 0
    assert artifact["missed_violation_results"]["count"] == 0
    assert artifact["latency_results"]["all_bounded"] is True
    assert artifact["action_byte_identity_results"]["all_identical"] is True

    for row in rows:
        assert row["guard_decision"] is True
        assert row["energy"] == 0
        assert row["agreement"] is True
        assert row["error_type"] == "none"
        assert row["action_byte_identity"] is True
        assert row["latency_s"] <= exp.LATENCY_BOUND_S
        assert row["diagnostics"]
        assert row["exact_external_label"]["source"] == "experiment_6843_exact_receipt_facts"
        assert row["truth"]["compatible"] is True


def test_scenario_6846_replay_determinism_and_negative_external_label(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-6846-REPLAY-DETERMINISM covers stable rows and false labels."""

    inventory = _read_json(REPO_ROOT / exp.LIVE_INVENTORY_PATH)
    inventory["rows"] = deepcopy(inventory["rows"][:2])
    negative = deepcopy(inventory["rows"][0])
    negative.update(
        {
            "game": "zz99",
            "run_id": "synthetic-negative-receipt",
            "stratum_identity": "synthetic-negative-receipt|zz99",
            "row_sha256": "sha256:" + "1" * 64,
            "trajectory_receipt_complete": False,
            "supervisor_receipt_complete": False,
            "tool_gap_receipt_complete": False,
            "tool_gap_receipt_count": 0,
        }
    )
    inventory["rows"][1] = negative

    paths = _source_paths(tmp_path, inventory_payload=inventory)
    first = exp.build_artifact(
        run_date="20260901",
        duration_s=0.02,
        source_paths=paths,
        root=tmp_path,
        environ={},
    )
    second = exp.build_artifact(
        run_date="20260901",
        duration_s=999.0,
        source_paths=paths,
        root=tmp_path,
        environ={},
    )

    assert exp.validate_artifact(first) == []
    assert first["reproducibility_checksum"] == second["reproducibility_checksum"]
    assert [row["row_sha256"] for row in first["per_game_results"]] == [
        row["row_sha256"] for row in second["per_game_results"]
    ]
    negative_row = next(row for row in first["per_game_results"] if row["game"] == "zz99")
    assert negative_row["truth"]["compatible"] is False
    assert negative_row["guard_decision"] is False
    assert negative_row["agreement"] is True
    assert negative_row["error_type"] == "external_violation_guard_blocked"


def test_scenario_6846_fail_closed_preconditions(tmp_path: Path) -> None:
    """SCENARIO-ARC-6846-FAIL-CLOSED-DIAGNOSTICS reports failed gates."""

    typed = _read_json(REPO_ROOT / exp.TYPED_PROGRAM_PATH)
    typed["typed_obligation_program_ready_score"] = 0
    paths = _source_paths(tmp_path, typed_payload=typed)

    artifact = exp.build_artifact(
        run_date="20260901",
        duration_s=0.02,
        source_paths=paths,
        root=tmp_path,
        environ={},
    )

    assert exp.validate_artifact(artifact) == []
    assert artifact["status"] == "complete_blocked_typed_arc_shadow_monitor"
    assert artifact["typed_arc_shadow_monitor_ready_score"] == 0
    assert artifact["gate_check_summary"]["failed_check"] == "typed_obligation_program_ready_score"
    assert artifact["gate_check_summary"]["observed"] == 0
    assert artifact["per_game_results"] == []

    blocked_by_config = exp.build_artifact(
        run_date="20260901",
        duration_s=0.02,
        source_paths=paths,
        root=tmp_path,
        environ={exp.FLAG_ENV: "1"},
    )
    assert blocked_by_config["gate_check_summary"]["failed_check"] in {
        "typed_obligation_program_ready_score",
        "default_off_config",
    }


def test_scenario_6846_missing_fields_block_before_replay(tmp_path: Path) -> None:
    """SCENARIO-ARC-6846-MISSING-FIELDS rejects incomplete frozen rows."""

    inventory = _read_json(REPO_ROOT / exp.LIVE_INVENTORY_PATH)
    inventory["rows"] = deepcopy(inventory["rows"][:1])
    inventory["rows"][0].pop("row_sha256", None)
    inventory["rows"][0].pop("source_path", None)
    paths = _source_paths(tmp_path, inventory_payload=inventory)

    artifact = exp.build_artifact(
        run_date="20260901",
        duration_s=0.02,
        source_paths=paths,
        root=tmp_path,
        environ={},
    )

    assert artifact["status"] == "complete_blocked_typed_arc_shadow_monitor"
    assert artifact["gate_check_summary"]["failed_check"] == "terminal_replay_rows"
    assert artifact["gate_check_summary"]["observed"]["missing_fields"] == [
        "row_sha256",
        "source_path",
    ]
    assert exp.validate_artifact(artifact) == []


def test_scenario_6846_canonical_source_identity_blocks_drift(tmp_path: Path) -> None:
    """SCENARIO-ARC-6846-CANONICAL-REACHABILITY rejects off-path source text."""

    paths = _source_paths(tmp_path, agent_source="def unrelated():\n    return None\n")

    artifact = exp.build_artifact(
        run_date="20260901",
        duration_s=0.02,
        source_paths=paths,
        root=tmp_path,
        environ={},
    )

    assert artifact["typed_arc_shadow_monitor_ready_score"] == 0
    assert artifact["gate_check_summary"]["failed_check"] == "canonical_source_identity"
    assert artifact["canonical_reachability_receipt"]["reachable"] is False


def test_req_6846_defensive_helper_branches(tmp_path: Path) -> None:
    """SCENARIO-ARC-6846-FAIL-CLOSED-DIAGNOSTICS covers malformed inputs."""

    assert exp._read_bytes(tmp_path / "missing.json")[1] == "FileNotFoundError"
    assert exp._load_json(b"not-json") == {}
    assert exp._load_json(b"[]") == {}
    assert exp._relative(Path("/outside-root-file"), tmp_path) == "/outside-root-file"
    assert exp._common_root(None) == exp.REPO_ROOT
    assert exp._common_root({"missing": tmp_path / "missing"}) == exp.REPO_ROOT
    left = _write_text(tmp_path / "common" / "left.txt", "left")
    right = _write_text(tmp_path / "common" / "right.txt", "right")
    assert exp._common_root({"left": left, "right": right}) == tmp_path / "common"
    assert exp.terminal_replay_rows_receipt({"rows": "bad"})["missing_fields"] == ["rows"]
    assert exp._gate_summary([{"passed": True}])["passed"] is True
    assert exp._check("x", 1, 1)["passed"] is True
    assert exp._error_type(True, False, False) == "missed_violation"
    assert exp._error_type(False, True, False) == "false_intervention"
    assert exp._valid_run_date("20261301") is False

    malformed = {
        "rows": [
            None,
            {
                "candidates": [
                    None,
                    {"exact_check": []},
                    {
                        "candidate_id": "c",
                        "exact_check": {
                            "arc_shadow_action_guard": False,
                            "energy": 1,
                            "diagnostics": [None],
                        },
                    },
                ]
            },
        ]
    }
    assert len(exp._candidate_pool(malformed)) == 1
    assert exp.atom_mapping_manifest(malformed)["mapping_count"] == 0
    with pytest.raises(exp.TypedArcShadowMonitorError, match="typed_candidate_missing"):
        exp._candidate_for_truth([], True)
    typed = _read_json(REPO_ROOT / exp.TYPED_PROGRAM_PATH)
    assert exp.replay_rows(typed, {"rows": ["bad"]}) == []


def test_req_6846_validator_and_cli_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-ARC-6846-ARTIFACT-NO-SOLVE validates the written artifact."""

    output = tmp_path / "experiment_6846.json"
    artifact = exp.execute(REPO_ROOT, "20260901", output)
    assert output.is_file()
    assert _read_json(output) == artifact

    assert exp.main(["--date", "20260901", "--output", str(output), "--validate"]) == 0
    assert exp.main(["--date", "20260901", "--output", str(output)]) == 0
    assert '"artifact"' in capsys.readouterr().out
    assert exp.main(["--date", "2026-09-01", "--output", str(output)]) == 2
    with pytest.raises(exp.TypedArcShadowMonitorError, match="invalid_run_date"):
        exp.execute(REPO_ROOT, "20261301", output)

    broken = deepcopy(artifact)
    broken["solve_claim"] = True
    broken["reproducibility_checksum"] = exp.reproducibility_checksum(broken)
    output.write_text(json.dumps(broken, indent=2, sort_keys=True), encoding="utf-8")
    assert exp.main(["--output", str(output), "--validate"]) == 1
    assert "solve_claim must be false" in capsys.readouterr().err

    output.write_text("not-json", encoding="utf-8")
    assert exp.main(["--output", str(output), "--validate"]) == 2

    many_errors = deepcopy(artifact)
    many_errors.pop("schema")
    many_errors["inference_substrate"] = "bad"
    many_errors["duration_s"] = True
    many_errors["verifier_is_oracle"] = True
    many_errors["verdict_class"] = "bad"
    many_errors["honest_verdict"] = "bad"
    many_errors["typed_arc_shadow_monitor_ready_score"] = 2
    many_errors["reproducibility_checksum"] = "sha256:" + "0" * 64
    errors = exp.validate_artifact(many_errors)
    assert "required artifact fields are missing" in errors
    assert "field principles do not cover every top-level field" in errors
    assert "schema mismatch" in errors
    assert "inference substrate mismatch" in errors
    assert "duration_s must be a nonnegative number" in errors
    assert "verifier_is_oracle must be false" in errors
    assert "verdict class is outside the closed set" in errors
    assert "honest_verdict must start with complete_" in errors
    assert "ready score must be 0 or 1" in errors
    assert "reproducibility checksum mismatch" in errors

    ready_bad = deepcopy(artifact)
    ready_bad["gate_check_summary"] = {"passed": False}
    ready_bad["per_game_results"] = []
    ready_bad["latency_results"]["all_bounded"] = False
    ready_bad["action_byte_identity_results"]["all_identical"] = False
    ready_bad["reproducibility_checksum"] = exp.reproducibility_checksum(ready_bad)
    ready_errors = exp.validate_artifact(ready_bad)
    assert "ready artifact has failed gate" in ready_errors
    assert "ready artifact emitted no replay rows" in ready_errors
    assert "ready artifact latency is unbounded" in ready_errors
    assert "ready artifact action identity failed" in ready_errors

    typed = _read_json(REPO_ROOT / exp.TYPED_PROGRAM_PATH)
    typed["typed_obligation_program_ready_score"] = 0
    blocked = exp.build_artifact(
        run_date="20260901",
        duration_s=0.02,
        source_paths=_source_paths(tmp_path / "blocked", typed_payload=typed),
        root=tmp_path / "blocked",
        environ={},
    )
    blocked["status"] = "wrong"
    blocked["gate_check_summary"] = {"passed": True}
    blocked["per_game_results"] = [{}]
    blocked["reproducibility_checksum"] = exp.reproducibility_checksum(blocked)
    blocked_errors = exp.validate_artifact(blocked)
    assert "blocked terminal verdict mismatch" in blocked_errors
    assert "blocked artifact lacks failed gate" in blocked_errors
    assert "blocked artifact emitted rows" in blocked_errors

    monkeypatch.setattr(exp, "build_artifact", lambda **_kwargs: {})
    with pytest.raises(exp.TypedArcShadowMonitorError, match="invalid_artifact"):
        exp.execute(REPO_ROOT, "20260901", output)

    def _raise_execute(_root: Path, _date: str, _output: Path) -> dict[str, Any]:
        raise exp.TypedArcShadowMonitorError("forced")

    monkeypatch.setattr(exp, "execute", _raise_execute)
    assert exp.main(["--date", "20260901", "--output", str(output)]) == 1
