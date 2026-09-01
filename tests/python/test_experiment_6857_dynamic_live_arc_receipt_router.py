"""Tests for the execution-time ARC receipt router.

Spec refs: REQ-ARC-6857 and every SCENARIO-ARC-6857-* section.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path

import pytest
import yaml

from carnot import experiment_6857_dynamic_live_arc_receipt_router as exp


REPO = Path(__file__).resolve().parents[2]


def _prepare_root(root: Path, *, ready: int = 1) -> Path:
    """Create only the read-only inputs that the router owns in a test."""

    (root / "results" / "receipts").mkdir(parents=True)
    (root / "ops").mkdir(parents=True)
    (root / "python" / "carnot" / "agentic").mkdir(parents=True)
    (root / exp.V599_PATH).write_text(
        json.dumps(
            {
                "v599_evidence_contract_ready_score": ready,
                "honest_verdict": "complete_test_contract",
            }
        ),
        encoding="utf-8",
    )
    (root / exp.REGISTRY_PATH).write_text(
        yaml.safe_dump({"schema_version": 1, "games": []}), encoding="utf-8"
    )
    (root / exp.FLAG_LEDGER_PATH).write_text(yaml.safe_dump({"flags": {}}), encoding="utf-8")
    generator = root / "python" / "carnot" / "agentic" / "generator.py"
    generator.write_text("GENERATOR = 'current'\n", encoding="utf-8")
    (root / exp.GENERATOR_SOURCE_PATH).write_text(
        'ARC_LIVE_GENERATOR_REPO_SUBSTR = "Qwen3.8-27B"\n', encoding="utf-8"
    )
    return generator


def _base_rows(family: str, attempt: str) -> list[dict]:
    """Build the smallest exact receipt chain for one artifact family."""

    if family == "supervisor":
        return [
            {
                "row_kind": "supervisor_action",
                "row_identity": f"{attempt}:supervisor",
                "attempt_identity": attempt,
                "action_identity": f"{attempt}:action",
                "next_action_identity": f"{attempt}:next",
                "transition_identity": f"{attempt}:transition",
                "later_outcome_identity": f"{attempt}:outcome",
                "matched_opportunity_identity": f"{attempt}:control",
            },
            {
                "row_kind": "next_action",
                "row_identity": f"{attempt}:next",
                "attempt_identity": attempt,
                "action_identity": f"{attempt}:action-next",
            },
            {
                "row_kind": "transition",
                "row_identity": f"{attempt}:transition",
                "attempt_identity": attempt,
                "state_before_sha256": "sha256:" + "1" * 64,
                "state_after_sha256": "sha256:" + "2" * 64,
            },
            {
                "row_kind": "exact_outcome",
                "row_identity": f"{attempt}:outcome",
                "attempt_identity": attempt,
                "levels_before": 0,
                "levels_after": 1,
                "level_ceiling": 2,
                "exact": True,
            },
        ]
    if family == "tool_loop":
        return [
            {
                "row_kind": "tool_event",
                "row_identity": f"{attempt}:tool",
                "attempt_identity": attempt,
                "request_identity": f"{attempt}:request",
                "response_identity": f"{attempt}:response",
                "next_action_identity": f"{attempt}:next",
                "later_outcome_identity": f"{attempt}:outcome",
                "agent_visible_receipt": "tool result shown to the agent",
                "first_party": True,
            },
            {
                "row_kind": "next_action",
                "row_identity": f"{attempt}:next",
                "attempt_identity": attempt,
                "action_identity": f"{attempt}:action-next",
            },
            {
                "row_kind": "exact_outcome",
                "row_identity": f"{attempt}:outcome",
                "attempt_identity": attempt,
                "levels_before": 0,
                "levels_after": 0,
                "level_ceiling": 1,
                "exact": True,
            },
        ]
    return [
        {
            "row_kind": {
                "lever_harness": "lever_event",
                "shadow": "shadow_event",
                "canonical_agent": "source",
            }[family],
            "row_identity": f"{attempt}:row",
            "attempt_identity": attempt,
        }
    ]


def _bundle(
    root: Path,
    family: str,
    attempt: str,
    *,
    rows: list[dict] | None = None,
    experiment_id: str | None = None,
) -> dict:
    """Build a provenance-complete native receipt bundle."""

    generator = root / "python" / "carnot" / "agentic" / "generator.py"
    payload = {
        "schema": exp.NATIVE_SCHEMA_BY_FAMILY[family],
        "artifact_family": family,
        "experiment_id": experiment_id or f"exp-{attempt}",
        "terminal_status": "complete",
        "honest_verdict": "complete_fixture_receipts",
        "completed_at": "2026-09-01T12:00:00Z",
        "attempt_identity": attempt,
        "generator_provenance": {
            "generator_id": "canonical-generator",
            "implementation_path": generator.relative_to(root).as_posix(),
            "implementation_sha256": exp.sha256_file(generator),
            "configuration_sha256": "sha256:" + "3" * 64,
            "model_id": "unsloth/Qwen3.8-27B-GGUF",
            "model_artifact_sha256": "sha256:" + "4" * 64,
            "tokenizer_sha256": "sha256:" + "5" * 64,
        },
        "live_reachability": {
            "reachable": True,
            "entrypoint": "make_carnot_agent -> E3AgentPolicy",
            "live_seam": "canonical_action_seam",
            "development_proxy": False,
            "outer_loop_re": False,
            "source_reading": False,
        },
        "configuration": {
            "game": "cd82",
            "policy_hash": "sha256:" + "6" * 64,
            "budget": 400,
            "supervisor_mode": "on" if family == "supervisor" else "off",
            "tool_mode": "on" if family == "tool_loop" else "off",
        },
        "verification_failure": False,
        "pooled_policy_change": False,
        "rows": deepcopy(rows) if rows is not None else _base_rows(family, attempt),
    }
    payload["declared_content_sha256"] = exp.payload_sha256(payload)
    return payload


def _write_bundle(root: Path, name: str, payload: dict) -> Path:
    path = root / "results" / "receipts" / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _build(root: Path) -> dict:
    return exp.build_artifact(
        root,
        run_date="20260901",
        duration_s=0.25,
        discovery_roots=[{"path": "results/receipts", "patterns": ["*.json"]}],
        process_observations=[{"pid": 77, "state": "S", "command": "arc-worker"}],
    )


def test_req_arc_6857_spec_declares_router_contract() -> None:
    """REQ-ARC-6857: OpenSpec owns all fields, schemas, and failure cases."""

    text = (REPO / exp.SPEC_PATH).read_text(encoding="utf-8")
    section = text.split("REQ-ARC-6857", 1)[1]
    for anchor in (
        "SCENARIO-ARC-6857-STALE-PATH-AND-AMBIGUITY",
        "SCENARIO-ARC-6857-PARTIAL-HASH-AND-GENERATOR",
        "SCENARIO-ARC-6857-EXACT-JOINS",
        "SCENARIO-ARC-6857-CONFIGURATION-SEPARATION",
        "SCENARIO-ARC-6857-DUPLICATE-ROW-IDENTITY",
        "SCENARIO-ARC-6857-HEADROOM-AND-TOOL-CHAIN",
        "SCENARIO-ARC-6857-NO-SOLVE-AND-PROCESS-OBSERVATION",
    ):
        assert anchor in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in section
    assert exp.INFERENCE_SUBSTRATE in section
    assert exp.OUTPUT_PATH.as_posix() in section


def test_req_arc_6857_routes_all_families_and_exact_chains(tmp_path: Path) -> None:
    """REQ-ARC-6857: valid families freeze with separate effect readiness."""

    _prepare_root(tmp_path)
    for family in exp.NATIVE_SCHEMA_BY_FAMILY:
        _write_bundle(tmp_path, f"{family}.json", _bundle(tmp_path, family, family))

    artifact = _build(tmp_path)

    assert exp.validate_artifact(artifact) == []
    assert artifact["arc_receipt_router_complete_score"] == 1
    assert artifact["supervisor_headroom_ready_score"] == 1
    assert artifact["tool_gap_first_party_receipts_ready_score"] == 1
    assert artifact["solve_claim"] is False
    assert artifact["game_level_solve_count"] == 0
    assert artifact["verifier_is_oracle"] is False
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("complete_")
    assert len(artifact["provenance_qualified_manifest"]) == 5
    assert len(artifact["generator_provenance_rows"]) == 5
    assert len(artifact["live_reachability_rows"]) == 5
    assert len(artifact["supervisor_headroom_rows"]) == 1
    assert artifact["supervisor_headroom_rows"][0]["exact_outcome_headroom"] == 2
    assert len(artifact["first_party_tool_gap_rows"]) == 1
    assert artifact["unmatched_receipt_rows"] == []
    assert {row["artifact_family"] for row in artifact["rows"]} == set(exp.NATIVE_SCHEMA_BY_FAMILY)
    assert len(artifact["configuration_strata"]) == 5
    assert set(artifact["field_principles"]) == set(artifact)
    assert artifact["process_observations"] == [{"pid": 77, "state": "S", "command": "arc-worker"}]


def test_scenario_arc_6857_stale_path_is_ignored_and_newest_ambiguity_is_quarantined(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-6857-STALE-PATH-AND-AMBIGUITY never selects by mtime."""

    _prepare_root(tmp_path)
    accepted = _bundle(tmp_path, "canonical_agent", "stable")
    accepted["stored_source_path"] = "/missing/old/experiment_6681.json"
    accepted["declared_content_sha256"] = exp.payload_sha256(accepted)
    _write_bundle(tmp_path, "stable.json", accepted)
    first = _write_bundle(
        tmp_path,
        "ambiguous-a.json",
        _bundle(tmp_path, "supervisor", "ambiguous", experiment_id="exp-a"),
    )
    second = _write_bundle(
        tmp_path,
        "ambiguous-b.json",
        _bundle(tmp_path, "supervisor", "ambiguous", experiment_id="exp-b"),
    )
    os.utime(first, (1, 1))
    os.utime(second, (2, 2))

    artifact = _build(tmp_path)

    assert [row["attempt_identity"] for row in artifact["provenance_qualified_manifest"]] == [
        "stable"
    ]
    assert artifact["provenance_qualified_manifest"][0]["stored_source_path_ignored"] is True
    ambiguous = [
        row
        for row in artifact["rejected_source_manifest"]
        if row["reason"] == "ambiguous_attempt_identity"
    ]
    assert len(ambiguous) == 2
    assert artifact["arc_receipt_router_complete_score"] == 1


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda payload: payload.update(terminal_status="partial"), "nonterminal_artifact"),
        (
            lambda payload: payload.update(declared_content_sha256="sha256:" + "0" * 64),
            "stale_declared_hash",
        ),
        (
            lambda payload: payload["generator_provenance"].update(
                implementation_sha256="sha256:" + "9" * 64
            ),
            "changed_generator",
        ),
        (
            lambda payload: payload["rows"][0].update(policy_hash="sha256:" + "8" * 64),
            "mixed_policy_configuration",
        ),
    ],
)
def test_scenario_arc_6857_rejects_partial_hash_generator_and_mixed_policy(
    tmp_path: Path, mutation, reason: str
) -> None:
    """SCENARIO-ARC-6857-PARTIAL-HASH-AND-GENERATOR fails closed per source."""

    _prepare_root(tmp_path)
    payload = _bundle(tmp_path, "canonical_agent", "bad")
    mutation(payload)
    if reason not in {"stale_declared_hash", "changed_generator"}:
        payload["declared_content_sha256"] = exp.payload_sha256(payload)
    _write_bundle(tmp_path, "bad.json", payload)

    artifact = _build(tmp_path)

    assert artifact["provenance_qualified_manifest"] == []
    assert artifact["rejected_source_manifest"][0]["reason"] == reason


@pytest.mark.parametrize(
    ("family", "mutate", "unmatched_reason"),
    [
        (
            "supervisor",
            lambda payload: payload["rows"][0].update(later_outcome_identity="missing"),
            "missing_exact_later_outcome",
        ),
        (
            "tool_loop",
            lambda payload: payload["rows"][0].update(agent_visible_receipt=""),
            "missing_agent_visible_receipt",
        ),
    ],
)
def test_scenario_arc_6857_preserves_missing_join_receipts(
    tmp_path: Path, family: str, mutate, unmatched_reason: str
) -> None:
    """SCENARIO-ARC-6857-EXACT-JOINS keeps incomplete chains as diagnostics."""

    _prepare_root(tmp_path)
    payload = _bundle(tmp_path, family, "unmatched")
    mutate(payload)
    payload["declared_content_sha256"] = exp.payload_sha256(payload)
    _write_bundle(tmp_path, "unmatched.json", payload)

    artifact = _build(tmp_path)

    assert artifact["arc_receipt_router_complete_score"] == 1
    assert artifact["supervisor_headroom_ready_score"] == 0
    assert artifact["tool_gap_first_party_receipts_ready_score"] == 0
    assert artifact["unmatched_receipt_rows"][0]["reason"] == unmatched_reason


def test_scenario_arc_6857_changed_configurations_stay_separate(tmp_path: Path) -> None:
    """SCENARIO-ARC-6857-CONFIGURATION-SEPARATION does not pool policies."""

    _prepare_root(tmp_path)
    first = _bundle(tmp_path, "canonical_agent", "one")
    second = _bundle(tmp_path, "canonical_agent", "two")
    second["configuration"].update(
        game="r11l",
        policy_hash="sha256:" + "7" * 64,
        budget=800,
        supervisor_mode="shadow",
        tool_mode="on",
    )
    second["declared_content_sha256"] = exp.payload_sha256(second)
    _write_bundle(tmp_path, "one.json", first)
    _write_bundle(tmp_path, "two.json", second)

    artifact = _build(tmp_path)

    assert len(artifact["configuration_strata"]) == 2
    assert {row["game"] for row in artifact["configuration_strata"]} == {"cd82", "r11l"}
    assert {row["budget"] for row in artifact["configuration_strata"]} == {400, 800}


def test_scenario_arc_6857_duplicate_row_identity_blocks_router(tmp_path: Path) -> None:
    """SCENARIO-ARC-6857-DUPLICATE-ROW-IDENTITY makes readiness fail closed."""

    _prepare_root(tmp_path)
    first = _bundle(tmp_path, "canonical_agent", "one")
    second = _bundle(tmp_path, "canonical_agent", "two")
    second["rows"][0]["row_identity"] = first["rows"][0]["row_identity"]
    second["declared_content_sha256"] = exp.payload_sha256(second)
    _write_bundle(tmp_path, "one.json", first)
    _write_bundle(tmp_path, "two.json", second)

    artifact = _build(tmp_path)

    assert artifact["arc_receipt_router_complete_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == "complete_blocked_dynamic_live_arc_receipt_router"
    failed = artifact["gate_check_summary"]["failed_checks"]
    assert failed[-1]["check"] == "duplicate_row_identity"
    assert failed[-1]["observed"] == ["one:row"]


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("development_proxy", True, "development_proxy"),
        ("outer_loop_re", True, "outer_loop_re"),
        ("source_reading", True, "source_reading"),
        ("reachable", False, "live_seam_unreachable"),
    ],
)
def test_req_arc_6857_quarantines_nonlive_provenance(
    tmp_path: Path, field: str, value: bool, reason: str
) -> None:
    """REQ-ARC-6857: only live, source-free rows qualify for effects."""

    _prepare_root(tmp_path)
    payload = _bundle(tmp_path, "canonical_agent", "quarantine")
    payload["live_reachability"][field] = value
    payload["declared_content_sha256"] = exp.payload_sha256(payload)
    _write_bundle(tmp_path, "quarantine.json", payload)

    artifact = _build(tmp_path)

    assert artifact["rejected_source_manifest"][0]["reason"] == reason


@pytest.mark.parametrize(
    ("field", "reason"),
    [
        ("pooled_policy_change", "pooled_policy_change"),
        ("verification_failure", "flagged_verification_failure"),
    ],
)
def test_req_arc_6857_quarantines_pooled_and_flagged_sources(
    tmp_path: Path, field: str, reason: str
) -> None:
    """REQ-ARC-6857: pooled policy and failed verification cannot qualify."""

    _prepare_root(tmp_path)
    payload = _bundle(tmp_path, "lever_harness", "quarantine")
    payload[field] = True
    payload["declared_content_sha256"] = exp.payload_sha256(payload)
    _write_bundle(tmp_path, "quarantine.json", payload)

    artifact = _build(tmp_path)

    assert artifact["rejected_source_manifest"][0]["reason"] == reason


def test_req_arc_6857_failed_precondition_writes_complete_blocked_artifact(
    tmp_path: Path,
) -> None:
    """REQ-ARC-6857: the V599 readiness gate reports its exact observed value."""

    _prepare_root(tmp_path, ready=0)

    artifact = _build(tmp_path)

    assert artifact["arc_receipt_router_complete_score"] == 0
    assert artifact["honest_verdict"] == "complete_blocked_dynamic_live_arc_receipt_router"
    assert artifact["gate_check_summary"]["failed_check"] == "v599_evidence_contract_ready_score"
    assert artifact["gate_check_summary"]["observed"] == 0
    assert artifact["rows"] == []


def test_req_arc_6857_unreadable_registry_and_unknown_schema_are_diagnostic(
    tmp_path: Path,
) -> None:
    """REQ-ARC-6857: unreadable owned inputs block while unknown JSON stays rejected."""

    _prepare_root(tmp_path)
    (tmp_path / exp.REGISTRY_PATH).write_text("games: [", encoding="utf-8")
    _write_bundle(tmp_path, "unknown.json", {"schema": "unknown", "status": "complete"})

    artifact = _build(tmp_path)

    assert artifact["gate_check_summary"]["failed_check"] == "arc_registry_readable"
    assert artifact["rejected_source_manifest"][0]["reason"] == "schema_not_accepted"


def test_req_arc_6857_validation_and_atomic_cli_write(tmp_path: Path, monkeypatch) -> None:
    """REQ-ARC-6857: validation and the CLI preserve the terminal artifact contract."""

    _prepare_root(tmp_path)
    _write_bundle(
        tmp_path,
        "canonical.json",
        _bundle(tmp_path, "canonical_agent", "cli"),
    )
    output = tmp_path / "out" / "router.json"
    monkeypatch.setattr(exp, "sample_live_processes", lambda: [])

    assert (
        exp.main(
            [
                "--root",
                str(tmp_path),
                "--date",
                "20260901",
                "--output",
                str(output),
                "--discovery-root",
                "results/receipts:*.json",
            ]
        )
        == 0
    )
    written = json.loads(output.read_text(encoding="utf-8"))
    assert exp.validate_artifact(written) == []
    broken = deepcopy(written)
    broken.pop("rows")
    broken["solve_claim"] = True
    broken["game_level_solve_count"] = 1
    broken["verifier_is_oracle"] = True
    broken["verdict_class"] = "invented"
    broken["honest_verdict"] = "pending"
    broken["inference_substrate"] = "wrong"
    broken["reproducibility_checksum"] = "sha256:stale"
    errors = exp.validate_artifact(broken)
    assert "missing top-level field: rows" in errors
    assert "solve_claim must be false" in errors
    assert "game_level_solve_count must be 0" in errors
    assert "verifier_is_oracle must be false" in errors
    assert "invalid verdict_class" in errors
    assert "honest_verdict must start with complete_" in errors
    assert "invalid inference_substrate" in errors
    assert "reproducibility_checksum mismatch" in errors


def test_req_arc_6857_discovery_handles_invalid_json_and_missing_root(tmp_path: Path) -> None:
    """REQ-ARC-6857: discovery reports unreadable input without inventing rows."""

    _prepare_root(tmp_path)
    invalid = tmp_path / "results" / "receipts" / "invalid.json"
    invalid.write_text("{not-json", encoding="utf-8")

    found = exp.discover_candidates(
        tmp_path,
        [
            {"path": "results/receipts", "patterns": ["*.json"]},
            {"path": "results/missing", "patterns": ["*.json"]},
        ],
    )

    assert found[0]["load_error"].startswith("JSONDecodeError:")
    assert found[1] == {
        "path": "results/missing",
        "exists": False,
        "load_error": "discovery_root_missing",
    }


def test_req_arc_6857_date_and_discovery_root_parsers() -> None:
    """REQ-ARC-6857: CLI parsing rejects dates and accepts explicit patterns."""

    assert exp.valid_run_date("20260901") is True
    assert exp.valid_run_date("2026-09-01") is False
    assert exp.parse_discovery_root("results:*.json,*.jsonl") == {
        "path": "results",
        "patterns": ["*.json", "*.jsonl"],
    }
    assert exp.parse_discovery_root("results") == {
        "path": "results",
        "patterns": ["*.json"],
    }
    with pytest.raises(ValueError, match="empty discovery root"):
        exp.parse_discovery_root(":*.json")
    with pytest.raises(SystemExit):
        exp.main(["--date", "bad"])


def test_req_arc_6857_legacy_canonical_agent_is_content_routed(tmp_path: Path) -> None:
    """REQ-ARC-6857: a legacy leaderboard row qualifies from runtime receipts."""

    _prepare_root(tmp_path)
    payload = {
        "experiment": "arc_leaderboard_eval",
        "policy": "e3",
        "budget": 800,
        "random_seed": 17,
        "status": "complete",
        "honest_verdict": "complete_leaderboard_eval_1_levels_0_gaps",
        "per_game": [
            {
                "game": "r11l",
                "actions": 11,
                "levels": 1,
                "oracle_levels": 2,
                "tool_loop_enabled": True,
                "trajectory_supervisor": {"enabled": True, "mode": "shadow"},
                "generator_provenance": {
                    "resolved": True,
                    "observed_server_model_path": "/cache/Qwen3.8-27B/model.gguf",
                    "reuse_model_check": "match",
                },
                "completions_consumed": {"completions": 2},
            }
        ],
    }
    _write_bundle(tmp_path, "legacy.json", payload)

    artifact = _build(tmp_path)

    assert artifact["provenance_qualified_manifest"][0]["schema"] == (
        "arc_leaderboard_eval.content.v1"
    )
    assert [row["row_kind"] for row in artifact["rows"]] == ["source", "exact_outcome"]
    assert artifact["configuration_strata"][0]["supervisor_mode"] == "shadow"
    assert artifact["configuration_strata"][0]["tool_mode"] == "on"


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (lambda payload: payload.pop("rows"), "partial_artifact_missing_fields"),
        (lambda payload: payload.update(generator_provenance=[]), "missing_generator_provenance"),
        (
            lambda payload: payload["generator_provenance"].pop("implementation_path"),
            "missing_generator_provenance",
        ),
        (
            lambda payload: payload["generator_provenance"].update(
                implementation_path="../../outside.py"
            ),
            "changed_generator",
        ),
        (lambda payload: payload.update(live_reachability=[]), "live_seam_unreachable"),
        (
            lambda payload: payload["configuration"].update(policy_hash=""),
            "mixed_policy_configuration",
        ),
    ],
)
def test_req_arc_6857_native_schema_defensive_rejections(
    tmp_path: Path, change, expected: str
) -> None:
    """REQ-ARC-6857: malformed native provenance has one stable reason."""

    _prepare_root(tmp_path)
    payload = _bundle(tmp_path, "canonical_agent", "defensive")
    change(payload)
    payload["declared_content_sha256"] = exp.payload_sha256(payload)
    _write_bundle(tmp_path, "defensive.json", payload)

    artifact = _build(tmp_path)

    assert artifact["rejected_source_manifest"][0]["reason"] == expected


@pytest.mark.parametrize(
    ("payload", "family"),
    [
        ({"policy": "explorer", "per_game": [{}]}, "canonical_agent"),
        ({"policy": "e3", "per_game": []}, "canonical_agent"),
        ({"policy": "e3", "per_game": [{}]}, "canonical_agent"),
        ({"policy": "e3", "per_game": [{}]}, "shadow"),
    ],
)
def test_req_arc_6857_legacy_generator_rejects_incomplete_receipts(
    tmp_path: Path, payload: dict, family: str
) -> None:
    """REQ-ARC-6857: a legacy model name without exact use is not provenance."""

    _prepare_root(tmp_path)
    assert exp._legacy_generator(payload, family, tmp_path) is None


def test_req_arc_6857_legacy_schema_rejections_are_explicit(tmp_path: Path) -> None:
    """REQ-ARC-6857: legacy proxy, RE, source, flag, and missing model stay visible."""

    _prepare_root(tmp_path)
    cases = [
        ("development_proxy", {}, "development_proxy"),
        ("outer_loop_re", {}, "outer_loop_re"),
        (None, {"read_game_source": True}, "source_reading"),
        (None, {"flagged_adversarial": True}, "flagged_verification_failure"),
        (None, {}, "missing_generator_provenance"),
    ]
    for index, (provenance, extra, reason) in enumerate(cases):
        payload = {
            "schema": "carnot.experiment_6846.typed_arc_shadow_monitor.v1",
            "status": "complete",
            "honest_verdict": "complete_legacy",
            "solve_provenance": provenance,
            **extra,
        }
        _write_bundle(tmp_path, f"legacy-{index}-{reason}.json", payload)

    artifact = _build(tmp_path)

    assert {row["reason"] for row in artifact["rejected_source_manifest"]} == {
        reason for _, _, reason in cases
    }


def test_scenario_arc_6857_all_unmatched_join_reasons_are_preserved(tmp_path: Path) -> None:
    """SCENARIO-ARC-6857-EXACT-JOINS covers every exact-link failure."""

    _prepare_root(tmp_path)
    base = _base_rows("supervisor", "joins")
    variants = []
    missing_transition = deepcopy(base)
    missing_transition[0]["transition_identity"] = "absent"
    variants.append(("missing-transition", missing_transition))
    missing_match = deepcopy(base)
    missing_match[0]["matched_opportunity_identity"] = ""
    variants.append(("missing-match", missing_match))
    missing_headroom = deepcopy(base)
    missing_headroom[-1]["level_ceiling"] = None
    variants.append(("missing-headroom", missing_headroom))
    for name, rows in variants:
        rows = json.loads(json.dumps(rows).replace("joins", name))
        payload = _bundle(tmp_path, "supervisor", name, rows=rows)
        _write_bundle(tmp_path, f"{name}.json", payload)

    tool_missing_outcome = _base_rows("tool_loop", "tool-outcome")
    tool_missing_outcome[0]["later_outcome_identity"] = "absent"
    _write_bundle(
        tmp_path,
        "tool-outcome.json",
        _bundle(tmp_path, "tool_loop", "tool-outcome", rows=tool_missing_outcome),
    )
    tool_missing_next = _base_rows("tool_loop", "tool-next")
    tool_missing_next[0]["next_action_identity"] = "absent"
    _write_bundle(
        tmp_path,
        "tool-next.json",
        _bundle(tmp_path, "tool_loop", "tool-next", rows=tool_missing_next),
    )
    tool_proxy = _base_rows("tool_loop", "tool-proxy")
    tool_proxy[0]["first_party"] = False
    _write_bundle(
        tmp_path,
        "tool-proxy.json",
        _bundle(tmp_path, "tool_loop", "tool-proxy", rows=tool_proxy),
    )

    reasons = {row["reason"] for row in _build(tmp_path)["unmatched_receipt_rows"]}

    assert reasons == {
        "missing_exact_action_transition_join",
        "missing_matched_opportunity",
        "missing_exact_headroom_fields",
        "missing_exact_later_outcome",
        "missing_exact_next_action_join",
        "not_first_party_tool_chain",
    }


def test_req_arc_6857_discovery_defensive_paths(tmp_path: Path) -> None:
    """REQ-ARC-6857: roots, duplicate globs, and non-object JSON are safe."""

    _prepare_root(tmp_path)
    (tmp_path / "results" / "receipts" / "list.json").write_text("[]", encoding="utf-8")
    found = exp.discover_candidates(
        tmp_path,
        [
            {"path": "../outside", "patterns": ["*.json"]},
            {"path": "results/receipts", "patterns": ["*.json", "list.*"]},
            {"path": "results/receipts", "patterns": ["list.json"]},
        ],
    )

    assert found[0]["load_error"] == "discovery_root_outside_repo"
    assert found[1]["load_error"].startswith("TypeError:")
    assert len(found) == 2


def test_req_arc_6857_generator_constant_can_be_missing_or_unset(tmp_path: Path) -> None:
    """REQ-ARC-6857: absent canonical identity cannot validate a legacy generator."""

    assert exp._current_generator_model(tmp_path) is None
    path = tmp_path / exp.GENERATOR_SOURCE_PATH
    path.parent.mkdir(parents=True)
    path.write_text("UNRELATED = True\n", encoding="utf-8")
    assert exp._current_generator_model(tmp_path) is None


def test_req_arc_6857_owned_reader_errors_and_scalar_documents(tmp_path: Path) -> None:
    """REQ-ARC-6857: missing and scalar owned files report exact read failures."""

    missing_json, json_error = exp._read_json(tmp_path / "missing.json")
    missing_yaml, yaml_error = exp._read_yaml(tmp_path / "missing.yaml")
    assert missing_json is None and json_error.startswith("FileNotFoundError:")
    assert missing_yaml is None and yaml_error.startswith("FileNotFoundError:")
    json_scalar = tmp_path / "scalar.json"
    yaml_scalar = tmp_path / "scalar.yaml"
    json_scalar.write_text("[]", encoding="utf-8")
    yaml_scalar.write_text("value", encoding="utf-8")
    assert exp._read_json(json_scalar) == (None, "top_level_not_object")
    assert exp._read_yaml(yaml_scalar) == (None, "top_level_not_mapping")


def test_req_arc_6857_process_observation_success_and_failure(monkeypatch) -> None:
    """REQ-ARC-6857: process sampling is one read with no signal or polling."""

    class Result:
        stdout = (
            "10 1 Mon Sep 1 12:00:00 2026 S 00:02 python arc-worker --game cd82\n"
            "11 1 Mon Sep 1 12:00:00 2026 S 00:01 python unrelated-worker\n"
            "bad line\n"
        )

    monkeypatch.setattr(exp.subprocess, "run", lambda *args, **kwargs: Result())
    rows = exp.sample_live_processes()
    assert len(rows) == 1
    assert rows[0]["pid"] == 10
    assert rows[0]["observation_only"] is True
    monkeypatch.setattr(
        exp.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("ps unavailable")),
    )
    assert exp.sample_live_processes()[0]["observation_error"] == "OSError:ps unavailable"


def test_req_arc_6857_cli_error_paths(tmp_path: Path, monkeypatch, capsys) -> None:
    """REQ-ARC-6857: bad roots and post-write validation return terminal errors."""

    _prepare_root(tmp_path)
    with pytest.raises(SystemExit):
        exp.main(["--date", "20260901", "--discovery-root", ":*.json"])
    monkeypatch.setattr(exp, "sample_live_processes", lambda: [])
    monkeypatch.setattr(exp, "validate_artifact", lambda artifact: ["forced invalid"])
    output = tmp_path / "invalid.json"
    assert exp.main(["--root", str(tmp_path), "--date", "20260901", "--output", str(output)]) == 2
    assert "forced invalid" in capsys.readouterr().out
