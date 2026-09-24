"""Tests for the V662 independent ARC panel audit.

Spec refs: REQ-ARC-WMTE-7584 and SCENARIO-ARC-WMTE-7584-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time

import pytest

from carnot import experiment_7584_v662_arc_independent_audit as audit


def _episode_rows() -> list[dict]:
    rows: list[dict] = []
    for arm, accepted, executed in (
        ("current_verifier", False, 0),
        ("integrity_guard", True, 1),
    ):
        rows.append(
            {
                "unit_id": "su15:7582001",
                "panel": "A",
                "game": "su15",
                "seed": 7582001,
                "arm": arm,
                "model_id": audit.HISTORICAL_MODEL_ID,
                "request_sha256": "sha256:" + "1" * 64,
                "transition_sha256": "sha256:" + "2" * 64,
                "verifier_sha256": "sha256:" + "3" * 64,
                "plan_sha256": "sha256:" + "4" * 64,
                "action_sha256": "sha256:" + "5" * 64,
                "prompt_transition_ids": ["p0", "p1"],
                "heldout_transition_ids": ["h0", "h1"],
                "raised_transition_ids": ["h1"],
                "heldout_denominator": 2,
                "heldout_attempted": 2,
                "accepted": accepted,
                "guard_registered": arm == "integrity_guard",
                "response_cap_hit": False,
                "generation_attempted": 1,
                "generation_completed": 1,
                "completion_tokens": 20,
                "cost_usd": 0.1,
                "frame_motion_count": 2,
                "executed_plan_progress_count": executed,
                "executed_plan_action_count": executed,
                "supervisor_redirects": [],
                "censored": False,
                "provenance": "live_agent_self_discovery",
            }
        )
    return rows


# REQ-ARC-WMTE-7584 / SCENARIO-ARC-WMTE-7584-INDEPENDENT-PANELS
def test_missing_panels_remain_independent_and_schema_complete(tmp_path: Path) -> None:
    checks, producers = audit.collect_preconditions(tmp_path)
    panel_checks = [row for row in checks if row["check"] == "producer_exists"]
    assert len(panel_checks) == 2
    assert all(row["passed"] is False for row in panel_checks)
    artifact = audit.build_blocked_artifact(
        tmp_path,
        checks,
        producers,
        protocol=audit.fixture_protocol(),
        source_hashes=[],
        receipts=audit.fixture_validation_receipts(tmp_path),
        duration_s=0.1,
        run_date="20260924",
    )
    assert artifact["honest_verdict"].startswith("complete_blocked_")
    assert artifact["verdict_class"] == "blocked"
    assert set(artifact["panel_dispositions"]) == {"panel_a", "panel_b"}
    assert artifact["panel_dispositions"]["panel_a"]["status"] == "blocked"
    assert artifact["panel_dispositions"]["panel_b"]["status"] == "blocked"
    assert len(artifact["rows"]) == 24
    assert all(row["censored"] is True for row in artifact["rows"])
    assert audit.validate_artifact(artifact, root=tmp_path)["valid"] is True


# REQ-ARC-WMTE-7584 / SCENARIO-ARC-WMTE-7584-ROW-INTEGRITY
def test_episode_reduction_retains_exceptions_caps_and_execution() -> None:
    rows = _episode_rows()
    reduced = audit.reduce_episode_rows(rows)
    assert reduced["episode_count"] == 1
    assert reduced["row_count"] == 2
    assert reduced["attempted_calls"] == 2
    assert reduced["completed_calls"] == 2
    assert reduced["cap_hit_count"] == 0
    assert reduced["raised_heldout_count"] == 2
    assert reduced["game_cluster_count"] == 1
    assert reduced["frame_motion_total"] == 4
    assert reduced["executed_plan_progress_total"] == 1
    assert reduced["paired_contrasts"]["acceptance"]["raw_denominator"] == 1


# REQ-ARC-WMTE-7584 / SCENARIO-ARC-WMTE-7584-ROW-INTEGRITY
def test_overlap_duplicate_and_order_dependent_rows_fail_closed() -> None:
    overlap = _episode_rows()
    overlap[0]["heldout_transition_ids"] = ["p0", "h1"]
    with pytest.raises(ValueError, match="transition_overlap"):
        audit.reduce_episode_rows(overlap)
    duplicated = _episode_rows()
    duplicated[0]["heldout_transition_ids"] = ["h0", "h0"]
    with pytest.raises(ValueError, match="duplicate_heldout"):
        audit.reduce_episode_rows(duplicated)
    order = _episode_rows()
    order[0]["engine_order_independent"] = False
    with pytest.raises(ValueError, match="order_dependent"):
        audit.reduce_episode_rows(order)


# REQ-ARC-WMTE-7584 / SCENARIO-ARC-WMTE-7584-PRIVATE-MUTATIONS
@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("dropped_exception", "raised_transition_missing"),
        ("duplicated_heldout", "duplicate_heldout"),
        ("wrong_model_id", "wrong_model_id"),
        ("unexecuted_plan", "unexecuted_plan_progress"),
        ("capped_response", "capped_response_accepted"),
    ],
)
def test_required_private_mutations_fail(mutation: str, error: str) -> None:
    rows = _episode_rows()
    audit.mutate_rows(rows, mutation)
    with pytest.raises(ValueError, match=error):
        audit.reduce_episode_rows(rows)


# REQ-ARC-WMTE-7584 / SCENARIO-ARC-WMTE-7584-PRIVATE-MUTATIONS
def test_private_mutation_panel_and_guard_attribution() -> None:
    controls = audit.run_private_mutations()
    assert {row["mutation"] for row in controls} == {
        "dropped_exception",
        "duplicated_heldout",
        "wrong_model_id",
        "unexecuted_plan",
        "capped_response",
    }
    assert all(row["passed"] is True for row in controls)
    rows = _episode_rows()
    reduced = audit.reduce_episode_rows(rows)
    attribution = reduced["paired_contrasts"]["acceptance"]
    assert attribution["attributed_to"] == "registered_research_guard"
    changed = deepcopy(rows)
    changed[1]["guard_registered"] = False
    with pytest.raises(ValueError, match="acceptance_change_unattributed"):
        audit.reduce_episode_rows(changed)


# REQ-ARC-WMTE-7584 / SCENARIO-ARC-WMTE-7584-SUPERVISOR-AND-B2
def test_supervisor_and_b2_floors_do_not_invent_refinement() -> None:
    no_firings = audit.reduce_supervisor([])
    assert no_firings["supervisor_refinement_supported"] is False
    assert no_firings["arm_change"] is None
    same_outcome = audit.reduce_supervisor(
        [
            {"game": "su15", "arm": "retry", "eligible": True, "useful": False},
            {"game": "sp80", "arm": "retry", "eligible": True, "useful": False},
        ]
    )
    assert same_outcome["supervisor_refinement_supported"] is False
    assert audit.b2_gate_fit_decision(1000, 99, {"su15": {"useful", "useless"}}) is False
    assert (
        audit.b2_gate_fit_decision(
            1000,
            100,
            {"su15": {"useful"}, "sp80": {"useless"}},
        )
        is True
    )


# REQ-ARC-WMTE-7584 / SCENARIO-ARC-WMTE-7584-TERMINAL
def test_terminal_schema_rejects_calls_claims_and_hash_drift(tmp_path: Path) -> None:
    artifact = audit.build_fixture_artifact(tmp_path)
    assert audit.validate_artifact(artifact, root=tmp_path)["valid"] is True
    assert artifact["MODEL_SPECS"] == []
    assert artifact["arc_claims_qualified_score"] == 0
    assert artifact["official_score_claimed"] is False
    assert artifact["solve_provenance"] == "no_new_solve_credit"
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    assert audit.cold_replay(path, root=tmp_path)["valid"] is True
    called = deepcopy(artifact)
    called["invocation_counts"]["generation_calls"]["attempted"] = 1
    with pytest.raises(ValueError, match="current_calls_nonzero"):
        audit.validate_artifact(called, root=tmp_path)
    claimed = deepcopy(artifact)
    claimed["official_score_claimed"] = True
    with pytest.raises(ValueError, match="official_score_claim"):
        audit.validate_artifact(claimed, root=tmp_path)


# REQ-ARC-WMTE-7584 / SCENARIO-ARC-WMTE-7584-TERMINAL
def test_source_authentication_and_absent_producer_distinction(tmp_path: Path) -> None:
    source = tmp_path / "producer.json"
    source.write_text("{}", encoding="utf-8")
    receipt = audit.source_receipt(source, tmp_path, "exp7582", "producer")
    assert audit.authenticate_source_receipt(receipt, tmp_path) == source
    absent = audit.absent_source_receipt(
        Path("results/experiment_7583_v662_arc_panel_b.json"), "exp7583"
    )
    assert absent["present"] is False
    assert absent["evidence_stage"] == "absent_producer"
    source.write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="source_hash_mismatch"):
        audit.authenticate_source_receipt(receipt, tmp_path)


# REQ-ARC-WMTE-7584 / SCENARIO-ARC-WMTE-7584-TERMINAL
def test_scoped_validation_and_capability_commands_are_exact(tmp_path: Path) -> None:
    scoped = audit.build_validation_commands(tmp_path, tmp_path / "private")
    assert [command.name for command in scoped] == list(audit.AFFECTED_CHECK_NAMES)
    assert all("tests/python" not in command.argv for command in scoped)
    assert any("--no-cov" in command.argv for command in scoped)
    capability = audit.capability_commands(tmp_path, tmp_path / "capability")
    assert [command.name for command in capability] == [
        "e2e_009",
        "e2e_010",
        "e2e_011",
        "e2e_012",
        "e2e_013",
        "llm_off_environment_smoke",
    ]
    assert all(command.timeout_s <= 600 for command in (*scoped, *capability))


# REQ-ARC-WMTE-7584 / SCENARIO-ARC-WMTE-7584-TERMINAL
def test_parser_principles_and_exact_terminal_plan(tmp_path: Path) -> None:
    args = audit.parse_args(["--root", str(tmp_path), "--date", "20260924"])
    assert args.root == tmp_path.resolve()
    assert args.date == "20260924"
    with pytest.raises(ValueError, match="run_date"):
        audit.parse_args(["--root", str(tmp_path), "--date", "20260923"])
    assert set(audit.REQUIRED_PRINCIPLE_FIELDS) <= set(audit.field_principles())
    commands = audit.terminal_commands(tmp_path / "candidate.json", tmp_path)
    assert [command.name for command in commands] == list(audit.TERMINAL_CHECK_NAMES)
    assert any("--strict" in command.argv for command in commands)


# REQ-ARC-WMTE-7584 / SCENARIO-ARC-WMTE-7584-INDEPENDENT-PANELS
def test_preconditions_read_present_producers_and_check_operators(tmp_path: Path) -> None:
    for label in ("AGENTS.md", "CODEX.md", "CLAUDE.md"):
        (tmp_path / label).write_text(label, encoding="utf-8")
    spec = tmp_path / audit.SPEC_PATH
    spec.parent.mkdir(parents=True)
    spec.write_text("REQ-ARC-WMTE-7584", encoding="utf-8")
    for relative in audit.PRODUCERS.values():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "honest_verdict": "complete_null_fixture",
                    "flagged_adversarial": False,
                    "verdict_class": "null",
                }
            ),
            encoding="utf-8",
        )
    checks, producers = audit.collect_preconditions(tmp_path)
    assert all(row["passed"] is True for row in checks)
    assert set(producers) == {"panel_a", "panel_b"}
    assert audit.check_row("in", "x", "p", "f", (1, 2), 2, "in")["passed"] is True
    assert (
        audit.check_row("prefix", "x", "p", "f", "complete_", "complete_x", "starts_with")["passed"]
        is True
    )
    with pytest.raises(ValueError, match="unknown_check_op"):
        audit.check_row("bad", "x", "p", "f", 1, 1, "bad")


# REQ-ARC-WMTE-7584 / SCENARIO-ARC-WMTE-7584-ROW-INTEGRITY
def test_additional_reducer_guards_and_empty_contrast() -> None:
    invalid_hash = _episode_rows()
    invalid_hash[0]["request_sha256"] = "bad"
    with pytest.raises(ValueError, match="request_sha256_invalid"):
        audit.reduce_episode_rows(invalid_hash)
    denominator = _episode_rows()
    denominator[0]["heldout_denominator"] = 3
    with pytest.raises(ValueError, match="heldout_denominator_invalid"):
        audit.reduce_episode_rows(denominator)
    provenance = _episode_rows()
    provenance[0]["provenance"] = "development_proxy"
    with pytest.raises(ValueError, match="solve_provenance_invalid"):
        audit.reduce_episode_rows(provenance)
    roster = _episode_rows()
    roster[0]["arm"] = "bad"
    with pytest.raises(ValueError, match="episode_arm_roster_invalid"):
        audit.reduce_episode_rows(roster)
    with pytest.raises(ValueError, match="episode_pair_incomplete"):
        audit.reduce_episode_rows(_episode_rows()[:1])
    assert audit._contrast([])["raw_denominator"] == 0
    with pytest.raises(ValueError, match="unknown_mutation"):
        audit.mutate_rows(_episode_rows(), "unknown")
    with pytest.raises(ValueError, match="blocked_artifact_requires_failed_check"):
        audit._blocked_summary([audit.check_row("ok", "x", "p", "f", True, True)])


# REQ-ARC-WMTE-7584 / SCENARIO-ARC-WMTE-7584-TERMINAL
def test_source_edge_cases_and_protocol_custody(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    assert audit.load_json(malformed) == {}
    assert audit.load_json(tmp_path / "absent.json") == {}
    scalar = tmp_path / "scalar.json"
    scalar.write_text("[]", encoding="utf-8")
    assert audit.load_json(scalar) == {}
    outside = tmp_path.parent / "exp7584-outside-source.json"
    outside.write_text("{}", encoding="utf-8")
    receipt = audit.source_receipt(outside, tmp_path, "outside", "fixture")
    assert receipt["path"] == outside.as_posix()
    with pytest.raises(ValueError, match="source_not_present"):
        audit.authenticate_source_receipt(audit.absent_source_receipt(Path("x"), "x"), tmp_path)
    local = tmp_path / "local.json"
    local.write_text("{}", encoding="utf-8")
    sized = audit.source_receipt(local, tmp_path, "local", "fixture")
    sized["bytes"] = 99
    with pytest.raises(ValueError, match="source_size_mismatch"):
        audit.authenticate_source_receipt(sized, tmp_path)
    checks: list[dict] = []
    hashes, protocol = audit._collect_source_hashes(audit.REPO_ROOT, checks)
    assert len(audit.protocol_rows(protocol)) == 24
    assert any(row["evidence_stage"] == "pre_gate_diagnostic_not_producer" for row in hashes)
    missing_checks: list[dict] = []
    assert audit._load_protocol(tmp_path, missing_checks, []) == {}
    assert missing_checks[0]["passed"] is False
    audit.progress(time.monotonic(), "fixture", "complete", units=1)
    assert "phase=fixture" in capsys.readouterr().out


def _rechecksum(value: dict) -> dict:
    value["reproducibility_checksum"] = audit.reproducibility_checksum(value)
    return value


# REQ-ARC-WMTE-7584 / SCENARIO-ARC-WMTE-7584-TERMINAL
@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda value: value.update(schema="bad"), "identity_mismatch"),
        (lambda value: value.update(milestone="bad"), "milestone_mismatch"),
        (lambda value: value.update(honest_verdict="blocked"), "terminal_prefix_missing"),
        (lambda value: value.update(verdict_class="bad"), "verdict_class_invalid"),
        (lambda value: value.update(MODEL_SPECS=["bad"]), "model_specs_not_empty"),
        (lambda value: value.update(inference_substrate_class="bad"), "substrate_mismatch"),
        (lambda value: value.update(solve_provenance="bad"), "solve_credit_invalid"),
        (lambda value: value.update(arc_claims_qualified_score=2), "arc_qualification_invalid"),
        (lambda value: value.update(field_principles={}), "field_principles_incomplete"),
        (lambda value: value.update(panel_dispositions={}), "panel_dispositions_incomplete"),
        (lambda value: value.update(gate_check_summary={}), "blocked_gate_summary_invalid"),
        (lambda value: value["rows"][0].pop("provenance"), "row_schema_invalid"),
        (lambda value: value.update(validation_receipts=[]), "validation_receipts_failed"),
        (lambda value: value.update(mutation_rows=[]), "mutation_panel_failed"),
    ],
)
def test_terminal_schema_mutations_fail_closed(
    tmp_path: Path, mutation: object, error: str
) -> None:
    artifact = audit.build_fixture_artifact(tmp_path)
    mutation(artifact)  # type: ignore[operator]
    _rechecksum(artifact)
    with pytest.raises(ValueError, match=error):
        audit.validate_artifact(artifact, root=tmp_path)


# REQ-ARC-WMTE-7584 / SCENARIO-ARC-WMTE-7584-TERMINAL
def test_cold_replay_and_independent_reduction_guards(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="artifact_not_object"):
        audit.cold_replay(tmp_path / "missing.json", root=tmp_path)
    artifact = audit.build_fixture_artifact(tmp_path)
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(artifact), encoding="utf-8")
    reduced = audit.independent_reduction(candidate, root=tmp_path)
    assert reduced == {
        "valid": True,
        "row_count": 24,
        "panel_counts": {"A": 12, "B": 12},
        "censored_count": 24,
    }
    duplicate = deepcopy(artifact)
    duplicate["rows"][1].update(duplicate["rows"][0])
    _rechecksum(duplicate)
    candidate.write_text(json.dumps(duplicate), encoding="utf-8")
    with pytest.raises(ValueError, match="support_row_duplicate"):
        audit.independent_reduction(candidate, root=tmp_path)
    short = deepcopy(artifact)
    short["rows"] = short["rows"][:-1]
    _rechecksum(short)
    candidate.write_text(json.dumps(short), encoding="utf-8")
    with pytest.raises(ValueError, match="blocked_roster_mismatch"):
        audit.independent_reduction(candidate, root=tmp_path)
    denominator = deepcopy(artifact)
    denominator["rows"][0]["raw_denominator"] = 2
    _rechecksum(denominator)
    candidate.write_text(json.dumps(denominator), encoding="utf-8")
    with pytest.raises(ValueError, match="support_denominator_mismatch"):
        audit.independent_reduction(candidate, root=tmp_path)


# REQ-ARC-WMTE-7584 / SCENARIO-ARC-WMTE-7584-TERMINAL
def test_manifest_pending_receipts_and_receipt_enrichment(tmp_path: Path) -> None:
    path, manifest = audit._manifest(tmp_path)
    assert path.is_file()
    assert manifest["changed_modules"] == [audit.MODULE_PATH.as_posix()]
    pending = audit._pending_terminal_receipts(tmp_path)
    assert [row["name"] for row in pending] == list(audit.TERMINAL_CHECK_NAMES)
    enriched = audit._enrich_receipts([{"name": "x"}], tmp_path)
    assert enriched[0]["cwd"] == str(tmp_path.resolve())
    assert enriched[0]["worktree"] == str(tmp_path.resolve())


# REQ-ARC-WMTE-7584 / SCENARIO-ARC-WMTE-7584-INDEPENDENT-PANELS
def test_partial_panel_and_missing_sidecar_custody_paths(tmp_path: Path) -> None:
    rows = audit._missing_rows(audit.fixture_protocol(), {"panel_a"})
    assert len(rows) == 12
    assert {row["panel"] for row in rows} == {"A"}

    protocol_producer = tmp_path / audit.PROTOCOL_PRODUCER_PATH
    protocol_producer.parent.mkdir(parents=True)
    protocol_producer.write_text(
        json.dumps({"live_panel_protocol_path": "results/raw/missing.json"}),
        encoding="utf-8",
    )
    panel_a = tmp_path / audit.PRODUCERS["panel_a"]
    panel_a.parent.mkdir(parents=True, exist_ok=True)
    panel_a.write_text("{}", encoding="utf-8")
    checks: list[dict] = []
    hashes, protocol = audit._collect_source_hashes(tmp_path, checks)
    assert protocol == {}
    assert any(row["check"] == "protocol_sidecar_exists" for row in checks)
    assert any(row["upstream"] == "panel_a" and row["present"] is True for row in hashes)
    assert any(row["evidence_stage"] == "absent_producer" for row in hashes)

    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    artifact = audit.build_fixture_artifact(tmp_path)
    artifact["source_artifact_hashes"] = [
        audit.source_receipt(source, tmp_path, "fixture", "producer")
    ]
    source.write_text('{"changed": true}', encoding="utf-8")
    _rechecksum(artifact)
    with pytest.raises(ValueError, match="source_hash_mismatch"):
        audit.validate_artifact(artifact, root=tmp_path)
