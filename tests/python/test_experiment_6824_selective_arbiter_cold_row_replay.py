"""Tests for the bounded selective-arbiter cold row replay.

Spec refs: REQ-CONSTRAINT-6824 and SCENARIO-CONSTRAINT-6824-PRECONDITIONS,
SCENARIO-CONSTRAINT-6824-PARSER, SCENARIO-CONSTRAINT-6824-ORDER-AND-IDENTITY,
SCENARIO-CONSTRAINT-6824-BUDGETS-AND-JOINS, SCENARIO-CONSTRAINT-6824-INTERVALS,
SCENARIO-CONSTRAINT-6824-ROW-FAULTS, and SCENARIO-CONSTRAINT-6824-AGGREGATION.
"""

from __future__ import annotations

import base64
from copy import deepcopy
import inspect
import json
from pathlib import Path

import pytest

from carnot import experiment_6824_cold_arbiter as arbiter
from carnot import experiment_6824_cold_parser as parser
from carnot import experiment_6824_cold_reducer as reducer
from carnot import experiment_6824_selective_arbiter_cold_row_replay as replay


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATHS = {
    name: REPO_ROOT / relative for name, relative in replay.SOURCE_RELATIVE_PATHS.items()
}


@pytest.fixture(scope="module")
def sources() -> dict[str, dict]:
    """Load immutable artifacts once because replay never edits the sources."""

    return replay.load_sources(SOURCE_PATHS)


@pytest.fixture(scope="module")
def artifact(sources: dict[str, dict]) -> dict:
    """Build the full cold artifact once for all row-owned assertions."""

    return replay.build_artifact(
        sources,
        source_paths=SOURCE_PATHS,
        run_date=replay.RUN_DATE,
        duration_s=1.25,
    )


def _valid_candidates() -> list[dict]:
    """Return two small strict candidates for ordering tests."""

    return [
        {
            "action": {"data": index, "kind": "ACT"},
            "authority_chain": ["system"],
            "candidate_id": f"candidate_{index}",
            "candidate_index": index,
            "parse_failure": None,
            "parse_state": "complete",
            "soft_progress": index,
        }
        for index in range(2)
    ]


def test_req_constraint_6824_spec_precedes_implementation() -> None:
    """REQ-CONSTRAINT-6824 owns every shard behavior and required field."""

    spec = (REPO_ROOT / replay.SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
    section = spec.split("### REQ-CONSTRAINT-6824", 1)[1]
    for scenario in replay.OPEN_SPEC_IDS[1:]:
        assert scenario in section
    for field in replay.TASK_REQUIRED_FIELDS:
        assert f"`{field}`" in section
    for path in (
        *replay.MODULE_RELATIVE_PATHS,
        replay.SCRIPT_RELATIVE_PATH,
        replay.RESULT_RELATIVE_PATH,
    ):
        assert path.as_posix() in section


def test_scenario_constraint_6824_parser_replays_sse_and_strict_json(
    sources: dict[str, dict],
) -> None:
    """SCENARIO-CONSTRAINT-6824-PARSER rebuilds raw proposal bytes without repair."""

    receipt = sources["exp6812"]["raw_output_manifest"][0]
    raw_api = base64.b64decode(receipt["raw_api_response_b64"], validate=True)
    raw_output = base64.b64decode(receipt["raw_output_b64"], validate=True)
    assert parser.decode_sse(raw_api) == raw_output
    parsed = parser.parse_proposal(raw_output)
    assert parsed["parse_state"] == "complete"
    assert [row["candidate_id"] for row in parsed["candidates"]] == [
        "candidate_0",
        "candidate_1",
    ]
    assert parser.parse_receipt(receipt)["raw_output_sha256"] == receipt["raw_output_sha256"]

    invalid = (
        (b"", "empty_output"),
        (b"\xff", "utf8_decode_error"),
        (b"[]", "invalid_top_level_fields"),
        (b'{"wrong":[]}', "invalid_top_level_fields"),
        (b'{"candidates":[]}', "invalid_candidate_count"),
        (b'{"candidates":[{},{}]}', "invalid_candidate_fields"),
    )
    for raw, reason in invalid:
        result = parser.parse_proposal(raw)
        assert result["parse_state"] == "incomplete"
        assert result["parse_failure"] == reason
        assert len(result["candidates"]) == 2

    with pytest.raises(parser.ColdParseError, match="data line"):
        parser.decode_sse(b"not an event stream")
    with pytest.raises(parser.ColdParseError, match="event JSON"):
        parser.decode_sse(b"data: nope\n\n")
    invalid_streams = (
        (b"\xff", "not UTF-8"),
        (b"", "data line"),
        (b"data: {}\n\n", "lacks choices"),
        (b'data: {"choices":[{}]}\n\n', "lacks delta"),
        (b'data: {"choices":[{"delta":{"content":1}}]}\n\n', "not text"),
    )
    for raw, message in invalid_streams:
        with pytest.raises(parser.ColdParseError, match=message):
            parser.decode_sse(raw)

    receipt_mutations = (
        ({key: value for key, value in receipt.items() if key != "raw_output_b64"}, "missing"),
        (dict(receipt, raw_output_b64="!"), "valid base64"),
        (dict(receipt, raw_api_response_len=-1), "raw API length"),
        (dict(receipt, raw_api_response_sha256="sha256:wrong"), "raw API hash"),
        (dict(receipt, raw_output_len=-1), "raw output length"),
        (dict(receipt, raw_output_sha256="sha256:wrong"), "raw output hash"),
    )
    for damaged, message in receipt_mutations:
        with pytest.raises(parser.ColdParseError, match=message):
            parser.parse_receipt(damaged)
    different = b"{}"
    damaged = dict(
        receipt,
        raw_output_b64=base64.b64encode(different).decode("ascii"),
        raw_output_len=len(different),
        raw_output_sha256=parser.sha256_bytes(different),
    )
    with pytest.raises(parser.ColdParseError, match="does not equal"):
        parser.parse_receipt(damaged)


def test_req_constraint_6824_parser_rejects_each_schema_drift() -> None:
    """REQ-CONSTRAINT-6824 keeps candidate parsing strict and non-coercive."""

    candidates = _valid_candidates()
    value = {
        "candidates": [
            {
                key: item
                for key, item in candidate.items()
                if key not in {"candidate_index", "parse_failure", "parse_state"}
            }
            for candidate in candidates
        ]
    }

    mutations: list[tuple[dict, str]] = []
    extra = deepcopy(value)
    extra["candidates"][0]["extra"] = True
    mutations.append((extra, "invalid_candidate_fields"))
    action = deepcopy(value)
    action["candidates"][0]["action"] = {"kind": "ACT"}
    mutations.append((action, "invalid_action"))
    kind = deepcopy(value)
    kind["candidates"][0]["action"]["kind"] = True
    mutations.append((kind, "invalid_action"))
    chain = deepcopy(value)
    chain["candidates"][0]["authority_chain"] = ["z", "a"]
    mutations.append((chain, "invalid_authority_chain"))
    candidate_id = deepcopy(value)
    candidate_id["candidates"][1]["candidate_id"] = "candidate_0"
    mutations.append((candidate_id, "invalid_candidate_id"))
    progress = deepcopy(value)
    progress["candidates"][0]["soft_progress"] = True
    mutations.append((progress, "invalid_soft_progress"))
    for mutation, reason in mutations:
        raw = json.dumps(mutation, separators=(",", ":")).encode("utf-8")
        assert parser.parse_proposal(raw)["parse_failure"] == reason


def test_scenario_constraint_6824_obligation_order_and_safe_identity(
    sources: dict[str, dict],
) -> None:
    """SCENARIO-CONSTRAINT-6824-ORDER-AND-IDENTITY fixes authority and safe bytes."""

    scenarios = {
        row["scenario_id"]: row for row in sources["exp6812"]["frozen_manifest"]["scenarios"]
    }
    competing = scenarios["competing_authorities-01"]
    obligations = arbiter.rebuild_obligations(competing)
    assert [row["priority_class"] for row in obligations] == ["hard", "binding"]
    assert [row["authority_order"] for row in obligations] == [0, 1]

    valid = {
        **deepcopy(competing["reference_candidates"][0]),
        "candidate_index": 0,
        "parse_failure": None,
        "parse_state": "complete",
    }
    exact = arbiter.evaluate_candidate(competing, valid)
    assert exact["hard_violation_count"] == 0
    assert exact["binding_violation_vector"] == [0]
    assert exact["legal_support"] is True
    assert exact["transition"]["add"] == ["bound_01", "done_01"]

    wrong = deepcopy(valid)
    wrong["authority_chain"] = ["operator_authority"]
    rejected = arbiter.evaluate_candidate(competing, wrong)
    assert rejected["hard_violation_count"] == 1
    assert rejected["first_conflict"] == "obl_01_hard"
    binding_wrong = deepcopy(valid)
    binding_wrong["authority_chain"] = ["system_authority"]
    rejected_binding = arbiter.evaluate_candidate(competing, binding_wrong)
    assert rejected_binding["binding_violation_vector"] == [1]
    assert rejected_binding["first_conflict"] == "obl_01_binding"

    safe = scenarios["already_safe_proposals-01"]
    incomplete = [parser.incomplete_candidate(index, "test") for index in range(2)]
    selected = arbiter.select_arm(
        safe,
        [arbiter.evaluate_candidate(safe, row) for row in incomplete],
        arm=arbiter.SELECTIVE_ARM,
        soft_clip=10,
    )
    expected = arbiter.canonical_action_bytes(safe["safe_proposal"])
    assert selected["selected_action_bytes"] == expected
    assert selected["safe_action_identity"] is True
    assert selected["false_intervention"] is False

    malformed = deepcopy(competing)
    del malformed["obligations"][0]["contract"]["fallback"]
    with pytest.raises(arbiter.ColdArbiterError, match="fallback"):
        arbiter.rebuild_obligations(malformed)

    malformed_contracts = []
    malformed_contracts.append(({"obligations": []}, "non-empty"))
    bad_row = deepcopy(competing)
    bad_row["obligations"][0] = {}
    malformed_contracts.append((bad_row, "record fields"))
    bad_contract = deepcopy(competing)
    bad_contract["obligations"][0]["contract"] = []
    malformed_contracts.append((bad_contract, "contract must"))
    for field, replacement, message in (
        ("authority", {}, "authority contract"),
        ("prerequisite", {}, "prerequisite contract"),
        ("priority", {}, "priority contract"),
        ("fallback", {}, "fallback contract"),
        ("execution_consequence", {}, "execution consequence"),
    ):
        changed = deepcopy(competing)
        changed["obligations"][0]["contract"][field] = replacement
        malformed_contracts.append((changed, message))
    unknown_priority = deepcopy(competing)
    unknown_priority["obligations"][0]["contract"]["priority"]["class"] = "unknown"
    malformed_contracts.append((unknown_priority, "priority class"))
    for changed, message in malformed_contracts:
        with pytest.raises(arbiter.ColdArbiterError, match=message):
            arbiter.rebuild_obligations(changed)


def test_req_constraint_6824_lexicographic_and_stable_selection() -> None:
    """REQ-CONSTRAINT-6824 prevents soft value from crossing hard authority."""

    scenario = {
        "fallback_action": {"data": "fallback", "kind": "FALLBACK"},
        "observed_facts": ["ready"],
        "safe_proposal": None,
    }
    candidates = _valid_candidates()
    evaluated = [
        {
            **candidates[0],
            "authority_preserved": True,
            "binding_violation_vector": [0],
            "first_conflict": "hard",
            "hard_violation_count": 1,
            "legal_support": False,
            "soft_score": 10**30,
        },
        {
            **candidates[1],
            "authority_preserved": True,
            "binding_violation_vector": [1],
            "first_conflict": None,
            "hard_violation_count": 0,
            "legal_support": True,
            "soft_score": -(10**30),
        },
    ]
    selected = arbiter.select_arm(scenario, evaluated, arm=arbiter.SELECTIVE_ARM, soft_clip=10)
    assert selected["selected_candidate_id"] == "candidate_1"

    tied = [deepcopy(evaluated[1]), deepcopy(evaluated[1])]
    tied[0].update(candidate_id="candidate_1", candidate_index=1)
    tied[1].update(candidate_id="candidate_0", candidate_index=0)
    assert (
        arbiter.select_arm(scenario, tied, arm=arbiter.SELECTIVE_ARM, soft_clip=10)[
            "selected_candidate_id"
        ]
        == "candidate_0"
    )
    with pytest.raises(ValueError, match="unknown arm"):
        arbiter.select_arm(scenario, tied, arm="oracle", soft_clip=10)


def test_scenario_constraint_6824_budgets_and_paired_joins_fail_closed() -> None:
    """SCENARIO-CONSTRAINT-6824-BUDGETS-AND-JOINS accepts each arm exactly once."""

    rows = [
        {
            "arm": arm,
            "candidate_count": 2,
            "cpu_allowance_us": 10,
            "exact_check_count": 2,
            "outcome_check_count": 1,
            "pair_id": "pair",
            "retry_cap": 1,
            "row_id": f"pair|{arm}",
            "row_type": "cold_replay",
            "work_units": 3,
        }
        for arm in reducer.ARMS
    ]
    assert list(reducer.join_pairs(rows)) == ["pair"]
    assert reducer.recompute_budgets(rows, expected_pair_count=1)["passed"] is True
    changed = deepcopy(rows)
    changed[1]["work_units"] = 4
    assert reducer.recompute_budgets(changed, expected_pair_count=1)["passed"] is False
    with pytest.raises(reducer.RowRosterError, match="paired arm roster"):
        reducer.join_pairs(rows[:1])
    with pytest.raises(reducer.RowRosterError, match="paired arm roster"):
        reducer.join_pairs([rows[0], deepcopy(rows[0])])
    assert reducer.recompute_budgets(rows[:1], expected_pair_count=1)["passed"] is False


def test_scenario_constraint_6824_intervals_are_deterministic() -> None:
    """SCENARIO-CONSTRAINT-6824-INTERVALS owns Wilson and paired arithmetic."""

    assert reducer.wilson_upper(0, 28, 0.05) == pytest.approx(0.12064330476584559)
    assert reducer.wilson_upper(0, 0, 0.05) is None
    first = reducer.paired_interval([1.0, 0.0, -1.0, 1.0], seed=44, resamples=200)
    second = reducer.paired_interval([1.0, 0.0, -1.0, 1.0], seed=44, resamples=200)
    assert first == second
    assert first["pair_count"] == 4
    assert reducer.paired_interval([], seed=44, resamples=10)["estimate"] is None


def test_scenario_constraint_6824_deletion_and_duplicate_audits() -> None:
    """SCENARIO-CONSTRAINT-6824-ROW-FAULTS rejects missing and repeated identities."""

    rows = [
        {"row_id": "a", "row_type": "cold_replay"},
        {"row_id": "b", "row_type": "cold_replay"},
    ]
    assert reducer.validate_roster(rows, ["a", "b"])["passed"] is True
    with pytest.raises(reducer.RowRosterError, match="missing"):
        reducer.validate_roster(rows[:1], ["a", "b"])
    with pytest.raises(reducer.RowRosterError, match="duplicate"):
        reducer.validate_roster([*rows, deepcopy(rows[0])], ["a", "b"])
    with pytest.raises(reducer.RowRosterError, match="extra"):
        reducer.validate_roster(rows, ["a"])
    cases = reducer.run_row_fault_audits(rows, ["a", "b"])
    assert [row["audit_case"] for row in cases] == ["row_deletion", "duplicate_row"]
    assert all(row["detected"] and row["aggregate_recompute_stopped"] for row in cases)


def test_req_constraint_6824_fault_audit_records_a_non_firing_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CONSTRAINT-6824 makes a silent row-fault guard visible as a failed audit."""

    monkeypatch.setattr(reducer, "validate_roster", lambda *_: {"passed": True})
    rows = [{"row_id": "a", "row_type": "cold_replay"}]
    cases = reducer.run_row_fault_audits(rows, ["a"])
    assert all(row["detected"] is False for row in cases)
    assert all(row["aggregate_recompute_stopped"] is False for row in cases)


def test_scenario_constraint_6824_full_replay_recomputes_rows_and_headlines(
    artifact: dict,
) -> None:
    """SCENARIO-CONSTRAINT-6824-AGGREGATION derives every claim from cold rows."""

    cold_rows = [row for row in artifact["rows"] if row["row_type"] == "cold_replay"]
    audit_rows = [row for row in artifact["rows"] if row["row_type"] == "row_fault_audit"]
    assert len(cold_rows) == replay.EXPECTED_REPLAY_ROW_COUNT == 576
    assert len(audit_rows) == 2
    assert artifact["row_coverage"]["passed"] is True
    assert artifact["row_coverage"]["expected_identity_count"] == 576
    assert artifact["row_coverage"]["observed_identity_count"] == 576
    assert artifact["budget_recomputation"]["passed"] is True
    assert artifact["safe_action_identity_recomputation"][arbiter.SELECTIVE_ARM]["rate"] == 1.0
    assert artifact["hard_violation_recomputation"][arbiter.SELECTIVE_ARM]["numerator"] == 0
    assert artifact["false_intervention_recomputation"][arbiter.SELECTIVE_ARM]["numerator"] == 0
    assert artifact["aggregate_recomputation"] == reducer.recompute_aggregates(
        cold_rows,
        held_scenario_ids=artifact["row_coverage"]["held_scenario_ids"],
        constants=artifact["row_coverage"]["public_constants"],
    )
    assert artifact["headline_differences"]["all_within_tolerance"] is True
    assert all(row["producer_row_match"] for row in cold_rows)
    assert artifact["source_verdict_supported"] is True
    assert artifact["cold_replay_shard_complete"] is True
    assert artifact["verdict_class"] == "positive"
    assert artifact["verifier_is_oracle"] is False
    assert set(artifact["field_principles"]) == set(artifact)
    assert replay.validate_artifact(artifact) == []


def test_req_constraint_6824_preconditions_and_blocked_shape(
    sources: dict[str, dict],
) -> None:
    """SCENARIO-CONSTRAINT-6824-PRECONDITIONS stops before incomplete replay."""

    passed = replay.check_preconditions(sources, SOURCE_PATHS)
    assert passed["passed"] is True
    assert passed["failed_checks"] == []
    assert {row["check"] for row in passed["checks"]} == {
        "raw_byte_manifests_readable",
        "complete_source_rows",
        "source_artifact_hashes",
        "frozen_arm_manifest",
        "selective_arbiter_ab_completed",
    }

    changed = deepcopy(sources)
    changed["exp6813"]["selective_arbiter_ab_completed"] = False
    changed["exp6812"]["rows"].pop()
    changed["exp6813"]["frozen_manifest"]["arms"] = ["changed"]
    blocked = replay.build_artifact(
        changed,
        source_paths=SOURCE_PATHS,
        run_date=replay.RUN_DATE,
        duration_s=0.25,
    )
    assert blocked["status"] == replay.BLOCKED_STATUS
    assert blocked["honest_verdict"].startswith(replay.BLOCKED_STATUS)
    assert blocked["verdict_class"] == "blocked"
    assert blocked["rows"] == []
    assert blocked["cold_replay_shard_complete"] is False
    assert blocked["gate_check_summary"]["failed_checks"] == [
        "complete_source_rows",
        "frozen_arm_manifest",
        "selective_arbiter_ab_completed",
    ]
    assert all(
        set(row) >= {"check", "expected", "observed", "passed"}
        for row in blocked["gate_check_summary"]["checks"]
    )
    corrupt = deepcopy(sources)
    corrupt["exp6812"]["raw_output_manifest"][0]["raw_output_sha256"] = "sha256:wrong"
    assert replay.check_preconditions(corrupt, SOURCE_PATHS)["failed_checks"] == [
        "raw_byte_manifests_readable"
    ]


def test_req_constraint_6824_modules_are_independent_and_content_addressed(
    artifact: dict,
) -> None:
    """REQ-CONSTRAINT-6824 records fresh module identities with no producer import."""

    for module, field in (
        (parser, "independent_parser_id"),
        (arbiter, "independent_arbiter_id"),
        (reducer, "independent_reducer_id"),
    ):
        source = inspect.getsource(module)
        assert "experiment_6813_selective_priority_arbiter_ab" not in source
        identity = artifact[field]
        assert identity["module"] == module.__name__
        assert identity["sha256"].startswith("sha256:")
        assert identity["imports_exp6813"] is False
    assert set(artifact["source_artifact_hashes"]) == {"exp6811", "exp6812", "exp6813"}


def test_req_constraint_6824_validator_and_terminal_decision(artifact: dict) -> None:
    """REQ-CONSTRAINT-6824 preserves positive, null, contradictory, and partial results."""

    assert replay.terminal_decision(True, True, True) == {
        "cold_replay_shard_complete": True,
        "source_verdict_supported": True,
        "verdict_class": "positive",
        "honest_verdict": "complete: cold row replay supports the positive producer verdict",
    }
    assert replay.terminal_decision(True, False, True)["verdict_class"] == "null"
    assert replay.terminal_decision(True, True, False)["source_verdict_supported"] is False
    assert replay.terminal_decision(False, True, True)["verdict_class"] == "partial"

    changed = deepcopy(artifact)
    changed["field_principles"] = {}
    changed["verifier_is_oracle"] = True
    changed["verdict_class"] = "invented"
    changed["honest_verdict"] = "unfinished"
    changed["duration_s"] = -1
    changed["row_coverage"]["passed"] = False
    findings = replay.validate_artifact(changed)
    for marker in (
        "field principle coverage mismatch",
        "verifier_is_oracle must be false",
        "verdict class outside closed enum",
        "honest verdict lacks terminal prefix",
        "duration_s must be non-negative",
        "row coverage is incomplete",
        "reproducibility checksum mismatch",
    ):
        assert marker in findings


def test_req_constraint_6824_writer_and_cli_use_explicit_output(
    artifact: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-CONSTRAINT-6824 validates one atomic write without touching results in tests."""

    target = tmp_path / "artifact.json"
    replay.write_output(artifact, target)
    assert replay.read_json_object(target) == artifact
    bad = tmp_path / "bad.json"
    bad.write_text("[]", encoding="utf-8")
    with pytest.raises(replay.ColdReplayError, match="object"):
        replay.read_json_object(bad)

    monkeypatch.setattr(replay, "build_from_repo", lambda **_: deepcopy(artifact))
    cli_target = tmp_path / "cli.json"
    assert replay.main(["--date", replay.RUN_DATE, "--output", str(cli_target)]) == 0
    assert replay.read_json_object(cli_target) == artifact
    assert artifact["honest_verdict"] in capsys.readouterr().out
    with pytest.raises(ValueError, match="YYYYMMDD"):
        replay.parse_run_date("2026-08-31")

    invalid = deepcopy(artifact)
    invalid["honest_verdict"] = "unfinished"
    with pytest.raises(replay.ColdReplayError, match="terminal prefix"):
        replay.write_output(invalid, tmp_path / "invalid.json")

    missing_path = tmp_path / "missing.json"
    assert replay.sha256_file(missing_path) == "missing"
    loaded = replay.load_sources({"missing": missing_path})
    assert "missing" in loaded["__load_errors__"]
    assert replay.validate_artifact({})[0].startswith("missing required fields")
