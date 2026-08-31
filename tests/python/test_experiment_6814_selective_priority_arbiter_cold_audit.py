"""Tests for the independent selective priority arbiter cold audit.

Spec refs: REQ-CONSTRAINT-6814, SCENARIO-CONSTRAINT-6814-PRECONDITIONS,
SCENARIO-CONSTRAINT-6814-INDEPENDENT, SCENARIO-CONSTRAINT-6814-AUTHORITY,
SCENARIO-CONSTRAINT-6814-SAFE-NO-OP, SCENARIO-CONSTRAINT-6814-AGGREGATION,
and SCENARIO-CONSTRAINT-6814-COMPLETION.
"""

from __future__ import annotations

import base64
from copy import deepcopy
import inspect
import json
from pathlib import Path

import pytest

from carnot import experiment_6814_selective_priority_arbiter_cold_audit as audit


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATHS = {
    name: REPO_ROOT / relative for name, relative in audit.SOURCE_RELATIVE_PATHS.items()
}


@pytest.fixture(scope="module")
def sources() -> dict[str, dict]:
    """Load immutable source objects once because cold replay does not edit them."""

    return audit.load_sources(SOURCE_PATHS)


@pytest.fixture(scope="module")
def artifact(sources: dict[str, dict]) -> dict:
    """Build one complete artifact for all row-derived assertions."""

    return audit.build_artifact(
        sources,
        source_paths=SOURCE_PATHS,
        run_date=audit.RUN_DATE,
        duration_s=1.25,
    )


def test_req_constraint_6814_spec_owns_every_required_field() -> None:
    """REQ-CONSTRAINT-6814 anchors code and evidence before implementation."""

    spec = (REPO_ROOT / audit.SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
    section = spec.split("### REQ-CONSTRAINT-6814", 1)[1]
    for marker in (
        "SCENARIO-CONSTRAINT-6814-PRECONDITIONS",
        "SCENARIO-CONSTRAINT-6814-INDEPENDENT",
        "SCENARIO-CONSTRAINT-6814-AUTHORITY",
        "SCENARIO-CONSTRAINT-6814-SAFE-NO-OP",
        "SCENARIO-CONSTRAINT-6814-AGGREGATION",
        "SCENARIO-CONSTRAINT-6814-COMPLETION",
        audit.MODULE_RELATIVE_PATH.as_posix(),
        audit.SCRIPT_RELATIVE_PATH.as_posix(),
        audit.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert marker in section
    for field in audit.REQUIRED_AUDIT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_constraint_6814_independent_source_has_no_producer_import() -> None:
    """SCENARIO-CONSTRAINT-6814-INDEPENDENT forbids producer implementation reuse."""

    source = inspect.getsource(audit)
    assert "experiment_6813_selective_priority_arbiter_ab import" not in source
    assert "from carnot import experiment_6813" not in source
    assert "experiment_6812_sota_operational_handoff_corpus_v2 import" not in source
    assert audit.INDEPENDENT_PARSER_ID in source
    assert audit.INDEPENDENT_ARBITER_ID in source
    assert audit.INDEPENDENT_REDUCER_ID in source


def test_scenario_constraint_6814_preconditions_pass_and_name_tampering(
    sources: dict[str, dict], tmp_path: Path
) -> None:
    """SCENARIO-CONSTRAINT-6814-PRECONDITIONS checks bytes, rows, budgets, and manifests."""

    summary = audit.check_preconditions(sources, SOURCE_PATHS)
    assert summary["all_passed"] is True
    assert summary["failed_checks"] == []
    assert {row["check"] for row in summary["checks"]} == {
        "selective_arbiter_ab_completed",
        "source_artifact_sha256s",
        "raw_bytes_and_hashes",
        "complete_source_rows",
        "complete_comparison_rows",
        "matched_budgets",
        "frozen_manifests",
    }

    changed = deepcopy(sources)
    changed["exp6813"]["selective_arbiter_ab_completed"] = False
    changed["exp6813"]["rows"].pop()
    changed["exp6813"]["budget_match_receipt"]["passed"] = False
    changed["exp6812"]["frozen_manifest"]["prompts_frozen_before_inference"] = False
    changed["exp6812"]["raw_output_manifest"][0]["raw_output_sha256"] = "sha256:wrong"
    copied = tmp_path / "exp6813.json"
    copied.write_text(json.dumps(changed["exp6813"]), encoding="utf-8")
    paths = dict(SOURCE_PATHS)
    paths["exp6813"] = copied
    failed = audit.check_preconditions(changed, paths)
    assert failed["all_passed"] is False
    assert failed["failed_checks"] == [
        "selective_arbiter_ab_completed",
        "source_artifact_sha256s",
        "raw_bytes_and_hashes",
        "complete_comparison_rows",
        "matched_budgets",
        "frozen_manifests",
    ]
    assert all("expected" in row and "observed" in row for row in failed["failures"])


def test_scenario_constraint_6814_raw_sse_and_strict_parser(sources: dict[str, dict]) -> None:
    """SCENARIO-CONSTRAINT-6814-INDEPENDENT reparses exact API and proposal bytes."""

    receipt = sources["exp6812"]["raw_output_manifest"][0]
    api_bytes = base64.b64decode(receipt["raw_api_response_b64"], validate=True)
    output_bytes = base64.b64decode(receipt["raw_output_b64"], validate=True)
    assert audit.decode_sse_bytes(api_bytes) == output_bytes
    parsed = audit.parse_raw_output(output_bytes)
    assert parsed["parse_state"] == "complete"
    assert [row["candidate_id"] for row in parsed["candidates"]] == [
        "candidate_0",
        "candidate_1",
    ]

    invalid_values = (
        (b"", "empty_output"),
        (b"\xff", "json_decode_error"),
        (b"[]", "invalid_top_level_fields"),
        (b'{"wrong":[]}', "invalid_top_level_fields"),
        (b'{"candidates":[]}', "invalid_candidate_count"),
        (b'{"candidates":[{},{}]}', "invalid_candidate_fields"),
    )
    for raw, reason in invalid_values:
        assert audit.parse_raw_output(raw)["parse_failure"] == reason
    with pytest.raises(audit.AuditInputError, match="data line"):
        audit.decode_sse_bytes(b"not an event stream")
    with pytest.raises(audit.AuditInputError, match="JSON"):
        audit.decode_sse_bytes(b"data: nope\n\n")


def test_req_constraint_6814_parser_rejects_each_candidate_schema_error() -> None:
    """REQ-CONSTRAINT-6814 does not extract, repair, or coerce proposal fields."""

    valid = {
        "candidates": [
            {
                "action": {"data": 0, "kind": "ACT"},
                "authority_chain": ["system"],
                "candidate_id": f"candidate_{index}",
                "soft_progress": index,
            }
            for index in range(2)
        ]
    }

    def encoded(value: object) -> bytes:
        return json.dumps(value, separators=(",", ":")).encode("utf-8")

    mutations = []
    wrong_fields = deepcopy(valid)
    wrong_fields["candidates"][0]["extra"] = True
    mutations.append((wrong_fields, "invalid_candidate_fields"))
    wrong_action = deepcopy(valid)
    wrong_action["candidates"][0]["action"] = {"kind": "ACT"}
    mutations.append((wrong_action, "invalid_action"))
    bool_kind = deepcopy(valid)
    bool_kind["candidates"][0]["action"]["kind"] = True
    mutations.append((bool_kind, "invalid_action"))
    authorities = deepcopy(valid)
    authorities["candidates"][0]["authority_chain"] = ["z", "a"]
    mutations.append((authorities, "invalid_authority_chain"))
    wrong_id = deepcopy(valid)
    wrong_id["candidates"][1]["candidate_id"] = "candidate_0"
    mutations.append((wrong_id, "invalid_candidate_id"))
    bool_progress = deepcopy(valid)
    bool_progress["candidates"][0]["soft_progress"] = True
    mutations.append((bool_progress, "invalid_soft_progress"))
    for value, reason in mutations:
        assert audit.parse_raw_output(encoded(value))["parse_failure"] == reason


def test_scenario_constraint_6814_rebuilds_authority_and_transitions(
    sources: dict[str, dict]
) -> None:
    """SCENARIO-CONSTRAINT-6814-AUTHORITY rebuilds contracts without source labels."""

    scenarios = {
        row["scenario_id"]: row for row in sources["exp6812"]["frozen_manifest"]["scenarios"]
    }
    competing = scenarios["competing_authorities-01"]
    rebuilt = audit.rebuild_obligations(competing)
    assert [row["priority_class"] for row in rebuilt] == ["hard", "binding"]
    assert [row["authority_order"] for row in rebuilt] == [0, 1]

    valid = deepcopy(competing["reference_candidates"][0])
    exact = audit.evaluate_candidate(competing, valid)
    assert exact["legal_support"] is True
    assert exact["hard_violation_count"] == 0
    assert exact["binding_violation_vector"] == [0]
    assert exact["transition"]["add"] == ["bound_01", "done_01"]

    wrong = deepcopy(valid)
    wrong["authority_chain"] = ["operator_authority"]
    rejected = audit.evaluate_candidate(competing, wrong)
    assert rejected["legal_support"] is False
    assert rejected["first_conflict"] == "obl_01_hard"
    assert rejected["hard_violation_count"] == 1

    malformed = deepcopy(competing)
    del malformed["obligations"][0]["contract"]["fallback"]
    with pytest.raises(audit.AuditInputError, match="fallback"):
        audit.rebuild_obligations(malformed)


def test_scenario_constraint_6814_selection_is_lexicographic_and_label_blind() -> None:
    """SCENARIO-CONSTRAINT-6814-AUTHORITY fixes priorities and blocks label influence."""

    def candidate(index: int, hard: int, binding: list[int], soft: int) -> dict:
        return {
            "candidate_id": f"candidate_{index}",
            "candidate_index": index,
            "action": {"data": index, "kind": "ACT"},
            "parse_state": "complete",
            "hard_violation_count": hard,
            "binding_violation_vector": binding,
            "soft_score": soft,
            "legal_support": hard == 0,
            "authority_preserved": True,
            "first_conflict": "hard" if hard else None,
        }

    candidates = [candidate(0, 1, [0], 10**30), candidate(1, 0, [1], -10**30)]
    clean = audit.select_arm(candidates, arm=audit.SELECTIVE_ARM, fallback_action={"kind": "NOOP"})
    assert clean["selected_candidate_id"] == "candidate_1"
    tied = [candidate(1, 0, [0], 5), candidate(0, 0, [0], 5)]
    assert audit.select_arm(
        tied, arm=audit.SELECTIVE_ARM, fallback_action={"kind": "NOOP"}
    )["selected_candidate_id"] == "candidate_0"

    attacked = deepcopy(candidates)
    attacked[0].update(
        model_id="preferred",
        exact_valid=True,
        exact_utility=10**100,
        future_outcome="win",
    )
    attacked[1].update(
        model_id="other",
        exact_valid=False,
        exact_utility=-(10**100),
        future_outcome="loss",
    )
    assert audit.select_arm(
        attacked, arm=audit.SELECTIVE_ARM, fallback_action={"kind": "NOOP"}
    )["selected_candidate_id"] == clean["selected_candidate_id"]
    with pytest.raises(ValueError, match="unknown arm"):
        audit.select_arm(candidates, arm="oracle", fallback_action={"kind": "NOOP"})


def test_req_constraint_6814_intervals_are_independent_and_deterministic() -> None:
    """REQ-CONSTRAINT-6814 owns Wilson and paired interval arithmetic."""

    assert audit.wilson_upper(0, 28, 0.05) == pytest.approx(0.12064330476584559)
    assert audit.wilson_upper(0, 0, 0.05) is None
    first = audit.paired_interval([1.0, 0.0, -1.0, 1.0], seed=44, resamples=200)
    second = audit.paired_interval([1.0, 0.0, -1.0, 1.0], seed=44, resamples=200)
    assert first == second
    assert first["pair_count"] == 4
    assert audit.paired_interval([], seed=44, resamples=10)["estimate"] is None


def test_scenario_constraint_6814_full_replay_recomputes_every_headline(
    artifact: dict,
) -> None:
    """SCENARIO-CONSTRAINT-6814-AGGREGATION derives claims from complete cold rows."""

    assert artifact["selective_arbiter_audit_completed"] is True
    assert len(artifact["rows"]) == audit.EXPECTED_REPLAY_ROW_COUNT + len(audit.ATTACK_IDS)
    assert artifact["source_artifact_sha256s"] == audit.EXPECTED_SOURCE_SHA256S
    assert artifact["independent_parser_id"] == audit.INDEPENDENT_PARSER_ID
    assert artifact["independent_arbiter_id"] == audit.INDEPENDENT_ARBITER_ID
    assert artifact["independent_reducer_id"] == audit.INDEPENDENT_REDUCER_ID
    assert artifact["headline_differences"]["all_within_tolerance"] is True
    assert all(
        row["within_tolerance"] for row in artifact["headline_differences"]["comparisons"]
    )
    replay_rows = [row for row in artifact["rows"] if row["row_type"] == "cold_replay"]
    assert artifact["aggregate_recomputation"] == audit.recompute_aggregates(
        replay_rows,
        interval_seed=audit.INTERVAL_SEED,
        interval_resamples=audit.INTERVAL_RESAMPLES,
    )
    assert artifact["budget_recomputation"]["passed"] is True
    assert artifact["hard_safety_supported"] is True
    assert artifact["utility_claim_supported"] is True
    assert artifact["source_verdict_supported"] is True
    assert artifact["verdict_class"] == "positive"
    assert artifact["verifier_is_oracle"] is False
    assert set(artifact["field_principles"]) == set(artifact)
    assert audit.validate_artifact(artifact) == []


def test_scenario_constraint_6814_attacks_and_safe_no_op_are_row_supported(
    artifact: dict,
) -> None:
    """SCENARIO-CONSTRAINT-6814-SAFE-NO-OP checks attacks, identity, and certificates."""

    attacks = artifact["priority_attack_results"]
    assert [row["attack_id"] for row in attacks] == list(audit.ATTACK_IDS)
    assert all(row["passed"] and row["failed_closed"] for row in attacks)
    assert len([row for row in artifact["rows"] if row["row_type"] == "attack"]) == len(
        audit.ATTACK_IDS
    )
    assert all(
        finding["proposal_time_influence"] is False
        for finding in artifact["prohibited_feature_findings"]
    )
    false = artifact["false_intervention_recomputation"]
    assert false[audit.SELECTIVE_ARM]["numerator"] == 0
    assert false[audit.SELECTIVE_ARM]["byte_identity_rate"] == 1.0
    assert artifact["certificate_findings"]["all_local_conflicts_true"] is True
    assert artifact["certificate_findings"]["all_first_priority_true"] is True


def test_scenario_constraint_6814_completion_does_not_depend_on_effect_sign(
    artifact: dict,
) -> None:
    """SCENARIO-CONSTRAINT-6814-COMPLETION preserves null or harmful effects."""

    positive = audit.terminal_decision(
        completed=True,
        authority_supported=True,
        rows_supported=True,
        utility_effect=0.2,
        source_claim_positive=True,
    )
    null = audit.terminal_decision(
        completed=True,
        authority_supported=True,
        rows_supported=True,
        utility_effect=0.0,
        source_claim_positive=True,
    )
    harmful = audit.terminal_decision(
        completed=True,
        authority_supported=True,
        rows_supported=True,
        utility_effect=-0.2,
        source_claim_positive=True,
    )
    assert positive["selective_arbiter_audit_completed"] is True
    assert null["selective_arbiter_audit_completed"] is True
    assert harmful["selective_arbiter_audit_completed"] is True
    assert [positive["verdict_class"], null["verdict_class"], harmful["verdict_class"]] == [
        "positive",
        "null",
        "null",
    ]
    assert null["source_verdict_supported"] is False
    disqualified = audit.terminal_decision(
        completed=True,
        authority_supported=False,
        rows_supported=True,
        utility_effect=1.0,
        source_claim_positive=True,
    )
    assert disqualified["verdict_class"] == "disqualified"
    partial = audit.terminal_decision(
        completed=True,
        authority_supported=True,
        rows_supported=False,
        utility_effect=1.0,
        source_claim_positive=True,
    )
    assert partial["verdict_class"] == "partial"
    assert artifact["selective_arbiter_audit_completed"] is True


def test_req_constraint_6814_blocked_shape_stops_before_replay(
    sources: dict[str, dict]
) -> None:
    """SCENARIO-CONSTRAINT-6814-PRECONDITIONS emits no fallback comparison rows."""

    changed = deepcopy(sources)
    changed["exp6813"]["selective_arbiter_ab_completed"] = False
    blocked = audit.build_artifact(
        changed,
        source_paths=SOURCE_PATHS,
        run_date=audit.RUN_DATE,
        duration_s=0.25,
    )
    assert set(blocked) == set(audit.REQUIRED_ARTIFACT_FIELDS)
    assert blocked["status"] == audit.BLOCKED_STATUS
    assert blocked["honest_verdict"].startswith(audit.BLOCKED_STATUS)
    assert blocked["verdict_class"] == "blocked"
    assert blocked["rows"] == []
    assert blocked["selective_arbiter_audit_completed"] is False
    assert blocked["source_verdict_supported"] is False
    assert blocked["gate_check_summary"]["failed_checks"]
    assert set(blocked["field_principles"]) == set(blocked)
    assert audit.validate_artifact(blocked) == []


def test_req_constraint_6814_validator_detects_row_and_schema_drift(artifact: dict) -> None:
    """REQ-CONSTRAINT-6814 rejects non-row-derived headlines and malformed terminals."""

    changed = deepcopy(artifact)
    changed["aggregate_recomputation"]["paired_progress_delta"]["estimate"] += 1.0
    changed["field_principles"] = {}
    changed["verifier_is_oracle"] = True
    changed["verdict_class"] = "invented"
    changed["honest_verdict"] = "unfinished"
    changed["duration_s"] = -1.0
    findings = audit.validate_artifact(changed)
    for marker in (
        "field principle coverage mismatch",
        "verifier_is_oracle must be false",
        "verdict class outside closed enum",
        "honest verdict lacks terminal prefix",
        "duration_s must be non-negative",
        "aggregate recomputation is not row-derived",
        "reproducibility checksum mismatch",
    ):
        assert marker in findings


def test_req_constraint_6814_writer_and_cli_use_explicit_test_output(
    artifact: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-CONSTRAINT-6814 validates before one atomic task-owned write."""

    target = tmp_path / "artifact.json"
    audit.write_output(artifact, target)
    assert audit.read_json_object(target) == artifact
    with pytest.raises(audit.AuditInputError, match="object"):
        bad = tmp_path / "bad.json"
        bad.write_text("[]", encoding="utf-8")
        audit.read_json_object(bad)

    monkeypatch.setattr(audit, "build_from_repo", lambda **_: deepcopy(artifact))
    cli_target = tmp_path / "cli.json"
    assert audit.main(["--date", audit.RUN_DATE, "--output", str(cli_target)]) == 0
    assert audit.read_json_object(cli_target) == artifact
    assert artifact["honest_verdict"] in capsys.readouterr().out
    with pytest.raises(ValueError, match="YYYYMMDD"):
        audit.parse_run_date("2026-08-31")

