"""Tests for the exact relation-program prefix-viability fixture.

Spec refs: REQ-CONSTRAINT-6919 and SCENARIO-CONSTRAINT-6919-*.
"""

from __future__ import annotations

import builtins
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6919_exact_prefix_viability_fixture as mod


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def graph_fixture() -> mod.RelationFixture:
    """Return one small family fixture for direct engine checks."""

    return mod.build_relation_fixture("graph_coloring", 0)


@pytest.fixture(scope="module")
def ready_artifact() -> dict[str, object]:
    """Build the full canary once because it invokes both exact engines."""

    return mod.build_artifact(date="20260903", repo_root=ROOT)


def _line(fixture: mod.RelationFixture, subject_index: int, value_index: int) -> str:
    return fixture.action_line(fixture.subjects[subject_index], fixture.values[value_index])


def test_req_constraint_6919_spec_owns_the_full_contract() -> None:
    """REQ-CONSTRAINT-6919 declares every field and required scenario before code."""

    text = (ROOT / mod.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("### REQ-CONSTRAINT-6919") :]
    scenarios = (
        "EMPTY",
        "COMPLETION",
        "IMPOSSIBLE",
        "LATE",
        "DUPLICATE",
        "UNSUPPORTED",
        "AMBIGUOUS",
        "NO-HEADROOM",
        "DISAGREEMENT",
        "ORDER",
        "RETIRED",
    )

    assert all(f"SCENARIO-CONSTRAINT-6919-{name}" in section for name in scenarios)
    assert all(f"`{field}`" in section for field in mod.REQUIRED_ARTIFACT_FIELDS)


def test_scenario_constraint_6919_empty_and_ambiguous_prefix(
    graph_fixture: mod.RelationFixture,
) -> None:
    """SCENARIO-CONSTRAINT-6919-EMPTY and -AMBIGUOUS keep all exact completions."""

    direct = mod.direct_prefix_viability(graph_fixture, ())
    final = mod.final_engine_prefix_viability(graph_fixture, ())

    assert direct.extendable is True
    assert direct.completion_count == 6
    assert direct.ambiguous is True
    assert direct.witness
    assert final.extendable == direct.extendable
    assert final.completion_count == direct.completion_count
    assert final.witness == direct.witness


def test_scenario_constraint_6919_valid_completion_and_order_are_set_based(
    graph_fixture: mod.RelationFixture,
) -> None:
    """SCENARIO-CONSTRAINT-6919-COMPLETION and -ORDER preserve set meaning."""

    prefix = (_line(graph_fixture, 0, 0), _line(graph_fixture, 1, 1))
    reversed_prefix = tuple(reversed(prefix))
    reports = [
        mod.direct_prefix_viability(graph_fixture, prefix),
        mod.direct_prefix_viability(graph_fixture, reversed_prefix),
        mod.final_engine_prefix_viability(graph_fixture, prefix),
        mod.final_engine_prefix_viability(graph_fixture, reversed_prefix),
    ]

    assert all(report.extendable for report in reports)
    assert all(report.completion_count == 1 for report in reports)
    assert all(report.available_headroom == 0 for report in reports)
    assert {report.witness for report in reports} == {tuple(sorted(prefix))}


def test_scenario_constraint_6919_immediate_and_late_impossibility(
    graph_fixture: mod.RelationFixture,
) -> None:
    """SCENARIO-CONSTRAINT-6919-IMPOSSIBLE and -LATE distinguish failure time."""

    first = _line(graph_fixture, 0, 0)
    forbidden = graph_fixture.action_line(graph_fixture.subjects[0], graph_fixture.forbidden_value)
    late = (first, _line(graph_fixture, 0, 1))
    first_report = mod.direct_prefix_viability(graph_fixture, (first,))
    immediate_reports = (
        mod.direct_prefix_viability(graph_fixture, (forbidden,)),
        mod.final_engine_prefix_viability(graph_fixture, (forbidden,)),
    )
    late_reports = (
        mod.direct_prefix_viability(graph_fixture, late),
        mod.final_engine_prefix_viability(graph_fixture, late),
    )

    assert first_report.extendable is True
    assert all(not report.extendable for report in immediate_reports)
    assert all(report.reason == "immediate_constraint" for report in immediate_reports)
    assert all(not report.extendable for report in late_reports)
    assert all(report.reason == "contradiction" for report in late_reports)


def test_scenario_constraint_6919_duplicate_and_unsupported_fail_closed(
    graph_fixture: mod.RelationFixture,
) -> None:
    """SCENARIO-CONSTRAINT-6919-DUPLICATE and -UNSUPPORTED reject invalid lines."""

    line = _line(graph_fixture, 0, 0)
    unsupported = f"{graph_fixture.subjects[0]} {graph_fixture.predicate} ultraviolet"
    reports = {
        "direct_duplicate": mod.direct_prefix_viability(graph_fixture, (line, line)),
        "final_duplicate": mod.final_engine_prefix_viability(graph_fixture, (line, line)),
        "direct_unsupported": mod.direct_prefix_viability(graph_fixture, (unsupported,)),
        "final_unsupported": mod.final_engine_prefix_viability(graph_fixture, (unsupported,)),
    }

    assert reports["direct_duplicate"].reason == "duplicate_relation"
    assert reports["final_duplicate"].reason == "duplicate_relation"
    assert reports["direct_unsupported"].reason == "unsupported_atom"
    assert reports["final_unsupported"].reason == "unsupported_atom"
    assert all(not report.extendable for report in reports.values())


def test_scenario_constraint_6919_no_headroom_rejects_invalid_full_program(
    graph_fixture: mod.RelationFixture,
) -> None:
    """SCENARIO-CONSTRAINT-6919-NO-HEADROOM rejects a full same-color program."""

    prefix = (_line(graph_fixture, 0, 0), _line(graph_fixture, 1, 0))
    direct = mod.direct_prefix_viability(graph_fixture, prefix)
    final = mod.final_engine_prefix_viability(graph_fixture, prefix)

    assert direct.available_headroom == 0
    assert final.available_headroom == 0
    assert direct.reason == "family_constraint"
    assert final.reason == "family_constraint"
    assert direct.extendable is False
    assert final.extendable is False


def test_req_constraint_6919_state_and_parser_boundaries(
    graph_fixture: mod.RelationFixture,
) -> None:
    """REQ-CONSTRAINT-6919 bounds states and rejects malformed or oversized prefixes."""

    line = _line(graph_fixture, 0, 0)
    state = mod.BoundedRelationProgramState(graph_fixture, (line,))
    oversized = (line, _line(graph_fixture, 1, 1), _line(graph_fixture, 1, 2))

    assert state.available_headroom == 1
    assert mod.BoundedRelationProgramState(graph_fixture, oversized).available_headroom == 0
    assert mod.direct_prefix_viability(graph_fixture, oversized).reason == "over_capacity"
    assert mod.direct_prefix_viability(graph_fixture, ("not a plain relation line",)).reason == (
        "unsupported_atom"
    )
    assert mod.clingo_final_validity(graph_fixture, (line,)).reason == "incomplete_program"
    assert mod._direct_final_valid(graph_fixture, (line,)) is False
    assert mod._completion_candidates(graph_fixture, oversized) == ()
    with pytest.raises(ValueError, match="unsupported_family"):
        mod.build_relation_fixture("unknown", 0)


def test_req_constraint_6919_artifact_has_balanced_rows_and_exact_receipts(
    ready_artifact: dict[str, object],
) -> None:
    """REQ-CONSTRAINT-6919 emits balanced canary and held evidence for every case."""

    artifact = ready_artifact
    prefix_rows = artifact["prefix_case_rows"]
    split_rows = artifact["source_group_split_rows"]
    parity_rows = artifact["exact_engine_parity_rows"]
    case_counts = Counter(row["case_type"] for row in prefix_rows)
    family_counts = Counter(row["family"] for row in prefix_rows)
    split_groups = {
        split: {row["source_group"] for row in split_rows if row["split"] == split}
        for split in ("canary", "held")
    }

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(artifact["field_principles"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert len(prefix_rows) >= 120
    assert set(family_counts) == set(mod.FAMILIES)
    assert min(family_counts.values()) >= mod.MIN_CASES_PER_FAMILY
    assert set(mod.REQUIRED_CASE_TYPES) <= set(case_counts)
    assert min(case_counts[name] for name in mod.REQUIRED_CASE_TYPES) >= mod.MIN_CASES_PER_TYPE
    assert split_groups["canary"].isdisjoint(split_groups["held"])
    assert {row["split"] for row in prefix_rows} == {"canary", "held"}
    assert len(parity_rows) == len(prefix_rows)
    assert all(row["engines_agree"] for row in parity_rows)
    assert all(row["proof"] or row["witness"] for row in prefix_rows)
    assert all(row["decision_latency_ms"] >= 0 for row in prefix_rows)
    assert all(row["final_available_headroom"] >= 0 for row in prefix_rows)


def test_req_constraint_6919_branch_rows_and_terminal_gate_are_consistent(
    ready_artifact: dict[str, object],
) -> None:
    """REQ-CONSTRAINT-6919 keeps branch costs, exact decisions, and gates replayable."""

    artifact = ready_artifact
    rows = artifact["rows"]
    prefix_ids = {row["prefix_case_id"] for row in artifact["prefix_case_rows"]}

    assert rows
    assert all(
        {"source_id", "prefix", "branch", "in_loop_extendable", "final_engine_extendable"}
        <= set(row)
        for row in rows
    )
    assert {row["prefix_case_id"] for row in rows} == prefix_ids
    assert artifact["feasible_branch_rows"]
    assert artifact["rejected_branch_rows"]
    assert artifact["witness_rows"]
    assert len(artifact["latency_rows"]) == len(artifact["prefix_case_rows"])
    assert artifact["retired_mechanism_activation_count"] == 0
    assert artifact["model_inference_call_count"] == 0
    assert artifact["exact_engine_disagreement_count"] == 0
    assert artifact["prefix_viability_canary_ready_score"] == 1
    assert artifact["verifier_is_oracle"] is True
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["gate_check_summary"]["passed"] is True
    assert mod.payload_checksum(artifact) == artifact["reproducibility_checksum"]


def test_scenario_constraint_6919_engine_disagreement_blocks_readiness() -> None:
    """SCENARIO-CONSTRAINT-6919-DISAGREEMENT refuses an injected wrong engine."""

    artifact = mod.build_artifact(
        date="20260903",
        repo_root=ROOT,
        final_validator=lambda _fixture, _lines: False,
    )

    assert artifact["exact_engine_disagreement_count"] > 0
    assert artifact["prefix_viability_canary_ready_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == "complete_blocked_exact_prefix_viability_fixture"
    assert "exact_engine_parity" in artifact["gate_check_summary"]["failed_checks"]


def test_scenario_constraint_6919_retired_mechanism_activation_blocks() -> None:
    """SCENARIO-CONSTRAINT-6919-RETIRED makes any retired activation terminally blocked."""

    artifact = mod.build_artifact(
        date="20260903",
        repo_root=ROOT,
        retired_mechanism_activations={"finite_answer_id": 1},
    )

    assert artifact["retired_mechanism_activation_count"] == 1
    assert artifact["prefix_viability_canary_ready_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert "retired_mechanism_activation_count" in artifact["gate_check_summary"]["failed_checks"]


def test_req_constraint_6919_missing_preconditions_write_complete_blocker(
    tmp_path: Path,
) -> None:
    """REQ-CONSTRAINT-6919 writes all required fields when exact sources are absent."""

    output = tmp_path / "result.json"
    artifact = mod.run(date="20260903", repo_root=tmp_path, output_path=output)

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["prefix_viability_canary_ready_score"] == 0
    assert artifact["honest_verdict"] == "complete_blocked_exact_prefix_viability_fixture"
    assert artifact["gate_check_summary"]["failed_check"] == "qualified_exp6274_compiler"


def test_req_constraint_6919_preconditions_fail_when_clingo_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CONSTRAINT-6919 requires the independent final engine to import."""

    original_import = builtins.__import__

    def rejecting_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "clingo":
            raise ImportError("clingo unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", rejecting_import)
    checks = mod.check_preconditions(ROOT)

    assert checks["two_independent_exact_engines"]["passed"] is False
    assert checks["all_passed"] is False


def test_req_constraint_6919_json_reader_and_default_output_fail_closed(tmp_path: Path) -> None:
    """REQ-CONSTRAINT-6919 rejects non-object JSON and uses its canonical output path."""

    not_object = tmp_path / "not_object.json"
    not_object.write_text("[]", encoding="utf-8")
    artifact = mod.run(date="20260903", repo_root=tmp_path)

    assert mod._read_json(not_object) is None
    assert (tmp_path / mod.RESULT_PATH).is_file()
    assert artifact["verdict_class"] == "blocked"


def test_req_constraint_6919_artifact_validation_rejects_inconsistent_claims(
    ready_artifact: dict[str, object],
) -> None:
    """REQ-CONSTRAINT-6919 rejects missing fields, bad gates, and positive oracle claims."""

    variants: list[tuple[dict[str, object], str]] = []
    missing = deepcopy(ready_artifact)
    missing.pop("rows")
    variants.append((missing, "missing_required_fields"))
    no_principle = deepcopy(ready_artifact)
    no_principle["field_principles"].pop("rows")
    variants.append((no_principle, "missing_field_principles"))
    bad_score = deepcopy(ready_artifact)
    bad_score["prefix_viability_canary_ready_score"] = 2
    variants.append((bad_score, "invalid_ready_score"))
    bad_gate = deepcopy(ready_artifact)
    bad_gate["gate_check_summary"]["passed"] = False
    variants.append((bad_gate, "ready_gate_disagreement"))
    positive = deepcopy(ready_artifact)
    positive["verdict_class"] = "positive"
    variants.append((positive, "oracle_verdict_cannot_be_positive"))
    unterminated = deepcopy(ready_artifact)
    unterminated["honest_verdict"] = "blocked"
    variants.append((unterminated, "honest_verdict_not_terminal"))

    for artifact, message in variants:
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(artifact)


def test_req_constraint_6919_main_reports_terminal_summary(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-CONSTRAINT-6919 exposes the required command-line run entry point."""

    summary_artifact = {
        "honest_verdict": "complete_fixture",
        "prefix_case_rows": [{"prefix_case_id": "one"}],
        "prefix_viability_canary_ready_score": 1,
    }
    monkeypatch.setattr(mod, "run", lambda *, date: summary_artifact)

    assert mod.main(["--date", "20260903"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["honest_verdict"] == "complete_fixture"
    assert output["prefix_case_count"] == 1
