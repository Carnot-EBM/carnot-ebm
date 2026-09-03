"""Tests for the deterministic span-first relation fixture.

Spec refs: REQ-VERIFY-6926 and SCENARIO-VERIFY-6926-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6926_span_first_relation_fixture as exp


def _proposal(span_a: str, span_b: str, relation: str) -> str:
    return f"SPAN_A: {span_a}\nSPAN_B: {span_b}\nRELATION: A -> {relation} -> B"


def _passing_checks() -> list[dict[str, object]]:
    return [exp.gate_check("test_preconditions", True, True)]


def test_req_verify_6926_spec_precedes_implementation() -> None:
    """REQ-VERIFY-6926 declares each failure boundary before code exists."""

    text = (exp.REPO_ROOT / exp.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("### REQ-VERIFY-6926") :]
    for suffix in (
        "PRECONDITIONS",
        "PROTOCOL",
        "BYTES",
        "AMBIGUITY",
        "SPAN-FAILURES",
        "SEMANTICS",
        "PARAPHRASE",
        "ASP",
        "ISOMORPHISM",
        "PARTITIONS",
        "REPLAY",
        "READINESS",
    ):
        assert f"SCENARIO-VERIFY-6926-{suffix}" in section


def test_unicode_spans_resolve_to_utf8_bytes_before_relation() -> None:
    """SCENARIO-VERIFY-6926-BYTES freezes bytes before relation semantics."""

    source = "Préface. Node Ω has color bleu."
    parsed = exp.parse_proposal(source.encode("utf-8"), _proposal("Node Ω", "bleu", "has color"))
    assert parsed["status"] == "accepted"
    assert parsed["span_a"]["start_utf8"] == source.encode("utf-8").index("Node Ω".encode())
    assert parsed["span_b"]["text"] == "bleu"
    assert parsed["relation_interpreted"] is True
    assert parsed["canonical_tuple"] == ["Node Ω", "has_color", "bleu", "positive"]


@pytest.mark.parametrize(
    ("proposal", "reason"),
    [
        ("SPAN_A: A\nSPAN_B: B", "line_count_not_three"),
        ("SPAN_A: A\nSPAN_B: B\nRELATION: A -> has color -> B\nEXTRA", "line_count_not_three"),
        ("SPAN_B: B\nSPAN_A: A\nRELATION: A -> has color -> B", "malformed_span_a_line"),
        ("SPAN_A:\nSPAN_B: B\nRELATION: A -> has color -> B", "malformed_span_a_line"),
        ("SPAN_A: A\nSPAN_B: B\nREL: A -> has color -> B", "malformed_relation_line"),
        ("SPAN_A: A\r\nSPAN_B: B\r\nRELATION: A -> has color -> B", "carriage_return_not_allowed"),
    ],
)
def test_plain_text_protocol_rejects_malformed_lines(proposal: str, reason: str) -> None:
    """SCENARIO-VERIFY-6926-PROTOCOL rejects malformed three-line output."""

    result = exp.parse_proposal(b"A then B", proposal)
    assert result["status"] == "rejected"
    assert result["reason"] == reason


def test_repeated_span_is_explicitly_ambiguous() -> None:
    """SCENARIO-VERIFY-6926-AMBIGUITY retains all duplicate byte offsets."""

    result = exp.parse_proposal(b"A links B and A links C", _proposal("A", "B", "precedes"))
    assert result["reason"] == "span_a_ambiguous"
    assert result["relation_interpreted"] is False
    assert result["span_a_candidates"] == [
        {"start_utf8": 0, "end_utf8": 1},
        {"start_utf8": 14, "end_utf8": 15},
    ]


@pytest.mark.parametrize(
    ("source", "proposal", "reason"),
    [
        (b"A then B", _proposal("missing", "B", "precedes"), "span_a_absent"),
        (b"A then B", _proposal("A", "missing", "precedes"), "span_b_absent"),
        (b"tokenAB", _proposal("tokenA", "AB", "precedes"), "spans_overlap"),
        (b"B appears before A", _proposal("A", "B", "precedes"), "span_direction_reversed"),
        (b"\xff A then B", _proposal("A", "B", "precedes"), "source_invalid_utf8"),
    ],
)
def test_span_failures_happen_before_relation_interpretation(
    source: bytes, proposal: str, reason: str
) -> None:
    """SCENARIO-VERIFY-6926-SPAN-FAILURES fails before reading labels."""

    result = exp.parse_proposal(source, proposal)
    assert result["reason"] == reason
    assert result["relation_interpreted"] is False


def test_relation_phrase_is_read_only_after_spans_resolve() -> None:
    """SCENARIO-VERIFY-6926-BYTES keeps unsupported labels after byte receipts."""

    result = exp.parse_proposal(b"A then B", _proposal("A", "B", "invented relation"))
    assert result["reason"] == "unsupported_relation_phrase"
    assert result["span_a"]["text"] == "A"
    assert result["span_b"]["text"] == "B"
    assert result["relation_interpreted"] is True


def test_fixture_has_every_family_label_cell_and_balanced_counts() -> None:
    """SCENARIO-VERIFY-6926-SEMANTICS covers the full family-label product."""

    rows = exp.generate_fixture_rows(exp.RANDOM_SEED)
    expected = {(family, label) for family in exp.FAMILIES for label in exp.LABELS}
    assert len(rows) == len(expected) == 15
    assert {(row["family"], row["label"]) for row in rows} == expected
    assert all(row["source_text_hash"].startswith("sha256:") for row in rows)
    assert all(row["content_hash"].startswith("sha256:") for row in rows)


def test_negation_unknown_and_wrong_label_semantics_are_exact() -> None:
    """SCENARIO-VERIFY-6926-SEMANTICS separates negative and unknown labels."""

    rows = exp.generate_fixture_rows(exp.RANDOM_SEED)
    negative = next(
        row for row in rows if row["family"] == "allocation" and row["label"] == "negative"
    )
    unknown = next(
        row for row in rows if row["family"] == "exclusion" and row["label"] == "unknown"
    )
    for row in (negative, unknown):
        parsed = exp.parse_proposal(row["source_bytes"], row["protocol_text"])
        checked = exp.check_relation_semantics(parsed, row)
        assert checked["passed"] is True
        assert checked["canonical_tuple"][3] == row["label"]

    wrong = exp.parse_proposal(
        negative["source_bytes"],
        _proposal(negative["span_a_text"], negative["span_b_text"], "is allocated to"),
    )
    checked = exp.check_relation_semantics(wrong, negative)
    assert checked["passed"] is False
    assert checked["reason"] == "relation_label_conflicts_source"


def test_registered_paraphrases_preserve_tuple_and_asp_effect() -> None:
    """SCENARIO-VERIFY-6926-PARAPHRASE maps aliases to the same exact effect."""

    for row in exp.generate_fixture_rows(exp.RANDOM_SEED):
        base = exp.parse_proposal(row["source_bytes"], row["protocol_text"])
        paraphrase = exp.parse_proposal(row["source_bytes"], row["paraphrase_protocol_text"])
        assert paraphrase["canonical_tuple"] == base["canonical_tuple"]
        assert exp.asp_program_for_tuple(paraphrase["canonical_tuple"], row) == row["asp_program"]


def test_exact_asp_engines_match_all_frozen_consequences() -> None:
    """SCENARIO-VERIFY-6926-ASP compares qualified and independent exact engines."""

    for row in exp.generate_fixture_rows(exp.RANDOM_SEED):
        receipt = exp.evaluate_asp_effect(row)
        assert receipt["solver_parity"] is True
        assert receipt["expected_effect_met"] is True
        assert receipt["primary_effect"] == row["expected_asp_effect"]
        assert receipt["primary_solver_receipt"]["status"] == "complete"
        assert receipt["independent_solver_receipt"]["status"] == "complete"


def test_entity_renaming_is_injective_and_effect_invariant() -> None:
    """SCENARIO-VERIFY-6926-ISOMORPHISM preserves effects after projection."""

    row = exp.generate_fixture_rows(exp.RANDOM_SEED)[0]
    receipt = exp.evaluate_isomorphism(row)
    assert receipt["injective"] is True
    assert receipt["solver_parity"] is True
    assert receipt["projected_effect"] == row["expected_asp_effect"]
    assert receipt["invariant"] is True
    with pytest.raises(ValueError, match="non_injective_renaming"):
        exp.rename_asp_program(
            row["asp_program"], {row["positive_atom"]: "same", row["negative_atom"]: "same"}
        )


def test_partitions_are_hash_frozen_and_held_prompts_hide_expected_outputs() -> None:
    """SCENARIO-VERIFY-6926-PARTITIONS keeps held answers outside prompts."""

    fixtures = exp.generate_fixture_rows(exp.RANDOM_SEED)
    partition_rows, protocol_rows, manifest = exp.build_partitions(fixtures, exp.RANDOM_SEED)
    assert {row["partition"] for row in partition_rows} == {"calibration", "heldout"}
    assert len({row["fixture_id"] for row in partition_rows}) == len(fixtures)
    assert manifest["count"] == sum(row["partition"] == "heldout" for row in partition_rows)
    held_ids = {row["fixture_id"] for row in partition_rows if row["partition"] == "heldout"}
    for prompt in protocol_rows:
        assert "expected_tuple" not in prompt
        assert "asp_program" not in prompt
        assert "expected_asp_effect" not in prompt
        if prompt["fixture_id"] in held_ids:
            assert prompt["expected_output_exposed"] is False


def test_invalid_mutations_are_retained_with_exact_reasons() -> None:
    """REQ-VERIFY-6926 requires one terminal row for every invalid mutation."""

    fixtures = exp.generate_fixture_rows(exp.RANDOM_SEED)
    mutations = exp.build_mutation_rows(fixtures)
    invalid = [row for row in mutations if row["valid"] is False]
    assert {row["reason"] for row in invalid} >= {
        "span_a_ambiguous",
        "spans_overlap",
        "span_direction_reversed",
        "span_a_absent",
        "line_count_not_three",
        "malformed_span_a_line",
        "malformed_relation_line",
        "unsupported_relation_phrase",
        "relation_label_conflicts_source",
    }
    assert all(
        row["terminal"] is True and row["content_hash"].startswith("sha256:") for row in mutations
    )
    assert len([row for row in mutations if row["mutation"] == "protocol_replay"]) == 15
    assert len([row for row in mutations if row["mutation"] == "paraphrase"]) == 15
    assert len([row for row in mutations if row["mutation"] == "entity_renaming"]) == 15


def test_ready_artifact_has_all_fields_rows_principles_and_replay() -> None:
    """SCENARIO-VERIFY-6926-READINESS gates complete per-row exact evidence."""

    artifact = exp.build_artifact(
        date="20260903", duration_s=0.5, precondition_checks=_passing_checks()
    )
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["span_relation_fixture_ready_score"] == 1
    assert artifact["verifier_is_oracle"] is True
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["honest_verdict"] == exp.READY_VERDICT
    assert len(artifact["rows"]) == len(artifact["fixture_rows"]) + len(artifact["mutation_rows"])
    assert exp.validate_artifact(artifact) == []
    assert exp.replay_artifact(artifact)["agreement"] is True


def test_replay_detects_hash_row_and_ready_score_drift() -> None:
    """SCENARIO-VERIFY-6926-REPLAY rejects changed deterministic evidence."""

    artifact = exp.build_artifact(
        date="20260903", duration_s=0.5, precondition_checks=_passing_checks()
    )
    changed = deepcopy(artifact)
    changed["fixture_rows"][0]["span_a_text"] = "changed"
    replay = exp.replay_artifact(changed)
    assert replay["agreement"] is False
    assert "fixture_rows" in replay["mismatched_fields"]
    assert "reproducibility_checksum" in exp.validate_artifact(changed)

    changed = deepcopy(artifact)
    changed["span_relation_fixture_ready_score"] = 0
    assert "ready_score_drift" in exp.validate_artifact(changed)


def test_blocked_artifact_names_failed_precondition_and_is_complete() -> None:
    """SCENARIO-VERIFY-6926-PRECONDITIONS emits a complete blocked receipt."""

    checks = [exp.gate_check("source_rows_readable", True, False)]
    artifact = exp.build_artifact(date="20260903", duration_s=0.1, precondition_checks=checks)
    assert artifact["span_relation_fixture_ready_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == exp.BLOCKED_VERDICT
    assert artifact["gate_check_summary"]["failed_check"] == "source_rows_readable"
    assert artifact["gate_check_summary"]["expected"] is True
    assert artifact["gate_check_summary"]["observed"] is False
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert exp.validate_artifact(artifact) == []


def test_real_preconditions_pin_sources_checkers_and_utf8() -> None:
    """SCENARIO-VERIFY-6926-PRECONDITIONS checks the qualified local inputs."""

    checks, hashes = exp.check_preconditions()
    summary = exp.gate_summary(checks)
    assert summary["passed"] is True
    assert set(hashes) == set(exp.SOURCE_PATHS)
    assert all(
        receipt["observed_sha256"] == receipt["expected_sha256"] for receipt in hashes.values()
    )


def test_run_and_cli_write_atomically_then_verify_in_fresh_process(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6926-REPLAY covers the required command surfaces."""

    output = tmp_path / "artifact.json"
    artifact = exp.run_experiment(date="20260903", output_path=output)
    assert output.exists()
    assert (
        json.loads(output.read_text(encoding="utf-8"))["reproducibility_checksum"]
        == artifact["reproducibility_checksum"]
    )
    assert exp.main(["--verify", str(output)]) == 0

    second = tmp_path / "artifact-main.json"
    assert exp.main(["--date", "20260903", "--output", str(second)]) == 0
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{}", encoding="utf-8")
    assert exp.main(["--verify", str(invalid)]) == 1


def test_secondary_parser_and_semantic_failures_are_explicit() -> None:
    """SCENARIO-VERIFY-6926-PROTOCOL retains secondary failure paths."""

    malformed_b = exp.parse_proposal(
        b"A then B", "SPAN_A: A\nSPAN_B:\nRELATION: A -> precedes -> B"
    )
    ambiguous_b = exp.parse_proposal(b"A then B and B", _proposal("A", "B", "precedes"))
    rejected_semantics = exp.check_relation_semantics(malformed_b, {})
    assert malformed_b["reason"] == "malformed_span_b_line"
    assert ambiguous_b["reason"] == "span_b_ambiguous"
    assert rejected_semantics["reason"] == "malformed_span_b_line"

    fixture = exp.generate_fixture_rows(exp.RANDOM_SEED)[0]
    parsed = exp.parse_proposal(fixture["source_bytes"], fixture["protocol_text"])
    parsed["family"] = "scheduling"
    assert (
        exp.check_relation_semantics(parsed, fixture)["reason"]
        == "relation_family_conflicts_source"
    )


def test_asp_helpers_reject_values_outside_the_frozen_contract() -> None:
    """SCENARIO-VERIFY-6926-ASP rejects bad tuples and renaming maps."""

    fixture = exp.generate_fixture_rows(exp.RANDOM_SEED)[0]
    with pytest.raises(ValueError, match="canonical_tuple_arity"):
        exp.asp_program_for_tuple(["too", "short"], fixture)
    with pytest.raises(ValueError, match="unsupported_relation_label"):
        exp.asp_program_for_tuple(["A", "has_color", "B", "maybe"], fixture)
    with pytest.raises(ValueError, match="empty_renaming"):
        exp.rename_asp_program(fixture["asp_program"], {})


def test_blocked_validation_reports_each_terminal_inconsistency() -> None:
    """SCENARIO-VERIFY-6926-READINESS rejects inconsistent blocked fields."""

    artifact = exp.build_artifact(
        date="20260903",
        duration_s=0.1,
        precondition_checks=[exp.gate_check("source_rows_readable", True, False)],
    )
    artifact["span_relation_fixture_ready_score"] = 1
    artifact["gate_check_summary"]["passed"] = True
    artifact["honest_verdict"] = "complete_wrong"
    artifact["reproducibility_checksum"] = exp.artifact_checksum(artifact)
    assert set(exp.validate_artifact(artifact)) >= {
        "blocked_ready_score",
        "blocked_gate_summary",
        "blocked_honest_verdict",
    }


def test_preconditions_fail_closed_on_bad_json_and_missing_solver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-6926-PRECONDITIONS covers environment failures."""

    assert exp.sha256_file(exp.REPO_ROOT / "missing-6926-input") == "missing"
    monkeypatch.setattr(exp, "_source_receipts", lambda: {})
    monkeypatch.setattr(exp, "_read_json", lambda _path: (_ for _ in ()).throw(OSError("bad")))
    checks, _ = exp.check_preconditions()
    assert exp.gate_summary(checks)["failed_check"] == "source_rows_readable"

    clean = {"rows": [{"terminal": True}], "clean_relation_corpus_ready_score": 1}
    prior = {"rows": [{"eligible": True}] * 8}
    values = iter((clean, prior))
    monkeypatch.setattr(exp, "_read_json", lambda _path: next(values))
    real_import = __import__

    def without_clingo(name: str, *args: object, **kwargs: object) -> object:
        if name == "clingo":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", without_clingo)
    checks, _ = exp.check_preconditions()
    solver = next(row for row in checks if row["check"] == "independent_clingo_available")
    assert solver["passed"] is False


def test_run_and_verify_fail_before_writing_invalid_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-VERIFY-6926-REPLAY refuses invalid output and missing input."""

    assert exp.main(["--verify", str(tmp_path / "absent.json")]) == 1
    monkeypatch.setattr(exp, "check_preconditions", lambda: (_passing_checks(), {}))
    monkeypatch.setattr(exp, "validate_artifact", lambda _artifact: ["injected_failure"])
    output = tmp_path / "never.json"
    with pytest.raises(RuntimeError, match="artifact_validation_failed:injected_failure"):
        exp.run_experiment(date="20260903", output_path=output)
    assert not output.exists()
