"""Tests for Exp 3286 clean verifier abstention root-cause audit.

Spec refs: REQ-VERIFY-3286, SCENARIO-VERIFY-3286.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import clean_verifier_abstention_root_cause_v1 as mod


REQUIRED_FIELDS = {
    "abstention_root_cause_audit_ready",
    "abstention_root_cause_identified",
    "prior_abstention_rate",
    "audited_exact_row_count",
    "answerable_row_count",
    "malformed_or_missing_answer_count",
    "threshold_or_policy_findings",
    "parser_or_extraction_findings",
    "calibrated_rerun_plan",
    "target_max_abstention_rate",
    "target_false_accept_rate",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path | str, payload: dict[str, Any]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_context_fixture(root: Path) -> None:
    rows = [
        {
            "context": "For this fixture only, mercury means banana.",
            "exact_checker_type": "exact_alias_string",
            "expected_answer": "banana",
            "family": "symbolic_aliases",
            "fixture_id": "ctx-001",
            "minimal_counterexample": {"candidate_answer": "planet"},
            "prior_bait_answer": "planet",
            "question": "What does mercury mean?",
        },
        {
            "context": "For this fixture only, python means blue screwdriver.",
            "exact_checker_type": "exact_alias_string",
            "expected_answer": "blue screwdriver",
            "family": "symbolic_aliases",
            "fixture_id": "ctx-002",
            "minimal_counterexample": {"candidate_answer": "snake"},
            "prior_bait_answer": "snake",
            "question": "What does python mean?",
        },
    ]
    path = root / mod.CONTEXT_FIXTURE_REL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")


def _write_exp3275(root: Path, rows: list[dict[str, Any]], *, abstention_rate: float) -> None:
    _write_json(
        root,
        mod.EXP3275_REL_PATH,
        {
            "experiment_id": "exp3275",
            "abstention_rate": abstention_rate,
            "false_accept_rate": 0.0,
            "false_reject_rate": 0.0,
            "gate_reasons": ["abstention_rate_above_threshold"],
            "n_eval": len(rows),
            "per_row_results": rows,
            "preconditions_checked": [{"name": "exact_row_fixture_availability", "passed": True}],
            "random_seed": 3275,
            "thresholds": {"abstention_threshold": 0.5, "false_accept_threshold": 0.1},
            "honest_verdict": "complete: clean local SOTA verifier rerun v14 not ready",
        },
    )


def _write_exp3268(root: Path) -> None:
    _write_json(
        root,
        mod.EXP3268_REL_PATH,
        {
            "experiment_id": "exp3268",
            "clean_sota_receipt_eligible": True,
            "models_used": [{"model_id": "unsloth/gemma-4-26B-A4B-it-GGUF"}],
        },
    )


def _abstain_row(
    row_id: str,
    fixture_id: str,
    expected_decision: str,
    output_text: str,
    source_kind: str,
) -> dict[str, Any]:
    return {
        "abstained": True,
        "decision": "abstain",
        "exact_authority": "context_exact_checker",
        "expected_decision": expected_decision,
        "fixture_id": fixture_id,
        "model_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "output_text": output_text,
        "row_id": row_id,
        "source_candidate_kind": source_kind,
    }


def test_req_verify_3286_spec_anchor_declares_machine_readable_audit() -> None:
    """REQ-VERIFY-3286: OpenSpec declares the required audit schema."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/verification/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-VERIFY-3286" in spec
    assert "SCENARIO-VERIFY-3286" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "target_false_accept_rate=0.0" in spec
    assert "target_max_abstention_rate<1.0" in spec


def test_scenario_verify_3286_abstain_all_rows_are_parser_contract_failure(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-3286: abstain-all v14 is traced to unparseable outputs."""

    _write_exp3268(tmp_path)
    _write_context_fixture(tmp_path)
    _write_exp3275(
        tmp_path,
        [
            _abstain_row(
                "ctx-001:expected",
                "ctx-001",
                "accept",
                " context: For this fixture only",
                "fixture_expected_answer",
            ),
            _abstain_row(
                "ctx-001:counterexample",
                "ctx-001",
                "reject",
                "ro判断 判断",
                "fixture_minimal_counterexample",
            ),
            _abstain_row(
                "ctx-002:expected",
                "ctx-002",
                "accept",
                " least-most degrees of definition",
                "fixture_expected_answer",
            ),
            _abstain_row(
                "ctx-002:counterexample",
                "ctx-002",
                "reject",
                "ro due to",
                "fixture_minimal_counterexample",
            ),
        ],
        abstention_rate=1.0,
    )

    artifact = mod.build_artifact(
        tmp_path,
        started_s=10.0,
        now_s=14.25,
        tests_run=["SCENARIO-VERIFY-3286"],
    )

    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["abstention_root_cause_audit_ready"] is True
    assert artifact["abstention_root_cause_identified"] is True
    assert artifact["prior_abstention_rate"] == 1.0
    assert artifact["audited_exact_row_count"] == 4
    assert artifact["answerable_row_count"] == 4
    assert artifact["malformed_or_missing_answer_count"] == 0
    assert artifact["row_class_counts"] == {
        "answerable": 4,
        "malformed_or_missing_answer": 0,
        "unanswerable": 0,
        "unknown": 0,
    }
    assert {row["answerability"] for row in artifact["exact_row_audit_table"]} == {"answerable"}
    assert {row["abstention_reason"] for row in artifact["exact_row_audit_table"]} == {
        "model_output_unparseable"
    }
    assert any(
        finding["category"] == "model_output_contract_mismatch"
        for finding in artifact["parser_or_extraction_findings"]
    )
    assert any(
        finding["category"] == "abstention_threshold"
        for finding in artifact["threshold_or_policy_findings"]
    )
    assert artifact["target_false_accept_rate"] == 0.0
    assert artifact["target_max_abstention_rate"] == 0.5
    assert artifact["calibrated_rerun_plan"]["experiment_id"] == "exp3287"
    assert (
        artifact["calibrated_rerun_plan"]["acceptance_criteria"]["minimum_decision_coverage"] == 0.5
    )
    assert artifact["duration_s"] == pytest.approx(4.25)
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["honest_verdict"].startswith("complete:")
    mod.validate_artifact(artifact)


def test_req_verify_3286_row_table_separates_data_quality_classes(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3286: answerable, unanswerable, malformed, and unknown are distinct."""

    path = tmp_path / mod.CONTEXT_FIXTURE_REL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "fixture_id": "good",
                        "expected_answer": "yes",
                        "minimal_counterexample": {"candidate_answer": "no"},
                    },
                    sort_keys=True,
                ),
                json.dumps(
                    {
                        "fixture_id": "missing",
                        "expected_answer": "",
                        "minimal_counterexample": {"candidate_answer": "bad"},
                    },
                    sort_keys=True,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_exp3268(tmp_path)
    _write_exp3275(
        tmp_path,
        [
            _abstain_row("good:expected", "good", "accept", "ACCEPT", "fixture_expected_answer"),
            _abstain_row("skip:row", "good", "abstain", "ABSTAIN", "fixture_expected_answer"),
            _abstain_row(
                "missing:expected",
                "missing",
                "accept",
                "ABSTAIN",
                "fixture_expected_answer",
            ),
            _abstain_row("mystery:row", "none", "maybe", "", "fixture_unknown"),
        ],
        abstention_rate=0.75,
    )

    artifact = mod.build_artifact(tmp_path)

    assert artifact["row_class_counts"] == {
        "answerable": 1,
        "malformed_or_missing_answer": 1,
        "unanswerable": 1,
        "unknown": 1,
    }
    assert artifact["answerable_row_count"] == 1
    assert artifact["malformed_or_missing_answer_count"] == 1
    assert artifact["abstention_root_cause_identified"] is True


def test_req_verify_3286_writer_helpers_and_validation_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-3286: helpers keep the artifact terminal and reproducible."""

    _write_exp3268(tmp_path)
    _write_context_fixture(tmp_path)
    _write_exp3275(tmp_path, [], abstention_rate=0.0)

    output = mod.write_artifact(
        tmp_path,
        output_path=Path("results/out.json"),
        started_s=3.0,
        now_s=2.0,
        tests_run=["writer"],
    )
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / "results/out.json"
    assert saved["audited_exact_row_count"] == 0
    assert saved["abstention_root_cause_identified"] is False
    assert saved["duration_s"] == 0.0
    assert saved["tests_run"] == ["writer"]
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{bad\n", encoding="utf-8")
    assert mod.read_json_object(bad) == {}
    mixed_jsonl = tmp_path / "mixed.jsonl"
    mixed_jsonl.write_text('\n{bad}\n{"ok": true}\n[]\n', encoding="utf-8")
    assert mod.read_jsonl_objects(mixed_jsonl) == [{"ok": True}]
    assert mod.read_jsonl_objects(tmp_path / "missing.jsonl") == []
    assert mod.normalize_output_decision("ACCEPT.") == "accept"
    assert mod.normalize_output_decision("I would accept") is None
    assert mod.normalize_output_decision("") is None
    assert (
        mod.classify_answerability(
            expected_decision="reject",
            expected_answer="yes",
            candidate_answer="",
            source_kind="fixture_minimal_counterexample",
        )
        == "malformed_or_missing_answer"
    )
    assert mod.abstention_reason({"decision": "accept"}, "ACCEPT") == "not_abstained"
    assert mod.abstention_reason({"decision": "abstain"}, "ACCEPT") == "reported_abstain"
    assert (
        mod.dominant_root_cause(
            [{"answerability": "malformed_or_missing_answer"}],
            [],
            [],
        )
        == "missing_or_malformed_exact_answers"
    )
    assert (
        mod.dominant_root_cause(
            [{"answerability": "answerable", "abstention_reason": "reported_abstain"}],
            [{"blocked": True}],
            [],
        )
        == "threshold_blocked_nonzero_abstention_rate"
    )
    assert (
        mod.dominant_root_cause(
            [{"answerability": "answerable", "abstention_reason": "not_abstained"}],
            [],
            [{"category": "row_extraction"}],
        )
        == "parser_or_extraction_findings_present"
    )
    assert (
        mod.dominant_root_cause(
            [{"answerability": "answerable", "abstention_reason": "not_abstained"}],
            [],
            [],
        )
        == "unknown"
    )
    assert mod.bounded_float("bad", default=0.25) == 0.25
    assert mod.rate(1, 0) == 0.0
    assert mod.duration(4.0, 2.0) == 0.0
    assert mod.sha256_file(tmp_path / "none") is None

    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({})
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(saved | {"honest_verdict": "blocked"})
    with pytest.raises(ValueError, match="prior_abstention_rate"):
        mod.validate_artifact(saved | {"prior_abstention_rate": 2.0})
    with pytest.raises(ValueError, match="audited_exact_row_count"):
        mod.validate_artifact(saved | {"audited_exact_row_count": -1})
    with pytest.raises(ValueError, match="threshold_or_policy_findings"):
        mod.validate_artifact(saved | {"threshold_or_policy_findings": "bad"})
    with pytest.raises(ValueError, match="calibrated_rerun_plan"):
        mod.validate_artifact(saved | {"calibrated_rerun_plan": []})
    with pytest.raises(ValueError, match="target_false_accept_rate"):
        mod.validate_artifact(saved | {"target_false_accept_rate": 0.1})
    with pytest.raises(ValueError, match="target_max_abstention_rate"):
        mod.validate_artifact(saved | {"target_max_abstention_rate": 1.0})
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(saved | {"reproducibility_checksum": "bad"})
