"""Behavior tests for the V651 contract and method-ingestion audit.

Spec refs: REQ-REPORT-7421 and SCENARIO-REPORT-7421-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_7421_v651_contract_ingestion as ingestion
from carnot.reporting.experiment_7303_validation_scope import REQUIRED_CHECK_NAMES


ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.memory_watchdog_skip


def _markdown_for(roadmap: dict[str, Any]) -> str:
    """Render an independent table fixture from explicit YAML values."""

    lines = [
        "# Fixture",
        "",
        f"**Milestone:** `{roadmap['milestone']}`",
        "",
        "## Exact Task Contract",
        "",
        "| Order | Task ID | Exact title | Deliverable | Phase | Substrate class | Structured gate |",
        "|---|---|---|---|---|---|---|",
    ]
    for order, task in enumerate(roadmap["tasks"], 1):
        gates = []
        for gate in task.get("gated_on") or []:
            value = json.dumps(gate["value"], separators=(",", ":"))
            gates.append(f"{gate['upstream']}.{gate['artifact_field']} {gate['op']} {value}")
        gate_text = "; ".join(gates) if gates else "None"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(order),
                    task["id"],
                    task["title"],
                    task["deliverable"],
                    str(task["phase"]),
                    task["inference_substrate_class"],
                    gate_text,
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _receipt(name: str, passed: bool = True) -> dict[str, Any]:
    """Build one exact-shaped validation receipt without a child process."""

    row: dict[str, Any] = {
        "name": name,
        "command": f"check {name}",
        "command_argv": ["check", name],
        "command_environment": {},
        "scope": "test_fixture",
        "exit_code": 0 if passed else 1,
        "duration_s": 0.01,
        "log_path": f"/tmp/{name}.log",
        "log_sha256": "sha256:" + "1" * 64,
        "passed": passed,
        "timed_out": False,
        "output_tail": "ok" if passed else "failed",
    }
    if name == "worktree_imports":
        row["resolved_imports"] = {
            "carnot.experiment_7421_v651_contract_ingestion": str(
                (ROOT / ingestion.MODULE_PATH).resolve()
            )
        }
    return row


def _validation(passed: bool = True) -> dict[str, Any]:
    """Return the frozen Exp7303 receipt set."""

    return {
        "validation_receipts": [
            _receipt(name, passed)
            for name in (*REQUIRED_CHECK_NAMES, *ingestion.TERMINAL_CHECK_NAMES)
        ],
        "required_checks_passed": passed,
        "terminal_validation_passed": passed,
        "plan_errors": [],
    }


def _source_rows() -> list[dict[str, Any]]:
    """Return deterministic access rows for artifact tests."""

    return ingestion.check_selected_sources(
        lambda source: {
            "access_state": "failed"
            if source["source_id"] == "semantic_scholar_citations"
            else "ok",
            "http_status": 429 if source["source_id"] == "semantic_scholar_citations" else 200,
            "detail": "fixture",
        }
    )


# REQ-REPORT-7421 / SCENARIO-REPORT-7421-CONTRACT
def test_real_authorities_fail_closed_on_stale_v650_markdown() -> None:
    """The V651 YAML cannot inherit the preserved twelve-row V650 table."""

    comparison = ingestion.load_contract_audit(ROOT)
    assert comparison["passed"] is False
    assert comparison["markdown_milestone"] == "2026.09.650"
    assert comparison["yaml_milestone"] == ingestion.MILESTONE
    assert len(comparison["markdown_task_rows"]) == 12
    assert [row["id"] for row in comparison["yaml_task_rows"]] == list(ingestion.EXPECTED_TASK_IDS)
    assert "markdown_milestone" in comparison["errors"]


# REQ-REPORT-7421 / SCENARIO-REPORT-7421-CONTRACT
def test_exact_thirteen_row_fixture_and_five_mutations() -> None:
    """Exact rows pass, while all required private defects fail."""

    roadmap = ingestion.load_yaml(ROOT / ingestion.ROADMAP_PATH)
    markdown = _markdown_for(roadmap)
    comparison = ingestion.compare_contract_authorities(markdown, roadmap)
    assert comparison["passed"] is True
    assert len(comparison["contract_rows"]) == 13
    assert all(row["passed"] for row in comparison["contract_rows"])

    mutations = ingestion.run_contract_mutation_controls(markdown, roadmap)
    assert [row["mutation"] for row in mutations] == [
        "count",
        "order",
        "milestone",
        "missing_producer_field",
        "quarantine_status",
    ]
    assert all(row["rejected"] is True for row in mutations)


# REQ-REPORT-7421 / SCENARIO-REPORT-7421-CONTRACT
def test_contract_parser_rejects_malformed_authorities() -> None:
    """A malformed YAML mapping or Markdown table remains invalid evidence."""

    with pytest.raises(ValueError, match="mapping"):
        ingestion.load_yaml(ROOT / "CODEX.md")
    result = ingestion.compare_contract_authorities("# no contract\n", {"tasks": []})
    assert result["passed"] is False
    assert result["contract_rows"] == []
    assert result["errors"]


# REQ-REPORT-7421 / SCENARIO-REPORT-7421-CONTRACT
def test_local_readers_reject_sequence_shaped_documents(tmp_path: Path) -> None:
    """A readable file is not an authority unless its top level is a mapping."""

    yaml_path = tmp_path / "sequence.yaml"
    yaml_path.write_text("- one\n- two\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        ingestion.load_yaml(yaml_path)
    json_path = tmp_path / "sequence.json"
    json_path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        ingestion.load_json(json_path)
    assert ingestion._prior_class("unknown", {}) == "disqualified"


# REQ-REPORT-7421 / SCENARIO-REPORT-7421-PRIORS
def test_v650_prior_rows_preserve_exact_negative_and_blocked_facts() -> None:
    """The prior reducer keeps null, blocked, and disqualified evidence distinct."""

    rows = ingestion.collect_prior_dispositions(ROOT)
    assert len(rows) == 12
    by_id = {row["task_id"]: row for row in rows}
    assert by_id["exp7413-source-calibration"]["original_verdict_class"] == "null"
    assert by_id["exp7414-selected-feedback"]["observed_facts"] == {
        "independent_groups": 149,
        "negative_labels": 5,
        "positive_labels": 144,
    }
    assert by_id["exp7416-anchored-extraction"]["observed_facts"]["attempted_calls"] == 0
    assert by_id["exp7416-anchored-extraction"]["observed_facts"]["block"] == (
        "available_slot_dictionary_exact_equality"
    )
    assert by_id["exp7417-extraction-audit"]["source_kind"] == "conductor_pre_gate_record"
    assert by_id["exp7417-extraction-audit"]["original_verdict_class"] == "blocked"
    assert by_id["exp7411-arc-call-budget"]["flagged_adversarial"] is True
    assert by_id["exp7411-arc-call-budget"]["observed_facts"]["typed_invocation_conflict"]
    assert ingestion.completion_ledger_state(ROOT)["latest_milestone"] == "2026.09.650"
    assert ingestion.completion_ledger_state(ROOT)["lag"] is True


# REQ-REPORT-7421 / SCENARIO-REPORT-7421-METHODS
def test_bounded_source_access_and_method_mapping_keep_failures() -> None:
    """Seven checks include citation failure and three bounded plan hooks."""

    rows = _source_rows()
    assert len(rows) == 7
    assert len(rows) <= 8
    assert (
        next(row for row in rows if row["source_id"] == "semantic_scholar_citations")[
            "access_state"
        ]
        == "failed"
    )
    mappings = ingestion.method_mapping_rows(rows)
    assert {row["method"] for row in mappings} == {
        "spline_locality",
        "partial_feedback",
        "claim_attribution",
    }
    assert all(row["primary_urls"] and row["method_limit"] for row in mappings)


# REQ-REPORT-7421 / SCENARIO-REPORT-7421-METHODS
def test_additive_spline_logit_equals_logistic_on_the_same_basis() -> None:
    """The two names produce the same dot product on a fixed design basis."""

    row = ingestion.additive_spline_logit_equivalence()
    assert row["passed"] is True
    assert row["maximum_absolute_difference"] == 0.0
    assert row["coefficient_count"] == 4
    note = ingestion.render_method_note(_source_rows(), row)
    assert "same fixed basis" in note
    assert "G1-G4" in note
    for deferred in (
        "proof memory",
        "external-text reranking",
        "generic Ising sweeps",
        "foundation-model training",
    ):
        assert deferred in note


# REQ-REPORT-7421 / SCENARIO-REPORT-7421-ARTIFACT
def test_artifact_is_complete_disqualified_when_markdown_is_stale() -> None:
    """Independent ingestion completes even though contract readiness is zero."""

    contract = ingestion.load_contract_audit(ROOT)
    priors = ingestion.collect_prior_dispositions(ROOT)
    sources = _source_rows()
    artifact = ingestion.build_artifact(
        ROOT,
        contract,
        ingestion.run_contract_mutation_controls(
            _markdown_for(ingestion.load_yaml(ROOT / ingestion.ROADMAP_PATH)),
            ingestion.load_yaml(ROOT / ingestion.ROADMAP_PATH),
        ),
        priors,
        sources,
        _validation(),
        started_at_utc="2026-09-19T13:00:00+00:00",
        ended_at_utc="2026-09-19T13:00:01+00:00",
        duration_s=1.0,
        phase_spans=ingestion.zero_test_phase_spans(),
    )
    assert artifact["schema"] == ingestion.SCHEMA
    assert artifact["status"] == "complete_disqualified_contract_authority"
    assert artifact["honest_verdict"] == (
        "complete_disqualified_stale_markdown_contract_with_method_ingestion"
    )
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["contract_ready_score"] == 0
    assert artifact["method_ingestion_complete_score"] == 1
    assert artifact["promotion_score"] == 0
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert set(artifact["invocation_counts"].values()) == {0}
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["execution_venue"] == "host"
    assert artifact["small_ebm_training"]["performed"] is False
    assert artifact["gate_check_summary"]["first_failure"]["field"] == "milestone"
    assert ingestion.validate_artifact(artifact, root=ROOT) == []


# REQ-REPORT-7421 / SCENARIO-REPORT-7421-ARTIFACT
def test_validator_rejects_score_prior_and_checksum_drift() -> None:
    """Cold reduction rejects fields that could change the conclusion."""

    artifact = ingestion.build_artifact(
        ROOT,
        ingestion.load_contract_audit(ROOT),
        ingestion.run_contract_mutation_controls(
            _markdown_for(ingestion.load_yaml(ROOT / ingestion.ROADMAP_PATH)),
            ingestion.load_yaml(ROOT / ingestion.ROADMAP_PATH),
        ),
        ingestion.collect_prior_dispositions(ROOT),
        _source_rows(),
        _validation(),
        started_at_utc="2026-09-19T13:00:00+00:00",
        ended_at_utc="2026-09-19T13:00:01+00:00",
        duration_s=1.0,
        phase_spans=ingestion.zero_test_phase_spans(),
    )

    def errors_after(change: Any, refresh: bool = True) -> list[str]:
        changed = deepcopy(artifact)
        change(changed)
        if refresh:
            changed["reproducibility_checksum"] = ingestion.reproducibility_checksum(changed)
        return ingestion.validate_artifact(changed, root=ROOT)

    assert "contract_score_mismatch" in errors_after(
        lambda value: value.__setitem__("contract_ready_score", 1)
    )
    assert "prior_dispositions_mismatch" in errors_after(
        lambda value: value["prior_dispositions"][0].__setitem__(
            "original_verdict_class", "positive"
        )
    )
    assert "reproducibility_checksum_mismatch" in errors_after(
        lambda value: value.__setitem__("status", "changed"), refresh=False
    )

    changed = deepcopy(artifact)
    changed["schema"] = "wrong"
    changed["MODEL_SPECS"] = ["wrong"]
    changed["task_contract_rows"] = []
    changed["contract_comparison"] = {}
    changed["source_access_rows"] = []
    changed["method_mapping_rows"] = ["wrong"]
    changed["spline_logit_equivalence"] = {}
    changed["method_ingestion_complete_score"] = 1
    changed["contract_mutation_rows"] = []
    changed["validation_receipts"] = []
    changed["source_artifact_hashes"] = {}
    changed["promotion_score"] = 1
    changed["field_principles"] = {}
    errors = ingestion.validate_artifact(changed, root=ROOT)
    assert {
        "identity_invalid",
        "model_contract_invalid",
        "task_contract_rows_mismatch",
        "contract_comparison_mismatch",
        "source_access_rows_invalid",
        "method_mapping_rows_mismatch",
        "spline_equivalence_mismatch",
        "method_score_mismatch",
        "contract_mutations_invalid",
        "affected_validation_invalid",
        "terminal_validation_invalid",
        "source_hash_mismatch",
        "promotion_score_nonzero",
        "field_principles_invalid",
    } <= set(errors)
    assert ingestion.validate_artifact([]) == ["artifact_mapping_required"]

    wrong_ids = deepcopy(artifact)
    wrong_ids["source_access_rows"][0]["source_id"] = "wrong"
    wrong_ids["reproducibility_checksum"] = ingestion.reproducibility_checksum(wrong_ids)
    assert "source_access_rows_invalid" in ingestion.validate_artifact(wrong_ids, root=ROOT)

    assert ingestion._hashes_match({"source_artifact_hashes": []}, ROOT) is False
    assert ingestion._hashes_match({"source_artifact_hashes": {"bad": "row"}}, ROOT) is False
    assert (
        ingestion._hashes_match(
            {"source_artifact_hashes": {"bad": {"path": "missing", "sha256": "x"}}},
            ROOT,
        )
        is False
    )
    malformed_hashes = deepcopy(artifact["source_artifact_hashes"])
    first_path = next(iter(malformed_hashes))
    malformed_hashes[first_path] = "wrong"
    assert ingestion._hashes_match({"source_artifact_hashes": malformed_hashes}, ROOT) is False
    missing_hashes = deepcopy(artifact["source_artifact_hashes"])
    missing_hashes[first_path] = {"path": "missing", "sha256": "sha256:" + "0" * 64}
    assert ingestion._hashes_match({"source_artifact_hashes": missing_hashes}, ROOT) is False

    summary = ingestion._failure_summary(
        {"markdown_milestone": ingestion.MILESTONE},
        [
            {
                "check": "affected_validation",
                "operator": "==",
                "expected": True,
                "observed": False,
                "passed": False,
            }
        ],
    )
    assert summary["first_failure"]["field"] == "affected_validation"


# REQ-REPORT-7421 / SCENARIO-REPORT-7421-ARTIFACT
def test_scoped_plan_and_public_arguments_are_fixed(tmp_path: Path) -> None:
    """Validation stays file-scoped and the public date cannot drift."""

    commands = ingestion.build_validation_plan(ROOT, tmp_path)
    assert ingestion.validate_validation_plan(ROOT, commands) == []
    assert [command.name for command in commands] == list(REQUIRED_CHECK_NAMES)
    assert all("tests/python" not in command.argv for command in commands)
    roadmap = ingestion.load_yaml(ROOT / ingestion.ROADMAP_PATH)
    rendered = ingestion._render_markdown_authority(roadmap)
    assert ingestion.compare_contract_authorities(rendered, roadmap)["passed"] is True
    terminal = ingestion._terminal_commands(ROOT, tmp_path / "candidate.json")
    assert [row.spec.name for row in terminal] == list(ingestion.TERMINAL_CHECK_NAMES)
    assert ingestion.parse_args(["--date", ingestion.RUN_DATE]).date == ingestion.RUN_DATE
    with pytest.raises(SystemExit):
        ingestion.parse_args(["--date", "20260918"])
