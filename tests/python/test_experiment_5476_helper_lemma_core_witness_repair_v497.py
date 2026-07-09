"""Tests for Exp5476 helper-lemma core witness repair.

Spec refs: REQ-VERIFY-5476, SCENARIO-VERIFY-5476.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5476_helper_lemma_core_witness_repair_v497 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_5476_helper_lemma_core_witness_repair_v497.py -q"
)


def _rows() -> list[dict[str, Any]]:
    return mod.select_witness_rows(mod.load_source_artifacts(REPO))


def _candidates() -> list[dict[str, Any]]:
    return mod.generate_helper_candidates(_rows())


def _candidate(row_id: str) -> dict[str, Any]:
    return next(candidate for candidate in _candidates() if candidate["row_id"] == row_id)


def test_req_verify_5476_spec_declares_helper_lemma_contract() -> None:
    """REQ-VERIFY-5476: OpenSpec anchors the helper repair artifact fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5476") : spec.index("### REQ-VERIFY-5462")]
    normalized = " ".join(section.split())
    searchable = f"{section}\n{normalized}"

    for marker in (
        "REQ-VERIFY-5476",
        "SCENARIO-VERIFY-5476",
        str(mod.RESULT_RELATIVE_PATH),
        "Exp5445",
        "Exp5458",
        "normalized verifier failure signature",
        "source/spec/context",
        "exact AST/KB recheck",
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in searchable
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_req_verify_5476_groups_failures_before_repair() -> None:
    """REQ-VERIFY-5476: repeated verifier signatures are measured before repair."""

    rows = _rows()
    groups = mod.group_failures_by_signature(rows)

    assert [row["row_id"] for row in rows] == [
        "fixture.nonexistent_json_method",
        "fixture.nonexistent_math_alias_method",
        "fixture.wrong_module_alias",
        "fixture.missing_bare_import",
        "fixture.imported_symbol_missing",
        "fixture.argument_intent_mismatch",
    ]
    assert len(groups) == 5
    assert mod.failure_signature(rows[0]) == "ast_kb:intent_mismatch+kb_missing_call"
    repeated = next(
        group
        for group in groups
        if group["failure_signature"] == "ast_kb:intent_mismatch+kb_missing_call"
    )
    assert repeated == {
        "failure_signature": "ast_kb:intent_mismatch+kb_missing_call",
        "row_ids": [
            "fixture.nonexistent_json_method",
            "fixture.wrong_module_alias",
        ],
        "count": 2,
        "repeated": True,
    }
    assert all(group["count"] == len(group["row_ids"]) for group in groups)


def test_scenario_verify_5476_accepts_helper_after_exact_recheck() -> None:
    """SCENARIO-VERIFY-5476: an import helper is accepted only after recheck."""

    candidate = _candidate("fixture.missing_bare_import")

    assert candidate["helper_kind"] == "api_import_binding_contract"
    assert candidate["generated_from"] == "witness_row_source_semantic_intent_and_kb_results"
    assert candidate["source_fields_used"] == [
        "source",
        "semantic_intent",
        "fully_qualified_call_sites",
        "imported_symbol_checks",
        "kb_lookup_results",
        "reject_reasons",
    ]
    assert candidate["helper_contract"]["expected_call_fqn"] == "json.loads"
    assert candidate["candidate_source"] == "import json\nresult = json.loads(payload)\n"
    assert candidate["accepted_after_exact_recheck"] is True
    assert candidate["exact_recheck"]["accepted"] is True
    assert candidate["exact_recheck"]["authority"] == "ast_kb_witness"
    assert candidate["false_accept"] is False
    assert candidate["semantics_changed_incorrectly"] is False


def test_scenario_verify_5476_rejects_unsupported_helper_after_exact_recheck() -> None:
    """SCENARIO-VERIFY-5476: a helper for an absent API member is rejected."""

    candidate = _candidate("fixture.nonexistent_math_alias_method")

    assert candidate["helper_kind"] == "api_member_precondition"
    assert candidate["helper_contract"]["expected_call_fqn"] == "math.relu"
    assert candidate["candidate_source"] == "import math\nresult = math.relu(x)\n"
    assert candidate["accepted_after_exact_recheck"] is False
    assert candidate["rejection_reason"] == "exact_recheck_rejected"
    assert candidate["false_accept"] is False
    assert candidate["semantics_changed_incorrectly"] is False
    assert "kb_missing_call:math.relu" in candidate["exact_recheck"]["failure_reasons"]


def test_scenario_verify_5476_artifact_fields_rates_and_rejections(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5476: deliverable JSON reports accepted and rejected helpers."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    artifact = mod.run(result_path=result_path, tests_run=[TEST_COMMAND], write=True)
    saved = json.loads(result_path.read_text(encoding="utf-8"))

    assert saved == artifact
    mod.validate_artifact(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["witness_row_count"] == 6
    assert artifact["failure_signature_count"] == 5
    assert artifact["helper_candidate_count"] == 6
    assert artifact["accepted_helper_count"] == 5
    assert artifact["exact_recheck_pass_rate"] == pytest.approx(5 / 6, abs=1e-6)
    assert artifact["false_accept_count"] == 0
    assert artifact["repeated_failure_reduction_rate"] == pytest.approx(1.0)
    assert artifact["helper_lemma_repair_ready"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["row_provenance_checksum"] == mod.row_provenance_checksum(
        artifact["witness_rows"]
    )
    assert {
        candidate["accepted_after_exact_recheck"] for candidate in artifact["helper_candidates"]
    } == {
        False,
        True,
    }
    assert artifact["semantic_change_rejections"] == []
    assert artifact["research_conductor_modified"] is False


def test_req_verify_5476_validation_fails_closed_on_artifact_drift() -> None:
    """REQ-VERIFY-5476: schema, provenance, and exact-authority drift are rejected."""

    artifact = mod.build_artifact(tests_run=[{"command": TEST_COMMAND, "outcome": "passed"}])

    missing = deepcopy(artifact)
    missing.pop("witness_row_count")
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "live_llm_inference"
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(bad_substrate)

    bad_false_accept = deepcopy(artifact)
    bad_false_accept["helper_candidates"][0]["false_accept"] = True
    with pytest.raises(ValueError, match="false_accept_count"):
        mod.validate_artifact(bad_false_accept)

    bad_candidate = deepcopy(artifact)
    bad_candidate["helper_candidates"][0]["accepted_after_exact_recheck"] = False
    with pytest.raises(ValueError, match="candidate acceptance"):
        mod.validate_artifact(bad_candidate)

    bad_checksum = deepcopy(artifact)
    bad_checksum["witness_rows"][0]["row_id"] = "tampered"
    with pytest.raises(ValueError, match="row_provenance_checksum"):
        mod.validate_artifact(bad_checksum)

    bad_ready = deepcopy(artifact)
    bad_ready["helper_lemma_repair_ready"] = True
    bad_ready["false_accept_count"] = 1
    with pytest.raises(ValueError, match="helper_lemma_repair_ready"):
        mod.validate_artifact(bad_ready)

    bad_conductor = deepcopy(artifact)
    bad_conductor["research_conductor_modified"] = True
    with pytest.raises(ValueError, match="research_conductor.py"):
        mod.validate_artifact(bad_conductor)

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"] = "done"
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(bad_verdict)


def test_req_verify_5476_defensive_helper_paths_are_deterministic() -> None:
    """REQ-VERIFY-5476: defensive helper branches fail or fall back deterministically."""

    source_artifacts = mod.load_source_artifacts(REPO)
    tampered_sources = deepcopy(source_artifacts)
    tampered_sources["ast_kb_witness"]["witness_rows"][2]["accepted"] = True
    with pytest.raises(ValueError, match="not a failed witness"):
        mod.select_witness_rows(tampered_sources)

    helper_row = deepcopy(_rows()[0])
    helper_row["reject_reasons"] = ["custom_guard_failed"]
    assert mod._helper_kind(helper_row) == "helper_invariant"
    assert mod._assignment_target("json.loads(payload)\n") == "result"
