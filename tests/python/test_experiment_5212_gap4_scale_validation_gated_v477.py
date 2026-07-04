"""Tests for Exp 5212 GAP-4 expanded-pool scale validation.

Spec refs: REQ-REPORT-5212, SCENARIO-REPORT-5212,
SCENARIO-REPORT-5212-BLOCKED-PROTOCOL-METADATA.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5212_gap4_scale_validation_gated_v477 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


def _candidate(index: int, **overrides: Any) -> JsonDict:
    row: JsonDict = {
        "accepted": True,
        "task_id": f"human_replay:test:{index}",
        "source": "unit",
        "code": "def transform(grid):\n    return [list(row) for row in grid]\n",
        "demo_perfect": True,
        "output_shape_matches": True,
        "guard_status": "accepted",
        "demos": [{"input": [[index]], "output": [[index]]}],
        "test_input": [[index]],
        "test_shape": [1, 1],
    }
    row.update(overrides)
    return row


def _protocol_row(index: int, *, vote: bool, gated: bool, domain: str = "arc1") -> JsonDict:
    return _candidate(
        index,
        task_id=f"{domain}:task:{index}",
        domain=domain,
        task=f"task_{index}",
        cluster_id=f"{domain}:task_{index}",
        vote_top2=vote,
        gated_top2=gated,
    )


def _exp5211_payload(rows: list[JsonDict]) -> JsonDict:
    return {
        "candidate_pool_n": 120,
        "gap4_expansion_usable": True,
        "candidate_rows": rows,
        "leakage_audit_passed": True,
    }


def test_req_report_5212_spec_declares_expanded_validation_contract() -> None:
    """REQ-REPORT-5212: OpenSpec declares the v477 validation artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-REPORT-5212",
        "SCENARIO-REPORT-5212",
        "SCENARIO-REPORT-5212-BLOCKED-PROTOCOL-METADATA",
        mod.RESULT_RELATIVE_PATH,
        mod.EXP5211_RELATIVE_PATH,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_report_5212_excludes_current_exp5211_rows_without_relabeling() -> None:
    """SCENARIO-REPORT-5212-BLOCKED-PROTOCOL-METADATA: no pass@2 labels means no score."""

    scored, exclusions = mod.score_expanded_pool(
        [_candidate(0), _candidate(1)],
        exp5197_task_ids=set(),
    )
    artifact = mod.build_artifact(
        scored_rows=scored,
        exclusions=exclusions,
        exp5211_candidate_pool_n=120,
        exp5211_gap4_expansion_usable=True,
        source_artifacts=[],
        duration_s=0.0,
        tests_run=["unit: pass"],
    )

    assert scored == []
    assert [row["reason"] for row in exclusions] == [
        "missing_protocol_pass2_fields",
        "missing_protocol_pass2_fields",
    ]
    assert artifact["n_scored"]["value"] == 0
    assert artifact["excluded_rows"]["value"] == 2
    assert artifact["exact_test_discordant_wins"]["value"] == 0
    assert artifact["exact_test_discordant_losses"]["value"] == 0
    assert artifact["exact_test_p_value_two_sided"]["value"] == 1.0
    assert artifact["exact_test_passes_min6_rule"]["value"] is False
    assert artifact["cluster_bootstrap_delta_ci95"]["value"] == [0.0, 0.0]
    assert artifact["gap4_status_recommendation"]["value"] == "blocked"
    assert artifact["failure_mode"] == "missing_protocol_pass2_fields"
    assert "floor_crossed" not in artifact["honest_verdict"]
    assert mod.artifact_schema_errors(artifact) == []


def test_req_report_5212_reuses_exp5197_exact_min6_rule_and_exp5177_bootstrap() -> None:
    """REQ-REPORT-5212: exact test and bootstrap stay on the established protocol."""

    scored, exclusions = mod.score_expanded_pool(
        [_protocol_row(i, vote=False, gated=True) for i in range(6)],
        exp5197_task_ids=set(),
    )
    artifact = mod.build_artifact(
        scored_rows=scored,
        exclusions=exclusions,
        exp5211_candidate_pool_n=120,
        exp5211_gap4_expansion_usable=True,
        source_artifacts=[],
        duration_s=1.0,
        tests_run=["unit: pass"],
    )

    assert exclusions == []
    assert artifact["n_scored"]["value"] == 6
    assert artifact["exact_test_discordant_wins"]["value"] == 6
    assert artifact["exact_test_discordant_losses"]["value"] == 0
    assert artifact["exact_test_p_value_two_sided"]["value"] == 0.03125
    assert artifact["exact_test_passes_min6_rule"]["value"] is True
    assert artifact["gap4_status_recommendation"]["value"] == "filled"
    assert artifact["honest_verdict"].startswith("success_")
    assert artifact["arc1_slice_result"]["labels_available"] is True
    assert artifact["arc2_heldout_slice_result"]["labels_available"] is False
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_report_5212_run_writes_blocked_artifact_for_unlabeled_expanded_pool(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5212: run writes the artifact and reports exclusions."""

    source_path = tmp_path / mod.EXP5211_RELATIVE_PATH
    source_path.parent.mkdir(parents=True)
    source_path.write_text(json.dumps(_exp5211_payload([_candidate(0)])), encoding="utf-8")

    artifact = mod.run(
        root=tmp_path,
        exp5197_task_loader=lambda _root: set(),
        tests_run=["targeted pytest: pass"],
        now=lambda: 5.0,
    )

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    assert result_path.exists()
    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert artifact["exp5211_candidate_pool_n"] == 120
    assert artifact["exp5211_gap4_expansion_usable"] is True
    assert artifact["n_scored"]["value"] == 0
    assert artifact["excluded_rows"]["value"] == 1
    assert artifact["tests_run"]["value"] == ["targeted pytest: pass"]
    assert artifact["source_artifacts"] == [{"path": mod.EXP5211_RELATIVE_PATH, "exists": True}]


def test_req_report_5212_malformed_and_leaky_rows_are_excluded_explicitly() -> None:
    """REQ-REPORT-5212: metadata and leakage audit failures are counted before scoring."""

    rows = [
        _candidate(0, accepted=False),
        _candidate(1, demo_perfect=False),
        _candidate(2, output_shape_matches=False),
        _candidate(3, guard_status="demo_mismatch"),
        _candidate(4, code=""),
        _candidate(40, code="def transform(:\n    return grid\n"),
        _candidate(5, code="def transform(grid):\n    return test_output\n"),
        _candidate(6, code="import os\ndef transform(grid):\n    return grid\n"),
        _candidate(7, test_output=[[7]]),
        _candidate(8, task_id="prior-task"),
        _protocol_row(9, vote=True, gated=False, domain="arc2"),
        _candidate(
            10,
            source="arc2_eval",
            vote_top2=True,
            gated_top2=True,
        ),
        _candidate(
            11,
            source="arc1_eval",
            vote_top2=True,
            gated_top2=True,
        ),
        _candidate(
            12,
            source="other",
            vote_top2=True,
            gated_top2=True,
        ),
    ]

    scored, exclusions = mod.score_expanded_pool(rows, exp5197_task_ids={"prior-task"})

    assert [row["reason"] for row in exclusions] == [
        "not_accepted",
        "demo_perfect_missing_or_false",
        "output_shape_missing_or_false",
        "guard_status_not_accepted",
        "missing_code",
        "code_parse_error",
        "leakage_token_in_code",
        "forbidden_ast",
        "leakage_metadata",
        "exp5197_task_leakage",
    ]
    assert len(scored) == 4
    assert scored[0]["domain"] == "arc2"
    assert [row["domain"] for row in scored[1:]] == ["arc2", "arc1", "unlabeled"]

    artifact = mod.build_artifact(
        scored_rows=scored[:1],
        exclusions=exclusions,
        exp5211_candidate_pool_n=120,
        exp5211_gap4_expansion_usable=True,
        source_artifacts=[],
        duration_s=1.0,
        tests_run=["unit: pass"],
    )
    assert artifact["gap4_status_recommendation"]["value"] == "retire_local_generation_path"
    assert artifact["arc2_heldout_slice_result"]["labels_available"] is True

    blocked = mod.build_artifact(
        scored_rows=[],
        exclusions=[],
        exp5211_candidate_pool_n=0,
        exp5211_gap4_expansion_usable=False,
        source_artifacts=[],
        duration_s=0.0,
        tests_run=[],
    )
    empty = mod.build_artifact(
        scored_rows=[],
        exclusions=[],
        exp5211_candidate_pool_n=120,
        exp5211_gap4_expansion_usable=True,
        source_artifacts=[],
        duration_s=0.0,
        tests_run=[],
    )
    assert blocked["failure_mode"] == "exp5211_pool_not_usable"
    assert empty["failure_mode"] == "no_scored_rows"


def test_artifact_schema_rejects_overclaims_and_bad_required_fields() -> None:
    """REQ-REPORT-5212: schema rejects filled overclaims and required-field drift."""

    artifact = mod.build_artifact(
        scored_rows=[_protocol_row(0, vote=True, gated=True)],
        exclusions=[],
        exp5211_candidate_pool_n=120,
        exp5211_gap4_expansion_usable=True,
        source_artifacts=[],
        duration_s=0.0,
        tests_run=["unit: pass"],
    )

    bad = dict(artifact)
    bad["n_scored"] = {"value": "1", "principle": mod.FIELD_PRINCIPLES["n_scored"]["principle"]}
    bad["exact_test_discordant_wins"] = {"value": 6}
    bad["exact_test_discordant_losses"] = {"value": 1}
    bad["exact_test_p_value_two_sided"] = {"value": True}
    bad["exact_test_passes_min6_rule"] = {"value": True}
    bad["cluster_bootstrap_delta_ci95"] = {"value": [0.0]}
    bad["gap4_status_recommendation"] = {"value": "filled"}
    bad["excluded_rows"] = {"value": "0"}
    bad["tests_run"] = {"value": "pytest"}
    bad["inference_substrate"] = {"value": "live_llm_inference"}
    bad["honest_verdict"] = "not_terminal_floor_crossed"
    bad["field_principles"] = {}
    bad["reproducibility_checksum"] = "sha256:bad"

    errors = mod.artifact_schema_errors(bad)

    assert "n_scored" in errors
    assert "exact_test_discordant_wins" in errors
    assert "exact_test_discordant_losses" in errors
    assert "exact_test_p_value_two_sided" in errors
    assert "exact_test_passes_min6_rule" in errors
    assert "cluster_bootstrap_delta_ci95" in errors
    assert "gap4_status_recommendation" in errors
    assert "excluded_rows" in errors
    assert "tests_run" in errors
    assert "inference_substrate" in errors
    assert "honest_verdict_terminal_prefix" in errors
    assert "honest_verdict_floor_overclaim" in errors
    assert "field_principles" in errors
    assert "reproducibility_checksum" in errors

    missing = dict(artifact)
    missing.pop("duration_s")
    missing["reproducibility_checksum"] = mod.payload_checksum(missing)
    assert "missing required field duration_s" in mod.artifact_schema_errors(missing)

    with pytest.raises(ValueError):
        mod.write_artifact(Path("/tmp"), bad)

    assert mod._wrapped_value("plain") == "plain"
    assert mod._read_json(Path("/tmp/definitely-missing-exp5212.json")) == {}
