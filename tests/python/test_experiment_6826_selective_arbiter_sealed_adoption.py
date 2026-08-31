"""Tests for the sealed selective-arbiter adoption receipt.

Spec refs: REQ-CONSTRAINT-6826, SCENARIO-CONSTRAINT-6826-PRECONDITIONS,
SCENARIO-CONSTRAINT-6826-MATRIX, SCENARIO-CONSTRAINT-6826-CRITERIA,
SCENARIO-CONSTRAINT-6826-DISQUALIFIERS, SCENARIO-CONSTRAINT-6826-COMPLETION,
and SCENARIO-CONSTRAINT-6826-ARTIFACT.
"""

from __future__ import annotations

import ast
from copy import deepcopy
import itertools
import json
from pathlib import Path

import pytest

from carnot import experiment_6826_selective_arbiter_sealed_adoption as adoption


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATHS = adoption.source_paths_for_root(REPO_ROOT)


@pytest.fixture(scope="module")
def sources() -> dict[str, dict]:
    """Load the three sealed inputs once because tests never edit disk state."""

    return adoption.load_sources(SOURCE_PATHS)


def _changed(sources: dict[str, dict], source: str) -> dict[str, dict]:
    """Copy one large source only, which keeps fault tests small and isolated."""

    return {**sources, source: deepcopy(sources[source])}


def test_req_constraint_6826_spec_precedes_implementation() -> None:
    """REQ-CONSTRAINT-6826 owns every path, field, and scenario."""

    spec = (REPO_ROOT / adoption.SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
    section = spec.split("### REQ-CONSTRAINT-6826", 1)[1]
    for requirement_id in adoption.OPEN_SPEC_IDS[1:]:
        assert requirement_id in section
    for field in adoption.TASK_REQUIRED_FIELDS:
        assert f"`{field}`" in section
    for path in (
        adoption.MODULE_RELATIVE_PATH,
        adoption.SCRIPT_RELATIVE_PATH,
        adoption.RESULT_RELATIVE_PATH,
    ):
        assert path.as_posix() in section


def test_scenario_constraint_6826_preconditions_accept_sealed_shards(
    sources: dict[str, dict],
) -> None:
    """SCENARIO-CONSTRAINT-6826-PRECONDITIONS accepts exact upstream seals."""

    summary = adoption.check_preconditions(sources, SOURCE_PATHS)
    assert summary["passed"] is True
    assert summary["failed_checks"] == []
    assert all(row["passed"] for row in summary["checks"])


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        ("completion", "upstream_completion_fields"),
        ("manifest", "complete_row_manifests"),
        ("hash", "source_artifact_hashes"),
        ("identity", "independent_code_identities"),
        ("verdict", "terminal_verdict_classes"),
    ],
)
def test_scenario_constraint_6826_preconditions_fail_closed(
    sources: dict[str, dict], mutation: str, failed_check: str
) -> None:
    """SCENARIO-CONSTRAINT-6826-PRECONDITIONS names each broken source gate."""

    source = (
        "exp6824" if mutation in {"completion", "manifest", "identity", "verdict"} else "exp6825"
    )
    changed = _changed(sources, source)
    if mutation == "completion":
        changed["exp6824"]["cold_replay_shard_complete"] = False
    elif mutation == "manifest":
        changed["exp6824"]["rows"].pop()
    elif mutation == "hash":
        changed["exp6825"]["source_artifact_hashes"]["exp6813"]["sha256"] = "sha256:wrong"
    elif mutation == "identity":
        changed["exp6824"]["independent_parser_id"]["imports_exp6813"] = True
    else:
        changed["exp6824"]["verdict_class"] = "invented"

    summary = adoption.check_preconditions(changed, SOURCE_PATHS)
    failed = next(row for row in summary["checks"] if row["check"] == failed_check)
    assert summary["passed"] is False
    assert failed["passed"] is False
    assert "expected" in failed and "observed" in failed

    artifact = adoption.build_artifact(
        changed,
        source_paths=SOURCE_PATHS,
        run_date="20260831",
        duration_s=0.25,
    )
    assert artifact["status"] == adoption.BLOCKED_STATUS
    assert artifact["rows"] == []
    assert artifact["selective_arbiter_audit_complete"] is False
    assert artifact["deployment_adoption_decision"] == "insufficient"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith(adoption.BLOCKED_STATUS)


def test_req_constraint_6826_unreadable_sources_are_blocked(tmp_path: Path) -> None:
    """REQ-CONSTRAINT-6826 records read failures instead of inventing evidence."""

    paths = {
        "exp6813": tmp_path / "missing-6813.json",
        "exp6824": tmp_path / "invalid-6824.json",
        "exp6825": tmp_path / "missing-6825.json",
    }
    paths["exp6824"].write_text("not-json", encoding="utf-8")
    loaded = adoption.load_sources(paths)
    summary = adoption.check_preconditions(loaded, paths)
    assert set(loaded) == set(paths)
    assert summary["passed"] is False
    assert "source_artifacts_readable" in summary["failed_checks"]


OUTCOMES = ("positive", "null", "harmful", "blocked", "disqualified", "partial")
OUTCOME_AUTHORITY = {
    "positive": ("enable", "positive"),
    "null": ("keep_shadow", "null"),
    "harmful": ("retire", "null"),
    "partial": ("insufficient", "partial"),
    "blocked": ("insufficient", "blocked"),
    "disqualified": ("redesign", "disqualified"),
}
CONSERVATIVE_ORDER = {
    "positive": 0,
    "null": 1,
    "harmful": 2,
    "partial": 3,
    "blocked": 4,
    "disqualified": 5,
}


@pytest.mark.parametrize(("cold_result", "attack_result"), itertools.product(OUTCOMES, repeat=2))
def test_scenario_constraint_6826_matrix_is_total_and_conservative(
    cold_result: str, attack_result: str
) -> None:
    """SCENARIO-CONSTRAINT-6826-MATRIX covers all 36 ordered shard pairs."""

    controlling = max((cold_result, attack_result), key=CONSERVATIVE_ORDER.__getitem__)
    decision = adoption.resolve_shard_outcomes(cold_result, attack_result)
    expected_adoption, expected_verdict = OUTCOME_AUTHORITY[controlling]
    assert decision == {
        "controlling_result": controlling,
        "deployment_adoption_decision": expected_adoption,
        "verdict_class": expected_verdict,
    }


def test_scenario_constraint_6826_matrix_rejects_unknown_results() -> None:
    """SCENARIO-CONSTRAINT-6826-MATRIX has no implicit result class."""

    with pytest.raises(adoption.SealedAdoptionError, match="unknown shard result"):
        adoption.resolve_shard_outcomes("positive", "invented")


def test_scenario_constraint_6826_criteria_recompute_distinct_components(
    sources: dict[str, dict],
) -> None:
    """SCENARIO-CONSTRAINT-6826-CRITERIA keeps four claims independent."""

    table = adoption.recompute_decision_table(
        sources["exp6813"], sources["exp6824"], sources["exp6825"]
    )
    assert table["component_decisions"] == {
        "hard_safety": "pass",
        "safe_action_preservation": "pass",
        "utility": "pass",
        "certificate_truth": "pass",
    }
    assert table["deployment_adoption_decision"] == "enable"
    assert table["verdict_class"] == "positive"
    assert table["selective_arbiter_audit_complete"] is True
    assert len(table["rows"]) == len(adoption.CRITERION_SOURCE_PAIRS)
    assert {(row["criterion"], row["evidence_source"]) for row in table["rows"]} == set(
        adoption.CRITERION_SOURCE_PAIRS
    )
    assert all(row["decision"] in adoption.COMPONENT_DECISIONS for row in table["rows"])


def _set_held_selective_progress(cold: dict, value: float) -> None:
    """Change only the held selective utility observations for sign tests."""

    held_ids = set(cold["row_coverage"]["held_scenario_ids"])
    for row in cold["rows"]:
        if (
            row.get("row_type") == "cold_replay"
            and row.get("arm") == "selective_priority"
            and row.get("scenario_id") in held_ids
        ):
            row["accepted_progress"] = value


def test_scenario_constraint_6826_utility_null_and_harmful_are_not_safety(
    sources: dict[str, dict],
) -> None:
    """SCENARIO-CONSTRAINT-6826-CRITERIA does not convert safety into utility."""

    null_sources = _changed(sources, "exp6824")
    pairs = {
        row["pair_id"]: row
        for row in null_sources["exp6824"]["rows"]
        if row.get("row_type") == "cold_replay" and row.get("arm") == "flat_reject_retry"
    }
    held_ids = set(null_sources["exp6824"]["row_coverage"]["held_scenario_ids"])
    for row in null_sources["exp6824"]["rows"]:
        if (
            row.get("row_type") == "cold_replay"
            and row.get("arm") == "selective_priority"
            and row.get("scenario_id") in held_ids
        ):
            row["accepted_progress"] = pairs[row["pair_id"]]["accepted_progress"]
    null_table = adoption.recompute_decision_table(
        null_sources["exp6813"], null_sources["exp6824"], null_sources["exp6825"]
    )
    assert null_table["component_decisions"]["hard_safety"] == "pass"
    assert null_table["component_decisions"]["utility"] == "insufficient"
    assert null_table["deployment_adoption_decision"] == "redesign"

    harmful_sources = _changed(sources, "exp6824")
    _set_held_selective_progress(harmful_sources["exp6824"], -1.0)
    harmful_table = adoption.recompute_decision_table(
        harmful_sources["exp6813"], harmful_sources["exp6824"], harmful_sources["exp6825"]
    )
    assert harmful_table["component_decisions"]["hard_safety"] == "pass"
    assert harmful_table["component_decisions"]["utility"] == "fail"
    assert harmful_table["deployment_adoption_decision"] == "redesign"


@pytest.mark.parametrize(
    ("fault", "reason"),
    [
        ("prohibited_feature", "prohibited_feature_influence"),
        ("hard_violation", "accepted_hard_violation"),
        ("incomplete_rows", "source_rows_incomplete"),
        ("arithmetic", "producer_cold_arithmetic_disagreement"),
    ],
)
def test_scenario_constraint_6826_disqualifiers_force_redesign(
    sources: dict[str, dict], fault: str, reason: str
) -> None:
    """SCENARIO-CONSTRAINT-6826-DISQUALIFIERS prevents unsafe enablement."""

    changed = _changed(sources, "exp6825" if fault == "prohibited_feature" else "exp6824")
    if fault == "prohibited_feature":
        row = next(
            row
            for row in changed["exp6825"]["rows"]
            if row["attack_id"] == "model_label_influence" and row["applicable"]
        )
        row["passed"] = False
    elif fault == "hard_violation":
        row = next(
            row
            for row in changed["exp6824"]["rows"]
            if row.get("row_type") == "cold_replay" and row.get("arm") == "selective_priority"
        )
        row["accepted_hard_violation"] = True
    elif fault == "incomplete_rows":
        changed["exp6824"]["rows"] = changed["exp6824"]["rows"][1:]
    else:
        changed["exp6824"]["headline_differences"]["all_within_tolerance"] = False

    table = adoption.recompute_decision_table(
        changed["exp6813"], changed["exp6824"], changed["exp6825"]
    )
    assert reason in table["disqualifiers"]
    assert table["deployment_adoption_decision"] == "redesign"
    assert table["verdict_class"] == "disqualified"


@pytest.mark.parametrize(
    ("utility", "expected"),
    [("pass", "enable"), ("insufficient", "keep_shadow"), ("fail", "retire")],
)
def test_scenario_constraint_6826_completion_is_effect_independent(
    utility: str, expected: str
) -> None:
    """SCENARIO-CONSTRAINT-6826-COMPLETION closes positive, null, and harmful tables."""

    components = {
        "hard_safety": "pass",
        "safe_action_preservation": "pass",
        "utility": utility,
        "certificate_truth": "pass",
    }
    decision = adoption.decide_deployment(components, [])
    assert decision == expected
    assert adoption.decision_table_complete(components, [], expected) is True


def test_scenario_constraint_6826_completion_closes_disqualification() -> None:
    """SCENARIO-CONSTRAINT-6826-COMPLETION treats redesign as terminal evidence."""

    components = {
        "hard_safety": "fail",
        "safe_action_preservation": "pass",
        "utility": "pass",
        "certificate_truth": "pass",
    }
    assert adoption.decide_deployment(components, ["accepted_hard_violation"]) == "redesign"
    assert adoption.decision_table_complete(components, ["accepted_hard_violation"], "redesign")


def test_scenario_constraint_6826_artifact_is_complete_and_self_checking(
    sources: dict[str, dict],
) -> None:
    """SCENARIO-CONSTRAINT-6826-ARTIFACT binds each required decision and source."""

    artifact = adoption.build_artifact(
        sources,
        source_paths=SOURCE_PATHS,
        run_date="20260831",
        duration_s=0.5,
    )
    assert set(artifact) == set(adoption.REQUIRED_ARTIFACT_FIELDS)
    assert set(artifact["field_principles"]) == set(artifact)
    assert set(artifact["source_artifact_hashes"]) == {"exp6813", "exp6824", "exp6825"}
    assert artifact["hard_safety_decision"] == "pass"
    assert artifact["safe_action_preservation_decision"] == "pass"
    assert artifact["utility_decision"] == "pass"
    assert artifact["certificate_truth_decision"] == "pass"
    assert artifact["deployment_adoption_decision"] == "enable"
    assert artifact["selective_arbiter_audit_complete"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["reproducibility_checksum"] == adoption.reproducibility_checksum(artifact)
    assert adoption.validate_artifact(artifact) == []


def test_req_constraint_6826_validator_rejects_detached_claims(sources: dict[str, dict]) -> None:
    """REQ-CONSTRAINT-6826 rejects malformed decisions and unbound output."""

    artifact = adoption.build_artifact(
        sources,
        source_paths=SOURCE_PATHS,
        run_date="20260831",
        duration_s=0.5,
    )
    changed = deepcopy(artifact)
    changed.pop("title")
    changed["field_principles"] = {}
    changed["duration_s"] = -1
    changed["verifier_is_oracle"] = True
    changed["verdict_class"] = "invented"
    changed["honest_verdict"] = "unfinished"
    changed["hard_safety_decision"] = "maybe"
    changed["deployment_adoption_decision"] = "ship"
    changed["selective_arbiter_audit_complete"] = True
    changed["rows"].append(deepcopy(changed["rows"][0]))
    findings = adoption.validate_artifact(changed)
    for marker in (
        "missing required fields",
        "field principle coverage mismatch",
        "duration_s must be non-negative",
        "verifier_is_oracle must be false",
        "verdict class outside closed enum",
        "honest verdict lacks terminal prefix",
        "component decision outside closed enum",
        "deployment decision outside closed enum",
        "criterion-source rows are not exact",
        "audit completion is not row-supported",
        "reproducibility checksum mismatch",
    ):
        assert any(marker in finding for finding in findings)


def test_req_constraint_6826_module_does_not_import_producer_code() -> None:
    """REQ-CONSTRAINT-6826 keeps producer implementation outside acceptance."""

    source = (REPO_ROOT / adoption.MODULE_RELATIVE_PATH).read_text(encoding="utf-8")
    imported = {
        alias.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("6813" in name for name in imported)


def test_req_constraint_6826_writer_and_cli_use_explicit_output(
    sources: dict[str, dict], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-CONSTRAINT-6826 writes atomically only to the caller-owned path."""

    artifact = adoption.build_artifact(
        sources,
        source_paths=SOURCE_PATHS,
        run_date="20260831",
        duration_s=0.5,
    )
    target = tmp_path / "direct.json"
    adoption.write_artifact(target, artifact)
    assert json.loads(target.read_text(encoding="utf-8")) == artifact

    cli_target = tmp_path / "cli.json"
    assert (
        adoption.main(
            [
                "--root",
                str(REPO_ROOT),
                "--date",
                "20260831",
                "--output",
                str(cli_target),
            ]
        )
        == 0
    )
    printed = json.loads(capsys.readouterr().out)
    assert printed["output"] == str(cli_target)
    assert printed["selective_arbiter_audit_complete"] is True
    assert json.loads(cli_target.read_text(encoding="utf-8"))["verdict_class"] == "positive"


def test_req_constraint_6826_writer_rejects_invalid_artifact(tmp_path: Path) -> None:
    """REQ-CONSTRAINT-6826 prevents an invalid receipt from replacing evidence."""

    with pytest.raises(adoption.SealedAdoptionError, match="missing required fields"):
        adoption.write_artifact(tmp_path / "bad.json", {})
