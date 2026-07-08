"""Tests for Exp5381 budget-memory tautology corrigendum.

Spec refs: REQ-LEARN-5381, SCENARIO-LEARN-5381-ROW-EVIDENCE,
SCENARIO-LEARN-5381-NEGATIVE-CONTROLS, SCENARIO-LEARN-5381-FAIL-CLOSED.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5381_budget_memory_tautology_corrigendum_v490 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/self-learning/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def test_req_learn_5381_spec_declares_corrigendum_contract() -> None:
    """REQ-LEARN-5381: OpenSpec anchors row-derived corrigendum behavior."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5381") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-LEARN-5381",
        "SCENARIO-LEARN-5381-ROW-EVIDENCE",
        "SCENARIO-LEARN-5381-NEGATIVE-CONTROLS",
        "SCENARIO-LEARN-5381-FAIL-CLOSED",
        str(mod.RESULT_RELATIVE_PATH),
        str(mod.SOURCE_ARTIFACT_RELATIVE_PATH),
        "`recomputed_fields_from_rows`",
        "latency proxy cost",
        "energy proxy cost",
    ):
        assert marker in section

    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_req_learn_5381_reviews_exact_source_tautology_reason() -> None:
    """REQ-LEARN-5381-1: the source TAUTOLOGY reason is preserved exactly."""

    source = mod.load_source_artifact(REPO)
    finding = mod.review_source_tautology(source, REPO)

    assert finding["source_flagged_adversarial"] is True
    assert finding["conductor_flagged_tautology"] is True
    assert finding["source_findings"] == [
        {
            "kind": "TAUTOLOGY",
            "severity": "critical",
            "detail": (
                "budget_bytes=400.0 and retained_bytes=400.0 agree to >5 sig "
                "figs. Two distinct metrics matching this precisely is more "
                "likely a bug than a finding."
            ),
        }
    ]


def test_scenario_learn_5381_row_evidence_recomputes_clean_artifact() -> None:
    """SCENARIO-LEARN-5381-ROW-EVIDENCE: aggregates derive from rows."""

    tests_run = [{"command": "unit exp5381", "outcome": "passed"}]
    artifact = mod.build_artifact(root=REPO, tests_run=tests_run)

    mod.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["budget_memory_corrigendum_clean"] is True
    assert artifact["source_artifact"] == str(mod.SOURCE_ARTIFACT_RELATIVE_PATH)
    assert artifact["row_count"] == 7
    assert artifact["negative_controls_count"] == 3
    assert artifact["negative_controls_passed"] == 3
    assert artifact["keep_share_trust_policy_ready"] is True
    assert artifact["rollback_supported"] is True
    assert artifact["no_weight_mutation"] is True
    assert artifact["unsafe_false_accepts"] == 0
    assert artifact["tests_run"] == tests_run
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES

    for field in mod.RECOMPUTED_FIELDS_FROM_ROWS:
        assert field in artifact["recomputed_fields_from_rows"]
    assert artifact["anti_tautology_controls"]["all_passed"] is True
    assert artifact["anti_tautology_controls"]["failed_controls"] == []

    for row in artifact["memory_evidence_rows"]:
        assert row["value_evidence"]["estimated_verifier_value"] >= 0.0
        assert row["cost_evidence"]["byte_cost"] > 0
        assert row["cost_evidence"]["latency_proxy_ms"] > 0.0
        assert row["cost_evidence"]["energy_proxy_mj"] > 0.0
        assert row["provenance"]["source_artifact"].startswith("results/")
        assert row["trust_label"]
        assert "risk" in row["stale_control"]
        assert "risk" in row["poison_control"]
        assert "available" in row["rollback_evidence"]
        assert row["decision_inputs_measured"]["all_required"] is True


def test_scenario_learn_5381_negative_controls_are_rejected() -> None:
    """SCENARIO-LEARN-5381-NEGATIVE-CONTROLS: bad memory is rejected."""

    rows = mod.build_evidence_rows(mod.load_source_artifact(REPO))
    controls = {row["control_kind"]: row for row in rows if row["control_kind"]}

    assert set(controls) == {
        "negative_high_cost_low_value",
        "negative_poisoned",
        "negative_stale",
    }
    assert controls["negative_stale"]["recomputed_keep_decision"] == "QUARANTINE"
    assert controls["negative_stale"]["recomputed_trust_decision"] == "UNTRUST"
    assert controls["negative_poisoned"]["recomputed_keep_decision"] == "QUARANTINE"
    assert controls["negative_poisoned"]["recomputed_trust_decision"] == "UNTRUST"
    assert controls["negative_high_cost_low_value"]["recomputed_keep_decision"] == "DROP"
    assert all(row["control_passed"] for row in controls.values())


def test_scenario_learn_5381_fail_closed_for_tampered_bad_memory() -> None:
    """SCENARIO-LEARN-5381-FAIL-CLOSED: poisoned accepts block readiness."""

    rows = mod.build_evidence_rows(mod.load_source_artifact(REPO))
    bad_rows = deepcopy(rows)
    poisoned = next(row for row in bad_rows if row["control_kind"] == "negative_poisoned")
    poisoned["recomputed_keep_decision"] = "KEEP"
    poisoned["recomputed_trust_decision"] = "TRUST"
    poisoned["recomputed_share_decision"] = "SHARE"
    poisoned["accepted_as_good"] = True
    poisoned["control_passed"] = False

    artifact = mod.build_artifact(
        root=REPO,
        tests_run=[{"command": "unit exp5381", "outcome": "passed"}],
        evidence_rows=bad_rows,
    )

    mod.validate_artifact(artifact)
    assert artifact["status"] == "blocked"
    assert artifact["budget_memory_corrigendum_clean"] is False
    assert artifact["negative_controls_passed"] == 2
    assert artifact["unsafe_false_accepts"] == 1
    assert artifact["honest_verdict"].startswith("blocked_")
    assert "negative_controls_rejected" in artifact["anti_tautology_controls"][
        "failed_controls"
    ]


def test_scenario_learn_5381_fail_closed_for_missing_cost_evidence() -> None:
    """REQ-LEARN-5381-2: missing row cost evidence blocks clean readiness."""

    rows = mod.build_evidence_rows(mod.load_source_artifact(REPO))
    bad_rows = deepcopy(rows)
    bad_rows[0]["cost_evidence"]["latency_proxy_ms"] = None
    bad_rows[0]["decision_inputs_measured"]["latency_proxy_cost"] = False
    bad_rows[0]["decision_inputs_measured"]["all_required"] = False

    artifact = mod.build_artifact(
        root=REPO,
        tests_run=[{"command": "unit exp5381", "outcome": "passed"}],
        evidence_rows=bad_rows,
    )

    assert artifact["budget_memory_corrigendum_clean"] is False
    assert artifact["keep_share_trust_policy_ready"] is False
    assert "row_level_evidence_present" in artifact["anti_tautology_controls"][
        "failed_controls"
    ]


def test_req_learn_5381_helper_branches_are_covered() -> None:
    """REQ-LEARN-5381-5: validation and deterministic helper branches close."""

    clean_artifact = mod.build_artifact(
        root=REPO,
        tests_run=[{"command": "unit exp5381", "outcome": "passed"}],
    )
    invalid_artifact = deepcopy(clean_artifact)
    invalid_artifact["row_count"] = True

    with pytest.raises(ValueError, match="row_count"):
        mod.validate_artifact(invalid_artifact)

    row = deepcopy(mod.load_source_artifact(REPO)["memory_decision_rows"][0])
    row["provenance"]["verified"] = False
    row["harmful"] = False

    assert mod._keep_decision(row, "UNTRUST", 0) == "DROP"
    assert mod._json_ready(Path("results/example.json")) == "results/example.json"


def test_req_learn_5381_run_writes_and_repository_artifact_is_stable(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-5381-3: run output matches deterministic row recomputation."""

    tests_run = [
        {
            "command": (
                ".venv/bin/pytest "
                "tests/python/test_experiment_5381_budget_memory_tautology_corrigendum_v490.py "
                "-q --no-cov"
            ),
            "outcome": "passed",
        },
        {
            "command": (
                ".venv/bin/coverage run "
                "--include=python/carnot/experiment_5381_budget_memory_tautology_corrigendum_v490.py "
                "-m pytest "
                "tests/python/test_experiment_5381_budget_memory_tautology_corrigendum_v490.py "
                "-q --no-cov -n 0 && .venv/bin/coverage report --fail-under=100"
            ),
            "outcome": "passed",
        },
        {"command": ".venv/bin/pytest tests/python -q", "outcome": "passed"},
    ]
    result_path = tmp_path / mod.RESULT_RELATIVE_PATH

    written = mod.run(root=REPO, result_path=result_path, tests_run=tests_run)
    replay = mod.build_artifact(root=REPO, tests_run=tests_run)
    checked_in = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert json.loads(result_path.read_text(encoding="utf-8")) == written
    assert written == replay
    assert checked_in == replay
    assert checked_in["budget_memory_corrigendum_clean"] is True
