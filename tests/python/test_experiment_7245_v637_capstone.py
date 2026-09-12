"""Tests for REQ-REPORT-7245 and SCENARIO-REPORT-7245-*.

The tests write only to temporary directories. Producer artifacts remain
read-only because the capstone must report their exact historical state.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7245_v637_capstone as capstone


ROOT = Path(__file__).resolve().parents[2]


def _publication_receipt() -> dict[str, object]:
    """Return the stable G1-G4 shape without starting a child process."""

    return {
        "paper_ready": False,
        "unmet_gates": ["G2"],
        "gates": {
            "G1": {"pass": True, "detail": "fixture"},
            "G2": {"pass": False, "detail": "fixture"},
            "G3": {"pass": True, "detail": "fixture"},
            "G4": {"pass": True, "detail": "fixture"},
        },
    }


@pytest.fixture(scope="module")
def built_artifact(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[dict[str, object], Path, Path]:
    """Build once because each build authenticates all twelve producers."""

    directory = tmp_path_factory.mktemp("exp7245-artifact")
    output = directory / "results/experiment_7245_v637_capstone.json"
    checkpoint = directory / "results/checkpoints/experiment_7245_v637_capstone.json"
    artifact = capstone.build_artifact(
        ROOT,
        capstone.RUN_DATE,
        output,
        checkpoint,
        publication_runner=lambda _root: _publication_receipt(),
    )
    return artifact, output, checkpoint


def test_req_report_7245_spec_and_exact_field_principles() -> None:
    """REQ-REPORT-7245: the requirement and prompt field reasons are durable."""

    text = (ROOT / capstone.SPEC_PATH).read_text(encoding="utf-8")
    assert "REQ-REPORT-7245" in text
    assert "SCENARIO-REPORT-7245-ARTIFACT" in text
    assert capstone.REQUIRED_FIELD_PRINCIPLES == capstone.PROMPT_FIELD_PRINCIPLES


def test_scenario_report_7245_contract_reads_frozen_thirteen_rows() -> None:
    """SCENARIO-REPORT-7245-CONTRACT: both frozen contracts agree exactly."""

    contract = capstone.load_contract(ROOT)
    assert contract["errors"] == []
    assert contract["task_ids"] == list(capstone.EXPECTED_TASK_IDS)
    assert [row["task_id"] for row in contract["task_contract_rows"]] == list(
        capstone.EXPECTED_TASK_IDS
    )
    assert all(row["hash_matches_receipt"] for row in contract["source_rows"])


def test_scenario_report_7245_intake_unwraps_only_explicit_wrapper() -> None:
    """SCENARIO-REPORT-7245-INTAKE: domain dictionaries stay dictionaries."""

    wrapped = {"principle": "why", "value": 1}
    domain = {"value": 1, "unit": "joule"}
    assert capstone.unwrap_principle(wrapped) == 1
    assert capstone.unwrap_principle(domain) is domain
    assert capstone.unwrap_principle(False) is False


def test_scenario_report_7245_intake_rejects_structured_quarantine() -> None:
    """SCENARIO-REPORT-7245-INTAKE: a passing numeric gate cannot clear quarantine."""

    contract = capstone.load_contract(ROOT)
    evidence = capstone.load_repository_payloads(ROOT, contract["tasks"])
    assert evidence["exp7237-mention-canary"]["payload"]["mention_canary_ready_score"] == 1
    assert evidence["exp7237-mention-canary"]["quarantine_receipt"]["quarantined"] is True
    assert evidence["exp7237-mention-canary"]["accepted_for_promoted_evidence"] is False
    assert evidence["exp7234-arc-scored-dryrun"]["accepted_for_promoted_evidence"] is False


def test_scenario_report_7245_gates_keep_quarantine_separate() -> None:
    """SCENARIO-REPORT-7245-INTAKE: replay records value and quarantine separately."""

    contract = capstone.load_contract(ROOT)
    evidence = capstone.load_repository_payloads(ROOT, contract["tasks"])
    gates = capstone.replay_gates(contract["tasks"], evidence)
    assert len(gates) == 5
    canary_gate = next(
        row
        for row in gates
        if row["consumer"] == "exp7238-mention-capture"
        and row["upstream"] == "exp7237-mention-canary"
    )
    assert canary_gate["observed_value"] == 1
    assert canary_gate["quarantined"] is True
    assert canary_gate["passed"] is False


def test_scenario_report_7245_claims_recompute_and_abstain() -> None:
    """SCENARIO-REPORT-7245-CLAIMS: clean rows promote and quarantine abstains."""

    contract = capstone.load_contract(ROOT)
    evidence = capstone.load_repository_payloads(ROOT, contract["tasks"])
    claims = capstone.recompute_claims(evidence)
    values = {(row["task_id"], row["metric"]): row for row in claims}
    assert values[("exp7233-contract", "source_contract_complete_score")]["recomputed_value"] == 0
    assert (
        values[("exp7234-arc-scored-dryrun", "scored_dryrun_complete_score")]["abstention"] is True
    )
    assert (
        values[("exp7236-mention-fixture", "mention_fixture_ready_score")]["recomputed_value"] == 1
    )
    assert (
        values[("exp7241-recurrence-learning", "recurrence_learning_value_score")][
            "recomputed_value"
        ]
        == 0
    )
    assert (
        values[("exp7241-recurrence-learning", "recurrence_run_complete_score")]["recomputed_value"]
        == 1
    )
    assert (
        values[("exp7242-recurrence-audit", "recurrence_promotion_score")]["recomputed_value"] == 0
    )
    assert (
        values[("exp7243-native-memory", "native_archive_cost_value_score")]["recomputed_value"]
        == 0
    )
    assert (
        values[("exp7244-board-disposition", "board_disposition_complete_score")][
            "recomputed_value"
        ]
        == 1
    )
    assert all(
        {"unit_id", "arm", "seed", "metric", "metric_value", "error", "abstention"} <= row.keys()
        for row in claims
    )
    assert all(row["matches"] is True for row in claims)


def test_scenario_report_7245_science_keeps_four_questions_independent() -> None:
    """SCENARIO-REPORT-7245-SCIENCE: narrow completion does not become value."""

    contract = capstone.load_contract(ROOT)
    evidence = capstone.load_repository_payloads(ROOT, contract["tasks"])
    questions = capstone.scientific_questions(capstone.recompute_claims(evidence), evidence)
    assert set(questions) == {
        "source_fidelity_and_verification_value",
        "prospective_recurrence_learning",
        "actual_scored_policy_model_use",
        "complete_native_deployment_cost",
    }
    assert questions["source_fidelity_and_verification_value"]["verdict_class"] == "blocked"
    assert questions["prospective_recurrence_learning"]["verdict_class"] == "null"
    assert questions["actual_scored_policy_model_use"]["verdict_class"] == "disqualified"
    assert questions["complete_native_deployment_cost"]["verdict_class"] == "null"
    assert questions["prospective_recurrence_learning"]["llm_verification_evidence"] is False


def test_scenario_report_7245_blocked_build_has_complete_scope(
    built_artifact: tuple[dict[str, object], Path, Path],
) -> None:
    """SCENARIO-REPORT-7245-BLOCKED: a complete matrix can retain a block."""

    artifact, output, checkpoint = built_artifact
    assert artifact["status"] == "complete"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["capstone_complete_score"] == 1
    assert len(artifact["task_contract_rows"]) == 13
    assert len(artifact["evidence_matrix"]) == 13
    assert artifact["evidence_matrix"][-1]["task_id"] == "exp7245-capstone"
    assert artifact["evidence_matrix"][-1]["artifact_sha256"] is None
    assert artifact["rows"] == artifact["recomputed_claim_rows"]
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["current_invocation_counters"] == {"loads": 0, "generations": 0, "calls": 0}
    assert output.is_file() and checkpoint.is_file()
    assert capstone.validate_artifact(artifact, root=ROOT) == []


def test_scenario_report_7245_decisions_preserve_exact_prior_failures(
    built_artifact: tuple[dict[str, object], Path, Path],
) -> None:
    """SCENARIO-REPORT-7245-DECISIONS: retirement signals stay task-specific."""

    decisions = built_artifact[0]["branch_decisions"]
    assert len(decisions) == 13
    assert {row["action"] for row in decisions} <= capstone.BRANCH_ACTIONS
    assert all(row["next_experiment_or_exact_condition"] is not None for row in decisions)
    assert all(row["broad_family_retirement_invented"] is False for row in decisions)
    assert all(
        row["retire_if_same_verdict_applied"] is False
        or row["exact_same_verdict_recurrence"] is True
        for row in decisions
    )


@pytest.mark.parametrize(
    ("field", "replacement", "error"),
    [
        ("capstone_complete_score", 0, "capstone_complete_score"),
        ("verdict_class", "partial", "verdict_class"),
        ("reproducibility_checksum", "sha256:bad", "reproducibility_checksum"),
    ],
)
def test_scenario_report_7245_artifact_detects_top_level_mutations(
    built_artifact: tuple[dict[str, object], Path, Path],
    field: str,
    replacement: object,
    error: str,
) -> None:
    """SCENARIO-REPORT-7245-ARTIFACT: terminal claims fail closed."""

    mutated = deepcopy(built_artifact[0])
    mutated[field] = replacement
    assert error in capstone.validate_artifact(mutated, root=ROOT)


def test_scenario_report_7245_artifact_detects_nested_mutations(
    built_artifact: tuple[dict[str, object], Path, Path],
) -> None:
    """SCENARIO-REPORT-7245-ARTIFACT: roster, claims, hashes, and actions are bound."""

    artifact = built_artifact[0]
    roster = deepcopy(artifact)
    roster["task_contract_rows"][0]["title"] = "changed"
    roster["reproducibility_checksum"] = capstone.reproducibility_checksum(roster)
    assert "task_contract_rows" in capstone.validate_artifact(roster, root=ROOT)
    claim = deepcopy(artifact)
    claim["rows"][0]["recomputed_value"] = 999
    assert "claim_rows" in capstone.validate_artifact(claim, root=ROOT)
    source = deepcopy(artifact)
    path = next(iter(source["source_artifact_hashes"]))
    source["source_artifact_hashes"][path] = "sha256:bad"
    assert "source_artifact_hashes" in capstone.validate_artifact(source, root=ROOT)
    action = deepcopy(artifact)
    action["branch_decisions"][0]["action"] = "invent"
    assert "branch_decisions" in capstone.validate_artifact(action, root=ROOT)


def test_req_report_7245_cli_and_failure_paths(
    tmp_path: Path,
    built_artifact: tuple[dict[str, object], Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-7245: lifecycle and both CLI paths fail closed."""

    output = tmp_path / "result.json"
    checkpoint = tmp_path / "checkpoint.json"
    with pytest.raises(ValueError, match="run date"):
        capstone.build_artifact(
            ROOT, "20260911", output, checkpoint, lambda _root: _publication_receipt()
        )
    with pytest.raises(RuntimeError, match="publication gate"):
        capstone.build_artifact(ROOT, capstone.RUN_DATE, output, checkpoint, lambda _root: {})

    output.write_text(json.dumps(built_artifact[0]), encoding="utf-8")
    assert capstone.main(["--root", str(ROOT), "--artifact-path", str(output), "--validate"]) == 0
    output.write_text("{}", encoding="utf-8")
    assert capstone.main(["--root", str(ROOT), "--artifact-path", str(output), "--validate"]) == 1

    called: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        capstone, "build_artifact", lambda *args, **_kwargs: called.append(args) or {}
    )
    assert capstone.main(["--root", str(ROOT), "--artifact-path", str(output)]) == 0
    assert called and called[0][1] == capstone.RUN_DATE


def test_req_report_7245_read_json_rejects_non_object(tmp_path: Path) -> None:
    """REQ-REPORT-7245: a non-object JSON root cannot become evidence."""

    path = tmp_path / "array.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON root"):
        capstone.read_json(path)
